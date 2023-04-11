import inspect
import os

import numpy as np
import pytorch_lightning as pl
import torch
from einops import rearrange

from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

from dset.training.models.architectures import CuboidTransformerModel
from dset.training.modules.optim import SequentialLR, warmup_lambda

from dset.training.models.utils import get_loss

DEFAULT_OPTIMISER = {
    "total_batch_size": 64,
    "micro_batch_size": 16,
    "seed": 0,
    "method": "adamw",
    "lr": 0.0001,
    "weight_decay": 1.0e-05,
    "gradient_clip_val": 1.0,
    "max_epochs": 100,
    # scheduler
    "lr_scheduler_mode": "cosine",
    "min_lr_ratio": 1.0e-3,
    "warmup_min_lr_ratio": 0.0,
    "warmup_percentage": 0.2,
    # early stopping
    "early_stop": True,
    "early_stop_mode": "min",
    "early_stop_patience": 5,
    "save_top_k": 5,
}


def get_parameter_names(model, forbidden_layer_types):
    r"""
    Returns the names of the model parameters that are not inside a forbidden layer.

    Borrowed from https://github.com/huggingface/transformers/blob/623b4f7c63f60cce917677ee704d6c93ee960b4b/src/transformers/trainer_pt_utils.py#L996
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result


class EarthFormer(pl.LightningModule):
    def __init__(
        self,
        model_params: dict,
        optimiser_params: dict = {},
        total_num_steps: int = 10000,
        loss_function: str = "MSELoss",
        loss_kwargs: dict = {},
        save_dir: str = None,
    ):
        super(EarthFormer, self).__init__()
        self.save_hyperparameters()

        self.optimiser_params = DEFAULT_OPTIMISER
        self.optimiser_params.update(optimiser_params)
        self.total_num_steps = total_num_steps

        num_blocks = len(model_params["enc_depth"])
        extra_params = {}
        if "self_pattern" in model_params:
            extra_params["enc_attn_patterns"] = [
                model_params.pop("self_pattern")
            ] * num_blocks

        if "cross_self_pattern" in model_params:
            extra_params["dec_self_attn_patterns"] = [
                model_params.pop("cross_self_pattern")
            ] * num_blocks

        if "cross_pattern" in model_params:
            extra_params["dec_cross_attn_patterns"] = [
                model_params.pop("cross_pattern")
            ] * num_blocks

        self.model = CuboidTransformerModel(**extra_params, **model_params)

        self.loss_obj = get_loss(loss_function, **loss_kwargs)
        self.in_len = model_params["input_shape"][0]
        self.out_len = model_params["target_shape"][0]

        # if oc_file is not None:
        #     oc_from_file = OmegaConf.load(open(oc_file, "r"))
        # else:
        #     oc_from_file = None
        # oc = self.get_base_config(oc_from_file=oc_from_file)
        # self.save_hyperparameters(oc)
        # self.oc = oc
        # # layout
        # self.in_len = oc.layout.in_len
        # self.out_len = oc.layout.out_len
        # self.layout = oc.layout.layout
        # self.channel_axis = self.layout.find("C")
        # self.batch_axis = self.layout.find("N")
        # self.channels = model_cfg["data_channels"]
        # # dataset
        # self.normalize_sst = oc.dataset.normalize_sst
        # # optimization
        # self.max_epochs = oc.optim.max_epochs
        # self.optim_method = oc.optim.method
        # self.lr = oc.optim.lr
        # self.wd = oc.optim.wd
        # # lr_scheduler
        # self.total_num_steps = total_num_steps
        # self.lr_scheduler_mode = oc.optim.lr_scheduler_mode
        # self.warmup_percentage = oc.optim.warmup_percentage
        # self.min_lr_ratio = oc.optim.min_lr_ratio
        # # logging
        # self.save_dir = save_dir
        # self.logging_prefix = oc.logging.logging_prefix
        # # visualization
        # self.train_example_data_idx_list = list(oc.vis.train_example_data_idx_list)
        # self.val_example_data_idx_list = list(oc.vis.val_example_data_idx_list)
        # self.test_example_data_idx_list = list(oc.vis.test_example_data_idx_list)
        # self.eval_example_only = oc.vis.eval_example_only

        # self.valid_mse = torchmetrics.MeanSquaredError()
        # self.valid_mae = torchmetrics.MeanAbsoluteError()
        # self.test_mse = torchmetrics.MeanSquaredError()
        # self.test_mae = torchmetrics.MeanAbsoluteError()

        # self.configure_save(cfg_file_path=oc_file)

    def configure_optimizers(self):
        # Configure the optimizer. Disable the weight decay for layer norm weights and all bias terms.
        decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters() if n in decay_parameters
                ],
                "weight_decay": self.optimiser_params["weight_decay"],
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if n not in decay_parameters
                ],
                "weight_decay": 0.0,
            },
        ]

        if self.optimiser_params["method"] == "adamw":
            optimizer = torch.optim.AdamW(
                params=optimizer_grouped_parameters,
                lr=self.optimiser_params["lr"],
                weight_decay=self.optimiser_params["weight_decay"],
            )
        else:
            raise NotImplementedError

        warmup_iter = int(
            np.round(self.optimiser_params["warmup_percentage"] * self.total_num_steps)
        )

        if self.optimiser_params["lr_scheduler_mode"] == "cosine":
            warmup_scheduler = LambdaLR(
                optimizer,
                lr_lambda=warmup_lambda(
                    warmup_steps=warmup_iter,
                    min_lr_ratio=self.optimiser_params["warmup_min_lr_ratio"],
                ),
            )
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=(self.total_num_steps - warmup_iter),
                eta_min=self.optimiser_params["min_lr_ratio"]
                * self.optimiser_params["lr"],
            )
            lr_scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_iter],
            )
            lr_scheduler_config = {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            }
        else:
            raise NotImplementedError
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    # def set_trainer_kwargs(self, **kwargs):
    # r"""
    # Default kwargs used when initializing pl.Trainer
    # """
    # checkpoint_callback = ModelCheckpoint(
    #     monitor="valid_loss_epoch",
    #     dirpath=os.path.join(self.save_dir, "checkpoints"),
    #     filename="model-{epoch:03d}",
    #     save_top_k=self.oc.optim.save_top_k,
    #     save_last=True,
    #     mode="min",
    # )
    # callbacks = kwargs.pop("callbacks", [])
    # assert isinstance(callbacks, list)
    # for ele in callbacks:
    #     assert isinstance(ele, Callback)
    # callbacks += [checkpoint_callback, ]
    # if self.oc.logging.monitor_lr:
    #     callbacks += [LearningRateMonitor(logging_interval='step'), ]
    # if self.oc.logging.monitor_device:
    #     callbacks += [DeviceStatsMonitor(), ]
    # if self.oc.optim.early_stop:
    #     callbacks += [EarlyStopping(monitor="valid_loss_epoch",
    #                                 min_delta=0.0,
    #                                 patience=self.oc.optim.early_stop_patience,
    #                                 verbose=False,
    #                                 mode=self.oc.optim.early_stop_mode), ]

    # logger = kwargs.pop("logger", [])
    # tb_logger = pl_loggers.TensorBoardLogger(save_dir=self.save_dir)
    # csv_logger = pl_loggers.CSVLogger(save_dir=self.save_dir)
    # logger += [tb_logger, csv_logger]
    # if self.oc.logging.use_wandb:
    #     wandb_logger = pl_loggers.WandbLogger(project=self.oc.logging.logging_prefix,
    #                                           save_dir=self.save_dir)
    #     logger += [wandb_logger, ]

    # log_every_n_steps = max(1, int(self.oc.trainer.log_step_ratio * self.total_num_steps))
    # trainer_init_keys = inspect.signature(Trainer).parameters.keys()
    # ret = dict(
    #     callbacks=callbacks,
    #     # log
    #     logger=logger,
    #     log_every_n_steps=log_every_n_steps,
    #     track_grad_norm=self.oc.logging.track_grad_norm,
    #     # save
    #     default_root_dir=self.save_dir,
    #     # ddp
    #     accelerator="gpu",
    #     # strategy="ddp",
    #     strategy=ApexDDPStrategy(find_unused_parameters=False, delay_allreduce=True),
    #     # optimization
    #     max_epochs=self.oc.optim.max_epochs,
    #     check_val_every_n_epoch=self.oc.trainer.check_val_every_n_epoch,
    #     gradient_clip_val=self.oc.optim.gradient_clip_val,
    #     # NVIDIA amp
    #     precision=self.oc.trainer.precision,
    # )
    # oc_trainer_kwargs = OmegaConf.to_object(self.oc.trainer)
    # oc_trainer_kwargs = {key: val for key, val in oc_trainer_kwargs.items() if key in trainer_init_keys}
    # ret.update(oc_trainer_kwargs)
    # ret.update(kwargs)
    # return ret

    def forward(self, batch):
        r"""
        inp
            shape = (N, in_len+out_len, lat, lon)
        tar
            shape = (N, out_len+NINO_WINDOW_T-1)
        """
        inp, tar = map(lambda x: x.to(dtype=torch.float), batch)

        pred_seq = self.model(inp)
        loss = self.loss_obj(pred_seq, tar)
        return pred_seq, loss, inp, tar, tar.float()

    def training_step(self, batch, batch_idx):
        pred_seq, loss, in_seq, target_seq, nino_target = self.forward(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        pred_seq, loss, in_seq, target_seq, nino_target = self.forward(batch)

        self.log("valid/loss", loss, on_step=True, on_epoch=False)

        return loss

    def predict_step(self, batch, batch_idx):
        pred_seq, loss, in_seq, target_seq, nino_target = self.forward(batch)
        return in_seq, pred_seq

    # def validation_epoch_end(self, outputs):
    #     valid_mse = self.valid_mse.compute()
    #     valid_mae = self.valid_mae.compute()
    #     nino_preds_list, nino_target_list = map(list, zip(*outputs))
    #     nino_preds_list = torch.cat(nino_preds_list, dim=0)
    #     nino_target_list = torch.cat(nino_target_list, dim=0)
    #     valid_acc, valid_nino_rmse = compute_enso_score(
    #         y_pred=nino_preds_list, y_true=nino_target_list,
    #         acc_weight=None)
    #     valid_weighted_acc, _ = compute_enso_score(
    #         y_pred=nino_preds_list, y_true=nino_target_list,
    #         acc_weight="default")
    #     valid_acc /= self.nino_out_len
    #     valid_nino_rmse /= self.nino_out_len
    #     valid_weighted_acc /= self.nino_out_len
    #     valid_loss = -valid_acc

    #     self.log('valid_loss_epoch', valid_loss, prog_bar=True, on_step=False, on_epoch=True)
    #     self.log('valid_mse_epoch', valid_mse, prog_bar=True, on_step=False, on_epoch=True)
    #     self.log('valid_mae_epoch', valid_mae, prog_bar=True, on_step=False, on_epoch=True)
    #     self.log('valid_corr_nino3.4_epoch', valid_acc, prog_bar=True, on_step=False, on_epoch=True)
    #     self.log('valid_corr_nino3.4_weighted_epoch', valid_weighted_acc, prog_bar=True, on_step=False, on_epoch=True)
    #     self.log('valid_nino_rmse_epoch', valid_nino_rmse, prog_bar=True, on_step=False, on_epoch=True)
    #     self.valid_mse.reset()
    #     self.valid_mae.reset()
