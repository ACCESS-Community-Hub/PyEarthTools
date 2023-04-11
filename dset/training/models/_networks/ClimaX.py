# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# credits: https://github.com/ashleve/lightning-hydra-template/blob/main/src/models/mnist_module.py
from typing import Any, Iterable

import torch
from pytorch_lightning import LightningModule

from dset.training.models.architectures.ClimaX.arch import ClimaX
from dset.training.models.architectures.ClimaX.lr_scheduler import (
    LinearWarmupCosineAnnealingLR,
)
from dset.training.models.architectures.ClimaX.metrics import (
    lat_weighted_acc,
    lat_weighted_mse,
    lat_weighted_mse_val,
    lat_weighted_rmse,
    mse,
)
from dset.training.models.architectures.ClimaX.pos_embed import interpolate_pos_embed
from dset.training.models.utils import get_loss


class ClimaX_Model(LightningModule):
    """Lightning module for global forecasting with the ClimaX model.

    Args:
        net (ClimaX): ClimaX model.
        pretrained_path (str, optional): Path to pre-trained checkpoint.
        lr (float, optional): Learning rate.
        beta_1 (float, optional): Beta 1 for AdamW.
        beta_2 (float, optional): Beta 2 for AdamW.
        weight_decay (float, optional): Weight decay for AdamW.
        warmup_epochs (int, optional): Number of warmup epochs.
        max_epochs (int, optional): Number of total epochs.
        warmup_start_lr (float, optional): Starting learning rate for warmup.
        eta_min (float, optional): Minimum learning rate.
    """

    def __init__(
        self,
        model_params: dict,
        in_variables: list[str],
        out_variables: list[str],
        pretrained_path: str = "",
        loss_function: str = "MSELoss",
        loss_kwargs: dict = {},
        lr: float = 5e-4,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        weight_decay: float = 1e-5,
        warmup_epochs: int = 10000,
        max_epochs: int = 200000,
        warmup_start_lr: float = 1e-8,
        eta_min: float = 1e-8,
    ):
        super().__init__()

        net = ClimaX(**model_params)

        self.save_hyperparameters(logger=False)
        self.net = net
        if len(pretrained_path) > 0:
            self.load_pretrained_weights(pretrained_path)

        self.loss_obj = get_loss(loss_function, **loss_kwargs)
        self.in_variables = in_variables
        self.out_variables = out_variables

    def load_pretrained_weights(self, pretrained_path):
        if pretrained_path.startswith("http"):
            checkpoint = torch.hub.load_state_dict_from_url(pretrained_path)
        else:
            checkpoint = torch.load(pretrained_path, map_location=torch.device("cpu"))
        print("Loading pre-trained checkpoint from: %s" % pretrained_path)
        checkpoint_model = checkpoint["state_dict"]
        # interpolate positional embedding
        interpolate_pos_embed(self.net, checkpoint_model, new_size=self.net.img_size)

        state_dict = self.state_dict()
        # checkpoint_keys = list(checkpoint_model.keys())
        for k in list(checkpoint_model.keys()):
            if "channel" in k:
                checkpoint_model[k.replace("channel", "var")] = checkpoint_model[k]
                del checkpoint_model[k]
        for k in list(checkpoint_model.keys()):
            if (
                k not in state_dict.keys()
                or checkpoint_model[k].shape != state_dict[k].shape
            ):
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # load pre-trained model
        msg = self.load_state_dict(checkpoint_model, strict=False)
        print(msg)

    def training_step(self, batch: Any, batch_idx: int):
        x, y, lead_times, variables, out_variables = batch
        x, y = map(lambda x: x.to(dtype=torch.float), (x, y))
        lead_times = lead_times.squeeze(-1)
        loss_dict, (predictions, y) = self.net.forward(
            x, y, lead_times, variables, out_variables, [mse], lat=None
        )

        # loss = self.loss_function(y, predictions)
        # self.log("train/loss", loss, on_step = True, prog_bar = True)
        # return loss

        loss_dict = loss_dict[0]

        for var in loss_dict.keys():
            self.log(
                "train/" + var,
                loss_dict[var],
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
        loss = loss_dict["loss"]

        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        x, y, lead_times, variables, out_variables = batch
        x, y = map(lambda x: x.to(dtype=torch.float), (x, y))
        lead_times = lead_times.squeeze(-1)

        all_loss_dicts, _ = self.net.forward(
            x,
            y,
            lead_times,
            variables,
            out_variables,
            # transform=self.denormalization,
            metric=[mse],
            # lat=self.lat,
            # clim=self.val_clim,
        )

        loss_dict = {}
        for d in all_loss_dicts:
            for k in d.keys():
                loss_dict[k] = d[k]

        for var in loss_dict.keys():
            self.log(
                "val/" + var,
                loss_dict[var],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
        return loss_dict

    def test_step(self, batch: Any, batch_idx: int):
        x, y, lead_times, variables, out_variables = batch
        x, y = map(lambda x: x.to(dtype=torch.float), (x, y))
        lead_times = lead_times.squeeze(-1)

        all_loss_dicts = self.net.evaluate(
            x,
            y,
            lead_times,
            variables,
            out_variables,
            # transform=self.denormalization,
            metrics=[mse],
            # lat=self.lat,
            # clim=self.test_clim,
        )

        loss_dict = {}
        for d in all_loss_dicts:
            for k in d.keys():
                loss_dict[k] = d[k]

        for var in loss_dict.keys():
            self.log(
                "test/" + var,
                loss_dict[var],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
        return loss_dict

    def predict_step(self, batch, batch_idx):
        x, y, lead_times, variables, out_variables = batch
        x, y = map(lambda a: a.to(dtype=torch.float), (x, y))
        loss_dict, (predictions, y) = self.net.forward(
            x, y, lead_times, variables, out_variables, [], lat=None
        )
        return batch, predictions

    def configure_optimizers(self):
        decay = []
        no_decay = []
        for name, m in self.named_parameters():
            if "var_embed" in name or "pos_embed" in name or "time_pos_embed" in name:
                no_decay.append(m)
            else:
                decay.append(m)

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": decay,
                    "lr": self.hparams.lr,
                    "betas": (self.hparams.beta_1, self.hparams.beta_2),
                    "weight_decay": self.hparams.weight_decay,
                },
                {
                    "params": no_decay,
                    "lr": self.hparams.lr,
                    "betas": (self.hparams.beta_1, self.hparams.beta_2),
                    "weight_decay": 0,
                },
            ]
        )

        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            self.hparams.warmup_epochs,
            self.hparams.max_epochs,
            self.hparams.warmup_start_lr,
            self.hparams.eta_min,
        )
        scheduler = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
