# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# reference: https://github.com/NVlabs/AFNO-transformer

# This file extends and is significantly based
# on the code from https://github.com/nci/FourCastNeXt which is made available
# under the Apache 2.0 license. That repository in turn extends the code from
# https://github.com/NVlabs/FourCastNet/, released under the BSD 3-Clause license.
# The FourCastNet model is described in detail at https://arxiv.org/abs/2202.11214.
# The FourCastNeXt model is described in detail at https://arxiv.org/abs/2401.05584,
# and a version of the FourCastNeXt code is bundled, adapted for compatibility
# and maintained within the PyEarthTools repository so it can continue to be a useful
# reference implementation and learning aid.

import lightning as pl
import numpy as np
import sys
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_optimizer import Lamb

from fourcastnext.architecture.afnonet import AFNONet

from pyearthtools.training.modules import get_loss


torch.set_float32_matmul_precision("medium")


class FourCastNextLM(pl.LightningModule):
    def __init__(
        self,
        model_params: dict = {},
        *,
        base_lr=1e-3,
        grad_accum_schedule=None,
        precision=32,
        loss_function: str = "L1Loss",
        loss_kwargs: dict = {},
    ):
        """
        FourCastNeXt model

        Expects data in (B,T,C,H,W, B,T_1,C,H,W)
        With the first element being the input and the second the target
        `T_1` can be any length thus indicating training up to that rollout.

        Args:
            model_params (dict):
                Model params to pass to `AFNONet`
            base_lr (_type_, optional):
                Base learning rate. Defaults to 1e-3.
            grad_accum_schedule (_type_, optional):
                _description_. Defaults to None.
            precision (int, optional):
                Float precision. Defaults to 32.
            loss_function (str, optional):
                Loss function to use. Defaults to "L1Loss".
            loss_kwargs (dict, optional):
                Kwargs to pass to the loss function. Defaults to {}.
        """
        super().__init__()
        self.save_hyperparameters()

        print("Setting up PyTorch Lightning Model")
        print(model_params)

        self.spatial_size = model_params.get("img_size", (128, 128))
        self.out_channels = model_params.get("out_channels", 10)

        grid = self.setup_grid()
        self.register_buffer("grid", grid, persistent=False)

        if precision == 16:
            self._dtype = torch.float16
        else:
            self._dtype = torch.float

        self.model = AFNONet(**model_params).to(dtype=self._dtype)
        self.model_params = model_params

        self.loss_obj = get_loss(loss_function, **loss_kwargs).to(dtype=self._dtype)

    # Can implement def load(file_to_load) here if wanted

    def forward(self, x, net):

        value, flow = net(x.to(dtype=self._dtype))

        x = x[:, -self.out_channels :]  # B, [t-1, t], H, W
        B, C, H, W = x.shape
        warp_coords = self.grid.repeat(B * C, 1, 1, 1) + flow.view(B * C, H, W, 2)
        x = x.reshape(B * C, 1, H, W)
        warped_x = F.grid_sample(x, warp_coords, mode="bilinear", align_corners=True)
        warped_x = warped_x.reshape(B, C, H, W)

        return warped_x + value

    def setup_grid(self):
        h, w = self.spatial_size
        xgrid = torch.arange(w)
        xgrid = 2 * xgrid / (w - 1) - 1

        ygrid = torch.arange(h)
        ygrid = 2 * ygrid / (h - 1) - 1
        coords = torch.meshgrid(ygrid, xgrid, indexing="ij")
        coords = torch.stack(coords[::-1], dim=0).float()
        return coords.permute(1, 2, 0).to(dtype=self._dtype)

    def get_teacher(self, device):
        model_name = "_teacher_model"
        if model_name not in sys.modules:
            teacher = AFNONet(**self.model_params).to(device=device, dtype=self._dtype)

            teacher.load_state_dict(self.model.state_dict())
            sys.modules[model_name] = teacher.eval()

        return sys.modules[model_name]

    def training_step(self, batch, batch_idx):
        """
        B T C H W
        """
        inp, tar = map(lambda x: x.to(dtype=self._dtype), batch)
        B, T, C, H, W = inp.shape
        if T > 1:
            input0 = inp[:, 0]
            input1 = inp[:, 1]
        else:
            input1 = inp[:, 0]

        # new_batch = self.preprocess(batch, is_training=True)
        n_pred_steps = tar.shape[1]
        target = tar[:, 0]

        # target = new_batch['target']
        # n_pred_steps = new_batch['n_pred_steps']

        if n_pred_steps > 1:
            if batch_idx % 2 == 0:
                output1 = self.forward(input1, self.model)
                total_loss = self.loss_obj(output1, target)
            else:
                tar_idx = 0
                with torch.inference_mode():
                    teacher = self.get_teacher(input1.device)
                    for i in range(np.random.choice(np.arange(1, n_pred_steps))):
                        output0 = self.forward(input1, teacher)
                        input1[:, : self.out_channels, ...] = input1[:, -self.out_channels :, ...]
                        input1[:, -self.out_channels :, ...] = output0
                        tar_idx = i
                output0 = self.forward(input1, self.model)
                total_loss = self.loss_obj(output0, tar[:, tar_idx + 1])
                self.log(
                    "teacher/index",
                    torch.Tensor([i]),
                    on_step=True,
                    prog_bar=False,
                )

        else:
            output1 = self.forward(input1, self.model)
            total_loss = self.loss_obj(output1, target)

        self.log(
            "train_loss",
            total_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        if self.trainer.is_global_zero and (
            batch_idx == 0
            or (self.trainer.global_step + 1) % self.trainer.log_every_n_steps == 0
            or self.trainer.global_step + 1 == self.trainer.max_steps - 1
        ):
            total_loss_val = total_loss.item()

            if not np.isfinite(total_loss_val):
                raise Exception(f"loss is not finite, loss={total_loss_val}, step={self.trainer.global_step+1}")

        self.schedule_accumulate_grads(self.trainer.global_step)

        return total_loss

    def schedule_accumulate_grads(self, step):
        if self.hparams.grad_accum_schedule is None:
            return

        accumulate_grad_batches = 1
        for sched_step in reversed(self.hparams.grad_accum_schedule.keys()):
            if step >= sched_step:
                accumulate_grad_batches = self.hparams.grad_accum_schedule[sched_step]
                break

        self.trainer.accumulate_grad_batches = accumulate_grad_batches

    def configure_optimizers(self):
        net_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = Lamb(net_params, lr=self.hparams.base_lr, weight_decay=self.hparams.base_lr**2)
        scheduler = CosineAnnealingLR(optimizer, self.trainer.max_steps, eta_min=self.hparams.base_lr * 0.1)
        return [
            optimizer,
        ], [
            scheduler,
        ]

    def validation_step(self, batch, batch_idx):
        # batch, batch_idx, _ = batch  # Issue caused by dataloader needed to replicate training dataloader
        # batch, batch_idx = batch  # Issue caused by dataloader needed to replicate training dataloader
        inp, tar = map(lambda x: x.to(dtype=self._dtype), batch)
        target = tar[:, 0]
        B, T, C, H, W = inp.shape
        if T > 1:
            input0 = inp[:, 0]
            input1 = inp[:, 1]
        else:
            input1 = inp[:, 0]

        output1 = self.forward(input1, self.model)
        total_loss = self.loss_obj(output1, target)
        self.log(
            "valid_loss",
            total_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return total_loss

    def predict_step(self, batch, batch_idx):

        try:
            # inp = batch.to(dtype=self._dtype)
            inp, tar = map(lambda x: x.to(dtype=self._dtype), batch)
        except Exception:
            inp = batch.to(dtype=self._dtype)

        B, T, C, H, W = inp.shape
        if T > 1:
            input0 = inp[:, 0]
            input1 = inp[:, 1]
        else:
            input1 = inp[:, 0]

        # FIXME: this wasn't autodetecting correctly. Current set to match tutorial.
        # If it's too hard to autodetect, set it in the init
        n_pred_steps = 4  # tar.shape[1]

        if n_pred_steps == 1:
            predictions = self.forward(input1, self.model)
            predictions = predictions.unsqueeze(1)
            return predictions

        predictions = []

        for i in range(n_pred_steps):
            output0 = self.forward(input1, self.model)
            input1[:, : self.out_channels, ...] = input1[:, -self.out_channels :, ...]
            input1[:, -self.out_channels :, ...] = output0
            predictions.append(output0)

        predictions = torch.stack(predictions, dim=1)

        return predictions

        # assert len(batch['input0'].shape) == 4

        # im_height, im_width = batch['input0'].shape[-2:]
        # roi_height, roi_width = self.hparams.spatial_size

        # out_h_stride = int(im_height / 2.0 + 0.5)
        # out_w_stride = int(im_width / 2.0 + 0.5)

        # assert out_h_stride <= roi_height
        # assert out_w_stride <= roi_width

        # n_steps = batch['n_pred_steps']

        # preds = torch.zeros(n_steps, self.hparams.out_channels, im_height, im_width, dtype=batch['input0'].dtype)

        # for hh in range(0, im_height, roi_height):
        #   for ww in range(0, im_width, roi_width):
        #     if hh + roi_height >= im_height:
        #       in_hh = im_height - roi_height
        #     else:
        #       in_hh = hh

        #     if ww + roi_width >= im_width:
        #       in_ww = im_width - roi_width
        #     else:
        #       in_ww = ww

        #     roi_input = batch['input0'][..., in_hh:in_hh+roi_height, in_ww:in_ww+roi_width].clone()

        #     roi_batch = self.preprocess({'input0': roi_input})

        #     inp = roi_batch['input0'].clone()

        #     for ii in range(n_steps):
        #       output = self.forward(inp, self.model)
        #       inp[:, :self.out_channels, ...] = inp[:, -self.out_channels:, ...]
        #       inp[:, -self.out_channels:, ...] = output

        #       unnormalized_out = output * self.stds + self.means
        #       unnormalized_out = unnormalized_out.to(device='cpu')

        #       if hh == 0 and ww == 0:
        #         preds[ii, :, :out_h_stride, :out_w_stride] = unnormalized_out[0, :, :out_h_stride, :out_w_stride]
        #       elif hh == 0 and ww == roi_width:
        #         preds[ii, :, :out_h_stride, -out_w_stride:] = unnormalized_out[0, :, :out_h_stride, -out_w_stride:]
        #       elif hh == roi_height and ww == 0:
        #         preds[ii, :, -out_h_stride:, :out_w_stride] = unnormalized_out[0, :, -out_h_stride:, :out_w_stride]
        #       elif hh == roi_height and ww == roi_width:
        #         preds[ii, :, -out_h_stride:, -out_w_stride:] = unnormalized_out[0, :, -out_h_stride:, -out_w_stride:]

        # return preds
