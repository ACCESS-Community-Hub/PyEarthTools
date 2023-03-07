"""
PhyDNet Specific Model Trainer
"""

import random

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

from dset.training.models.architectures import PhyDNet as PhyDNetArch
from dset.training.models.networks.phydnet.constrain_moments import K2M
from dset.training.modules.loss import extremes


class PhyDNet(pl.LightningModule):
    """
    PhyDNet Specific
    """

    def __init__(
        self,
        model_params,
        loss_function="MSELoss",
        optimiser="Adam",
        lr=0.01,
        num_predictions=12,
        loss_kwargs: dict = {},
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = PhyDNetArch(**model_params).to(dtype=torch.float)

        if loss_function == "extreme":
            self.loss_obj = extremes.ExtremeLoss(**loss_kwargs)
        else:
            self.loss_obj = getattr(torch.nn, loss_function)(**loss_kwargs)

        self.optimiser = optimiser
        self.lr = float(lr)
        self.num_predictions = num_predictions

        self.cache = None

    def configure_optimizers(self):
        return getattr(torch.optim, self.optimiser)(self.parameters(), self.lr)
        # return torch.optim.Adam(self.parameters(), lr=0.02)

    def make_constraints(self):
        if hasattr(self, "_constraints"):
            return self._constraints

        constraints = torch.zeros((49, 7, 7)).to(self.device)
        ind = 0
        for i in range(0, 7):
            for j in range(0, 7):
                constraints[ind, i, j] = 1
                ind += 1
        self._constraints = constraints

        return constraints

    def forward(self, x, first_timestep=False):
        return self.model(x, first_timestep)

    def training_step(self, batch, batch_idx):

        difference_flag = False
        if len(batch) == 3:
            difference_flag = True
            inp, tar, initial = map(lambda x: x.to(dtype=torch.float), batch)
        else:
            inp, tar = map(lambda x: x.to(dtype=torch.float), batch)

        teacher_forcing_ratio = np.maximum(0, 1 - self.current_epoch * 0.003)

        input_length = inp.size(1)
        target_length = tar.size(1)

        loss = 0

        for ei in range(input_length - 1):
            encoder_output, encoder_hidden, output_image, _, _ = self.forward(
                inp[:, ei, :, :, :], (ei == 0)
            )
            loss += self.loss_obj(output_image, inp[:, ei + 1, :, :, :])

        decoder_input = inp[
            :, -1, :, :, :
        ]  # first decoder input = last image of input sequence

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        for di in range(target_length):
            decoder_output, decoder_hidden, output_image, _, _ = self.forward(
                decoder_input
            )
            target = tar[:, di, :, :, :]
            if difference_flag:
                output_image = output_image + initial[:, di]

            loss += self.loss_obj(output_image, target)

            if use_teacher_forcing:
                decoder_input = target  # Teacher forcing
            else:
                decoder_input = output_image

        # Moment regularization  # encoder.phycell.cell_list[0].F.conv1.weight # size (nb_filters,in_channels,7,7)
        k2m = K2M([7, 7]).to(self.device)
        for b in range(0, self.model.phycell.cell_list[0].input_dim):
            filters = self.model.phycell.cell_list[0].F.conv1.weight[
                :, b, :, :
            ]  # (nb_filters,7,7)
            m = k2m(filters.double())
            m = m.float()
            loss += nn.MSELoss()(
                m, self.make_constraints()
            )  # constrains is a precomputed matrix
        if loss == np.nan:
            raise KeyboardInterrupt("Loss went to Nan")

        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        difference_flag = False
        if len(batch) == 3:
            difference_flag = True
            inp, tar, initial = map(lambda x: x.to(dtype=torch.float), batch)
        else:
            inp, tar = map(lambda x: x.to(dtype=torch.float), batch)
        teacher_forcing_ratio = np.maximum(0, 1 - self.current_epoch * 0.003)

        input_length = inp.size(1)
        target_length = tar.size(1)

        loss = 0

        for ei in range(input_length - 1):
            encoder_output, encoder_hidden, output_image, _, _ = self.forward(
                inp[:, ei, :, :, :], (ei == 0)
            )
            loss += self.loss_obj(output_image, inp[:, ei + 1, :, :, :])

        decoder_input = inp[
            :, -1, :, :, :
        ]  # first decoder input = last image of input sequence

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        for di in range(target_length):
            decoder_output, decoder_hidden, output_image, _, _ = self.forward(
                decoder_input
            )
            target = tar[:, di, :, :, :]
            if difference_flag:
                output_image = output_image + initial[:, di]

            loss += self.loss_obj(output_image, target)

            if use_teacher_forcing:
                decoder_input = target  # Teacher forcing
            else:
                decoder_input = output_image

        # Moment regularization  # encoder.phycell.cell_list[0].F.conv1.weight # size (nb_filters,in_channels,7,7)
        k2m = K2M([7, 7]).to(self.device)
        for b in range(0, self.model.phycell.cell_list[0].input_dim):
            filters = self.model.phycell.cell_list[0].F.conv1.weight[
                :, b, :, :
            ]  # (nb_filters,7,7)
            m = k2m(filters.double())
            m = m.float()
            loss += nn.MSELoss()(
                m, self.make_constraints()
            )  # constrains is a precomputed matrix

        self.log(
            "valid/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return loss

    def predict_step(self, batch, batch_idx):
        difference_flag = False
        if len(batch) == 3:
            difference_flag = True
            inp, _, initial = map(lambda x: x.to(dtype=torch.float), batch)
        else:
            inp, _ = map(lambda x: x.to(dtype=torch.float), batch)

        input_length = inp.size(1)
        predictions = []

        for ei in range(max(1, input_length - 1)):
            encoder_output, encoder_hidden, _, _, _ = self.model(
                inp[:, ei, :, :, :], (ei == 0)
            )

        decoder_input = inp[
            :, -1, :, :, :
        ]  # first decoder input= last image of input sequence

        for di in range(self.num_predictions):
            decoder_output, decoder_hidden, output_image, _, _ = self.model(
                decoder_input, False, False
            )
            if difference_flag:
                output_image = output_image + initial[:, di]

            decoder_input = output_image  # <- RNN
            predictions.append(output_image.squeeze(1))

        predictions = torch.stack(predictions, 1)
        if len(predictions.shape) == 4:
            predictions = predictions.unsqueeze(2)
        return inp, predictions
