# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

from __future__ import annotations

import logging
import warnings

from typing import Any

import numpy as np
from pytorch_lightning.core.module import LightningModule

from edit.data.patterns.utils import parse_root_dir

from edit.pipeline.controller import Pipeline
from edit.training.data.datamodule import PipelineDataModule
from edit.training.wrapper.lightning.wrapper import LightningWrapper

PREDICT_KWARGS = {"enable_progress_bar": False, "logger": None}


class LoggingContext:
    """Quiet lightning warnings"""

    def __init__(self, change: bool = True) -> None:
        self.change = change

    def __enter__(self, *args, **kwargs):
        if self.change:
            logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
            warnings.simplefilter(action="ignore", category=UserWarning)

    def __exit__(self, *args, **kwargs):
        if self.change:
            logging.getLogger("pytorch_lightning").setLevel(logging.INFO)
            warnings.simplefilter(action="default", category=UserWarning)


class LightingPrediction(LightningWrapper):
    """
    Pytorch Lightning ModelWrapper with prediction enabled.
    """

    def __init__(
        self,
        model: LightningModule,
        data: Pipeline | PipelineDataModule,
        trainer_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        path, self.temp_dir = parse_root_dir("temp")

        super().__init__(model, data, path, trainer_kwargs, **kwargs)
        self.record_initialisation(ignore="model")

        self.trainer_kwargs.update(PREDICT_KWARGS)

    def predict(self, data):
        """
        Run forward pass with `model` on `data`

        Args:
            data (Any):
                Data to run prediction on

        Returns:
            (Any):
                Predicted data
        """
        if isinstance(data, str):
            data = self.get_sample(data)

        from torch.utils.data import DataLoader, IterableDataset

        class FakeDataLoader(IterableDataset):
            def __init__(self, data):
                self.data = data

            def __iter__(self):
                yield data

        fake_data = DataLoader(
            FakeDataLoader(data),
            batch_size=None,
        )
        warnings.filterwarnings("ignore", ".*does not have many workers.*")
        with LoggingContext():
            predictions_raw = self.trainer.predict(model=self.model, dataloaders=fake_data)
            if predictions_raw is None:
                raise RuntimeError("Predictions were None, cannot parse, try runninng prediction on only one gpu.")

        prediction = np.vstack(predictions_raw)
        return prediction
