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


from __future__ import annotations

from functools import cached_property
import logging
import warnings

from typing import Any

import numpy as np
import lightning as L

from pyearthtools.data.patterns.utils import parse_root_dir

from pyearthtools.pipeline.controller import Pipeline
from pyearthtools.training.data.lightning import PipelineLightningDataModule
from pyearthtools.training.wrapper.lightning.wrapper import LightningWrapper

PREDICT_KWARGS = {"enable_progress_bar": False, "logger": None}


class LoggingContext:
    """Quiet lightning warnings"""

    def __init__(self, change: bool = True) -> None:
        self.change = change

    def __enter__(self, *args, **kwargs):
        if self.change:
            logging.getLogger("lightning").setLevel(0)
            warnings.simplefilter(action="ignore", category=UserWarning)

    def __exit__(self, *args, **kwargs):
        if self.change:
            logging.getLogger("lightning").setLevel(logging.INFO)
            warnings.simplefilter(action="default", category=UserWarning)


class LightingPrediction(LightningWrapper):
    """
    Pytorch Lightning ModelWrapper with prediction enabled.
    """

    def __init__(
        self,
        model: L.LightningModule,
        data: (
            dict[str, Pipeline | str | tuple[Pipeline, ...]]
            | tuple[Pipeline | str, ...]
            | str
            | Pipeline
            | PipelineLightningDataModule
        ),
        trainer_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        """
        Lightning Prediction Wrapper

        Allows for prediction with a pytorch lightning model upon `pyearthtools` data.

        Args:
            model (L.LightningModule):
                Lightning Model to use for prediction.
            data (dict[str, Pipeline | str | tuple[Pipeline, ...]] | tuple[Pipeline | str , ...] | str | Pipeline | PipelineLightningDataModule):
                Pipeline to use to get data. Will be converted into a `PipelineLightningDataModule`.
            trainer_kwargs (dict[str, Any] | None, optional):
                Kwargs to provide to Lightning Trainer. Defaults to None.
        """
        path, self.temp_dir = parse_root_dir("temp")
        logging.getLogger("lightning").setLevel(0)

        super().__init__(model, data, path, trainer_kwargs, **kwargs)
        self.record_initialisation(ignore="model")

        self.trainer_kwargs.update(PREDICT_KWARGS)

    @cached_property
    def trainer(self) -> L.Trainer:
        return super().trainer

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
            data = self.get_sample(data, fake_batch_dim=True)

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
                raise RuntimeError("Predictions were None, cannot be parsed, try running prediction on only one gpu.")

        prediction = np.vstack(predictions_raw)
        return prediction
