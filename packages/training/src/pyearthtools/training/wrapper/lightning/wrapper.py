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
from pathlib import Path
from typing import Any, Optional
import warnings

import lightning as L
import torch

from pyearthtools.data.utils import parse_path

from pyearthtools.pipeline.controller import Pipeline
from pyearthtools.training.data.lightning import PipelineLightningDataModule
from pyearthtools.training.wrapper.wrapper import ModelWrapper


class LightningWrapper(ModelWrapper):
    """
    Pytorch Lightning ModelWrapper

    For prediction use
        `pyearthtools.training.lightning.Predict`
    For training use
        `pyearthtools.training.lightning.Train`
    """

    model: L.LightningModule
    _default_datamodule = PipelineLightningDataModule
    _loaded_file: str | Path

    _pyearthtools_repr = {
        "expand_attr": ["trainer_kwargs", "pipelines", "splits"],
    }

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
        path: str | Path,
        trainer_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Base pytorch lightning model wrapper

        Args:
            model (L.LightningModule):
                Lightning Model to use for prediction.
            data (dict[str, Pipeline | str | tuple[Pipeline, ...]] | tuple[Pipeline | str , ...] | str | Pipeline | PipelineLightningDataModule):
                Pipeline to use to get data. Will be converted into a `PipelineLightningDataModule`.
            path (str | Path):
                Root path
            trainer_kwargs (Optional[dict[str, Any]], optional):
                Kwargs for `L.Trainer`. Defaults to None.
        """
        super().__init__(model, data)

        self.path = Path(parse_path(path))
        self.datamodule.save(self.path / "DataModule")

        self._trainer_kwargs = trainer_kwargs or dict(kwargs)
        self._trainer_kwargs["default_root_dir"] = path

    @property
    def trainer_kwargs(self):
        return self._trainer_kwargs

    @property
    def trainer(self) -> L.Trainer:
        return self.get_trainer(**self.trainer_kwargs)

    def get_trainer(self, **kwargs) -> L.Trainer:
        """Get Lightning trainer updated with `kwargs`."""
        kwargs = dict(self._trainer_kwargs)
        kwargs.update(kwargs)

        if not kwargs.get("enable_progress_bar", True):
            kwargs["callbacks"] = list(
                t for t in kwargs.pop("callbacks", []) if not t.__class__.__name__ == "TQDMProgressBar"
            )

        return L.Trainer(**kwargs)

    def load(self, file: str | Path, only_state: bool = False):
        """Load Model from Checkpoint File.

        Can either be PyTorch Lightning Checkpoint or torch checkpoint.

        Args:
            file (str, optional):
                Path to checkpoint
            only_state (bool, optional):
                If only the model state should be loaded. Defaults to False.

        Returns:
            (Path | None):
                Path of checkpoint being loaded, or None if no path found.
        """
        file_to_load = file

        warnings.warn(f"Loading checkpoint: {file_to_load}", UserWarning)
        self._loaded_file = file_to_load

        ## If model has implementation, let it handle it.
        if hasattr(self.model, "load"):
            self.model.load(file_to_load)

        if only_state:
            state = torch.load(file_to_load)
            if "state_dict" in state:
                state = state["state_dict"]
                new_state = {}
                for key, variable in state.items():
                    if "model" in key or "net" in key:
                        new_state[key.replace("model.", "").replace("net.", "")] = variable
                    else:
                        new_state[key] = variable
                state = new_state

            self.model.model.load_state_dict(state)  # TODO improve finding of model

        try:
            self.model = self.model.__class__.load_from_checkpoint(file_to_load)
        except (RuntimeError, KeyError):
            warnings.warn(
                "A KeyError arose when loading from checkpoint, will attempt to load only the model state.",
                RuntimeWarning,
            )
            self.load(file=file, only_state=True)

    def save(self, path: str):
        self.trainer.save_checkpoint(path)

    def predict(self, data):
        raise NotImplementedError(
            "This is the base lightning wrapper, use `pyearthtools.training.lightning.Predict` for prediction."
        )
