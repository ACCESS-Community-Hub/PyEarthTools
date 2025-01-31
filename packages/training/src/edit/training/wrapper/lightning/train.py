# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional
import warnings
import importlib.util

import lightning as L
from lightning.pytorch import callbacks
from lightning.pytorch import loggers

from edit.pipeline.controller import Pipeline
from edit.training.data.lightning import PipelineLightningDataModule
from edit.training.wrapper.lightning.wrapper import LightningWrapper
from edit.training.wrapper.train import TrainingWrapper

DEFAULT_CALLBACKS = {
    "Checkpoint": dict(
        monitor="epoch",
        mode="max",
        dirpath="{path}/Checkpoints",
        filename="model-{epoch:02d}-{step:02d}",
        every_n_train_steps=1000,
    )
}


def get_logger(logger: str | bool, path: str, **kwargs):
    """Get logger"""
    tensorboard_installed = importlib.util.find_spec("tensorboard") is not None

    if isinstance(logger, bool):
        if logger:
            logger = "tensorboard"
        else:
            return

    if logger == "tensorboard" and not tensorboard_installed:
        warnings.warn(
            "Logger was set to 'tensorboard' but 'tensorboard' is not installed.\nDefaulting to csv logging ..."
        )
        logger = "csv"

    if logger == "tensorboard":
        return loggers.TensorBoardLogger(path, **kwargs)

    elif logger == "csv":
        return loggers.CSVLogger(path, **kwargs)


def make_callback(callback: str, kwargs: dict[str, Any], **formats):
    """Make Lightning callback from `kwargs` formatted with `formats`."""
    for key, val in kwargs.items():
        if isinstance(val, str):
            for format_str in (format_str for format_str in formats.items() if "{" + f"{format_str[0]}" + "}" in val):
                kwargs[key] = val.replace("{" + f"{format_str[0]}" + "}", str(format_str[1]))

    return getattr(callbacks, callback)(**kwargs)


class LightingTraining(LightningWrapper, TrainingWrapper):

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
        trainer_kwargs: dict[str, Any] | None = None,
        *,
        checkpointing: Optional[dict[str, Any] | tuple[dict[str, Any], ...] | bool] = None,
        logger: Optional[str | dict[str, Any] | bool] = None,
        **kwargs,
    ):
        """
        Pytorch Lightning Training Logger

        Args:
            model (L.LightningModule):
                Lightning Model to use for prediction.
            data (dict[str, Pipeline | str | tuple[Pipeline, ...]] | tuple[Pipeline | str , ...] | str | Pipeline | PipelineLightningDataModule):
                Pipeline to use to get data. Will be converted into a `PipelineLightningDataModule`.
            path (str | Path):
                Root path to save logs and checkpoints into.
            trainer_kwargs (dict[str, Any] | None, optional):
                Kwargs for `L.Trainer`, i.e. `max_epochs`, .... Defaults to None.
            checkpointing (Optional[dict[str, Any] | tuple[dict[str, Any], ...] | bool], optional):
                Checkpointing config, can be True to use default epoch checkpointing, or dictionary of config / tuple of dictionaries. Defaults to None.
            logger (Optional[str | dict[str, Any]], optional):
                Logging config, can be True to use default `tensorboard` config, or dictionary of config. Defaults to None.
        """

        super().__init__(model, data, path, trainer_kwargs, **kwargs)
        self.record_initialisation(ignore="model")

        callbacks: list = self.trainer_kwargs.pop("callbacks", [])

        if checkpointing is not None:
            if isinstance(checkpointing, (tuple, list)):
                callbacks.extend(
                    tuple(map(lambda x: make_callback("ModelCheckpoint", x, path=self.path), checkpointing))
                )
            elif isinstance(checkpointing, bool):
                if checkpointing:
                    callbacks.append(make_callback("ModelCheckpoint", DEFAULT_CALLBACKS["Checkpoint"], path=self.path))
            else:
                callbacks.append(make_callback("ModelCheckpoint", checkpointing, path=self.path))
        self.trainer_kwargs["callbacks"] = callbacks

        if logger is not None:
            if isinstance(logger, dict):
                self.trainer_kwargs["logger"] = get_logger(path=str(self.path), **logger)
            else:
                self.trainer_kwargs["logger"] = get_logger(logger, path=str(self.path))

    @property
    def callbacks(self):
        return self.trainer_kwargs.get("callbacks", [])

    def fit(self, load: bool | str = True, **kwargs):
        """Using Pytorch Lightning `.fit` to train model, auto fills model and dataloaders

        Args:
            load (bool | str, optional):
                Whether to load most recent checkpoint file in checkpoint dir, or specified checkpoint file. Defaults to True.
        """

        if load:
            if isinstance(load, bool):
                latest_path = self._find_latest_path(self.path)
                if latest_path is not None:
                    self.load(latest_path)
            else:
                self.load(load)

        data_config = {}
        if "train_dataloaders" in kwargs:
            data_config["train_dataloaders"] = kwargs.pop("train_dataloaders")
            data_config["val_dataloaders"] = kwargs.pop("valid_dataloaders", None)
        else:
            data_config["train_dataloaders"] = kwargs.pop("datamodule", self.datamodule).train_dataloader()
            try:
                data_config["val_dataloaders"] = kwargs.pop("datamodule", self.datamodule).val_dataloader()
            except ValueError:
                pass

        if hasattr(self, "_loaded_file") and self._loaded_file is not None:
            kwargs["ckpt_path"] = str(self._loaded_file)

        self.trainer.fit(
            model=self.model,
            **data_config,
            **kwargs,
        )

    def _find_latest_path(self, path: str | Path, suffix=".ckpt") -> Path | None:
        """Find latest file or folder inside a given folder

        Args:
            path (str | Path):
                Folder to search in
        Returns:
            (Path):
                Path of latest file or folder
        """
        latest_item = None
        latest_time = -1
        for item in Path(path).rglob(f"*{suffix}"):
            time = max(os.stat(item))
            if not Path(item).suffix == suffix:
                continue

            if time > latest_time:
                latest_time = time
                latest_item = item
        return latest_item
