# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

from __future__ import annotations

import functools
import logging
import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

import xarray as xr
import matplotlib.pyplot as plt

from edit.data.patterns.utils import parse_root_dir

import edit.training
from edit.training.trainer.template import EDIT_AutoInference, EDIT_Training
from edit.pipeline.templates import DataIterator, DataStep

from edit.utils.context import PrintOnError


class LoggingContext:
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


class Inference(EDIT_AutoInference):
    _loaded_file = None

    def __init__(
        self,
        model: "pytorch_lightning.LightningModule",
        pipeline: DataStep,
        *,
        path: str | Path = "temp",
        batch_size: int = 1,
        num_workers: int = 0,
        **kwargs,
    ):
        super().__init__(pipeline)

        self.model = model
        self.batch_size = batch_size

        self.datamodule = self._get_data(
            batch_size,
            train_data=pipeline,
            num_workers=num_workers,
        )

        self.path, _ = parse_root_dir(path)
        self.pipeline.save(Path(self.path) / "pipeline.yaml")
        self.trainer_kwargs = kwargs

    @functools.cached_property
    def trainer(self) -> "pytorch_lightning.Trainer":
        return self.load_trainer()

    def load_trainer(self, logger=False, **kwargs) -> "pytorch_lightning.Trainer":
        import pytorch_lightning as pl

        trainer_kwargs = dict(self.trainer_kwargs)
        trainer_kwargs.update(dict(default_root_dir=self.path))
        trainer_kwargs.update(kwargs)

        if not kwargs.get("enable_progress_bar", True):
            trainer_kwargs["callbacks"] = list(
                t for t in trainer_kwargs.pop("callbacks", []) if not t.__class__.__name__ == "TQDMProgressBar"
            )

        trainer = pl.Trainer(logger=trainer_kwargs.pop("logger", logger), **trainer_kwargs)

        return trainer

    def load(self, file: str | bool = True, only_state: bool = False) -> Path | str | None:
        """Load Model from Checkpoint File.

        Can either be PyTorch Lightning Checkpoint or torch checkpoint.

        Args:
            file (str | bool, optional):
                Path to checkpoint, or boolean to find latest file. Defaults to True.
            only_state (bool, optional):
                If only the model state should be loaded. Defaults to False.

        Returns:
            (Path | None):
                Path of checkpoint being loaded, or None if no path found.
        """
        import torch

        file_to_load: str | Path

        if isinstance(file, bool):
            if not file:
                return

            if self.checkpoint_path.exists() and len(list(Path(self.checkpoint_path).iterdir())) > 0:
                file_to_load = max(Path(self.checkpoint_path).iterdir(), key=os.path.getmtime)
            else:
                warnings.warn(f"No file located to load from.\nSearched {self.checkpoint_path}", UserWarning)
                return
        else:
            file_to_load = str(file)

        warnings.warn(f"Loading checkpoint: {file_to_load}", UserWarning)
        self._loaded_file = file_to_load

        ## If model has implementation, let it handle it.
        if hasattr(self.model, "load"):
            return self.model.load(file_to_load)

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

            self.model.model.load_state_dict(state)
            return file_to_load

        try:
            self.model = type(self.model).load_from_checkpoint(file_to_load)
        except (RuntimeError, KeyError) as e:
            warnings.warn(
                "A KeyError arose when loading from checkpoint, will attempt to load only the model state.",
                RuntimeWarning,
            )
            return self.load(file=file, only_state=True)

        return file_to_load

    def save(self, path: str, directory: str | Path | None = None):
        directory = directory or self.checkpoint_path
        self.trainer.save_checkpoint(Path(directory) / path)

    def _predict_from_data(self, data: np.ndarray | tuple[np.ndarray, np.ndarray], batch_size=None):
        """
        Using the loaded model, and given data make a prediction
        """
        from torch.utils.data import DataLoader, IterableDataset

        class FakeDataLoader(IterableDataset):
            def __init__(self, data: np.ndarray | tuple[np.ndarray]):
                self.data = data

            def __iter__(self):
                if isinstance(self.data, (list, tuple)):
                    for data in zip(*self.data):
                        yield data
                else:
                    for data in self.data:
                        yield data

        batch_size = batch_size or self.batch_size
        if isinstance(data, (list, tuple)):
            batch_size = min(self.batch_size, len(data[0]))

        fake_data = DataLoader(
            FakeDataLoader(data),
            batch_size=batch_size,
            # num_workers=kwargs.pop("num_workers", self.num_workers), #Apparently this reproduces data on small scales
        )
        warnings.filterwarnings("ignore", ".*does not have many workers.*")

        predictions_raw = self.trainer.predict(model=self.model, dataloaders=fake_data)
        if predictions_raw is None:
            raise RuntimeError(f"Predictions were None, cannot parse, try runninng prediction on only one gpu.")

        if isinstance(predictions_raw[0], (list, tuple)):
            prediction = tuple(
                map(
                    np.vstack,
                    zip(*predictions_raw),
                )
            )
        else:
            prediction = np.vstack(predictions_raw)
        return prediction

    def predict(self, *args, quiet: bool = False, **kwargs):
        with LoggingContext(quiet):
            if quiet:
                self.trainer = self.load_trainer(enable_progress_bar=False)
            return super().predict(*args, **kwargs)

    def recurrent(self, *args, quiet: bool = True, **kwargs):
        with LoggingContext(quiet):
            if quiet:
                self.trainer = self.load_trainer(enable_progress_bar=False)
            return super().recurrent(*args, **kwargs)

    def _get_data(self, *args, **kwargs):
        import pytorch_lightning as pl

        class EDITDataModule(pl.LightningDataModule):
            def __init__(self, batch_size, train_data, valid_data=None, num_workers=0) -> None:
                super().__init__()
                self.batch_size = batch_size
                self.num_workers = num_workers
                self.train_data = train_data
                self.valid_data = valid_data

            def train_dataloader(self):
                from torch.utils.data import DataLoader

                if isinstance(self.train_data, DataLoader):
                    return self.train_data

                return DataLoader(
                    self.train_data,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    pin_memory=False,
                )

            def val_dataloader(self):
                from torch.utils.data import DataLoader

                if isinstance(self.valid_data, DataLoader):
                    return self.valid_data

                return DataLoader(
                    self.valid_data or self.train_data,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    pin_memory=False,
                )

        return EDITDataModule(*args, **kwargs)

    def __getattr__(self, key):
        if key == "trainer" or self.trainer is None:
            raise AttributeError(f"{self.__class__.__name__} has no attribute {key!r}")
        return getattr(self.trainer, key)

    def _find_latest_path(self, path: str | Path, file: bool = True) -> Path | None:
        """Find latest file or folder inside a given folder

        Args:
            path (str | Path):
                Folder to search in
            file (bool, optional):
                Take only files. Defaults to True.

        Returns:
            (Path):
                Path of latest file or folder
        """
        latest_item = None
        latest_time = -1
        for item in Path(path).iterdir():
            if file:
                time = max(os.stat(item))
            else:
                if item.is_file() and not file:
                    continue
                time = max(os.stat(file).st_mtime for file in item.iterdir())
            if time > latest_time:
                latest_time = time
                latest_item = item
        return latest_item

    def __repr__(self):
        repr_string = [super().__repr__()]

        repr_string.append("\nModel:")
        repr_string.append(f"{repr(self.model)}")

        return "\n".join(repr_string)


class Training(Inference, EDIT_Training):
    """
    Pytorch Lightning Trainer Wrapper.
    """

    _loaded_file = None

    def __init__(
        self,
        model: "pl.LightningModule",
        pipeline: DataIterator | "torch.data.DataLoader",
        *,
        path: str,
        valid_data: DataIterator | None = None,
        batch_size: int = 1,
        num_workers: int | None = None,
        find_batch_size: bool = False,
        EarlyStopping: bool | str = True,
        **kwargs,
    ) -> None:
        """Pytorch Lightning Trainer Wrapper.
        Provides fit, predict overrides to work with edit.training

        Args:
            model (pl.LightningModule):
                Pytorch Lightning Module to use as model
            pipeline (DataLoader | DataIterator):
                Dataloader to use for Training,
            path (str):
                Path to save Models and Logs, can also provide `default_root_dir`.
            valid_data (DataIterator, optional):
                Dataloader to use for validation. Defaults to None.
            find_batch_size (bool, optional):
                Auto find the best batch size. Defaults to False.
            EarlyStopping (bool, optional):
                Add in EarlyStopping callback. Defaults to True
            **kwargs (Any, optional):
                All passed to trainer __init__, will intercept 'logger' to update from str if given

        """
        if isinstance(pipeline, DataStep) and "PytorchIterable" not in pipeline.steps:
            pipeline = edit.training.loader.PytorchIterable(pipeline)
            valid_data = edit.training.loader.PytorchIterable(valid_data) if valid_data else valid_data

        super().__init__(model, valid_data or pipeline, path=path, batch_size=batch_size, num_workers=num_workers)

        import pytorch_lightning as pl
        import torch

        torch.set_float32_matmul_precision("high")

        self.datamodule = self._get_data(
            batch_size,
            train_data=pipeline,
            valid_data=valid_data,
            num_workers=num_workers,
        )

        self.checkpoint_path = (Path(self.path) / "Checkpoints").resolve()
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            save_top_k=10,
            monitor="step",
            mode="max",
            dirpath=self.checkpoint_path,
            filename="model-{step}-{epoch:02d}",
            every_n_train_steps=500,
        )

        self.callbacks = kwargs.pop("callbacks", [])
        self.callbacks.append(checkpoint_callback)

        if EarlyStopping and not (isinstance(EarlyStopping, str) and EarlyStopping == "True"):
            self.callbacks.append(
                pl.callbacks.EarlyStopping(
                    monitor=EarlyStopping if isinstance(EarlyStopping, str) else "valid/loss",
                    min_delta=0.02,
                    patience=6,
                    verbose=True,
                    mode="min",
                )
            )

        self.log_path = Path(self.path)
        self.logger = None

        if "logger" not in kwargs:
            kwargs["logger"] = "tensorboard"

        TENSORBOARD_INSTALLED = True
        try:
            import tensorboard
        except ModuleNotFoundError:
            TENSORBOARD_INSTALLED = False

        if "logger" in kwargs and isinstance(kwargs["logger"], str):
            self.logger = str(kwargs.pop("logger")).lower()
            if self.logger == "tensorboard" and not TENSORBOARD_INSTALLED:
                warnings.warn(
                    f"Logger was set to 'tensorboard' but 'tensorboard' is not installed.\nDefaulting to csv logging"
                )
                kwargs["logger"] = "csv"

            if self.logger == "tensorboard":
                kwargs["logger"] = pl.loggers.TensorBoardLogger(self.path, name=kwargs.pop("name", None))

            elif self.logger == "csv":
                kwargs["logger"] = pl.loggers.CSVLogger(self.path, name="csv_logs")
                self.log_path = self.log_path / "csv_logs"

        kwargs["limit_val_batches"] = int(kwargs.pop("limit_val_batches", 10))

        if isinstance(find_batch_size, str):
            find_batch_size = True if find_batch_size == "True" else False
        self.find_batch_size = find_batch_size

        self.trainer_kwargs.update(kwargs)
        self.trainer_kwargs.update(callbacks=list(self.callbacks))

    def load_trainer(self, **kwargs):
        import pytorch_lightning as pl

        trainer = super().load_trainer(**kwargs)

        if self.find_batch_size:
            tuner = pl.tuner.Tuner(trainer)
            tuner.scale_batch_size(self.model, mode="power", datamodule=self.datamodule)
        return trainer

    def fit(self, load: bool = True, **kwargs):
        """Using Pytorch Lightning `.fit` to train model, auto fills model and dataloaders

        Args:
            load (bool | str, optional):
                Whether to load most recent checkpoint file in checkpoint dir, or specified checkpoint file. Defaults to True.
        """

        self.load(load)

        data_config = {}
        if "train_dataloaders" in kwargs:
            data_config["train_dataloaders"] = kwargs.pop("train_dataloaders")
            data_config["valid_dataloaders"] = kwargs.pop("valid_dataloaders", None)
        else:
            data_config = {"datamodule": self.datamodule}

        # with PrintOnError(lambda: f"An error arose getting: {self.pipeline.current_index}"):
        self.trainer.fit(
            model=self.model,
            ckpt_path=str(self._loaded_file),
            **data_config,
            **kwargs,
        )

    def __flatten_metrics(self, data: pd.DataFrame):
        return data

    def graph(self, x: str = "step", y: str = "train/loss", path: str | Path | None = None) -> plt.Axes:
        """Create Plots of metrics file

        Args:
            x (str, optional):
                X axis column. Defaults to step.
            y (str, optional):
                Y axis column. Defaults to train/loss.
            path (str | Path, optional):
                Override for path to search in. Defaults to None

        Raises:
            FileNotFoundError:
                If metrics file/s could not be found

        Returns:
            (plt.Axes):
                Matplotlib Axes of metrics plot
        """
        if self.logger == "tensorboard":
            raise KeyError(f"Model was logged with TensorBoard, run `tensorboard --logdir [dir]` in cmd to view")

        metrics = None
        for folder in Path(path or self.log_path).iterdir():
            if folder.is_file():
                continue

            csv_file = Path(folder) / "metrics.csv"
            if not csv_file.exists():
                continue

            if metrics is None:
                metrics = pd.read_csv(csv_file)
            else:
                metrics = pd.concat([metrics, pd.read_csv(csv_file)])

        if metrics is None:
            raise FileNotFoundError(f"No metrics.csv files could be found at {path or self.log_path!r}")

        metrics = self.__flatten_metrics(metrics)
        ax = metrics.sort_values(x).plot(y=y, x=x)
        ax.set_xlabel(x)
        ax.set_ylabel(y)

        return ax


__all__ = [
    "Inference",
    "Training",
    "LoggingContext",
]
