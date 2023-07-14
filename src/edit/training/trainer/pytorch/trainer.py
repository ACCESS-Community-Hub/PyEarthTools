from __future__ import annotations

import copy
import functools
import logging
import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

import xarray as xr
import matplotlib.pyplot as plt


TENSORBOARD_INSTALLED = True
try:
    import tensorboard
except ModuleNotFoundError:
    TENSORBOARD_INSTALLED = False

from edit.data import Collection
from edit.training.trainer.template import EDITTrainer
from edit.pipeline.templates import DataIterator


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


class EDITLightningTrainer(EDITTrainer):
    """
    Pytorch Lightning Trainer Wrapper.
    """

    def __init__(
        self,
        model: "pl.LightningModule",
        train_data: DataLoader | DataIterator,
        path: str = None,
        valid_data: DataLoader | DataIterator = None,
        find_batch_size: bool = False,
        EarlyStopping: bool = True,
        **kwargs,
    ) -> None:
        """Pytorch Lightning Trainer Wrapper.
        Provides fit, predict overrides to work with edit.training

        Args:
            model (pl.LightningModule):
                Pytorch Lightning Module to use as model
            train_data (DataLoader | DataIterator):
                Dataloader to use for Training,
            path (str, optional):
                Path to save Models and Logs, can also provide `default_root_dir`. Defaults to None
            valid_data (DataIterator, optional):
                Dataloader to use for validation. Defaults to None.
            find_batch_size (bool, optional):
                Auto find the best batch size. Defaults to False.
            EarlyStopping (bool, optional):
                Add in EarlyStopping callback. Defaults to True
            **kwargs (Any, optional):
                All passed to trainer __init__, will intercept 'logger' to update from str if given

        """
        super().__init__(model, train_data=train_data, valid_data=valid_data, path=path)

        import pytorch_lightning as pl
        import torch

        torch.set_float32_matmul_precision("high")

        num_workers = kwargs.pop("num_workers", 0)
        self.num_workers = num_workers
        batch_size = kwargs.pop("batch_size", 1)
        self.batch_size = batch_size

        try:
            train_data = copy.copy(train_data)
        except Exception:
            pass

        self.datamodule = self._get_data(
            batch_size,
            train_data=train_data,
            valid_data=valid_data,
            num_workers=num_workers,
        )

        path = kwargs.pop("default_root_dir", path)
        if path is None:
            raise ValueError(
                f"Path cannot be None, either provide `default_root_dir` or `path`"
            )
        self.path = path
        self.checkpoint_path = (Path(path) / "Checkpoints").resolve()

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            save_top_k=5,
            monitor="step",
            mode="max",
            dirpath=self.checkpoint_path,
            filename="model-{step}-{epoch:02d}",
            every_n_train_steps=500,
        )
        self.callbacks = kwargs.pop("callbacks", [])
        self.callbacks.append(checkpoint_callback)
        
        if EarlyStopping:
            self.callbacks.append(
                pl.callbacks.early_stopping.EarlyStopping(
                    monitor=EarlyStopping if isinstance(EarlyStopping, str) else "valid/loss",
                    min_delta=0.00,
                    patience=4,
                    verbose=False,
                    mode="min",
                )
            )

        self.log_path = Path(path)
        self.logger = None

        if "logger" not in kwargs:
            kwargs["logger"] = "tensorboard"

        if "logger" in kwargs and isinstance(kwargs["logger"], str):
            self.logger = str(kwargs.pop("logger")).lower()
            if self.logger == "tensorboard" and not TENSORBOARD_INSTALLED:
                warnings.warn(
                    f"Logger was set to 'tensorboard' but 'tensorboard' is not installed.\nDefaulting to csv logging"
                )
                kwargs["logger"] = "csv"

            if self.logger == "tensorboard":
                kwargs["logger"] = pl.loggers.TensorBoardLogger(
                    path, name=kwargs.pop("name", None)
                )

            elif self.logger == "csv":
                kwargs["logger"] = pl.loggers.CSVLogger(path, name="csv_logs")
                self.log_path = self.log_path / "csv_logs"

        kwargs["limit_val_batches"] = int(kwargs.pop("limit_val_batches", 10))

        if isinstance(find_batch_size, str):
            find_batch_size = True if find_batch_size == "True" else False
        self.find_batch_size = find_batch_size

        self.trainer_kwargs = kwargs
        self.trainer_kwargs.update(dict(default_root_dir=path))

        self.load_trainer()

    def load_trainer(self, **kwargs):
        import pytorch_lightning as pl

        trainer_kwargs = dict(self.trainer_kwargs)
        trainer_kwargs.update(callbacks=list(self.callbacks), **kwargs)

        self.trainer = pl.Trainer(**trainer_kwargs)

        if self.find_batch_size:
            tuner = pl.tuner.Tuner(self.trainer)
            tuner.scale_batch_size(self.model, mode="power", datamodule=self.datamodule)

    def _get_data(self, *args, **kwargs):
        import pytorch_lightning as pl

        class EDITDataModule(pl.LightningDataModule):
            def __init__(
                self, batch_size, train_data, valid_data=None, num_workers=0
            ) -> None:
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
        if key == "trainer":
            raise AttributeError(f"{self!r} has no attribute {key!r}")
        return getattr(self.trainer, key)

    def fit(self, load: bool = True, *args, **kwargs):
        """Using Pytorch Lightning `.fit` to train model, auto fills model and dataloaders

        Args:
            load (bool | str, optional):
                Whether to load most recent checkpoint file in checkpoint dir, or specified checkpoint file. Defaults to True.
        """

        file = self.load(load)

        self.trainer.fit(
            model=self.model,
            train_dataloaders=kwargs.pop("train_dataloaders", None),
            val_dataloaders=kwargs.pop("valid_dataloaders", None),
            datamodule=self.datamodule,
            # ckpt_path = file,
            *args,
            **kwargs,
        )

    def load(self, file: str | bool = True, only_state: bool = False):
        """Load Model from Checkpoint File.

        Can either be PyTorch Lightning Checkpoint or torch checkpoint.

        Args:
            file (str | bool, optional):
                Path to checkpoint, or boolean to find latest file. Defaults to True.
            only_state (bool, optional):
                If only the model state should be loaded. Defaults to False.
        """
        import torch

        if isinstance(file, bool):
            if (
                file
                and self.checkpoint_path.exists()
                and len(list(Path(self.checkpoint_path).iterdir())) > 0
            ):
                file = max(Path(self.checkpoint_path).iterdir(), key=os.path.getctime)
            else:
                return

        warnings.warn(f"Loading checkpoint: {file}", UserWarning)

        if only_state:
            state = torch.load(file)
            if "state_dict" in state:
                state = state["state_dict"]
                new_state = {}
                for key, variable in state.items():
                    if "model" in key or "net" in key:
                        new_state[
                            key.replace("model.", "").replace("net.", "")
                        ] = variable
                    else:
                        new_state[key] = variable
                state = new_state

            self.model.model.load_state_dict(state)
            return file

        try:
            self.model = self.model.load_from_checkpoint(file)
        except (RuntimeError, KeyError) as e:
            warnings.warn(
                "A KeyError arose when loading from checkpoint, will attempt to load only the model state.",
                RuntimeWarning,
            )
            return self.load(file=file, only_state=True)

        return file

    def _predict_from_data(self, data: np.ndarray | tuple, **kwargs):
        """
        Using the loaded model, and given data make a prediction
        """
        from torch.utils.data import DataLoader, IterableDataset

        class FakeDataLoader(IterableDataset):
            def __init__(self, data: tuple[np.ndarray]):
                self.data = data

            def __iter__(self):
                for data in zip(*self.data):
                    yield data

        batch_size = self.batch_size
        if isinstance(data, (list, tuple)):
            batch_size = min(self.batch_size, len(data[0]))

        fake_data = DataLoader(
            FakeDataLoader(data),
            batch_size=kwargs.pop("batch_size", batch_size),
            # num_workers=kwargs.pop("num_workers", self.num_workers), #Apparently this reproduces data on small scales
        )
        warnings.filterwarnings("ignore", ".*does not have many workers.*")

        prediction = tuple(
            map(
                np.vstack,
                zip(*self.trainer.predict(model=self.model, dataloaders=fake_data)),
            )
        )

        return prediction

    def _find_latest_path(self, path: str | Path, file: bool = True) -> Path:
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

    @functools.wraps(EDITTrainer.predict)
    def predict(self, *args, quiet: bool = False, **kwargs) -> tuple:
        with LoggingContext(quiet):
            if quiet:
                self.load_trainer(enable_progress_bar=False)
            return super().predict(*args, **kwargs)

    @functools.wraps(EDITTrainer.predict_recurrent)
    def predict_recurrent(self, *args, quiet: bool = False, **kwargs) -> tuple:
        with LoggingContext(quiet):
            if quiet:
                self.load_trainer(enable_progress_bar=False)
            return super().predict_recurrent(*args, **kwargs)

    def __flatten_metrics(self, data: pd.DataFrame):
        return data

    def graph(
        self, x: str = "step", y: str = "train/loss", path: str | Path = None
    ) -> plt.Axes:
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
            raise KeyError(
                f"Model was logged with TensorBoard, run `tensorboard --logdir [dir]` in cmd to view"
            )

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
            raise FileNotFoundError(
                f"No metrics.csv files could be found at {path or self.log_path!r}"
            )

        metrics = self.__flatten_metrics(metrics)
        ax = metrics.sort_values(x).plot(y=y, x=x)
        ax.set_xlabel(x)
        ax.set_ylabel(y)

        return ax
