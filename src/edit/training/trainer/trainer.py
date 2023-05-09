from __future__ import annotations

import copy
from logging import warning
import os
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import xarray as xr
import matplotlib.pyplot as plt

TENSORBOARD_INSTALLED = True
try:
    import tensorboard
except ModuleNotFoundError:
    TENSORBOARD_INSTALLED = False

from torch.utils.data import DataLoader, IterableDataset

from edit.data import Collection
from edit.training.trainer.template import EDITTrainer
from edit.training.data.templates import DataIterator


class EDITTrainerWrapper(EDITTrainer):
    """
    Pytorch Lightning Trainer Wrapper.
    """

    def __init__(
        self,
        model: pl.LightningModule,
        train_data: Union[DataLoader, DataIterator],
        path: str = None,
        valid_data: Union[DataLoader, DataIterator] = None,
        **kwargs,
    ) -> None:
        """Pytorch Lightning Trainer Wrapper.
        Provides fit, predict overrides to work with edit.training

        Args:
            model (pl.LightningModule):
                Pytorch Lightning Module to use as model
            train_data (Union[DataLoader, DataIterator]):
                Dataloader to use for Training,
            path (str, optional):
                Path to save Models and Logs, can also provide `default_root_dir`. Defaults to None
            valid_data (DataIterator, optional):
                Dataloader to use for validation. Defaults to None.
            **kwargs (Any, optional):
                All passed to trainer __init__, will intercept 'logger' to update from str if given

        """
        self.model = model

        num_workers = kwargs.pop("num_workers", 0)
        self.num_workers = num_workers
        batch_size = kwargs.pop("batch_size", 1)
        self.batch_size = batch_size

        try:
            train_data = copy.copy(train_data)
        except Exception:
            pass

        self.train_iterator = train_data
        if not isinstance(train_data, DataLoader):
            self.train_data = DataLoader(
                train_data,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
            )
        else:
            self.train_data = train_data

        self.valid_iterator = valid_data
        self.valid_data = None
        if valid_data:
            if not isinstance(valid_data, DataLoader):
                self.valid_data = DataLoader(
                    valid_data,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory=True,
                )
            else:
                self.valid_data = valid_data

        path = kwargs.pop("default_root_dir", path)
        if path is None:
            raise ValueError(
                f"Path cannot be None, either provide `default_root_dir` or `path`"
            )
        self.path = path
        self.checkpoint_path = (Path(path) / "Checkpoints").resolve()

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            save_top_k=10,
            monitor="train/loss",
            dirpath=self.checkpoint_path,
            mode="min",
            filename="{epoch:02d}",
            every_n_train_steps=2500,
        )
        callbacks = kwargs.pop("callbacks", [])
        callbacks.append(checkpoint_callback)

        self.log_path = Path(path)
        self.logger = None

        if "logger" not in kwargs:
            kwargs["logger"] = "tensorboard"

        if "logger" in kwargs and isinstance(kwargs["logger"], str):
            self.logger = str(kwargs.pop("logger")).lower()
            if self.logger == "tensorboard" and not TENSORBOARD_INSTALLED:
                warning.warn(f"Logger was set to 'tensorboard' but 'tensorboard' is not installed.\nDefaulting to csv logging")
                kwargs["logger"] = "csv"

            
            if self.logger == "tensorboard":
                kwargs["logger"] = pl.loggers.TensorBoardLogger(
                    path, name=kwargs.pop("name", None)
                )

            elif self.logger == "csv":
                kwargs["logger"] = pl.loggers.CSVLogger(path, name="csv_logs")
                self.log_path = self.log_path / "csv_logs"

        kwargs["limit_val_batches"] = int(kwargs.pop("limit_val_batches", 10))

        self.trainer = pl.Trainer(
            default_root_dir=path,
            callbacks=callbacks,
            **kwargs,
        )

    def __getattr__(self, key):
        if key == "trainer":
            raise AttributeError(f"{self!r} has no attribute {key!r}")
        return getattr(self.trainer, key)

    def fit(self, resume: bool = True, *args, **kwargs):
        """Using Pytorch Lightning `.fit` to train model, auto fills model and dataloaders

        Args:
            resume (bool | str, optional): 
                Whether to resume most recent checkpoint file in checkpoint dir, or specified checkpoint file. Defaults to True.
        """        

        if isinstance(resume, str):
            kwargs["ckpt_path"] = resume

        elif resume and Path(self.checkpoint_path).exists() and "ckpt_path" not in kwargs:
            kwargs["ckpt_path"] = max(
                Path(self.checkpoint_path).iterdir(), key=os.path.getctime
            )

        self.trainer.fit(
            model=self.model,
            train_dataloaders=kwargs.pop("train_dataloaders", self.train_data),
            val_dataloaders=kwargs.pop("valid_dataloaders", self.valid_data),
            *args,
            **kwargs,
        )

    def load(self, file: str | bool = True, only_state: bool = False):
        """Load Model from Checkpoint File.

        Can either be PytorchLightning Checkpoint or torch checkpoint.

        Args:
            file (str | bool, optional): 
                Path to checkpoint, or boolean to find latest file. Defaults to True.
            only_state (bool, optional): 
                If only the model state should be loaded. Defaults to False.
        """        

        if isinstance(file, bool) and file:
            file = max(Path(self.checkpoint_path).iterdir(), key=os.path.getctime)

        print(f"Loading checkpoint: {file}")
        if only_state:
            state = torch.load(file)
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
            return
        self.model = self.model.load_from_checkpoint(file)

    def _predict_from_data(self, data : np.ndarray, **kwargs):
        """
        Using the loaded model, and given data make a prediction
        """
        class FakeDataLoader(IterableDataset):
            def __init__(self, data: tuple[np.ndarray]):
                self.data = data

            def __iter__(self):
                for data in zip(*self.data):
                    yield data

        batch_size = self.batch_size
        if isinstance(data, list):
            batch_size = min(self.batch_size, len(data[0]))

        fake_data = DataLoader(
            FakeDataLoader(data),
            batch_size=kwargs.pop("batch_size", batch_size),
            # num_workers=kwargs.pop("num_workers", self.num_workers), #Apparently this reproduces data on small scales
        )

        prediction = tuple(
            map(
                np.vstack,
                zip(*self.trainer.predict(model=self.model, dataloaders=fake_data)),
            )
        )

        return prediction

    def predict(
        self,
        index: str,
        undo: bool = True,
        data_iterator: DataIterator = None,
        resume: bool | str = True,
        only_state: bool = True,
        **kwargs,
    ) -> tuple[np.array] | tuple[xr.Dataset]:
        """Pytorch Lightning Prediction Override

        Uses [edit.training][edit.training.data] DataStep to get data at given index.
        Can automatically try to rebuild the xarray Dataset.

        !!! Warning
            If number of patches is not divisible by the `batch_size`, issues may arise.
            Solution: batch_size = 1

        Args:
            index (str): 
                Index to get from the validation or training data loader or given `data_iterator`
            undo (bool, optional): 
                Rebuild Data using DataStep.undo. Defaults to True.
            data_iterator (DataIterator, optional): 
                Override for DataStep to us. Defaults to None.
            resume (bool | str, optional): 
                Path to checkpoint, or boolean to find latest file. Defaults to True.
            only_state (bool, optional): 
                Load only the model state. Defaults to True.

        Returns:
            (tuple[np.array] | tuple[xr.Dataset]): 
                Either xarray datasets or np arrays, [truth data, predicted data]
        """    
        data_source = data_iterator or self.valid_iterator or self.train_iterator
        data = data_source[index]

        if isinstance(resume, str):
            self.load(
                resume,
                only_state=only_state,
            )

        elif resume and Path(self.checkpoint_path).exists():
            self.load(
                True,
                only_state=only_state,
            )

        prediction = self._predict_from_data(data, **kwargs)

        truth_data = None
        if undo:
            prediction = data_source.undo(prediction)
            if isinstance(prediction, (tuple, list)):
                truth_data = prediction[0]
                prediction = prediction[-1]

            if "Coordinate 1" in prediction:
                prediction = prediction.rename({"Coordinate 1": "time"})
            if hasattr(data_source, "rebuild_time"):
                prediction = data_source.rebuild_time(prediction, index)

            return Collection(truth_data or data_source.undo(data)[1], prediction)
        return Collection(data[1], prediction[1])

    def predict_recurrent(
        self,
        start_index: str,
        recurrence: int,
        data_iterator: DataIterator = None,
        resume: bool = True,
        only_state: bool = True,
        truth_step: int = 0,
        **kwargs,
    ) -> tuple[np.array] | tuple[xr.Dataset]:
        """Uses [predict][edit.training.trainer.EDITTrainerWrapper.predict] to predict timesteps and then feed back through recurrently.

        Uses [edit.training][edit.training.data] DataStep to get data at given index.
        Can automatically try to rebuild the xarray Dataset.

        !!! Warning
            If number of patches is not divisible by the `batch_size`, issues may arise.
            Solution: batch_size = 1

        Args:
            start_index (str):
                Starting Index of Prediction
            recurrence (int):
                Number of times to recur
            data_iterator (DataIterator, optional):
                Override for initial data retrieval. Defaults to None.
            resume (bool, optional):
                Resume from checkpoint. Defaults to True.
            only_state (bool, optional):
                Resume only_state. Defaults to True.
            truth_step (int, optional):
                Data Pipeline step to use to retrieve Truth data. Defaults to 0
        Returns:
            (tuple[np.array] | tuple[xr.Dataset]):
                Either xarray datasets or np arrays, [truth data, predicted data]
        """
        data_source = data_iterator or self.valid_iterator or self.train_iterator
        data = list(data_source[start_index])

        if isinstance(resume, str):
            self.load(
                resume,
                only_state=only_state,
            )

        elif resume and Path(self.checkpoint_path).exists():
            self.load(
                True,
                only_state=only_state,
            )

        predictions = []
        index = start_index

        for i in range(recurrence):
            input_data = None
            prediction = self._predict_from_data(data, **kwargs)

            fixed_predictions = data_source.undo(prediction)

            if isinstance(fixed_predictions, (tuple, list)):
                input_data = fixed_predictions[0]
                fixed_predictions = fixed_predictions[-1]

            if "Coordinate 1" in fixed_predictions:
                fixed_predictions = fixed_predictions.rename({"Coordinate 1": "time"})
            if hasattr(data_source, "rebuild_time"):
                fixed_predictions = data_source.rebuild_time(
                    fixed_predictions,
                    index,
                    offset=1 if i >= 1 else 0,
                )

            predictions.append(fixed_predictions)

            # data[0] = fixed_predictions
            # #data.reverse()
            input_data = input_data or data_source.undo(data)[0]
            new_input = xr.merge((input_data, fixed_predictions)).isel(
                time=slice(-1 * len(input_data.time), None)
            )
            index = new_input.time.values[-1]
            data[0] = data_source.apply(new_input)

        predictions = xr.merge(predictions)

        if truth_step is None:
            return predictions

        return Collection(
            self.train_iterator.step(truth_step)(predictions), predictions
        )

    def data(self, index : str, undo=False) -> np.array | xr.Dataset:
        """Get data which is fed into model

        Args:
            index (str): 
                Index to retrieve at
            undo (bool, optional): 
                Rebuild Data using DataStep.undo. Defaults to False.

        Returns:
            (np.array | xr.Dataset): 
                Retrieved Data
        """        
        data = self.train_iterator[index]

        if undo:
            data = self.train_iterator.undo(data)
        return data

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
