import copy
import os
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import xarray as xr
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, IterableDataset

from edit.training.trainer.template import EDITTrainer
from edit.training.data.templates import DataIterator


class EDITTrainerWrapper(EDITTrainer):
    """
    Pytorch Lightning Trainer Wrapper.
    Provides fit, predict overrides to work with edit.training
    """

    def __init__(
        self,
        model,
        path: str,
        train_data: Union[DataLoader, DataIterator],
        valid_data: DataIterator = None,
        **kwargs,
    ) -> None:
        """
        Pytorch Lightning Trainer Wrapper.
        Provides fit, predict overrides to work with edit.training


        Parameters
        ----------
        model
            Model to use
        path
            
        train_data
            Training data to use, can either be DataIterator, or pytorch DataLoader
        valid_data, optional
            Validation data to use, can either be DataIterator, or pytorch DataLoader,
            by default None

        **kwargs, optional
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
            kwargs["logger"] = "csv"

        if "logger" in kwargs and isinstance(kwargs["logger"], str):
            self.logger = str(kwargs.pop("logger")).lower()
            if self.logger == "tensorboard":
                kwargs["logger"] = pl.loggers.TensorBoardLogger(
                    path, name=kwargs.pop("name", None)
                )

            elif self.logger == "csv":
                kwargs["logger"] = pl.loggers.CSVLogger(path, name="csv_logs")
                self.log_path = self.log_path / "csv_logs"

        kwargs['limit_val_batches'] = kwargs.pop('limit_val_batches', 10)

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
        """
        Using Pytorch Lightning .fit to train model, auto fills model and dataloaders

        Parameters
        ----------
        resume
            Whether to resume most recent checkpoint file in checkpoint dir
        """
        if resume and Path(self.checkpoint_path).exists() and "ckpt_path" not in kwargs:
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

    def load(self, file: str, only_state: bool = False):
        """
        Load Model from Checkpoint File.

        Can either be PytorchLightning Checkpoint or torch checkpoint.

        Parameters
        ----------
        file
            Path to checkpoint
        only_state, optional
            If only the model state should be loaded, by default False
        """

        if only_state:
            state = torch.load(file)
            if "state_dict" in state:
                state = state["state_dict"]
                new_state = {}
                for key, variable in state.items():
                    if "model" in key:
                        new_state[key.replace("model.", "")] = variable
                    else:
                        new_state[key] = variable
                state = new_state

            self.model.model.load_state_dict(state)
            return
        self.model = self.model.load_from_checkpoint(file)

    def _predict_from_data(self, data, **kwargs):
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
        undo: bool = False,
        data_iterator: DataIterator = None,
        resume: bool = True,
        **kwargs,
    ) -> Union[tuple[np.array], tuple[xr.Dataset]]:
        """
        Pytorch Lightning Prediction Override

        Uses edit.training DataIterator to get data at given index.
        Can automatically try to rebuild the xarray Dataset.

        !!! Warning
            If number of patches is not divisible by the `batch_size`, issues may arise.
            Solution: batch_size = 1

        Parameters
        ----------
        index
            Index into DataIterator, usually str
        undo, optional
            Whether to pass prediction through DataIterator.undo, by default False
        data_iterator, optional
            Override for DataIterator to use, by default None

        Returns
        -------
            Either xarray datasets or np arrays of input data, predicted data

        """
        data_source = data_iterator or self.valid_iterator or self.train_iterator
        data = data_source[index]

        if resume and Path(self.checkpoint_path).exists():
            self.load(
                max(Path(self.checkpoint_path).iterdir(), key=os.path.getctime),
                only_state=True,
            )

        prediction = self._predict_from_data(data, **kwargs)

        if undo:
            fixed_predictions = list(data_source.undo(prediction))

            if "Coordinate 1" in fixed_predictions[1]:
                fixed_predictions[1] = fixed_predictions[1].rename(
                    {"Coordinate 1": "time"}
                )

            fixed_predictions[1] = data_source.rebuild_time(
                fixed_predictions[-1], index
            )

        return tuple(fixed_predictions)

    def predict_recurrent(
        self,
        start_index: str,
        recurrence: int,
        data_iterator: DataIterator = None,
        resume: bool = True,
        **kwargs,
    ):
        """Uses [predict][edit.training.trainer.trainer.predict] to predict timesteps and then feed back through recurrently.

        
        Uses edit.training DataIterator to get data at given start index.
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

        Returns:
            (xr.Dataset): 
                Combined Predictions
        """        
        data_source = data_iterator or self.valid_iterator or self.train_iterator
        data = list(data_source[start_index])

        if resume and Path(self.checkpoint_path).exists():
            self.load(
                max(Path(self.checkpoint_path).iterdir(), key=os.path.getctime),
                only_state=True,
            )

        predictions = []
        index = start_index

        for i in range(recurrence):
            prediction = self._predict_from_data(data, **kwargs)
            fixed_predictions = list(data_source.undo(prediction))

            if "Coordinate 1" in fixed_predictions[1]:
                fixed_predictions[1] = fixed_predictions[1].rename(
                    {"Coordinate 1": "time"}
                )

            fixed_predictions[1] = data_source.rebuild_time(
                fixed_predictions[-1], index
            )
            index = fixed_predictions[1].time.values[-1]
            predictions.append(fixed_predictions[1])

            data = fixed_predictions
            data.reverse()
            data = data_source.apply(data)

        return xr.merge(predictions)

    def data(self, index, undo=False):
        """
        Get data which is fed into model

        Parameters
        ----------
        index
            Index to retrieve at
        undo, optional
            Whether to undo data transforms, by default False

        Returns
        -------
            np.array or xarray.Dataset
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
                f"No metrics.csv files could be found at {self.log_path}"
            )

        metrics = self.__flatten_metrics(metrics)
        ax = metrics.sort_values(x).plot(y=y, x=x)
        ax.set_xlabel(x)
        ax.set_ylabel(y)

        return ax
