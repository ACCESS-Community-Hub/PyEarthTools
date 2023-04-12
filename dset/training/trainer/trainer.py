import copy
import os
from pathlib import Path
from typing import Union

import numpy as np
import pytorch_lightning as pl
import torch
import xarray as xr
from torch.utils.data import DataLoader, IterableDataset

from dset.training.trainer.template import DSETTrainer
from dset.training.data.templates import DataIterator
from dset.training.data.loaders import PytorchIterable#, DALILoader


class DSETTrainerWrapper(DSETTrainer):
    """
    Pytorch Lightning Trainer Wrapper.
    Provides fit, predict overrides to work with dset.training
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
        Provides fit, predict overrides to work with dset.training


        Parameters
        ----------
        model
            Model to use
        train_data
            Training data to use, can either be DataIterator, or pytorch DataLoader
        valid_data, optional
            Validation data to use, can either be DataIterator, or pytorch DataLoader,
            by default None
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
                train_data, batch_size=batch_size, num_workers=num_workers, pin_memory=True
            )
        else:
            self.train_data = train_data

        self.valid_iterator = valid_data
        self.valid_data = None
        if valid_data:
            if not isinstance(valid_data, DataLoader):
                self.valid_data = DataLoader(
                    valid_data, batch_size=batch_size, num_workers=num_workers, pin_memory=True
                )
            else:
                self.valid_data = valid_data

        path = kwargs.pop("default_root_dir", path)
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

        self.trainer = pl.Trainer(
            default_root_dir=kwargs.pop("default_root_dir", path),
            callbacks=callbacks,
            **kwargs,
        )

    def __getattr__(self, key):
        if key == "trainer":
            raise AttributeError(f"{self!r} has no attribute {key!r}")
        return getattr(self.trainer, key)

    def fit(self, resume: bool = False, *args, **kwargs):
        """
        Using Pytorch Lightning .fit to train model, auto fills model and dataloaders

        Parameters
        ----------
        resume
            Whether to resume most recent checkpoint file in checkpoint dir
        """
        if resume and Path(self.checkpoint_path).exists() and 'ckpt_path' not in kwargs:
            kwargs['ckpt_path'] = max(Path(self.checkpoint_path).iterdir(), key=os.path.getctime)

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

        Uses dset.training DataIterator to get data at given index.
        Can automatically try to rebuild the xarray Dataset.

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
            self.load(max(Path(self.checkpoint_path).iterdir(), key=os.path.getctime), only_state=True)


        class FakeDataLoader(IterableDataset):
            def __init__(self, data: tuple[np.ndarray]):
                self.data = data

            def __iter__(self):
                for data in zip(*self.data):
                    yield data

        fake_data = DataLoader(
            FakeDataLoader(data),
            batch_size=kwargs.pop("batch_size", self.batch_size),
            # num_workers=kwargs.pop("num_workers", self.num_workers), #Apparently this reproduces data on small scales
        )
        self.fake_data = fake_data
        prediction = tuple(
            map(
                np.vstack,
                zip(*self.trainer.predict(model=self.model, dataloaders=fake_data)),
            )
        )
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
        return prediction


    def data(self, index, undo = False):
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
