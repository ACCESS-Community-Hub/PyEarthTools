import copy
from pathlib import Path
from typing import Union

import numpy as np
import pytorch_lightning as pl
import xarray as xr
from torch.utils.data import DataLoader, IterableDataset

from dset.training.trainer.template import DSETTrainer
from dset.training.data.templates import DataIterator


class DSETTrainerWrapper(DSETTrainer):
    def __init__(
        self,
        model,
        path: str,
        train_data: Union[DataLoader, DataIterator],
        valid_data: DataIterator = None,
        **kwargs
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
        batch_size = kwargs.pop("batch_size", 1)

        try:
            train_data = copy.copy(train_data)
        except Exception:
            pass

        self.train_iterator = train_data
        if not isinstance(train_data, DataLoader):
            self.train_data = DataLoader(
                train_data, batch_size=batch_size, num_workers=num_workers
            )
        else:
            self.train_data = train_data

        self.valid_iterator = valid_data
        self.valid_data = None
        if valid_data:
            if not isinstance(valid_data, DataLoader):
                self.valid_data = DataLoader(
                    valid_data, batch_size=batch_size, num_workers=num_workers
                )
            else:
                self.valid_data = valid_data

        path = kwargs.pop("default_root_dir", path)
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            save_top_k=10,
            monitor="train/loss",
            #dirpath=Path(path) / 'Checkpoints',
            mode="min",
            filename="{epoch:02d}",
            every_n_train_steps=1000,
        )
        callbacks = kwargs.pop("callbacks", [])
        callbacks.append(checkpoint_callback)

        self.trainer = pl.Trainer(
            default_root_dir=kwargs.pop("default_root_dir", path),
            callbacks=callbacks,
            **kwargs
        )

    def __getattr__(self, key):
        return getattr(self.trainer, key)

    def fit(self, *args, **kwargs):
        """
        Using Pytorch Lightning .fit to train model, auto fills model and dataloaders
        """
        self.trainer.fit(
            model=self.model,
            train_dataloaders=kwargs.pop("train_dataloaders", self.train_data),
            val_dataloaders=kwargs.pop("valid_dataloaders", self.valid_data),
            *args,
            **kwargs
        )

    def load(self, file: str):
        """
        Load model from checkpoint file
        """
        self.model = self.model.load_from_checkpoint(file)

    def predict(
        self,
        index: str,
        undo: bool = False,
        data_iterator: DataIterator = None,
        **kwargs
    ) -> Union[np.array, xr.Dataset]:
        """
        Pytorch Lightning Prediction Override

        Uses dset.training DataIterator to get data

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
            Either xarray datasets or np arrays

        """
        data_source = data_iterator or self.train_iterator
        data = data_source[index]

        class FakeDataLoader(IterableDataset):
            def __init__(self, data):
                self.data = data

            def __iter__(self):
                for data in zip(*self.data):
                    yield data

        fake_data = DataLoader(
            FakeDataLoader(data), batch_size=kwargs.pop("batch_size", 1)
        )
        prediction = tuple(
            map(
                np.vstack,
                zip(*self.trainer.predict(model=self.model, dataloaders=fake_data)),
            )
        )

        if undo:
            fixed_predictions = list(data_source.undo(prediction))
            fixed_predictions[1] = data_source.rebuild_time(
                fixed_predictions[-1], index
            )
            return tuple(fixed_predictions)
        return prediction
