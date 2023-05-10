from __future__ import annotations
from abc import abstractmethod

from pathlib import Path
import numpy as np
import xarray as xr


from edit.training.data.templates import DataStep
from edit.data import Collection


class EDITTrainer:
    """
    Template for EDITTrainer Wrapper
    """

    def __init__(self, model, train_data: DataStep, valid_data: DataStep = None, path : str | Path = None, **kwargs) -> None:
        self.model = model
        self.train_data = train_data
        self.valid_data = valid_data
        self.path = path

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
        data = self.train_data[index]

        if undo:
            data = self.train_data.undo(data)
        return data

    @abstractmethod
    def fit(self):
        raise NotImplementedError()

    @abstractmethod
    def _predict_from_data(self, data, **kwargs):
        raise NotImplementedError


    def predict(
        self,
        index: str,
        *,
        undo: bool = True,
        data_iterator: DataStep = None,
        load: bool | str = False,
        load_kwargs: dict = {},
        fake_batch_dim: bool = False,
        **kwargs,
    ) -> tuple[np.array] | tuple[xr.Dataset]:
        """Predict using the model a particular index

        Uses [edit.training][edit.training.data] DataStep to get data at given index.
        Can automatically try to rebuild the data.

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
            load (bool | str, optional): 
                Path to checkpoint, or boolean to find latest file. Defaults to False.
            load_kwargs (dict, optional): 
                Keyword arguments to pass to loading function. Defaults to {}.
            fake_batch_dim (bool, optional):
                If the batch dimension needs to be faked. Defaults to False.

        Returns:
            (tuple[np.array] | tuple[xr.Dataset]): 
                Either xarray datasets or np arrays, [truth data, predicted data]
        """    
        self.load(load,**load_kwargs)

        data_source = data_iterator or self.valid_data or self.train_data
        data = data_source[index]
        truth = data[0]

        if fake_batch_dim:
            if isinstance(data, tuple):
                data = tuple(map(lambda x: np.expand_dims(x,axis = 0), data))
            else:
                data = np.expand_dims(data, axis = 0)


        prediction = self._predict_from_data(data, **kwargs)

        if fake_batch_dim:
            if isinstance(data, tuple):
                prediction = list(map(lambda x: np.squeeze(x,axis = 0), prediction))
            else:
                prediction = np.squeeze(prediction, axis = 0)
        
        truth_data = None

        if not undo:
            return Collection(truth, prediction[1])

        prediction = data_source.undo(prediction)
        if isinstance(prediction, (tuple, list)):
            truth_data = prediction[0]
            prediction = prediction[-1]

        if not isinstance(prediction, xr.Dataset):
            return Collection(truth_data, prediction)

        if "Coordinate 1" in prediction:
            prediction = prediction.rename({"Coordinate 1": "time"})
        if hasattr(data_source, "rebuild_time"):
            prediction = data_source.rebuild_time(prediction, index)

        return Collection(truth_data or data_source.undo(data)[1], prediction)

    def predict_recurrent(
        self,
        start_index: str,
        recurrence: int,
        data_iterator: DataStep = None,
        load: bool = False,
        load_kwargs: dict = {},
        truth_step: int = 0,
        fake_batch_dim: bool = False,
        **kwargs,
    ) -> tuple[np.array] | tuple[xr.Dataset]:
        """Uses [predict][edit.training.trainer.template.EDITTrainer.predict] to predict timesteps and then feed back through recurrently.

        Uses [edit.training][edit.training.data] DataStep to get data at given index.
        Can automatically try to rebuild the data.

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
            load (bool, optional):
                Resume from checkpoint. Defaults to False.
            load_kwargs (dict, optional): 
                Keyword arguments to pass to loading function
            truth_step (int, optional):
                Data Pipeline step to use to retrieve Truth data. Defaults to 0
            fake_batch_dim (bool, optional):
                If the batch dimension needs to be faked. Defaults to False.
        Returns:
            (tuple[np.array] | tuple[xr.Dataset]):
                Either xarray datasets or np arrays, [truth data, predicted data]
        """
        data_source = data_iterator or self.valid_data or self.train_data
        data = list(data_source[start_index])

        if isinstance(load, str):
            self.load(load,**load_kwargs)

        elif load and Path(self.path).exists():
            self.load(True, **load_kwargs)

        predictions = []
        index = start_index

        for i in range(recurrence):
            if fake_batch_dim:
                if isinstance(data, (list,tuple)):
                    data = list(map(lambda x: np.expand_dims(x,axis = 0), data))
                else:
                    data = np.expand_dims(data, axis = 0)

            input_data = None
            prediction = self._predict_from_data(data, **kwargs)
            if fake_batch_dim:
                if isinstance(data, (list, tuple)):
                    prediction = list(map(lambda x: np.squeeze(x,axis = 0), prediction))
                else:
                    prediction = np.squeeze(prediction, axis = 0)

            fixed_predictions = data_source.undo(prediction)

            if isinstance(fixed_predictions, (tuple, list)):
                input_data = fixed_predictions[0]
                fixed_predictions = fixed_predictions[-1]

            if not isinstance(fixed_predictions, xr.Dataset):
                raise TypeError(f"Unable to recurrently merge data of type {type(fixed_predictions)}")

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
            self.train_data.step(truth_step)(predictions), predictions
        )


    ## Model State Functions 
    @abstractmethod
    def load(self, path : str | Path | bool):
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str | Path):
        raise NotImplementedError
        
    