from __future__ import annotations
from abc import abstractmethod, ABCMeta
import functools
import math

from pathlib import Path
import warnings
from typing import Any, Callable
import numpy as np
import xarray as xr
import logging

from tqdm.auto import trange

import edit.pipeline
from edit.pipeline.templates import DataStep

from edit.data import Collection, LabelledCollection, IndexWarning, patterns, TimeDelta, EDITDatetime

import edit.training
from edit.training.trainer.dataindex import MLDataIndex

LOG = logging.getLogger('edit.training')

def parse_recurrent(interval: int | TimeDelta = 1):
    """
    Parse kwargs given to a recurrent function.

    Converts to `steps`, number of model steps to run

    !!! Supported

        | Kwarg | Description |
        | ----- | ----------- |
        | steps | Default value, number of steps of model, all are converted to this |
        | time  | Time value given in same units as interval |
        | to_time  | Time to predict up to, uses `interval` to get steps from current. |
    
    !!! Examples
        ```python
        @parse_recurrent(interval = 6) # 6 hour interval
        def func(*args, steps, **kwargs):
            ....
        func(steps = 10) # Nothing, run 10 steps
        func(time = 48) # Time of 48 hours, becomes `steps = 8`
        ```

    Args:
        interval (int | TimeDelta, optional): 
            Time interval to convert with. Defaults to 1.
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def parse(*args, **kwargs):
            if 'steps' in kwargs:
                pass
            elif 'time' in kwargs:
                time = kwargs.pop('time')
                kwargs['steps'] = math.ceil(time / interval)
            elif 'to_time' in kwargs:
                to_time = kwargs.pop('time')
                kwargs['steps'] = math.ceil((EDITDatetime('current') - to_time) / interval)
            return func(*args, **kwargs)
        return parse
    return decorator

class EDIT_Inference(metaclass = ABCMeta):
    def __init__(self, pipeline: DataStep):
        self.pipeline = pipeline

    @abstractmethod
    def _predict_from_data(self, data: Any, **kwargs) -> np.ndarray:
        """
        Must be implemented by a child class to actually predict from data

        !!! Tip
            Function must return prediction as a [numpy array][np.array] of the same shape as target
        """
        raise NotImplementedError()
    
    @abstractmethod
    def predict(self, idx: Any, *args, **kwargs):
        raise NotImplementedError()
    
    @abstractmethod
    def recurrent(self, idx: Any, steps: int, *args, **kwargs):
        raise NotImplementedError()
    
    @functools.wraps(MLDataIndex)
    def as_index(self, **kwargs):
        """
        Convert this trainer to an `MLDataIndex`

        Passes across all kwargs
        """
        if isinstance(self.pipeline, DataStep) and hasattr(self.pipeline, '_interval'):
            kwargs['data_interval'] = kwargs.get('data_interval', self.pipeline._interval)
        return edit.training.MLDataIndex(self, **kwargs)
    
    def data(self, idx: Any, undo=False) -> np.ndarray | xr.Dataset | Collection:
        """
        Get data from pipeline

        Args:
            index (str):
                Index to retrieve at
            undo (bool, optional):
                Rebuild Data using DataStep.undo. Defaults to False.

        Returns:
            (np.array | xr.Dataset):
                Retrieved Data
        """
        data = self.pipeline[idx]

        if undo:
            data = self.pipeline.undo(data)

        if isinstance(data, (tuple, list)):
            data = Collection(*data)
        return data
    
    ## Model State Functions
    @abstractmethod
    def load(self, path: str | Path | bool, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def save(self, path: str | Path, **kwargs):
        raise NotImplementedError()


    def __repr__(self):
        repr_string = []
        repr_string.append(f"===== {self.__class__.__module__} {self.__class__.__name__} =====")
        repr_string.append("Pipeline:")
        repr_string.append(f"{repr(self.pipeline)}")

        return '\n'.join(repr_string)
        
class EDIT_Training(EDIT_Inference):
    @abstractmethod
    def fit(self):
        """Abstract fit function which needs to be wrapped by the child."""
        raise NotImplementedError()   
    
class EDIT_AutoInference(EDIT_Inference):

    ###
    ##  Prediction Wrappers
    ###
    def predict(
        self,
        index: str,
        *,
        undo: bool = True,
        data_iterator: DataStep | None = None,
        load: bool | str = False,
        load_kwargs: dict = {},
        fake_batch_dim: bool | None = None,
        quiet: bool = False,
        **kwargs,
    ) -> np.ndarray | xr.Dataset:
        """Predict using the model a particular index

        Uses [edit.pipeline][edit.pipeline] to get data at given index.
        Can automatically try to rebuild the data.

        !!! Warning
            Uses child classes implementation of `_predict_from_data` to run the predictions.

        !!! Warning
            If number of patches is not divisible by the `batch_size`, issues may arise.

            Solution: `batch_size = 1`

        !!! Tip
            If RuntimeError's arise with incorrect number of channels, try setting `fake_batch_dim`

        Args:
            index (str):
                Index to get from the validation or training data loader or given `data_iterator`
            undo (bool, optional):
                Rebuild Data using DataStep.undo. Defaults to True.
            data_iterator (DataIterator, optional):
                Override for DataStep to use. Defaults to None.
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
        data_source = data_iterator or self.pipeline

        if 'Patch' in data_source.steps and 'patch_update' not in kwargs:    
            with edit.pipeline.context.PatchingUpdate(data_source, kernel_size = kwargs.pop('kernel_size', None), stride_size = kwargs.pop('stride_size', None)):
                return self.predict(index, data_iterator=data_source, undo = undo, load = load, load_kwargs=load_kwargs, fake_batch_dim=fake_batch_dim, patch_update = True, **kwargs)
        
        kwargs.pop('patch_update', None)

        self.load(load, **load_kwargs)

        if 'ToNumpy' in self.pipeline.steps or 'FakeData' in self.pipeline.steps:
            fake_batch_dim = True if fake_batch_dim is None else fake_batch_dim

        if fake_batch_dim is None:
            fake_batch_dim = False

        data = data_source[index]
    
        if fake_batch_dim:
            data = expand_dims(data)

        prediction = self._predict_from_data(data, **kwargs)

        if fake_batch_dim:
            prediction = squeeze_dims(prediction)
            data = squeeze_dims(data)

        if not undo:
            if isinstance(prediction, (tuple, list)):
                prediction = prediction[-1]
            return prediction

        if isinstance(data, (tuple, list)) and not isinstance(prediction, (tuple, list)):
            prediction = (*data[:-1], prediction)

        prediction = data_source.undo(prediction)

        if isinstance(prediction, (tuple, list)):
            prediction = prediction[-1]
        if hasattr(data_source, "rebuild_time"):
            prediction = data_source.rebuild_time(prediction, index, offset = 0)
        
        return prediction # Just return prediction
        
        # truth = data_source.undo(data)
        # if isinstance(truth, (tuple, list)):
        #     truth = truth[1]

        # if not isinstance(prediction, xr.Dataset):
        #     return Collection(truth, prediction)

        # if "Coordinate 1" in prediction:
        #     prediction = prediction.rename({"Coordinate 1": "time"})
            
        # if hasattr(data_source, "rebuild_time"):
        #     truth, prediction = map(lambda x: data_source.rebuild_time(x, index, offset = 0), (truth, prediction))

        # return Collection(truth, prediction)
    
    def recurrent(
        self,
        start_index: str,
        steps: int,
        interval: str | TimeDelta | tuple | int | None = None,
        *,
        data_iterator: DataStep | None = None,
        load: bool = False,
        load_kwargs: dict = {},
        truth_step: int | None = None,
        fake_batch_dim: bool | None = None,
        trim_time_dim: int | None = None,
        verbose: bool = True,
        quiet: bool = False,
        cache: bool | str | Path = False,
        save_location: str | Path | None = None,
        use_output: bool = False,
        **kwargs,
    ) -> np.ndarray | xr.Dataset:
        """Time wise recurrent prediction

        Uses [edit.pipeline][edit.pipeline] to get data at given index.

        !!! Warning
            Uses child classes implementation of `_predict_from_data` to run the predictions.

        !!! Warning
            If number of patches is not divisible by the `batch_size`, issues may arise.
            Solution: batch_size = 1

        Args:
            start_index (str):
                Starting Index of Prediction
            steps (int):
                Number of times to recur
            data_iterator (DataIterator, optional):
                Override for initial data retrieval. Defaults to None.
            load (bool, optional):
                Resume from checkpoint. Defaults to False.
            load_kwargs (dict, optional):
                Keyword arguments to pass to loading function
            truth_step (int | None, optional):
                Data Pipeline step to use to retrieve Truth data. Defaults to None
            fake_batch_dim (bool, optional):
                If the batch dimension needs to be faked. Defaults to False.
            trim_time_dim (int, optional):
                Number of sample in time to use of prediction. Defaults to None.
            verbose (bool, optional):
                Show progress of recurrent predictions. Defaults to True.
            cache (bool | str | Path, optional):
                Whether to cache intermediate data to directory, if True, set up temp directory. Defaults to False.
            save_location (str | Path, optional):
                Location to save merged data, if not given, and `cache` is, all data will be loaded into memory than returned. 
                Therefore, if large datasets are in use, and `cache` given, it is best to set this as well.
                Defaults to None
            use_output (bool, optional):
                Whether to use output directly for input, skips extra `.undo` and `.apply` calls. Defaults to False.
            
        Returns:
            (tuple[np.array] | tuple[xr.Dataset]):
                Either xarray datasets or np arrays, [truth data, predicted data]
        """
        def select_if_tuple(item, index: int):
            if isinstance(item, (list, tuple)):
                return item[index]
            return item
        
        data_source = data_iterator or self.pipeline
        
        if isinstance(interval, (str, tuple, int)):
            interval = TimeDelta(interval)

        if 'Patch' in data_source.steps and 'patch_update' not in kwargs:    
            with edit.pipeline.context.PatchingUpdate(data_source, kernel_size = kwargs.pop('kernel_size', None), stride_size = kwargs.pop('stride_size', None)):
                return self.recurrent(start_index, steps, data_iterator=data_iterator, load = load, load_kwargs=load_kwargs, truth_step=truth_step, fake_batch_dim=fake_batch_dim, trim_time_dim=trim_time_dim,verbose=verbose, patch_update = True, **kwargs)
        kwargs.pop('patch_update', None)

        # Retrieve Initial Input Data
        data = data_source[start_index]

        # Load Model
        if load:
            self.load(load, **load_kwargs)

        if "ToNumpy" in self.pipeline.steps:
            fake_batch_dim = True if fake_batch_dim is None else fake_batch_dim
        if fake_batch_dim is None:
            fake_batch_dim = False

        predictions = []
        index = start_index

        cache_pattern, save_pattern = None, None
        if cache:
            if isinstance(cache, bool) and cache:
                cache = None
            cache_pattern = patterns.ArgumentExpansion(cache or 'temp', extension='.nc', filename_as_arguments = False)
        
        if save_location:
            save_pattern = patterns.Direct(root_dir = save_location or 'temp', extension='.nc')

        # Begin steps
        for i in trange(math.ceil(steps), disable=not verbose, desc="Predicting Recurrently"):
            if fake_batch_dim:  # Fake the Batch Dimension, for use with ToNumpy
                data = expand_dims(data)

            input_data = None
            prediction = self._predict_from_data(data, **kwargs)  # Prediction

            if fake_batch_dim:  # Squeeze again if faking the batch dim
                prediction = squeeze_dims(prediction)
                data = squeeze_dims(data)
            
            # if not isinstance(prediction, (tuple, list)):
            #     prediction = (*(data[:-1] if isinstance(data, (tuple, list)) else [data]), prediction)

            fixed_predictions = data_source.undo(prediction)  # Undo Pipeline

            # Separate components
            if isinstance(fixed_predictions, (tuple, list)):
                prediction = prediction[-1]

                input_data = fixed_predictions[0]
                fixed_predictions = fixed_predictions[-1]

            if not isinstance(fixed_predictions, xr.Dataset):
                raise TypeError(
                    f"Unable to recurrently merge data of type {type(fixed_predictions)}"
                )

            if hasattr(data_source, "rebuild_time"):
                fixed_predictions = data_source.rebuild_time(
                    fixed_predictions,
                    index,
                    offset=1 if i >= 1 else 0,
                )
            elif interval is not None:
                encoding = fixed_predictions['time'].encoding
                fixed_predictions['time'] = fixed_predictions.time + interval * ((i+1) if use_output else 1)
                fixed_predictions['time'].encoding.update(encoding)

            # ## Save out input, and fixed predictions
            # if cache_pattern is not None:
            #     cache_pattern.save(fixed_predictions, i, 'fixed')
            #     cache_pattern.save(input_data, i, 'input')

            #     fixed_predictions = cache_pattern(i, 'fixed')
            #     input_data = cache_pattern(i, 'input')

            ## Record Prediction
            append_prediction = type(fixed_predictions)(fixed_predictions)
            if trim_time_dim:
                append_prediction = fixed_predictions.isel(
                    time=slice(None, trim_time_dim)
                )

            index = append_prediction.time.values[-1]

            if cache_pattern is not None:
                cache_pattern.save(append_prediction, i,)
                # append_prediction = cache_pattern(i)
                predictions.append(cache_pattern.search(i))

            else:
                predictions.append(append_prediction)

            # Setup recurrent input data
            if use_output:
                if isinstance(data, (tuple, list)):
                    data = list(data)
                    data[0] = prediction
                else:
                    data = prediction
            else:
                def add_predictions(input_data, prediction_data):
                    if trim_time_dim:
                        prediction_data = prediction_data.isel(
                            time=slice(None, trim_time_dim)
                        )

                    new_input = xr.merge((input_data, prediction_data))

                    new_input = new_input.isel(time=slice(-1 * len(input_data.time), None))
                    return new_input

                # index = new_input.time.values[-1]
                new_input_data = add_predictions(
                    input_data or select_if_tuple(data_source.undo(data), 0), fixed_predictions
                )
                new_input_data = data_source.apply((new_input_data, fixed_predictions))

                new_input_data = select_if_tuple(new_input_data, 0)
                if isinstance(data, (tuple, list)):
                    data = list(data)
                    data[0] = new_input_data
                else:
                    data  = new_input_data                

        LOG.info("Merging Predictions")

        try:
            import dask
            dask.config.set({"array.slicing.split_large_chunks": False})
        except (ModuleNotFoundError, ImportError):
            pass                

        if cache_pattern:
            predictions = xr.open_mfdataset(predictions, chunks = 'auto')
        else:
            predictions = xr.concat(predictions, dim = 'time')
        
        if save_location:
            save_pattern.save(predictions, start_index)
            predictions = save_pattern(start_index)            

        if hasattr(cache_pattern, 'temp_dir') and cache_pattern.temp_dir:
            if not save_pattern:
                predictions = predictions.compute()
            cache_pattern.cleanup(safe = True)

        return predictions # Just return prediction
        # if truth_step is None:
        #     return predictions

        # LOG.info("Recovering Truth")

        # try:
        #     with warnings.catch_warnings():
        #         warnings.simplefilter(action="ignore", category=IndexWarning)
        #         truth_pipe_step = data_source.step(truth_step)
        #         # if "CachingIndex" in data_source.steps:
        #         #     truth_step = data_source.step("CachingIndex")
        #         truth_data = truth_pipe_step(predictions)
        # except Exception as e:
        #     warnings.warn(f"An error occured getting truth data, setting to None\n. {e}", RuntimeWarning)
        #     truth_data = None
        
        # return LabelledCollection(truth = truth_data, predictions = predictions)

## Prediction Utilities
def expand_dims(data: np.ndarray | tuple | list) -> np.ndarray | tuple | list:
    if isinstance(data, (list, tuple)):
        return type(data)(map(expand_dims, data))
    return np.expand_dims(data, axis=0)

def squeeze_dims(data: np.ndarray | tuple | list) -> np.ndarray | tuple | list:
    if isinstance(data, (list, tuple)):
        return type(data)(map(squeeze_dims, data))
    return np.squeeze(data, axis=0)


class EDIT_AutoInference_Training(EDIT_AutoInference, EDIT_Training):
    pass
