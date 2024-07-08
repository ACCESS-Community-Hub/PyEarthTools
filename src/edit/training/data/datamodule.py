# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

"""
Training DataModule from Pipelines
"""

import functools
from typing import Union, Optional, Callable, Any

import numpy as np

from edit.pipeline import Pipeline, Iterator
from edit.utils.initialisation import InitialisationRecordingMixin

class PipelineDataModule(InitialisationRecordingMixin):
    def __init__(self, pipelines: Union[dict[str,Union[Pipeline, tuple[Pipeline,...]]], tuple[Pipeline,...],Pipeline], train_split: Optional[Iterator] = None, valid_split: Optional[Iterator] = None):
        """
        Setup `Pipeline`'s for use with ML Training

        Args:
            pipelines (Union[dict[str,Union[Pipeline, tuple[Pipeline,...]]], tuple[Pipeline,...],Pipeline]): 
                Pipelines for data retrieval, can be dictionary and/or list/tuple of `Pipelines` or a single `Pipeline`
            train_split (Optional[Iterator], optional): 
                Iterator to use for training. Defaults to None.
            valid_split (Optional[Iterator], optional): 
                Iterator to use for validation. Defaults to None.
        """        
        super().__init__()
        self.record_initialisation()

        self._pipelines = pipelines
        self._train_split = train_split
        self._valid_split = valid_split

    @property
    def pipelines(self):
        return self._pipelines

    @classmethod
    def map_function(cls, obj, function: Callable[[Any], Any], **kwargs):
        recur_function = functools.partial(PipelineDataModule.map_function, function = function, **kwargs)
        if isinstance(obj, dict):
            return {key: recur_function(val) for key, val in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(map(recur_function, obj))
        return function(obj, **kwargs)

    def map_function_to_pipelines(self, function: Callable[[Pipeline], Any], **kwargs):
        """
        Map a function over `Pipelines`
        """
        return self.map_function(self._pipelines, function, **kwargs)
    
    def train(self):
        """
        Set `Pipeline`s to iterate over `train_split`
        """
        if self._train_split is None:
            raise ValueError("Cannot enter training mode as `train_split` is None.")
        def set_iterator(obj: Pipeline):
            obj.iterator = self._train_split
        self.map_function_to_pipelines(set_iterator)

    def eval(self):
        """
        Set `Pipeline`s to iterate over `valid_split`
        """
        if self._valid_split is None:
            raise ValueError("Cannot enter training mode as `valid_split` is None.")
        def set_iterator(obj: Pipeline):
            obj.iterator = self._valid_split
        self.map_function_to_pipelines(set_iterator)

    def get_sample(self, idx, *, fake_batch_dim: bool = False):
        """Get sample from `pipeline`s"""
        if fake_batch_dim:
            def add_batch_dim(obj):
                if isinstance(obj, (list, tuple)):
                    return type(obj)(map(add_batch_dim, obj))
                return np.expand_dims(obj, 0)
            return self.map_function_to_pipelines(lambda x: add_batch_dim(x[idx]))
        return self.map_function_to_pipelines(lambda x: x[idx])

    @classmethod
    def find_shape(cls, obj):
        """Find shape of `obj`"""
        if isinstance(obj, dict):
            return {key: PipelineDataModule.find_shape(val) for key, val in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(map(PipelineDataModule.find_shape, obj))
        return obj.shape
