"""
Provide a Machine Learning Model as an [edit.data.DataIndex][edit.data.DataIndex].

This will allow data to be retrieved as normal, with the user not having to worry about it being an ML Model
"""
from __future__ import annotations

import os
from pathlib import Path
from edit.data import DataIndex, CachingIndex

from edit.training.data.context import PatchingUpdate
from edit.training.trainer import from_yaml, EDITLightningTrainer
from edit.training.trainer.template import EDITTrainer


class MLDataIndex(CachingIndex):
    def __init__(self, trainer: EDITTrainer, stride_override: int = None, cache: str | Path = None, recurrent_config: dict = {}, **kwargs):
        """Setup ML Data Index from defined trainer

        !!! Info
            This can be used just like a [DataIndex][edit.data.DataIndex] from [edit.data][edit.data.index],
            so calling or indexing into this object work, as well as supplying transforms.

        Args:
            trainer (EDITTrainer): 
                EDITTrainer to use to retrieve data
            stride_override (int, optional): 
                Values to override stride with, if using `PatchingDataIndex`. Defaults to None.
            cache (str | Path, optional):
                Location to cache outputs, if not supplied don't cache.
            recurrent_config (dict, optional):
                Configuration if Model must be run recurrently
            **kwargs (dict, optional):
                Any keyword arguments to pass to [DataIndex][edit.data.DataIndex]
        """        
        super().__init__(cache = cache, **kwargs)
        self.trainer = trainer
        self.stride_override = stride_override
        self.recurrent_config= recurrent_config

    def get(
        self,
        querytime : str,
    ):  # transforms: Union[Callable, TransformCollection, Transform]= None
        """
        Get Data from given timestep
        """
        with PatchingUpdate(self.trainer, stride_size=self.stride_override):
            if self.recurrent_config:
                _, predicted_ds = self.trainer.predict_recurrent(querytime, **self.recurrent_config)
            else:
                _, predicted_ds = self.trainer.predict(querytime, undo=True)

        return predicted_ds

    def input_data(self, querytime: str):
        """
        Get input data at given timestep
        """
        with PatchingUpdate(self.trainer, stride_size=self.stride_override):
            input_data = self.trainer.train_iterator.undo(
                self.trainer.train_iterator[querytime]
            )
        return input_data

    @property
    def data(self):
        with PatchingUpdate(self.trainer, stride_size=self.stride_override):
            return self.trainer.train_data

    @staticmethod
    def from_yaml(
        yaml_config: str | Path,
        checkpoint_path: str = None,
        *,
        auto_load: bool = False,
        only_state: bool = False,
        stride_override: int = None,
        **kwargs,
    ):
        """Setup ML Data Index from yaml file config and pretrained model 

        Args:
            yaml_config (str | Path): 
                Path to yaml config
            checkpoint_path (str, optional): 
                Path to pretrained checkpoint. Defaults to None.
            auto_load (bool, optional): 
                Find latest `checkpoint_path` automatically . Defaults to False.
            only_state (bool, optional): 
                Only load the state of the model. Defaults to False.
            stride_override (int, optional): 
                Values to override stride with, if using `PatchingDataIndex`. Defaults to None.

        Raises:
            RuntimeError: 
                If no `checkpoint_path` is given

        Returns:
            (MLDataIndex): 
                MLDataIndex to use to get data with
        """    
        trainer: EDITLightningTrainer
        trainer = from_yaml(
            yaml_config,
            strategy=kwargs.pop("strategy", "dp"),
            logger=kwargs.pop("logger", False),
            **kwargs,
        )

        if auto_load and Path(trainer.checkpoint_path).exists() and not checkpoint_path:
            checkpoint_path = max(
                Path(trainer.checkpoint_path).iterdir(), key=os.path.getctime
            )

        if not checkpoint_path:
            raise RuntimeError(
                f"MLDataIndex must load a trained model, if no checkpoint_path given, use auto_load."
            )
        print(f"Loading {checkpoint_path}...")
        trainer.load(checkpoint_path, only_state=only_state)

        return MLDataIndex(trainer, stride_override = stride_override)
