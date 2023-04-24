"""
ML Data Indexes
"""

import os
from pathlib import Path
from typing import Callable, Union
from dset.data import OperatorIndex, DataIndex
from dset.data.transform import apply, Transform, TransformCollection

from dset.training.data.context import PatchingUpdate
from dset.training.trainer import from_yaml, DSETTrainerWrapper
from dset.training.trainer.template import DSETTrainer


class MLDataIndex(DataIndex):
    def __init__(self, trainer: DSETTrainer, stride_override: int = None):
        """
        Setup ML Data Index from defined trainer

        Parameters
        ----------
        trainer
            DSETTrainer to use to retrieve data
        """
        self.trainer = trainer
        self.stride_override = stride_override

    def get(
        self,
        query_time,
    ):  # transforms: Union[Callable, TransformCollection, Transform]= None
        """
        Get Data from given timestep
        """
        with PatchingUpdate(self.trainer, stride_size=self.stride_override):
            _, predicted_ds = self.trainer.predict(query_time, undo=True)
        # predicted_ds = apply(transforms)(predicted_ds)
        return predicted_ds

    def input_data(self, query_time):
        """
        Get input data at given timestep
        """
        with PatchingUpdate(self.trainer, stride_size=self.stride_override):
            input_data = self.trainer.train_iterator.undo(
                self.trainer.train_iterator[query_time]
            )
        return input_data
    
    @property
    def iterator(self):
        with PatchingUpdate(self.trainer, stride_size=self.stride_override):
            return self.trainer.train_iterator

    @staticmethod
    def from_yaml(
        yaml_config: str,
        checkpoint_path: str = None,
        *,
        auto_load: bool = False,
        only_state: bool = False,
        stride_override: int = None,
        **kwargs
    ):
        """
        Setup ML Data Index from yaml config and pretrained model

        Parameters
        ----------
        yaml_config
            Path to yaml config
        checkpoint_path
            Path to pretrained checkpoint
        **kwargs
            All passed to trainer.load_from_yaml
        """
        trainer: DSETTrainerWrapper
        trainer = from_yaml(
            yaml_config,
            strategy=kwargs.pop("strategy", "dp"),
            logger=kwargs.pop("logger", False),
            **kwargs
        )

        if auto_load and Path(trainer.checkpoint_path).exists() and not checkpoint_path:
            checkpoint_path = max(Path(trainer.checkpoint_path).iterdir(), key=os.path.getctime)

        if not checkpoint_path:
            raise RuntimeError(f"MLDataIndex must load a trained model, if no checkpoint_path given, use auto_load.")
        print(f"Loading {checkpoint_path}...")
        trainer.load(checkpoint_path, only_state=only_state)
    

        return MLDataIndex(trainer, stride_override)

    