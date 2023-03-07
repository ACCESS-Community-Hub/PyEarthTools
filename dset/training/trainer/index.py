"""
ML Data Indexes
"""

from typing import Callable, Union
from dset.data import OperatorIndex, DataIndex
from dset.data.transform import apply, Transform, TransformCollection

from dset.training.data.context import PatchingUpdate
from dset.training.trainer import load_from_yaml, DSETTrainerWrapper
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

    
    def single(self, query_time, transforms: Union[Callable, TransformCollection, Transform]= None):
        """
        Get Data from given timestep
        """
        with PatchingUpdate(self.trainer, stride_size=self.stride_override):
            _, predicted_ds = self.trainer.predict(query_time, undo = True)
        predicted_ds = apply(transforms)(predicted_ds)
        return predicted_ds

    def input_data(self, query_time, transforms: Union[Callable, TransformCollection, Transform]= None):
        """
        Get input data at given timestep
        """
        return tuple(map(apply(transforms), self.trainer.train_iterator.undo(self.trainer.train_iterator[query_time])))

    @staticmethod
    def from_yaml(yaml_config : str, checkpoint_path: str, only_state: bool = False, **kwargs):
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
        trainer = load_from_yaml(yaml_config, **kwargs)
        trainer.load(checkpoint_path, only_state = only_state)

        return MLDataIndex(trainer)
