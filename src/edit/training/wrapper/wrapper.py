# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

from abc import ABCMeta, abstractmethod

from typing import Union

from edit.utils.initialisation import InitialisationRecordingMixin
from edit.pipeline import Pipeline

from edit.training.data import PipelineDataModule


class ModelWrapper(InitialisationRecordingMixin, metaclass=ABCMeta):
    """
    Base Model Wrapper

    Defines the interface in which to use a `model`, and `datamodule/Pipeline`
    """

    def __init__(self, model, data: Union[Pipeline, PipelineDataModule]):
        """
        Construct Base model wrapper

        Args:
            model (Any):
                Model to use.
            data (Union[Pipeline, PipelineDataModule]):
                Data to use. If not `PipelineDataModule` will be made into base.
                Will only then have `get_sample`.
        """
        super().__init__()

        if not isinstance(data, PipelineDataModule):
            data = PipelineDataModule(data)

        self.model = model
        self.datamodule = data

    def get_sample(self, idx, *, fake_batch_dim: bool = False):
        """Get sample from `datamodule`."""
        return self.datamodule.get_sample(idx, fake_batch_dim=fake_batch_dim)

    @property
    def pipelines(self):
        return self.datamodule.pipelines

    @abstractmethod
    def load(self, *args, **kwargs): ...

    @abstractmethod
    def save(self, *args, **kwargs): ...

    @abstractmethod
    def predict(self, data, *args, **kwargs): ...
