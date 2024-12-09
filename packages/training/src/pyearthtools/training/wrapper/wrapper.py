# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.
from __future__ import annotations

from abc import ABCMeta, abstractmethod

from pyearthtools.utils.initialisation import InitialisationRecordingMixin
from pyearthtools.pipeline import Pipeline

from pyearthtools.training.data import PipelineDataModule


class ModelWrapper(InitialisationRecordingMixin, metaclass=ABCMeta):
    """
    Base Model Wrapper

    Defines the interface in which to use a `model`, and `datamodule/Pipeline`
    """

    _default_datamodule: type[PipelineDataModule] = PipelineDataModule
    _record_model: bool = False

    def __init__(
        self,
        model,
        data: (
            dict[str, Pipeline | str | tuple[Pipeline, ...]]
            | tuple[Pipeline | str, ...]
            | str
            | Pipeline
            | PipelineDataModule
        ),
    ):
        """
        Construct Base model wrapper

        `model` will not be recorded in the initialisation by default, set `_record_model` to change
        this behaviour.

        Args:
            model (Any):
                Model to use.
            data (dict[str, Pipeline | tuple[Pipeline, ...]] | tuple[Pipeline, ...] | Pipeline | PipelineDataModule):
                Data to use. If not `PipelineDataModule` will be made into `_default_datamodule`.
                Will only then have `get_sample`.
        """
        super().__init__()
        self.record_initialisation(ignore="model" if self._record_model else None)

        if not isinstance(data, PipelineDataModule):
            data = self._default_datamodule(data)
        if not isinstance(data, self._default_datamodule):
            data = self._default_datamodule(data.pipelines, train_split=data._train_split, valid_split=data._valid_split)  # type: ignore

        self.model = model
        self.datamodule = data

    def get_sample(self, idx, *, fake_batch_dim: bool = False):
        """Get sample from the `datamodule`."""
        return self.datamodule.get_sample(idx, fake_batch_dim=fake_batch_dim)

    @property
    def pipelines(self):
        """Get pipelines from the `datamodule`."""
        return self.datamodule.pipelines

    @property
    def splits(self):
        """Training and Validation split as configured by the `datamodule`."""
        return {"training": self.datamodule._train_split, "validation": self.datamodule._valid_split}

    @abstractmethod
    def load(self, *args, **kwargs):
        """Load model"""

    @abstractmethod
    def save(self, *args, **kwargs):
        """Save model"""

    @abstractmethod
    def predict(self, data, *args, **kwargs):
        """
        Run a forward pass with the model

        Args:
            data:
                Data to run prediction with
        """
