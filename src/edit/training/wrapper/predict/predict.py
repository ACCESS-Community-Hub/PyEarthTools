# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

from __future__ import annotations

from typing import Optional, Any

from abc import ABCMeta


from edit.utils.initialisation import InitialisationRecordingMixin

from edit.pipeline.controller import Pipeline
from edit.training.wrapper.wrapper import ModelWrapper


class PredictionWrapper(InitialisationRecordingMixin, metaclass=ABCMeta):
    """
    Model wrapper to enable prediction

    Hooks:
        `after_predict` (prediction) -> prediction:
            Function executed after data has been reversed from prediction.

    Usage:
        ```python
        model = ModelWrapper(MODEL_GOES_HERE, DATA_PIPELINE)
        predictor = PredictionWrapper(model)
        predictor.predict('2000-01-01T00')
        ```
    """

    def __init__(self, model: ModelWrapper, reverse_pipeline: Optional[Pipeline | int | str] = None):
        """
        Use a `model` to run a prediction.

        Retrieves initial conditions for `model.get_sample`, so set it's `Pipeline` accordingly.

        Args:
            model (ModelWrapper):
                Model and Data source to use.
            reverse_pipeline (Optional[Pipeline | int | str], optional):
                Override for `Pipeline` to use on the undo operation.
                If not given, will default to using `model.pipelines`.
                If `str` or `int` use value to index into `model.pipelines`. Useful if `model.pipelines`
                is a dictionay or tuple.
                Or can be `Pipeline` it self to use. If `reverse_pipeline.has_source()` is True, run `reverse_pipeline.undo`. otherwise
                apply pipeline with `reverse_pipeline.apply`
                Defaults to None.
        """
        super().__init__()
        self.model = model
        self._reverse_pipeline = reverse_pipeline

    def _predict(self, data, *args, **kwargs) -> Any:
        """
        Run prediction with `model` on given `data`, calling `self.model.predict`.

        """
        return self.model.predict(data, *args, **kwargs)

    def get_sample(self, idx, *, fake_batch_dim: bool = False):
        return self.model.get_sample(idx, fake_batch_dim=fake_batch_dim)

    @property
    def pipelines(self):
        return self.model.pipelines

    @property
    def datamodule(self):
        return self.model.datamodule

    @property
    def reverse_pipeline(self) -> Pipeline:
        if self._reverse_pipeline is None:
            if not isinstance(self.pipelines, Pipeline):
                raise TypeError("`reverse_pipeline` was not given but `datamodule` is not a simple `Pipeline`.")
            return self.pipelines
        elif isinstance(self._reverse_pipeline, Pipeline):
            return self._reverse_pipeline
        elif isinstance(self._reverse_pipeline, (str, int)):
            if not isinstance(self.pipelines, (tuple, dict, list)):
                raise TypeError(
                    f"Cannot index into underlying `Pipelines` with {self._reverse_pipeline!r} as they are not indexable."
                )
            return self.pipelines[self._reverse_pipeline]  # type: ignore #TODO better error messaging
        raise TypeError(f"Cannot parse `reverse_pipeline` of {type(self._reverse_pipeline)}.")

    def reverse(self, data):
        """
        Run `reverse_pipeline` on `data`.
        """
        reverse_pipeline = self.reverse_pipeline

        if reverse_pipeline.has_source():
            return reverse_pipeline.undo(data)
        return reverse_pipeline.apply(data)

    def predict(self, idx: Any, fake_batch_dim: bool = False, **kwargs) -> Any:
        """
        Run prediction with `model` with data from `idx`

        Args:
            idx (Any):
                Index to get initial conditions from
            fake_batch_dim (bool, optional):
                Whether to fake the batch dim. Defaults to True.

        Returns:
            (Any):
                Prediction data after being run through `reverse` and `after_predict`.
        """
        predicted_data = self._predict(self.get_sample(idx, fake_batch_dim=fake_batch_dim), **kwargs)
        if fake_batch_dim:
            predicted_data = predicted_data[0]
        return self.after_predict(self.reverse(predicted_data))

    def after_predict(self, prediction):
        """Hook to modify prediction, post `predict`."""
        return prediction
