# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

from __future__ import annotations
import functools

from typing import Literal, TypeVar, Any, Optional

from abc import abstractmethod

import xarray as xr
import numpy as np
import tqdm.auto as tqdm
import logging


from edit.data import TimeDelta, EDITDatetime, TimeRange

from edit.pipeline.controller import Pipeline
from edit.training.wrapper.wrapper import ModelWrapper
from edit.training.wrapper.predict.predict import Predictor

from edit.training.manage import Variables

XR_TYPE = TypeVar("XR_TYPE", xr.Dataset, xr.DataArray)
LOG = logging.getLogger("edit.training")


class TimeSeriesPredictor(Predictor):
    """
    Temporal predictions

    Adds `recurrent`, which is expected to be implemented by subclass.

    Provides `prepare_output` as a function hook to modify model outputs before using
    as input.

    Usage:
        ```python
        model = ModelWrapper(MODEL_GOES_HERE, DATA_PIPELINE)
        predictor = TimeSeriesPredictionWrapper(model)
        predictor.recurrent('2000-01-01T00', steps = 10)
        ```
    """

    def __init__(
        self,
        model: ModelWrapper,
        reverse_pipeline: Pipeline | int | str | None = None,
        *,
        fix_time_dim: bool = True,
        interval: int | str | TimeDelta = 1,
        time_dim: str = "time",
    ):
        """
        Predict with a `model` a time series.

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
            fix_time_dim (bool, optional):
                Fix time dimension after prediction. Defaults to True.
            interval (int | str | TimeDelta, optional):
                Interval of temporal predictions, must be passable by `edit.data.TimeDelta`. Defaults to 1.
            time_dim (str, optional):
                Name of time dimension in undone data. Defaults to "time".
        """
        super().__init__(model, reverse_pipeline)  # type: ignore
        self.record_initialisation()

        self._interval = TimeDelta(interval)
        self._time_dim = time_dim
        self._fix_time_dim = fix_time_dim

    def fix_time_dim(self, idx, data: XR_TYPE, *, offset: int = 1) -> XR_TYPE:
        """
        Time dimension is usually wrong after running out, so this attempts to fix it.

        Uses `interval` and `time_dim` from `__init__`.

        Args:
            idx (Any):
                Starting index
            data (XR_TYPE):
                Data to fix time dimension of
            offset (int, optional):
                Offset of idx. Defaults to 1.

        Returns:
            (XR_TYPE):
                Data with fixed time dim
        """
        if not self._fix_time_dim:
            return data

        interval = TimeDelta(self._interval)

        idx = EDITDatetime(idx) + (interval * offset)
        len_time = len(data[self._time_dim])

        time_coord = list(map(lambda x: x.datetime64(), TimeRange(idx, idx + (interval * len_time), interval)))
        encoding = data[self._time_dim].encoding
        attributes = {"long_name": "time"}

        fixed_time = data.assign_coords({self._time_dim: time_coord})
        fixed_time[self._time_dim].encoding.update(encoding)  # Maintain time encoding
        fixed_time[self._time_dim].attrs.update(attributes)

        return fixed_time

    def predict(self, idx: Any, fake_batch_dim: bool = True, **kwargs) -> Any:
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
        reversed_data = super().predict(idx, fake_batch_dim=fake_batch_dim, **kwargs)

        if isinstance(reversed_data, tuple):
            return tuple(map(functools.partial(self.fix_time_dim, idx), reversed_data))
        return self.fix_time_dim(idx, reversed_data)

    @abstractmethod
    def recurrent(self, idx, steps: int, **kwargs): ...

    def prepare_output(self, output):
        """Hook to prepare output for inputs"""
        return output


class ManualTimeSeriesPredictor(TimeSeriesPredictor):
    """
    Interface for TimeSeries prediction in which the `model` itself handles all of the recurrency.
    """

    def recurrent(self, *_, **__):
        raise NotImplementedError("Model handles the recurrency itself, call `predict` instead")


class TimeSeriesAutoRecurrentPredictor(TimeSeriesPredictor):
    """
    AutoRecurrent temporal predictions.
    """

    def __init__(
        self,
        model: ModelWrapper,
        reverse_pipeline: Pipeline | int | str | None = None,
        *,
        fix_time_dim: bool = True,
        interval: int | str | TimeDelta = 1,
        time_dim: str = "time",
        combine: Optional[Literal["stack", "concat"]] = "concat",
        combine_axis: int = 0,
    ):
        """
        Predict with a `model` a time series.


        `combine` and `combine_axis` can be used to modify how timesteps are combined,
        if model predictions have a leading time dim, use `concat`, or if time dim at 2nd axis,
        set `combine_axis = 1`.
        If no time dim included, set `combine` to `stack`.

        If data must be reversed before being combined, set `combine = None`.
        Will be undone, and `xr.combine_by_coords` used.

        ## Warning:
            The pipeline that is used to undo the predictions, if `combine` must allow a change in the time dimension,
            i.e. no squish's or expand's on that dim.

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
            fix_time_dim (bool, optional):
                Fix time dimension after prediction. Defaults to True.
            interval (int | str | TimeDelta, optional):
                Interval of temporal predictions, must be passable by `edit.data.TimeDelta`. Defaults to 1.
            time_dim (str, optional):
                Name of time dimension in undone data. Defaults to "time".
            combine (Optional[Literal['stack', 'combine']], optional):
                How to combine timesteps, either stack on `combine_axis` or concat.
                If `None`, do not combine before undo operation and use `xr.combine_by_coords` after.
                `concat` concatenates on existing axis, whereas `stack` stacks on new axis.
                Defaults to 'concat'.
            combine_axis (int, optional):
                If to `combine` which axis to combine on. Will remove the batch dim, so 0 is actually 1 with batch dim included.
        """
        super().__init__(model, reverse_pipeline, fix_time_dim=fix_time_dim, interval=interval, time_dim=time_dim)
        self.record_initialisation()

        self._combine_func = combine
        self._combine_axis = combine_axis

    def _combine(self, idx: Any, outputs: list):
        """
        Combine timesteps together
        """

        if self._combine_func is not None:
            LOG.debug(f"Combining steps with {self._combine_func!r}.")

            combine_func = np.stack if self._combine_func == "stack" else np.concatenate
            LOG.debug(f"Prior to combine outputs[0] had shape {outputs[0].shape}.")
            stacked_outputs = combine_func(outputs, axis=self._combine_axis)

            LOG.debug(f"Combined data was of shape: {stacked_outputs.shape}")

            reversed_data = self.reverse(stacked_outputs)
            return self.after_predict(self.fix_time_dim(idx, reversed_data))  # type: ignore

        reversed_outputs = []
        for step, out in enumerate(outputs):
            reversed_data = self.reverse(out)
            reversed_outputs.append(self.fix_time_dim(idx, reversed_data, offset=step * out.shape[self._combine_axis]))  # type: ignore

        return self.after_predict(xr.combine_by_coords(reversed_outputs))

    def recurrent(
        self,
        idx: Any,
        steps: int,
        *,
        fake_batch_dim: bool = True,
        verbose: bool = False,
    ) -> Any:
        """
        Predict autorecurrently

        Requires model `inputs` == `outputs @ t+1`

        Runs for n`steps` ahead, feeding model outputs back in to predict at the next step.


        Args:
            idx (Any):
                Index to get initial conditions at
            steps (int):
                Number of steps to roll out for. Model iterations
            fake_batch_dim (bool, optional):
                Fake batch dim when getting a sample of data. Defaults to True.
            verbose (bool, optional):
                Show progress. Defaults to False.

        Returns:
            (Any):
                Combined temporal data
        """

        input_data = self.get_sample(idx, fake_batch_dim=fake_batch_dim)
        _ = self.reverse_pipeline

        outputs = []
        for step in tqdm.trange(steps, disable=not verbose, desc="Predicting Autorecurrently"):
            model_output = self._predict(input_data)

            LOG.debug(f"At step {step} model output was of shape {model_output.shape}")

            if fake_batch_dim:
                outputs.append(model_output[0])
            else:
                outputs.append(model_output)
            input_data = self.prepare_output(model_output)
        return self._combine(idx, outputs)


class TimeSeriesManagedPredictor(TimeSeriesAutoRecurrentPredictor):
    """
    AutoRecurrent prediction where output != input.

    Uses `Variables` to manage data shape, and can either retrieve missing data from `Pipelines` or take from input.
    If not `take_missing_from_input`, expects `model.datamodule.pipelines` to be a dictionary, and `variable_manager` to use the same names.

    If `datamodule` returns data, `take_missing_from_input` must be True, as data cannot be retrieved otherwise.

    Examples:
        Say a model takes 10 prognostics, 4 forcings, and predicts only the prognostics
        ```python
        variable_manager = Variables(prognostics = 10, forcings = 4)
        # model has pipeline datamodule as a dictionary with `prognostics` and `forcings`.
        predictor = TimeSeriesManagedRecurrent(model, variable_manager, output_order = 'P', reverse_pipeline = 'prognostics')
        predictor.recurrent('2000-01-01T00', 10)
        ```
        If diagnostics are given back by the model, and not given in the inputs.

    If `reverse_pipeline` is not given and `pipelines` data is not a dictionary, put missing data at the end of the order.

    ## Note:
        If diagnostic type variables are returned, it is unlikely that `reverse_pipeline` referencing an input pipeline will work,
        so it is best to pass in a new pipeline built to undo model outputs.

    """

    def __init__(
        self,
        model: ModelWrapper,
        variable_manager: Variables,
        output_order: str,
        reverse_pipeline: Pipeline | str | None = None,
        *,
        input_order: Optional[str] = None,
        variable_axis: int = 0,
        take_missing_from_input: bool = False,
        fix_time_dim: bool = True,
        interval: int | str | TimeDelta = 1,
        time_dim: str = "time",
        combine: None | Literal["stack"] | Literal["concat"] = "concat",
        combine_axis: int = 1,
        **extra_pipelines: Pipeline,
    ):
        """
        AutoRecurrent predictions where output != input.

        Expects `model.datamodule.pipelines` to be a dictionary, and `variable_manager` to use the same names.
        Based on `output_order` finds the missing data needed for a prediction, and queries the `datamodule` for it
        if `take_missing_from_input` is False, otherwise pull from input.

        `combine_axis` is used to identify number of time steps predicted in one pass of the model.

        Args:
            model (ModelWrapper):
                Model and Data source to use.
            variable_manager (Variables):
                Variable manager, used to extract components from `output` data, and `input` if `datamodule` is not a dictionary.
            output_order (str):
                Order of output for use with `variable_manager`.
                E.g. variable_manager.split(model_output, output_order)
                If model outputs inputs, and diagnostics, `output_order` would be `ID`.
            input_order (str, Optional):
                Override for order of input data, if incoming data is not a dictionary.
                If not given, and `incoming data` is array will use default order from `variable_manager`.
                Defaults to None.
            variable_axis (int, Optional):
                Axis of tensor of variables. Used to ensure seperation of according to `output_order`.
                Only used if model returns a tensor.
                Defaults to 0.
            take_missing_from_input (bool):
                Whether to take missing data from the input. Defaults to False.
            extra_pipelines (Pipeline, optional):
                Extra pipelines to use for missing data retrieval instead of `datamodule` if `take_missing_from_input` is False.
                Expected to have the same names as `variable_manager` and `datamodule`.
                Defaults to {}.

        See `TimeSeriesAutoRecurrent` for docs for the rest of the args.
        """
        super().__init__(
            model,
            reverse_pipeline,
            fix_time_dim=fix_time_dim,
            interval=interval,
            time_dim=time_dim,
            combine=combine,
            combine_axis=combine_axis,
        )
        self.record_initialisation()
        self.variable_manager = variable_manager
        self._variable_axis = variable_axis

        self._input_order = input_order
        self._output_order = output_order

        self._take_missing_from_input = take_missing_from_input
        self._extra_pipelines = extra_pipelines

        if not isinstance(self.pipelines, dict) and not take_missing_from_input and extra_pipelines is None:
            raise TypeError(
                "Cannot manage an autorecurrent prediction if `pipelines` is not a dictionary, and `take_missing_from_input` is False."
            )

    def recurrent(
        self,
        idx: Any,
        steps: int,
        *,
        fake_batch_dim: bool = True,
        verbose: bool = False,
    ) -> Any:
        """
        Predict autorecurrently

        outputs do not have to equal inputs.

        Will split the outputs based on `output_order`, find missing keys, and get from `pipelines` or `input`.

        Can be used with `datamodules` that return dictionarys or data,

        If `model` returns a dictionary, will look for a key `prediction` for predictions to pass to outputs.

        Args:
            idx (Any):
                Index to get initial conditions at
            steps (int):
                Number of steps to roll out for. Model iterations
            fake_batch_dim (bool, optional):
                Fake batch dim when getting a sample of data. Defaults to True.
            verbose (bool, optional):
                Show progress. Defaults to False.

        Returns:
            (Any):
                Combined temporal data
        """

        input_data = self.get_sample(idx, fake_batch_dim=fake_batch_dim)
        input_as_dict = isinstance(input_data, dict)

        input_dict = (
            dict(input_data)
            if isinstance(input_data, dict)
            else self.variable_manager.split(
                input_data if not fake_batch_dim else input_data[0], order=self._input_order
            )
        )

        _ = self.reverse_pipeline

        outputs = []
        for step in tqdm.trange(steps, disable=not verbose, desc="Predicting Autorecurrently"):

            model_output = self._predict(input_data)

            if isinstance(model_output, dict):
                output_components = dict(model_output)
                model_output = model_output["prediction"]
                if fake_batch_dim:
                    model_output = model_output[0]
            else:
                if fake_batch_dim:
                    model_output = model_output[0]

                LOG.debug(f"At step {step} model output was of shape {model_output.shape}")

                model_output_shaped = np.moveaxis(model_output, self._variable_axis, 0)
                output_components = {
                    key: np.moveaxis(val, 0, self._variable_axis)
                    for key, val in self.variable_manager.split(model_output_shaped, self._output_order).items()
                }

            current_time_step = EDITDatetime(idx) + (self._interval * step * model_output.shape[self._combine_axis])
            outputs.append(model_output)

            output_components = self.prepare_output(output_components)

            output_components_shape = {key: val.shape for key, val in output_components.items()}
            LOG.debug(f"At step {step} model output components was of shape {output_components_shape}")

            for data_name in input_dict.keys():  # type: ignore
                if data_name not in output_components:
                    if self._take_missing_from_input:
                        output_components[data_name] = input_dict[data_name]  # type: ignore
                    else:
                        if self._extra_pipelines is not None and data_name in self._extra_pipelines:
                            pipeline_for_data = self._extra_pipelines[data_name]
                        else:
                            pipeline_for_data = self.pipelines[data_name]  # type: ignore
                        output_components[data_name] = pipeline_for_data[str(current_time_step)]  # type: ignore

            output_components = {key: output_components[key] for key in input_dict.keys()}

            if input_as_dict:
                if fake_batch_dim:
                    input_data = {key: self.datamodule.fake_batch_dim(val) for key, val in output_components.items()}
                else:
                    input_data = output_components
            else:
                input_data = self.variable_manager.join(**output_components)
                if fake_batch_dim:
                    input_data = self.datamodule.fake_batch_dim(input_data)

        return self._combine(idx, outputs)
