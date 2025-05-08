# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Model wrapper functions.

Allows easy usage of any model for prediction or data retrieval.
"""
# pylint: disable=R0914,R0915

from __future__ import annotations

from pathlib import Path
import logging
from typing import Any
import warnings

import pyearthtools.zoo
from pyearthtools.zoo.commands import utils as command_utils
from pyearthtools.zoo import utils, exceptions
from pyearthtools.zoo.register import dynamic_import

LOG = logging.getLogger("pyearthtools.zoo")


def _prefunctions():
    """Run functions before evaluating the prediction arguments"""

    dynamic_import()


def _initialise_model(  # pylint: disable=R0913
    model: str,
    pipeline_name: str,
    output: Path | str | None,
    data_cache: Path | str | None = None,
    config_path: Path | str | None = None,
    quiet: bool = False,
    **kwargs,
) -> pyearthtools.zoo.BaseForecastModel:
    """
    Get initialised model class from config

    Args:
        model (str):
            Model name to load
        pipeline_name (str):
            Used to determine the filename for loading the pipeline config
        output (Path | str):
            Location to save data
        data_cache (Path | str | None):
            Where to cache data. Defaults to None
        config_path (Path | str | None):
           Override for config path. Defaults to None
        quiet (bool):
            Whether to print documentation of model.
        kwargs (dict[Any, Any]):
            Extra keyword arguments to send to the model.

    Raises:
        ModelException:
            If `model` is unknown
        ValueError:
            If `pipeline` is unknown
        TypeError:
            If arguments for the `model` are missing

    Returns:
        pyearthtools.zoo.BaseForecastModel:
            Initalised model class.

    """

    LOG.info("Loading %s", model)
    _prefunctions()

    model_class = getattr(pyearthtools.zoo.Models, model, None)
    if model_class is None:
        raise exceptions.ModelException(f"`pyearthtools.zoo` cannot find {model!r}.\n{pyearthtools.zoo.Models}")

    doc = model_class.__doc__
    if doc and not quiet:
        print(doc)

    config_path = Path(config_path).resolve() if config_path else config_path

    # Get list of available pipelines from the config path (more than one may be present)
    valid_pipelines = list(model_class._valid_pipeline(config_path=config_path).keys())  # pylint: disable=W0212
    if not model_class.is_valid_pipeline(pipeline_name, config_path=config_path):
        raise ValueError(f"Cannot parse pipeline: {pipeline_name!r}. Valid: {valid_pipelines}")

    required, _ = utils.get_arguments(model_class)
    for done in set(["pipeline", "output", "data_cache", "kwargs", "config_path", *kwargs.keys()]).intersection(
        set(required.keys())
    ):
        required.pop(done)

    if any(map(lambda x: x in pipeline_name.lower(), pyearthtools.zoo.LIVE_SUBSTRINGS)) and data_cache is None:
        warnings.warn(
            "Downloaded data is specified but the `data_cache` has been left unset. This will use a default location.",
            RuntimeWarning,
        )

    if len(required.keys()) > 0:
        raise TypeError(
            f"Some required keyword arguments are missing, {list(required.keys())}.\nPlease specify them as keywords."
        )

    return model_class(
        pipeline_name,
        output=output,
        data_cache=data_cache,
        config_path=config_path,
        **kwargs,
    )


def data(
    model: str,
    time: str,
    pipeline: str,
    data_cache: Path | str | None = None,
    config_path: Path | str | None = None,
    **kwargs,
) -> list[Any]:
    """
    Get data needed for model to run,

    Can be used to precache data for 'live' runs.

    Args:
        model (str):
            Model name to load
        time (str):
            Isoformat of time to get data for
        pipeline (str):
            Pipeline config to use
        data_cache (Path | str | None):
            Where to cache data. Defaults to None
        config_path (Path | str | None):
            Override for config path. Defaults to None
        kwargs (dict[Any, Any]):
            Extra keyword arguments to send to the model.

    Raises:
        RuntimeError:
            If an error occured, catch it with nice error message.

    Returns:
        list[Any]:
            Loaded data needed for the model.
    """

    initialised_model = _initialise_model(model, pipeline, None, data_cache, config_path, **kwargs)

    try:
        return initialised_model.data(time)
    except Exception as e:
        raise RuntimeError(
            "Something went wrong, please raise an issue with this trace "
            "and the associated arguments back to `pyearthtools.zoo`"
        ) from e


def predict(  # pylint: disable=R0913
    model: str,
    time: str,
    pipeline_name: str,
    output: Path | str,
    data_cache: Path | str | None = None,
    config_path: Path | str | None = None,
    **kwargs,
) -> Any:
    """
    Run a prediction for a given model, pipeline, and time.

    Args:
        model (str):
            Model name to load
        time (str):
            Isoformat of time to run prediction for
        pipeline (str):
            Pipeline config to use
        output (Path | str):
            Location to save data
        data_cache (Path | str | None):
            Where to cache data. Defaults to None
        config_path (Path | str | None):
            Override for config path. Defaults to None
        kwargs (dict[Any, Any]):
            Extra keyword arguments to send to the model.

    Raises:
        RuntimeError:
            If an error occured, catch it with nice error message.

    Returns:
        (Any):
            Loaded Predictions.
    """
    import pyearthtools.data  # pylint: disable=W0621

    extra_kwargs_str = " ".join(f"--{key} {value!r}" for key, value in kwargs.items())
    data_cache_str = " " if data_cache is None else f" --data_cache {data_cache}"

    # Add a transform to add the command line to the history attributes
    post_transforms = pyearthtools.data.transforms.attributes.set_attributes(
        history=f"pet predict {model!r} --pipeline_name {pipeline_name!r} --output {output!s}"
        f"{f' --config_path {config_path}' if config_path else ''} "
        f"--time {time}{data_cache_str} {extra_kwargs_str}".replace("  ", " ")
    ) + kwargs.pop("post_transforms", pyearthtools.data.TransformCollection())

    initialised_model = _initialise_model(
        model,
        pipeline_name,
        output,
        data_cache,
        config_path,
        post_transforms=post_transforms,
        **kwargs,
    )

    try:
        return initialised_model.run(time)
    except Exception as e:
        raise RuntimeError(
            "Something went wrong, please raise an issue with this trace "
            "and the associated arguments back to `pyearthtools.zoo`"
        ) from e


def _get_predict_kwargs_interactively(**kwargs: Any) -> dict[str, Any]:
    print("pyearthtools.zoo - Getting model kwargs interactively...\n")

    def prompt(
        key,
        prompt,
        values: list[str] | str | None = None,
        file: bool = False,
        show_options: bool = True,
    ) -> Any:
        if key in kwargs:
            if kwargs.get(key, None) is not None:
                return kwargs.pop(key)
            kwargs.pop(key, None)

        import readline

        readline.set_completer_delims("\t")
        readline.parse_and_bind("tab: complete")

        if file:
            readline.set_completer(utils.TabCompleter().path_completer)
        elif values is not None:
            readline.set_completer(utils.TabCompleter().create_list_completer(values))

        if values == []:
            values = None

        if isinstance(values, str):
            values = [values]

        parsed_result = None
        while parsed_result is None or parsed_result == "":
            parsed_result = utils.parse_str(
                input(
                    f"{prompt}?{(' ' + str(list(values))) if values is not None and show_options else ''},"
                    f"(Key: {key}): "
                )
            )
            if parsed_result is None or parsed_result == "":
                print("Must provide an entry...")

        return parsed_result

    _prefunctions()

    if kwargs.get("model", None) is None:
        print(pyearthtools.zoo.Models)
        print("(Specify model path with a '/' seperation.)")

    model = prompt(
        "model",
        "Which model would you like to use",
        list(pyearthtools.zoo.available_models()),
        show_options=False,
    )

    LOG.info("Loading %s", model)

    model_class = getattr(pyearthtools.zoo.Models, model)
    doc = model_class.__doc__  # pylint: disable=W0212

    if doc:
        print(doc)

    config_path = kwargs.pop("config_path", None)
    config_path = Path(config_path).resolve() if config_path else config_path

    valid_pipelines = list(model_class._valid_pipeline(config_path=config_path).keys())  # pylint: disable=W0212
    pipeline = prompt("pipeline", "Which pipeline / data source do you want to use", valid_pipelines)
    if not model_class.is_valid_pipeline(pipeline, config_path=config_path):
        raise ValueError(f"Cannot use pipeline: {pipeline}. Valid: {valid_pipelines}")

    data_cache = kwargs.pop("data_cache", None)
    if any(map(lambda x: x in pipeline.lower(), pyearthtools.zoo.LIVE_SUBSTRINGS)):
        data_cache = prompt("data_cache", "Where should 'downloaded' data be stored/found", file=True)

    output = prompt("output", "Where would you like to save the output (path)", file=True)

    required, noted_kwargs = utils.get_arguments(model_class)
    for done in set(["pipeline", "output", "data_cache", "kwargs", "config_path"]).intersection(set(required.keys())):
        required.pop(done)

    required_kwargs = {}
    for name, options in required.items():
        if name not in kwargs:
            print("Some required keyword arguments are missing...")
        required_kwargs[name] = prompt(name, f"{name!r}", options, file=False)

    remaining_kwargs = list(set(noted_kwargs.keys()).difference(kwargs.keys()))
    if "config_path" in remaining_kwargs:
        remaining_kwargs.remove("config_path")

    model_kwargs = input(f"Other kwargs {remaining_kwargs}? (Click Format): ")
    model_kwargs = command_utils.parse_str_to_dict(model_kwargs)

    time = prompt(
        "time",
        f"Basetime to {'predict' if not data else 'get data'} for",
        "isoformat",
        file=False,
    )

    extra_kwargs = dict(required_kwargs)
    extra_kwargs.update(model_kwargs)
    extra_kwargs.update(kwargs)

    return dict(  # pylint: disable=R1735
        model=model,
        pipeline=pipeline,
        output=output,
        config_path=str(config_path),
        time=time,
        data_cache=data_cache,
        **extra_kwargs,
    )  # pylint: disable=R1735


def interactive(
    **kwargs: str,
) -> Any:
    """
    Interactively run a prediction for a given model, pipeline, and time.

    Args:
        kwargs (dict[str, Any]):
            Specified kwargs to not prompt for.
            See `.predict` for a list of base ones.
    Raises:
        RuntimeError:
            If an error occured, catch it with nice error message.

    Returns:
        (Any):
            Loaded predictions.
    """
    predict_kwargs = _get_predict_kwargs_interactively(**kwargs)

    return predict(quiet=predict_kwargs.pop("quiet", True), **predict_kwargs)
