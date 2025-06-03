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

# pylint: disable=C0116,R0913,E1120,R1735

"""
`pyearthtools-models` commands

Available sub commands
```
- predict           - Predict
- interactive       - Predict Interactively
- models            - See what models are available
```
"""

from __future__ import annotations

import sys
from pathlib import Path
import logging

import click

import pyearthtools.zoo
from pyearthtools.zoo.commands import utils as command_utils
from pyearthtools.zoo.utils import AvailableModels

available_models = AvailableModels()

LOG = logging.getLogger("pyearthtools.zoo")


@click.group("pyearthtools-models")
@click.option("--debug", is_flag=True, help="Log debug")
@click.option("--info", is_flag=True, help="Log info")
def entry_point(debug, info):
    """
    `pyearthtools.zoo` commands
    """
    if debug or info:
        log_level = "DEBUG" if debug else "INFO"

        import pyearthtools.utils

        pyearthtools.utils.config.set({"logger.default.stream_logger_level": log_level})
        pyearthtools.utils.config.set({"logger.models.stream_logger_level": log_level})

        pyearthtools.utils.logger.initiate_logging("models")


@entry_point.command("models", help="Print available models.")
def models():
    from pyearthtools.zoo.register import dynamic_import

    dynamic_import()

    _models = pyearthtools.zoo.Models
    if len(_models.available) == 0:
        print("NO MODELS AVAILABLE.")
        sys.exit(0)

    print(_models)
    print("(Specify category with a '/' seperation.)")
    sys.exit(0)


HELP_STR = f"""
    Run Prediction of registered model

    \b
{available_models!s}

    * EACH MODEL MAY REQUIRE SPECIFIC KWARGS NOT HERE LISTED *

    All other keyword arguments will be passed to underlying model

    MODEL: Model to use

    """


@entry_point.command(
    "predict",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    ),
    help=HELP_STR,
)
@click.pass_context
@click.argument("model", type=str)
@click.option("--time", type=str, required=True, help="Basetime to predict from")
@click.option("--pipeline_name", type=str, required=True, help="Pipeline config")
@click.option("--output", type=click.Path(file_okay=False), required=True, help="Output directory")
@click.option(
    "--data_cache",
    type=click.Path(file_okay=False),
    required=False,
    help="Data cache location",
)
@click.option(
    "--config_path",
    type=click.Path(file_okay=False),
    default=None,
    help="Override for config path",
)
def run_predict(
    ctx,
    model: str,
    time: str,
    output: str,
    pipeline_name: str,
    data_cache: str,
    config_path: str,
):
    ctx_kwargs = command_utils.get_keyword_from_ctx(ctx)

    if "data" in ctx_kwargs:
        raise ValueError("data has been deprecated as an argument for `predict`, use `data`.")

    predictions = pyearthtools.zoo.predict(
        model,
        time,
        pipeline_name=pipeline_name,
        output=output,
        data_cache=data_cache,
        config_path=config_path,
        **ctx_kwargs,
    )

    # FIXME: This is probably very much the wrong way to save data, but something
    # is needed as a stopgap while the train/predict/save cycle is restored
    # post migration, and the more appropriate use of the output directory
    # through pipelines will be revisited afterwards

    if output is not None:
        import os

        filename = "standard_filename.nc"
        filename = os.path.join(output, filename)
        predictions.to_netcdf(filename)

    print(f"Model Predictions saved underneath {output!r}.")


INTERACTIVE_HELP_STR = f"""
    Interactive Prediction

    Run Prediction of registered model, getting arguments interactively

    \b
{available_models!s}

    If a keyword argument is not given, this command will prompt the user,
    however, if passed, will skip.
    """


@entry_point.command(
    "interactive",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    ),
    help=INTERACTIVE_HELP_STR,
)
@click.pass_context
@click.option("--model", type=str, required=False, help="Model to use", default=None)
@click.option("--time", type=str, required=False, help="Basetime to predict from", default=None)
@click.option("--pipeline_name", type=str, required=False, help="Pipeline config", default=None)
@click.option(
    "--output",
    type=click.Path(file_okay=False),
    required=False,
    help="Output directory",
    default=None,
)
@click.option(
    "--data_cache",
    type=click.Path(file_okay=False),
    required=False,
    help="Data cache location",
    default=None,
)
@click.option(
    "--config_path",
    type=click.Path(file_okay=False),
    default=None,
    help="Override for config path",
)
def interactive(
    ctx,
    model: str,
    time: str,
    pipeline_name: str,
    output: Path | str,
    data_cache: Path | str,
    config_path: str | Path,
):
    ctx_kwargs = command_utils.get_keyword_from_ctx(ctx)

    if "data" in ctx_kwargs:
        raise ValueError("data has been deprecated as an argument for `predict`, use `data`.")

    from pyearthtools.zoo.predict import _get_predict_kwargs_interactively

    predict_kwargs = _get_predict_kwargs_interactively(
        model=model,
        time=time,
        pipeline_name=pipeline_name,
        output=output,
        data_cache=data_cache,
        config_path=config_path,
        **ctx_kwargs,
    )

    kwargs_str = " ".join(
        f"--{key} {value!r}" for key, value in predict_kwargs.items() if value is not None and not key == "model"
    )

    print("Use this command to run non interactively:")
    print(f"\tpyearthtools-models predict {predict_kwargs['model']!r} {kwargs_str}".replace("  ", " "))
    pyearthtools.zoo.predict(quiet=predict_kwargs.pop("quiet", True), **predict_kwargs)

    print(f"Model Predictions saved underneath {predict_kwargs['output']!r}.")


DATA_HELP_STR = f"""
    Get data for model

    \b
{available_models!s}

    * EACH MODEL MAY REQUIRE SPECIFIC KWARGS NOT HERE LISTED *

    All other keyword arguments will be passed to underlying model

    MODEL: Model to use

    """


@entry_point.command(
    "data",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    ),
    help=DATA_HELP_STR,
)
@click.pass_context
@click.argument("model", type=str)
@click.option("--time", type=str, required=True, help="Basetime to get data for")
@click.option("--pipeline_name", type=str, default=None, required=True, help="Pipeline config")
@click.option(
    "--data_cache",
    type=click.Path(file_okay=False),
    required=False,
    help="Data cache location",
)
@click.option(
    "--config_path",
    type=click.Path(file_okay=False),
    default=None,
    help="Override for config path",
)
def data(ctx, model, time, pipeline, data_cache, config_path: str | Path):
    ctx_kwargs = command_utils.get_keyword_from_ctx(ctx)
    _ = pyearthtools.zoo.data(model, time, pipeline, data_cache, config_path=config_path, **ctx_kwargs)

    print(f"All data was retrieved. {'Cached at ' + data_cache if data_cache is not None else ''}")
    print("Model ready for prediction.")


if __name__ == "__main__":
    entry_point()
