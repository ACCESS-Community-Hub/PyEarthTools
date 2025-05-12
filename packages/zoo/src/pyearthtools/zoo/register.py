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
Allow models to be registered.

Any registered model is accessible underneath `pyearthtools.zoo.*`

Useful for using built models outside of the commands.
"""

from __future__ import annotations

from typing import Callable, Any, Literal
import importlib

import sys
import warnings

from pathlib import Path

import pyearthtools.zoo


def register(name: str, exists: Literal["warn", "ignore", "error"] = "warn") -> Callable[..., Any]:
    """
    Register a custom model for `pyearthtools.zoo`.

    Any registered model is accessible underneath `pyearthtools.zoo.Models.*`

    By setting the key with '/' the categories of the model can be set.

    Example:
        >>> register('Category/MODEL')(MODEL)
        >>> # Accessible at `pyearthtools.zoo.Models.Category.MODEL`

    Args:
        name (str):
            Name under which the model should be registered. A warning is issued
            if this name conflicts with a preexisting model.
    """

    def decorator(
        registered_model: Callable[..., Any],
    ) -> Callable[..., Callable[..., Any]]:
        """Register `accessor` under `name` in `pyearthtools.zoo.Models`"""
        if name in pyearthtools.zoo.Models:
            msg = (
                f"Registration of model {registered_model!r} under name {name!r}"
                "is overriding a preexisting registered model with the same name."
            )

            if exists == "warn":
                warnings.warn(
                    msg,
                    pyearthtools.zoo.AccessorRegistrationWarning,
                    stacklevel=2,
                )
            elif exists == "error":
                raise pyearthtools.zoo.ModelRegistrationException(msg)

        if getattr(registered_model, "_name", None) is None:
            registered_model._name = name  # pylint: disable=W0212
        pyearthtools.zoo.Models[name] = registered_model
        return registered_model

    return decorator


def dynamic_import():
    """
    Dynamically import modules specified in config under `models.imports`.
    Therefore can be configured by the user in `~/.config/pyearthtools/models.yaml`,
    or by setting `pyearthtools_MODELS__IMPORTS` in the environment.

    Allows for custom models which are simply `registered` to be used in the commands with `pyearthtools-models`.

    Syntax:
        Seperate modules by ':', and within each specification, split the name and path by '@'.

            i.e.

            ```shell
            export pyearthtools_MODELS__IMPORTS=MODULE_NAME@PATH_TO_MODULE_GOES_HERE:
            ```
    """
    import pyearthtools.utils

    imports = pyearthtools.utils.config.get("models.imports")
    if isinstance(imports, str):
        imports = imports.replace('"', "").replace("'", "").split(":")

    if imports is None:
        return

    for import_spec in imports:
        module_name, path = import_spec.split("@", maxsplit=1)
        sys.path.append(str(Path(path).expanduser().resolve().absolute()))
        try:
            importlib.import_module(module_name)
        except (ImportError, ModuleNotFoundError) as e:
            warnings.warn(
                f"Could not import {module_name!r} with {path!r} added to path, due to \n {type(e).__name__}: {e}",
                UserWarning,
            )
