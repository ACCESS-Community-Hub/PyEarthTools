# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

from typing import Literal, Sequence, Any, Union, Optional, TypeVar
from collections.abc import Mapping

from pathlib import Path
import yaml

import pyearthtools.utils
from pyearthtools.utils.initialisation.imports import dynamic_import
from pyearthtools.utils.initialisation.mixin import InitialisationRecordingMixin

Self = TypeVar("Self", Any, Any)

YAML_KEY = "!pyearthtools@"


# define the representer, responsible for serialization
def pyearthtools_initialisation_representer(dumper: yaml.Dumper, data: InitialisationRecordingMixin):
    type_data = type(data)

    extra_params = getattr(data, pyearthtools.utils.initialisation.OVERRIDE_KEY, {})

    initialisation_dict = data.initialisation
    if data._property is not None:
        initialisation_dict["__property"] = data._property

    if "class" in extra_params:
        module_path = f"{YAML_KEY}{extra_params['class']}"
    else:
        module_path = f"{YAML_KEY}{str(type_data.__module__)}.{type_data.__name__}"

    return dumper.represent_mapping(
        module_path,
        initialisation_dict,
    )


def pyearthtools_initialisation_constructor(
    loader: Union[yaml.loader.Loader, yaml.loader.FullLoader, yaml.loader.UnsafeLoader], tag_suffix: str, node
):
    tag_suffix = tag_suffix.replace(YAML_KEY, "")
    kwarg_dict = loader.construct_mapping(node, deep=True)
    kwarg_dict = {str(k): v for k, v in kwarg_dict.items()}

    property = kwarg_dict.pop("__property", None)

    obj = dynamic_import(tag_suffix)(*kwarg_dict.pop("__args", []), **kwarg_dict)  # type: ignore
    if property is not None:
        return getattr(obj, property)
    return obj


class Loader(yaml.Loader):

    def include(self, node):

        filename = Path(self.construct_scalar(node))  # type: ignore

        with open(filename, "r") as f:
            return yaml.load(f, Loader)


class Dumper(yaml.Dumper):
    """pyearthtools yaml dumper"""


Loader.add_constructor("!include", Loader.include)
Loader.add_multi_constructor(YAML_KEY, pyearthtools_initialisation_constructor)

Dumper.add_multi_representer(InitialisationRecordingMixin, pyearthtools_initialisation_representer)


__all__ = ["Loader", "Dumper"]
