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

# ruff: noqa: F401

"""
`pyearthtools.zoo`

Contains the PyEarthTools model registry and provides the command-line interaction
for easy inference with bundled models.

Each model can setup data and pipeline configurations to allow a user to choose the data location,
and use live data.

# Registration
    There are two ways to register a model for use with `pyearthtools.zoo`.

    One allows it to be dynamically imported, and the other only visible when the other package
    is manually imported.

## Entrypoints - Dynamic
    This method allows use of the model once its installed with no other action needed,

    Assign an entry point underneath `pyearthtools.zoo.model`.
    If this model needs to be categorised, seperate the category with `_`.

    Example: - inside the `pyproject.toml`.
    ```toml
        [project.entry-points."pyearthtools.zoo.model"]
        NESM_Model = "MODULE_NAME.registered_model:ModelClass"
    ```
    This will be available underneath `pyearthtools.zoo.Models.NESM.Model`, and when using the command line,
    at `pyearthtools-models predict 'NESM/Model' ...`

## Manual Registration
    This method will only expose the model once it has been manually imported.
    It takes use of decorators to register the class.

    If this model needs to be categorised, seperate the category with `/`.

    Example:
    ```python
    import pyearthtools.zoo

    @pyearthtools.zoo.register('Category/Name`)
    class MODEL():
        ....
    ```

    Once the model is imported, it will be accessible at
    `pyearthtools.zoo.Models.Category.Name` or `pyearthtools.zoo.Models['Category/Name']`

    The model will also be visible when using the `pyearthtools.zoo.predict/interactive/data` functions.

## Dynamic Importing
    Setting $pyearthtools_MODELS_IMPORTS can allow modules not in the environment to be imported when running `pyearthtools-models`
    to allow the commands to access custom models.

    Seperate modules by ':', and within each specification, split the name and path by '@'.

        i.e.

        ```shell
        export pyearthtools_MODELS_IMPORTS=MODULE_NAME@PATH_TO_MODULE_GOES_HERE:
        ```
    Be aware that depending on what is being imported, this may drastically reduce command responsiveness.
"""

from pyearthtools.zoo import config as _
from pyearthtools.zoo import logger as __

from pyearthtools.zoo.register import register
from pyearthtools.zoo.model import BaseForecastModel

from pyearthtools.zoo.warnings import AccessorRegistrationWarning
from pyearthtools.zoo.exceptions import ModelException, ModelRegistrationException

from pyearthtools.zoo import utils

Models = utils.AvailableModels()

from pyearthtools.zoo.predict import predict, data, interactive  # pylint: disable=C0413  # noqa: E402
from pyearthtools.zoo.commands import commands  # pylint: disable=C0413 # noqa: E402

__version__ = "0.2.0"


def available_models() -> tuple[str, ...]:
    """Get available models"""
    return Models.available


LIVE_SUBSTRINGS = ["live", "cds"]

__all__ = ["register", "BaseForecastModel", "Models", "predict", "data", "interactive"]
