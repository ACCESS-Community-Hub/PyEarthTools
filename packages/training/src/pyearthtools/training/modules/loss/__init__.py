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


import importlib

from pyearthtools.training.modules.loss.extremes import ExtremeLoss
from pyearthtools.training.modules.loss.centre_weighted import centre_weighted
from pyearthtools.training.modules.loss.rmse import RMSELoss
from pyearthtools.training.modules.loss.structure import SSIMLoss
from pyearthtools.training.modules.loss.component import ComponentLoss

from pyearthtools.training import modules


def _get_callable(module: str):
    """
    Provide dynamic import capability

    Parameters
    ----------
        module
            String of path the module, either module or specific function/class

    Returns
    -------
        Specified module or function
    """
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError:
        module = module.split(".")
        return getattr(_get_callable(".".join(module[:-1])), module[-1])
    except ValueError as e:
        raise ModuleNotFoundError("End of module definition reached")


def get_loss(loss_function: str, **loss_kwargs):
    """
    Get loss functions.
    Can either be name of one included in `torch.nn`, `piqa` or pyearthtools.training.modules.loss, or
    fully qualified import path

    Will attempt to load from in order, torch, piqa, pyearthtools.training, and full import path

    Refer to each packages documentation for kwargs and best use cases.

    Parameters
    ----------
    loss_function
        loss function to use
    **loss_kwargs
        kwargs to pass to init loss function

    Returns
    -------
        Initialised loss function
    """
    try:
        import torch.nn as nn

        torch_imported = True
    except (ModuleNotFoundError, ImportError):
        torch_imported = False

    try:
        import piqa

        piqa_imported = True
    except (ModuleNotFoundError, ImportError):
        piqa_imported = False

    if torch_imported and hasattr(nn, loss_function):
        return getattr(nn, loss_function)(**loss_kwargs)

    elif piqa_imported and hasattr(piqa, loss_function):
        return getattr(piqa, loss_function)(**loss_kwargs)

    elif hasattr(modules.loss, loss_function):
        return getattr(modules.loss, loss_function)(**loss_kwargs)
    return _get_callable(loss_function)(**loss_function)
