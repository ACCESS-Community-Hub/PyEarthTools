# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

import importlib

from edit.training.modules.loss.extremes import ExtremeLoss
from edit.training.modules.loss.centre_weighted import centre_weighted
from edit.training.modules.loss.rmse import RMSELoss
from edit.training.modules.loss.structure import SSIMLoss
from edit.training.modules.loss.component import ComponentLoss

from edit.training import modules


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
    Can either be name of one included in `torch.nn`, `piqa` or edit.training.modules.loss, or
    fully qualified import path

    Will attempt to load from in order, torch, piqa, edit.training, and full import path

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
