# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty 
# of any kind including, but not limited to, fitness for a particular 
# purpose. The user assumes the entire risk as to the use and 
# performance of the software. In no event shall the copyright holder 
# be held liable for any claim, damages or other liability arising 
# from the use of the software.

import importlib

TORCH_IMPORTED = True
try:
    from torch import nn
except (ModuleNotFoundError, ImportError):
    TORCH_IMPORTED = False

from edit.training import modules


def get_loss(loss_function: str, **loss_kwargs):
    """
    Get loss functions.
    Can either be name of one included in torch.nn or edit.training.modules.loss, or
    fully qualified import path

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
    if TORCH_IMPORTED:
        if hasattr(nn, loss_function):
            return getattr(nn, loss_function)(**loss_kwargs)
    if hasattr(modules.loss, loss_function):
        return getattr(modules.loss, loss_function)(**loss_kwargs)
    return get_callable(loss_function)(**loss_function)
