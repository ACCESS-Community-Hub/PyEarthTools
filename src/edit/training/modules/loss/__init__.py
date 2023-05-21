
import importlib

from .extremes import ExtremeLoss
from .centre_weighted import centre_weighted
from .rmse import RMSELoss

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
    import torch.nn as nn

    if hasattr(nn, loss_function):
        return getattr(nn, loss_function)(**loss_kwargs)
    elif hasattr(modules.loss, loss_function):
        return getattr(modules.loss, loss_function)(**loss_kwargs)
    return _get_callable(loss_function)(**loss_function)
