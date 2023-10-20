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
