import importlib
import logging
from typing import Union

from dset.training.data.templates import (
    DataIterationOperator,
    DataIterator,
    SequentialIterator,
)

logger = logging.getLogger('DSET_Training')

def get_callable(module: str):
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
        return getattr(get_callable(".".join(module[:-1])), module[-1])


@SequentialIterator
class Catch(DataIterationOperator):
    """
    Error Catching
    """

    def __init__(
        self, iterator: DataIterator, error: Union[tuple[Exception], Exception]
    ) -> None:
        """
        Catch Errors in iteration and continue

        Parameters
        ----------
        iterator
            Underlying Iterator
        error
            Error/s to catch
        """

        super().__init__(iterator)
        if not isinstance(error, (tuple, list)):
            error = (error,)
        error = list(error)

        for i, err in enumerate(error):
            if isinstance(err, str):
                error[i] = get_callable(err)

        self._error_to_catch = tuple(error)

        self.__doc__ = f"Catch Errors: {self._error_to_catch}"

    def __iter__(self):
        while True:
            try:
                yield next(self.iterator.__iter__())
            except StopIteration:
                break
            except self._error_to_catch as excep:
                logger.warn(f"In iteration an exception was caught. {excep}")
