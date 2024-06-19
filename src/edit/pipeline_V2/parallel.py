# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

"""
Parallel Interfaces for `edit.pipeline`

Provides a class exposing main parallel functions, with the actual implementation
abstracted away.

Therefore, if dask is available and enabled, it can be automatically used, but if not
no code is needed to be changed to run in serial.
"""

from abc import abstractmethod
import functools
from typing import Callable, TypeVar, Any

from edit.utils.decorators import classproperty
from edit.utils.context import ChangeValue

from edit.pipeline_V2 import config

Future = TypeVar("Future", Any, Any)


enable = ChangeValue(config, "RUN_PARALLEL", True)
disable = ChangeValue(config, "RUN_PARALLEL", False)


class ParallelInterface:
    """
    Interface for parallel computation.
    Allows for the system to define how to parallelise or if to, without the user changing code.

    Mimic's the `dask` interface

    """

    def __new__(cls, *_a, **_k):
        if not config.RUN_PARALLEL:
            cls = SerialInterface
            return super().__new__(cls)  # type: ignore
        try:
            import dask.distributed

            cls = DaskParallelInterface
        except (ModuleNotFoundError, ImportError):
            cls = SerialInterface
        return super().__new__(cls)  # type: ignore

    @abstractmethod
    def submit(self, func, *args, **kwargs) -> Future:
        pass

    @abstractmethod
    def map(self, func, *iterables, **kwargs) -> list[Future]:
        pass

    @abstractmethod
    def gather(self, futures, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def wait(self, futures, **kwargs) -> Any:
        pass

    @abstractmethod
    def collect(self, futures, **kwargs) -> Any:
        pass

    @abstractmethod
    def fire_and_forget(self, futures) -> None:
        pass


class SerialInterface(ParallelInterface):
    """Execute things in serial, with all the api of a ParallelInterface"""

    def _wrap_in_result(self, obj):
        class FutureFaker:
            def __init__(self, obj):
                self._obj = obj

            def result(self, *args):
                return self._obj

        return FutureFaker(obj)

    def submit(self, func, *args, **kwargs):
        return self._wrap_in_result(func(*args, **kwargs))

    def map(self, func, iterables, *iter, **kwargs) -> Future:
        return tuple(map(lambda i: self._wrap_in_result(func(i, **kwargs)), iterables, *iter))  # type: ignore

    def gather(self, futures, *args, **kwargs):
        return type(futures)(map(lambda x: x.result(), futures))

    def wait(self, futures, **kwargs):
        return futures

    def collect(self, futures):
        return type(futures)(map(lambda x: x.result(), futures))

    def fire_and_forget(self, futures):
        pass


class DaskParallelInterface(ParallelInterface):
    """
    Wrapper for the dask Client
    """

    @classproperty
    def client(cls):
        """Get dask client"""
        from dask.distributed import Client
        from distributed.client import _get_global_client

        dask_config = config.DASK_CONFIG
        dask_config["processes"] = dask_config.pop("processes", False)

        if _get_global_client() is None and not config.START_DASK:
            raise RuntimeError(f"Cannot start dask cluster if `config.START_DASK` is False.")

        return _get_global_client() or Client(**dask_config)

    def defer_to_client(func: Callable):  # type: ignore
        wrapped = func
        try:
            from dask.distributed import Client

            wrapped = getattr(Client, func.__name__).__doc__
        except:
            pass

        def wrapper(self, *args, **kwargs):
            return getattr(DaskParallelInterface.client, func.__name__)(*args, **kwargs)

        return functools.update_wrapper(wrapper, wrapped)

    def __getattr__(self, key):
        return getattr(self.client, key)

    @defer_to_client
    def submit(self, *args, **kwargs): ...

    @defer_to_client
    def map(self, func, *iterables, **kwargs):
        """Map function across iterables"""
        ...

    @defer_to_client
    def gather(self, *args, **kwargs): ...

    def wait(self, futures, **kwargs):
        from dask.distributed import wait

        return wait(futures, **kwargs)

    def collect(self, futures):
        type_to_make = type(futures)
        if type_to_make == type((i for i in [])):
            type_to_make = tuple
        return type_to_make(map(lambda x: x.result(), futures))

    def fire_and_forget(self, futures, **kwargs):
        from dask.distributed import fire_and_forget

        return fire_and_forget(futures)


def get_parallel() -> ParallelInterface:
    """
    Get parallel interface

    Args:
        start_dask (bool, optional):
            Whether to use dask even if a client doesn't already exist.
            Will start a new cluster is True.
            Defaults to True.

    Returns:
        (ParallelInterface):
            Parallel Interface

            Abstracts away the parallelisation, so that if actually serial,
            no code change is needed.
    """
    from multiprocessing.process import current_process

    if not config.RUN_PARALLEL:
        return SerialInterface()
    try:
        import dask.distributed
        from distributed.client import _get_global_client

        if _get_global_client() is None and not config.START_DASK:
            return SerialInterface()

        return DaskParallelInterface()
    except (ModuleNotFoundError, ImportError):
        return SerialInterface()


class ParallelEnabledMixin:
    @property
    def parallel_interface(self):
        return get_parallel()
