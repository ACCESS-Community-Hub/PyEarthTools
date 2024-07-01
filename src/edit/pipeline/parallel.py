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

from importlib.util import find_spec

from typing import Callable, Literal, Optional, Type, TypeVar, Any, Union

from edit.utils.decorators import classproperty

import edit.utils

Future = TypeVar("Future", Any, Any)


class ParallelToggle:
    """Parallel Toggle Context Manager"""

    _enter_state: bool

    def __init__(self, state: Literal["enable", "disable"]):
        self._state = state

    def __enter__(self):
        self._enter_state = edit.utils.config.get("pipeline_V2.run_parallel")
        edit.utils.config.set({"pipeline_V2.run_parallel": self._state == "enable"})

    def __exit__(self, *args):
        edit.utils.config.set({"pipeline_V2.run_parallel": self._enter_state})

    def __repr__(self):
        return f"Context Manager to toggle parallelisation {'on' if self._state == 'enable' else 'off'}."


enable = ParallelToggle("enable")
disable = ParallelToggle("disable")

PARALLEL_INTERFACES = Literal["Futures", "Delayed", "Serial"]


class FutureFaker:
    def __init__(self, obj):
        self._obj = obj

    def result(self, *args):
        return self._obj


class ParallelInterface:
    """
    Interface for parallel computation.
    Allows for the system to define how to parallelise or if to, without the user changing code.

    Mimic's the `dask` interface
    """

    _interface_kwargs: dict[str, Any]

    def __init__(self, **kwargs: Any):
        self._interface_kwargs = kwargs

    @classmethod
    def check(cls):
        return None

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

    @property
    def config(self):
        return self._interface_kwargs.get("Serial", {})

    def submit(self, func, *args, **kwargs):
        return FutureFaker(func(*args, **kwargs))

    def map(self, func, iterables, *iter, **kwargs) -> Future:
        return tuple(map(lambda i: FutureFaker(func(i, **kwargs)), iterables, *iter))  # type: ignore

    def gather(self, futures, *args, **kwargs):
        if isinstance(futures, FutureFaker):
            return futures.result()
        return type(futures)(map(lambda x: x.result(), futures))

    def wait(self, futures, **kwargs):
        return futures

    def collect(self, futures):
        if isinstance(futures, FutureFaker):
            return futures.result()
        return type(futures)(map(lambda x: x.result(), futures))

    def fire_and_forget(self, futures):
        pass


class DaskParallelInterface(ParallelInterface):
    """
    Wrapper for the dask Client
    """

    @property
    def config(self):
        return self._interface_kwargs.get("Futures", {})

    @classproperty
    def client(cls) -> "distributed.Client":  # type: ignore  # noqa: F821
        """Get dask client"""
        from dask.distributed import Client
        import distributed

        try:
            client = distributed.get_client()
        except ValueError:
            client = None

        dask_config = edit.utils.config.get("pipeline_V2.parallel.dask.config")
        dask_config["processes"] = dask_config.pop("processes", False)

        if client is None and not edit.utils.config.get("pipeline_V2.parallel.dask.start"):
            raise RuntimeError("Cannot start dask cluster when `pipeline_V2.parallel.dask.start` is False.")

        return client or Client(**dask_config)

    @classmethod
    def check(cls):
        if find_spec("distributed") is None:
            return "Cannot import dask."

    def defer_to_client(func: Callable):  # type: ignore
        wrapped = func
        try:
            from dask.distributed import Client
            wrapped = getattr(Client, func.__name__).__doc__
        except AttributeError:
            pass
        except (ImportError, ModuleNotFoundError):
            return func

        def wrapper(self, *args, **kwargs):
            return getattr(DaskParallelInterface.client, func.__name__)(*args, **kwargs)

        return functools.update_wrapper(wrapper, wrapped)

    def __getattr__(self, key):
        return getattr(self.client, key)

    @defer_to_client
    def submit(self, func, *args, **kwargs): ...

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
        return DaskParallelInterface.client.gather(futures)  # type: ignore
        type_to_make = type(futures)
        if type_to_make == type((i for i in [])):
            type_to_make = tuple
        return type_to_make(map(lambda x: x.result(), futures))

    def fire_and_forget(self, futures, **kwargs):
        from dask.distributed import fire_and_forget

        return fire_and_forget(futures)


class DaskDelayedInterface(ParallelInterface):
    """
    Wrap all functions with `dask.delayed`

    Config (Delayed):
        `name`: Override for name of delayed
        `pure`: Whether function is pure or not.
    """

    @classmethod
    def check(cls):
        if find_spec("distributed") is None:
            return "Cannot import dask."

    @property
    def config(self):
        return self._interface_kwargs.get("Delayed", {})

    def run_delayed(self, func, *args, **kwargs):
        print("Running delayed", func)
        from dask.delayed import tokenize, delayed

        name = self.config.get("name", None)
        if name is not None:
            name += f"-{tokenize(args, kwargs)}"

        pure = self.config.get("pure", None)

        # if len(args) == 1:
        #     return delayed(func, name=name, pure=pure)(args[0], **kwargs)

        return delayed(func, name=name, pure=pure)(*args, **kwargs)

    def submit(self, func, *args, **kwargs):
        return FutureFaker(self.run_delayed(func, *args, **kwargs))

    def map(self, func, iterables, *iter, **kwargs) -> Future:
        return tuple(map(lambda i: FutureFaker(self.run_delayed(func, i, **kwargs)), iterables, *iter))  # type: ignore

    def gather(self, futures):
        if isinstance(futures, FutureFaker):
            return futures.result()
        return type(futures)(map(lambda x: x.result(), futures))

    def wait(self, futures):
        return futures

    def collect(self, futures):
        if isinstance(futures, FutureFaker):
            return futures.result()
        return type(futures)(map(lambda x: x.result(), futures))

    def fire_and_forget(self, futures):
        pass


def get_parallel(interface: Optional[PARALLEL_INTERFACES] = None, **interface_kwargs: Any) -> ParallelInterface:
    """
    Get parallel interface

    Args:
        interface (Optional[PARALLEL_INTERFACES], optional):
            Manual specification of interface.
        **interface_kwargs (Any):
            Extra kwargs to pass to Interface.

    Returns:
        (ParallelInterface):
            Parallel Interface

            Abstracts away the parallelisation, so that if actually serial,
            no code change is needed.

    Raises:
        ImportError:
            If cannot use specified `interface` due to its check failing.
    """
    if not edit.utils.config.get("pipeline_V2.run_parallel"):
        return SerialInterface(**interface_kwargs)

    if interface:
        interface_dict: dict[PARALLEL_INTERFACES, Type[ParallelInterface]] = {
            "Futures": DaskParallelInterface,
            "Delayed": DaskDelayedInterface,
            "Serial": SerialInterface,
        }
        if interface == "Dask":
            raise Exception("Use Futures instead")
        check = interface_dict[interface].check()
        if check is not None:
            raise ImportError(f"Unable to use {interface} as it's check failed.\n{check}")
        return interface_dict[interface](**interface_kwargs)

    import distributed

    try:
        client = distributed.get_client()
    except ValueError:
        client = None

    if client is None and not edit.utils.config.get("pipeline_V2.parallel.dask.start"):
        return SerialInterface(**interface_kwargs)

    return get_parallel(edit.utils.config.get("pipeline_V2.parallel.default"), **interface_kwargs)


class ParallelEnabledMixin:
    """
    Parallel Mixin

    Provides `parallel_interface` to get an interface to run parallel computing

    Properties:
        `parallel_interface`: Interface which is decided from `config` and `_override_interface`, exposes submit, map, collect , ... .
        `_override_interface`: List of interfaces to try and get, will fall over to `SerialInterface`.
        `_interface_kwargs`: Kwargs to provide to the interface. See each for available config.
            Ensure key of references an Interface with it's value being the kwargs.

    Methods:
        `get_parallel_interface`: Get a specific interface directly.

    """

    _override_interface: Optional[Union[PARALLEL_INTERFACES, list[PARALLEL_INTERFACES]]] = None
    _interface_kwargs: dict[str, Any]

    @property
    def parallel_interface(self):
        """
        Get parallel interface according to `config` and `_override_interface`.

        Will fail over to `SerialInterface` if an error occurs
        """
        try:
            return self.get_parallel_interface(self._override_interface, **getattr(self, "_interface_kwargs", {}))
        except ImportError:
            return SerialInterface(**getattr(self, "_interface_kwargs", {}))

    def get_parallel_interface(
        self,
        interface: Optional[Union[PARALLEL_INTERFACES, list[PARALLEL_INTERFACES]]] = None,
        **interface_kwargs: dict[str, Any],
    ) -> ParallelInterface:
        """
        Get parallel interface.

        If `interface` is list, will iterate through until a successful candidate is hit.

        Args:
            interface (Optional[Union[PARALLEL_INTERFACES, list[PARALLEL_INTERFACES]]], optional):
                Interface to get, or list to try. Defaults to None.
            **interface_kwargs (dict[str, Any]):
                Extra kwargs to pass to Interface. Ensure key of references an Interface with its
                value being the kwargs.

        Raises:
            ValueError:
                If unable to successfully get an interface.

        Returns:
            (ParallelInterface):
                Interface to use.

        Examples:
            >>> get_parallel_interface() # Default
            >>> get_parallel_interface('Serial') # Get serial interface
            >>> get_parallel_interface(['Delayed', 'Serial']) # Try and get delayed or serial interface in order
            >>> get_parallel_interface(['Delayed', 'Serial'], interface_kwargs = {'Delayed': {'pure': False'}}) # Provide kwargs to Delayed


        """
        class_interface_kwargs = getattr(self, "_interface_kwargs", {})
        for key in interface_kwargs.keys():
            if key not in class_interface_kwargs:
                class_interface_kwargs[key] = {}
            class_interface_kwargs[key].update(interface_kwargs[key])

        if isinstance(interface, list):
            errors = []
            for inter in interface:
                return get_parallel(inter, **class_interface_kwargs)
            raise ValueError(f"Unable to get any of {interface}, as they each failed.\n{errors}")

        return get_parallel(interface, **class_interface_kwargs)
