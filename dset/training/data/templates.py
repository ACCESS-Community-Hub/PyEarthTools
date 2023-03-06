import functools
import importlib
from abc import abstractmethod
from typing import Union

import yaml
from torch.utils.data import IterableDataset


from dset.data.default import DataIndex, OperatorIndex


def get_callable(module: str) -> "DataIterator":
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


def SequentialIterator(func):
    """
    Decorator to allow Iterator's to not be fully specified,
    such that the first element of a (DataIterator,DataInterface) is missing.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if args and isinstance(args[0], (DataIterator, DataInterface, DataIndex)):
            return func(*args, **kwargs)

        def add_iterator(iterator: DataIterator):
            return func(iterator, *args, **kwargs)
        return add_iterator
    return wrapper


def Sequential(*args: list["DataIterator"]) -> "DataIterator":
    """
    From a list of DataIterators missing only an iterator,
    build a full DataIterator

    *args
        DataIterator - with @SequentialIterator

    Returns
    -------
        DataIterator
    """
    iterator = args[0]
    for i in range(1, len(args)):
        iterator = args[i](iterator)
    return iterator


def from_dict(data_specifications: Union[str, dict]) -> "DataIterator":
    """
    Create DataIterator from a dictionary.

    Use keys as class names, if not found will auto try dset.training.data.~

    Specify order to set order

    Parameters
    ----------
    data_specifications
        Dictionary containg Data Specifications

    Returns
    -------
        DataIterator

    Raises
    ------
    TypeError
        If imported class cannot be understood
    """
    if isinstance(data_specifications, str):
        with open(data_specifications, 'r') as file:
            data_specifications = yaml.safe_load(file)
        if "data" in data_specifications:
            data_specifications = data_specifications["data"]

    if "order" in data_specifications:
        order = data_specifications.pop("order")
    else:
        order = list(data_specifications.keys())

    data_list = []
    for data_iter_name in order:
        data_iter_name = str(data_iter_name)

        data_iter = None
        for alterations in ["", "dset.training.data.", "dset.data."]:
            try:
                data_iter = get_callable(alterations + data_iter_name)
            except (ModuleNotFoundError, ImportError, AttributeError, ValueError):
                pass
            if data_iter:
                break

        # TODO Add checking back
        # Wasnt working
        # if not isinstance(data_iter, (DataInterface, DataIterator, DataIndex, OperatorIndex)):
        #    raise TypeError(f"{data_iter_name!r} gave {data_iter!r} which cannot be used with DataIterators.")

        data_list.append(data_iter(**data_specifications[data_iter_name]))
    return Sequential(*data_list)


class DataInterface:
    """
    Interface with dset.data
    """

    def __init__(self) -> None:
        raise NotImplementedError()


class DataIterator(IterableDataset):
    """
    Provide an interface between a DataInterface and what is required for ML Training.

    Must implement __iter__, __getitem__ & undo
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError

    @abstractmethod
    def undo(self, data, *args, **kwargs):
        raise NotImplementedError


class DataOperation(DataIterator):
    def __init__(self, iterator: DataIterator) -> None:
        """
        Run Operations on Data as it is being used.

        Parameters
        ----------
        iterator
            Underlying iterator to use
        """
        super().__init__()
        # functools.update_wrapper(self, iterator)
        self.iterator = iterator

    def __getattr__(self, key):
        if key == "iterator":
            raise AttributeError(f"{self.__class__} has no attribute {key}")
        return getattr(self.iterator, key)

    def undo(self, data, *args, **kwargs):
        return self.iterator.undo(data, *args, **kwargs)

    def __getitem__(self, idx):
        raise NotImplementedError(
            "Using base DataOperation implements no __getitem__ operation, must use a child class."
        )

    def __iter__(self):
        raise NotImplementedError(
            "Using base DataOperation implements no __iter__ operation, must use a child class."
        )

    def _formatted_name(self):
        padding = lambda name, length_: name + "".join([" "] * (length_ - len(name)))
        desc = self.__doc__ or "No Docstring"
        desc = desc.replace("\n", "").replace("\t", "").strip()
        return f"{padding(self.__class__.__name__, 30)}{desc}\n{self.iterator._formatted_name()}"

    def __repr__(self):
        string = "DataIterator with the following Operations:"
        operations = self._formatted_name()
        operations = operations.split("\n")
        operations.reverse()
        operations = "\n".join(["\t* " + oper for oper in operations])
        return f"{string}\n{operations}"

class DataIterationOperator(DataOperation):
    def __getitem__(self, idx):
        return self.iterator[idx]
