from typing import Any, Union
from dset.training.trainer.template import DSETTrainer


class PatchingUpdate:
    """
    Patching Update Context Manager.

    So that any changes to patching can be reversed.
    """

    def __init__(
        self,
        iterator: Union["PatchingDataIndex", DSETTrainer],
        kernel_size: Union[tuple[int, int], int] = None,
        stride_size: Union[tuple[int, int], int] = None,
    ):
        """
        Update Patching Config

        Parameters
        ----------
        iterator
            Iterator in which to update
        kernel_size, optional
            New kernel_size, by default None
        stride_size, optional
            New Stride size, by default None

        Raises
        ------
        RuntimeError
            If iterator is not a PatchingDataIndex
        """
        if isinstance(iterator, DSETTrainer):
            iterator = iterator.valid_iterator or iterator.train_iterator

        if not hasattr(iterator, "get_patching"):
            raise RuntimeError("DataIterator does not seem to be a PatchingDataIndex.")

        self.iterator = iterator

        self._patching_initial = iterator.get_patching()
        self._saved_tesselators = iterator._tesselators
        self._new_patching = (kernel_size, stride_size)

    def __enter__(self):
        self.iterator.update_patching(*self._new_patching)

    def __exit__(self, *args):
        self.iterator.update_patching(*self._patching_initial)
        self.iterator._tesselators = self._saved_tesselators


class ChangeValue:
    """
    Context Manager to change attribute of object and revert after
    """

    def __init__(self, object: Any, key: str, value: Any):
        """
        Update Attribute

        Parameters
        ----------
        object
            Object to update
        key
            Attribute Name
        value
            Value to update key to

        Raises
        ------
        AttributeError
            If object has no attribute key
        """
        if not hasattr(object, key):
            raise AttributeError(f"{type(object)!r} has no attribute {key!r}")

        self.object = object
        self.key = key
        self.value = value

        self.original_value = getattr(object, key)

    def __enter__(self):
        setattr(self.object, self.key, self.value)
        return self.object

    def __exit__(self, *args):
        setattr(self.object, self.key, self.original_value)
