from typing import Any, Union
from edit.training.trainer.template import EDITTrainer
from edit.training.data.operations import PatchingDataIndex
from edit.training.data.templates import DataStep


class PatchingUpdate:
    """
    Patching Update Context Manager.

    So that any changes to patching configuration can be reversed.
    """

    def __init__(
        self,
        iterator: "PatchingDataIndex | EDITTrainer | DataStep",
        kernel_size: tuple[int, int] | int = None,
        stride_size: tuple[int, int] | int = None,
    ):
        """Update Patching Config

        Args:
            iterator (PatchingDataIndex | EDITTrainer): Iterator to update
            kernel_size (tuple[int, int] | int, optional): New kernel_size. Defaults to None.
            stride_size (tuple[int, int] | int, optional): New Stride size. Defaults to None.

        Raises:
            RuntimeError: If iterator does not contain a PatchingDataIndex
        """
        if isinstance(iterator, EDITTrainer):
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
        """Update Attribute

        Args:
            object (Any): Object to update
            key (str): Attribute Name
            value (Any): Value to update key to

        Raises:
            AttributeError: If object has no attribute key
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
