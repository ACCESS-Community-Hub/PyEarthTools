

from typing import Union
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
        self._new_patching = (kernel_size, stride_size)

    def __enter__(self):
        self.iterator.update_patching(*self._new_patching)

    def __exit__(self, *args):
        self.iterator.update_patching(*self._patching_initial)

