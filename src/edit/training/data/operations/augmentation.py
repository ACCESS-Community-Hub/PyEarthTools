"""
Augmentation Strategies
"""

from __future__ import annotations

import numpy as np

from edit.training.data.templates import DataOperation, DataStep
from edit.training.data.sequential import SequentialIterator

@SequentialIterator
class Rotate(DataOperation):
    """
    Rotation Augmentation by 90 degrees in the plane specified by axes.

    !!! Example
        ```python
        Rotate(PipelineStep, seed = 10)

        ## As this is decorated with @SequentialIterator, it can be partially initialised

        partialRotate = Rotate(seed = 10)
        partialRotate(PipelineStep)
        ```
    """
    def __init__(self, index: DataStep | DataOperation, seed: int = 42, axis: tuple[int] = (-2, -1)):
        """
        Rotation Augmentation by 90 degrees in the plane specified by axes.

        Generates a random number between 0 & 3 inclusive, for number of times to rotate.

        Args:
            index (DataStep | DataOperation): 
                Underlying index
            seed (int, optional): 
                Random Number seed. Defaults to 42.
            axis (tuple[int], optional): 
                Rotation plane. Axes must be different. Defaults to (-2, -1).
        """
        super().__init__(index, apply_func=self._apply_rotation, undo_func=None, apply_get=False, split_tuples=True, recognised_types=[np.ndarray])
        self.rng = np.random.default_rng(seed)
        if not isinstance(axis, (list, tuple)):
            raise TypeError(f"'axis' must be a tuple or list")
        self.axis = axis

        self.__doc__ = "Rotation Augmentation"
        self._info_ = dict(seed = seed, axis = axis)

    def _apply_rotation(self, data: np.ndarray):
        random_num = self.rng.integers(0, 3, endpoint = True)
        return np.rot90(data, k=random_num, axes=self.axis).copy()

@SequentialIterator
class Flip(DataOperation):
    """
    Flip Augmentation on the specified axes.

    !!! Example
        ```python
        Flip(PipelineStep, seed = 10)

        ## As this is decorated with @SequentialIterator, it can be partially initialised

        partialFlip = Flip(seed = 10)
        partialFlip(PipelineStep)
        ```
    """
    def __init__(self, index: DataStep | DataOperation, seed: int = 42, axis: int = -1):
        """
        Flip Augmentation by 90 degrees in the plane specified by axes.

        Generates a random boolean, if True, flip, otherwise not

        Args:
            index (DataStep | DataOperation): 
                Underlying index
            seed (int, optional): 
                Random Number seed. Defaults to 42.
            axis (tuple[int], optional): 
                Axis to flip data in. Defaults to -1.
        """
        super().__init__(index, apply_func=self._apply_flip, undo_func=None, apply_get=False, split_tuples=True, recognised_types=[np.ndarray])
        self.rng = np.random.default_rng(seed)
        self.axis = axis

        self.__doc__ = "Flip Augmentation"
        self._info_ = dict(seed = seed, axis = axis)

    def _apply_flip(self, data: np.ndarray):
        random_num = self.rng.integers(0, 1, endpoint = True)
        if random_num > 0:
            return np.flip(data, axis=self.axis).copy()
        return data

class Transform(DataOperation):
    """
    Flip & Rotation Augmentation.

    !!! Example
        ```python
        Transform(PipelineStep, seed = 10)

        ## As this is decorated with @SequentialIterator, it can be partially initialised

        partialTransform = Transform(seed = 10)
        partialTransform(PipelineStep)
        ```
    """
    def __init__(self, index: DataStep | DataOperation, seed: int = 42, axis: tuple[int] = (-2, -1)):
        """
        Apply both Flip & Rotation Augmentations, will rotate on given axis, and flip on both

        Args:
            index (DataStep | DataOperation): 
                Underlying index
            seed (int, optional): 
                Random Number seed. Defaults to 42.
            axis (tuple[int], optional): 
                Rotation plane primarily. Axes must be different. Will also flip on each given axis. Defaults to (-2, -1).
        """
        super().__init__(index, apply_func=self._apply_transform, undo_func=None, apply_get=False, split_tuples=True, recognised_types=[np.ndarray])

        self.transforms = [Rotate([], seed = seed, axis = axis)]
        for i, ax in enumerate(axis):
            self.transforms.append(Flip([], seed = seed * i, axis = ax))

        self.__doc__ = "Flip & Rotation Augmentation"
        self._info_ = dict(seed = seed, axis = axis)

    def _apply_transform(self, data: np.ndarray):
        for trans in self.transforms:
            data = trans.apply_func(data)
        return data
