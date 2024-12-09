# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

import numpy as np

from pyearthtools.pipeline.operation import Operation


class Rotate(Operation):
    """
    Rotation Augmentation by 90 degrees in the plane specified by axes.
    """

    _override_interface = ["Delayed", "Serial"]
    _interface_kwargs = {"Delayed": {"pure": False, "name": "Rotate"}}

    def __init__(
        self,
        seed: int = 42,
        axis: tuple[int, int] = (-2, -1),
    ):
        """
        Rotation Augmentation by 90 degrees in the plane specified by axes.

        Generates a random number between 0 & 3 inclusive, for number of times to rotate.

        Args:
            seed (int, optional):
                Random Number seed. Defaults to 42.
            axis (tuple[int, int], optional):
                Rotation plane. Axes must be different. Defaults to (-2, -1).
        """
        super().__init__(
            split_tuples=True,
            recursively_split_tuples=True,
            operation="apply",
            recognised_types=(np.ndarray),
        )
        self.record_initialisation()

        self.rng = np.random.default_rng(seed)
        if not isinstance(axis, (list, tuple)):
            raise TypeError(f"'axis' must be a tuple or list, not {axis}.")
        self.axis = axis

    def apply_func(self, sample: np.ndarray) -> np.ndarray:
        random_num = self.rng.integers(0, 3, endpoint=True)
        return np.rot90(sample, k=random_num, axes=self.axis).copy()


class Flip(Operation):
    """
    Flip Augmentation on the specified axes.
    """

    _override_interface = ["Delayed", "Serial"]
    _interface_kwargs = {"Delayed": {"pure": False, "name": "Flip"}}

    def __init__(self, seed: int = 42, axis: int = -1):
        """
        Flip Augmentation by 90 degrees in the plane specified by axes.

        Generates a random boolean, if True, flip, otherwise not

        Args:
            seed (int, optional):
                Random Number seed. Defaults to 42.
            axis (tuple[int], optional):
                Axis to flip data in. Defaults to -1.
        """
        super().__init__(
            split_tuples=True,
            recursively_split_tuples=True,
            operation="apply",
            recognised_types=(np.ndarray),
        )
        self.record_initialisation()

        self.rng = np.random.default_rng(seed)
        self.axis = axis

    def apply_func(self, sample: np.ndarray) -> np.ndarray:
        random_num = self.rng.integers(0, 1, endpoint=True)
        if random_num > 0:
            return np.flip(sample, axis=self.axis).copy()
        return sample


class Transform(Operation):
    """
    Flip & Rotation Augmentation.
    """

    _override_interface = ["Delayed", "Serial"]
    _interface_kwargs = {"Delayed": {"pure": False, "name": "FlipRotate"}}

    def __init__(
        self,
        seed: int = 42,
        axis: tuple[int, int] = (-2, -1),
    ):
        """
        Apply both Flip & Rotation Augmentations, will rotate on given axis, and flip on both

        Args:
            seed (int, optional):
                Random Number seed. Defaults to 42.
            axis (tuple[int], optional):
                Rotation plane primarily. Axes must be different. Will also flip on each given axis. Defaults to (-2, -1).
        """
        super().__init__(
            split_tuples=True,
            recursively_split_tuples=True,
            operation="apply",
            recognised_types=(np.ndarray),
        )
        self.record_initialisation()

        self.transforms: list[Operation] = [Rotate(seed=seed, axis=axis)]

        for i, ax in enumerate(axis):
            self.transforms.append(Flip(seed=seed * i, axis=ax))

    def apply_func(self, sample: np.ndarray) -> np.ndarray:
        for trans in self.transforms:
            sample = trans.apply_func(sample)
        return sample
