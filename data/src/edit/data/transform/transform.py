# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty 
# of any kind including, but not limited to, fitness for a particular 
# purpose. The user assumes the entire risk as to the use and 
# performance of the software. In no event shall the copyright holder 
# be held liable for any claim, damages or other liability arising 
# from the use of the software.

from __future__ import annotations

from abc import ABCMeta, abstractmethod, abstractproperty
from types import FunctionType
from typing import Any, Callable
import warnings
from edit.data.collection import Collection

import xarray as xr

import edit.data
import edit.data.transform
from edit.data.transform.default import get_default_transforms

import edit.utils



#TODO Add in init args capturing
class Transform(metaclass=ABCMeta):
    """
    Base Transform Class to obfuscate a transform process.

    A child class must implement `.apply(self, dataset: xr.Dataset)`, and `.info`.

    When using this transform, simply call it like a function.
    Can also add another transform to this.
    """

    def __init__(self, docstring: str | None = None) -> None:
        """Initalise root `Transform` class

        Cannot be used as is, a child must implement the `.apply` function.

        Args:
            docstring (str, optional):
                Docstring to set this `Transform` to. Defaults to None.

        Raises:
            TypeError:
                If cannot parse `docstring`
        """
        if not isinstance(docstring, str) and docstring is not None:
            raise TypeError(f"Cannot parse `docstring` of type {type(docstring)}")
        if docstring:
            self.__doc__ = docstring

    @abstractproperty
    def _info_(self) -> dict[str, Any]:
        """Info attribute

        Must contain all neccessary kwargs to rebuild the transform
        """
        raise NotImplementedError("Transform class must implement this method.")

    @abstractmethod
    def apply(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Apply transformation to Dataset

        Args:
            dataset (xr.Dataset):
                Dataset to apply transform to

        Raises:
            NotImplementedError:
                Base Transform does not implement this function

        Returns:
            xr.Dataset:
                Transformed Dataset
        """
        raise NotImplementedError

    def __call__(
        self, dataset: xr.Dataset | tuple[xr.Dataset] | list[xr.Dataset] | dict[str, xr.Dataset], **kwargs
    ) -> Any:
        """
        Apply Transformation to given dataset

        Args:
            dataset (xr.Dataset | tuple[xr.Dataset] | list[xr.Dataset] | dict[str, xr.Dataset]):
                Dataset/s to apply transformation to

        Returns:
            (Any):
                Same as input type with transforms applied
        """

        if isinstance(dataset, (xr.DataArray, xr.Dataset)):
            return self.apply(dataset, **kwargs)

        elif (
            isinstance(dataset, (tuple, list))
            and len(dataset) > 0
            # and isinstance(dataset[0], (xr.DataArray, xr.Dataset))
        ):
            applied_to_data = map(lambda x: self.__call__(x, **kwargs), dataset)
            if isinstance(dataset, Collection):
                return Collection(*applied_to_data)
            return tuple(applied_to_data)  # type: ignore

        elif isinstance(dataset, dict):
            return {x: self.__call__(dataset[x]) for x in dataset.keys()}  # type: ignore

        try:
            return self.apply(dataset, **kwargs)  # type: ignore
        except TypeError:
            warnings.warn(f"Cannot apply transform on object of {type(dataset)}", UserWarning)
            return dataset

    ##Operations
    def __add__(self, other: "FunctionType | Transform | TransformCollection"):
        return TransformCollection(self) + other

    def __and__(self, other: FunctionType | Transform | TransformCollection) -> TransformCollection:
        return self + other

    ##Representation
    def __repr__(self) -> str:
        padding = lambda name, length_: "".join([" "] * (length_ - len(name)))
        return_string = "Transform:"
        name = self.__class__.__name__
        desc = self._doc_
        return_string += f"\n   {name}{padding(name, 30)}{desc}"
        return return_string

    @property
    def _doc_(self) -> str:
        desc = self.__doc__ or "No docstring"
        desc = desc.replace("\n", "").replace("\t", "").strip()
        return desc

    def _repr_html_(self) -> str:
        return edit.utils.repr_utils.provide_html(
            self,
            name="Transform",
            documentation_attr="_doc_",
            info_attr="_info_",
            backup_repr=self.__repr__(),
        )


class FunctionTransform(Transform):
    """Transform Function which applies a given function"""

    def __init__(self, function: Callable) -> None:
        """
        Transform Function to apply a user given function

        Args:
            function (Callable): User given function to apply
        """
        super().__init__()
        self.function = function

    @property
    def _info_(self):
        return {'function': str(function)}

    def apply(self, dataset: xr.Dataset):
        return self.function(dataset)

    @property
    def __doc__(self):
        return f"Implementing: {self.function.__name__}: {self.function.__doc__ or 'No Docstring given'}"


class TransformCollection:
    """
    A Collection of Transforms to be applied to Data

    Can be added to or appended to & called to apply all transforms in order.
    """

    def __init__(
        self,
        *transforms: "Transform | TransformCollection | Callable | None | list[Transform] | tuple[Transform] | dict[str, dict]",
        apply_default: bool = False,
        intelligence_level: int = 100,
    ):
        """
        Setup new TransformCollection

        Args:
            *transforms (Transform | TransformCollection, Callable | None | list):
                Transforms to include
            apply_default (bool, optional):
                Apply default transforms. Defaults to False.
            intelligence_level (int, optional):
                Intelligence level of default transforms. Defaults to 100.
        """
        self.apply_default = apply_default

        self.intelligence_level = intelligence_level
        self._transforms: list[Transform]
        self._transforms = []

        if transforms:
            self.append(transforms)

    def apply(self, dataset: xr.Dataset | tuple[xr.Dataset] | list[xr.Dataset] | dict[str, xr.Dataset]) -> Any:
        """
        Apply Transforms to a Dataset

        Args:
            dataset (xr.Dataset): Dataset to apply transforms to

        Returns:
            (Any):
                Same as input type with transforms applied
        """
        return self.__call__(dataset)

    def __call__(self, dataset: xr.Dataset | tuple[xr.Dataset] | list[xr.Dataset] | dict[str, xr.Dataset]) -> Any:
        for transform in self._transforms:
            dataset = transform(dataset)
        return dataset

    def append(self, transform: "None | list | tuple | dict | FunctionType | Transform | TransformCollection"):
        """
        Append a transform/s to the collection

        Args:
            transform (list | dict | FunctionType | Transform | TransformCollection):
                Transform/s to add

        Raises:
            TypeError:
                If transform cannot be understood
        """
        if isinstance(transform, Transform):
            self._transforms.append(transform)

        elif isinstance(transform, TransformCollection):
            for transf in transform._transforms:
                self.append(transf)
            self.apply_default = self.apply_default & transform.apply_default
            self.intelligence_level = min(self.intelligence_level, transform.intelligence_level)

        elif isinstance(transform, (list, tuple)):
            for transf in list(transform):
                self.append(transf)

        elif isinstance(transform, FunctionType):
            self._transforms.append(FunctionTransform(transform))

        elif transform is None:
            pass

        elif isinstance(transform, dict):
            transform = dict(transform)
            for transf in TransformCollection(edit.data.transform.utils.get_transforms(transform)):
                self.append(transf)
        else:
            raise TypeError(f"'transform' cannot be type {type(transform)!r}")

    ###Operations
    def __add__(self, other: "list | FunctionType | Transform | TransformCollection"):
        new_collection = TransformCollection(self._transforms)
        new_collection.append(other)
        return new_collection

    def pop(self, index=-1) -> Transform:
        """
        Remove and return item at index (default last).
        Raises IndexError if list is empty or index is out of range.

        Args:
            index (int, optional): Index to pop from list at. Defaults to -1.

        Returns:
            Transform: Transform popped out
        """
        return self._transforms.pop(index)

    def remove(self, key: type | str | Transform):
        """
        Remove first occurrence of value.

        Args:
            key (type | str | Transform): Key to search for

        Raises:
            ValueError: If the value is not present.
        """

        for transf in self._transforms:
            if isinstance(key, str) and transf.__class__.__name__ == key:
                self._transforms.remove(transf)
                return
            elif isinstance(key, type) and isinstance(transf, key):
                self._transforms.remove(transf)
                return
            elif transf == key:
                self._transforms.remove(transf)
                return
        raise ValueError(f"{key} not in TransformCollection")

    def to_dict(self):
        return edit.data.transform.utils.parse_transforms(self)

    def __iter__(self):
        for transf in self._transforms:
            yield transf

    def __getitem__(self, index):
        return self._transforms[index]

    def __len__(self):
        return len(self._transforms)

    def __contains__(self, key) -> bool:
        """
        Return if key in [TransformCollection][edit.data.transform.transform.TransformCollection]
        """
        if isinstance(key, str):
            return key in [transf.__class__.__name__ for transf in self._transforms]
        elif isinstance(key, type):
            return key in [type(transf) for transf in self._transforms]
        else:
            return key in self._transforms

    ##Representation
    def __repr__(self) -> str:
        padding = lambda name, length_: "".join([" "] * (length_ - len(name)))
        return_string = "Transform Collection:"
        if self.apply_default:
            return_string += "\nDefault Transforms:"
            for i in get_default_transforms(self.intelligence_level):
                name = i.__class__.__name__
                desc = i._doc_
                return_string += f"\n   {name}{padding(name, 30)}{desc}"
            return_string += "\n:"
        if len(self._transforms) == 0:
            return_string += f"\n   Empty"

        for i in self._transforms:
            name = i.__class__.__name__
            desc = i._doc_
            return_string += f"\n   {name}{padding(name, 30)}{desc}"
        return return_string

    def _repr_html_(self) -> str:

        return edit.utils.repr_utils.provide_html(
            *self._transforms,
            name="Transform Collection",
            documentation_attr="_doc_",
            info_attr="_info_",
            backup_repr=self.__repr__(),
        )
