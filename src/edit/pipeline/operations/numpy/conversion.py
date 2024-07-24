# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

from typing import Any, Optional, Union, Hashable

import numpy as np
import xarray as xr

import edit.data

from edit.pipeline.operation import Operation

ATTRIBUTES_IGNORE = ["license", "summary"]


class ToXarray(Operation):
    """
    Numpy -> Xarray Converter
    """

    _override_interface = "Serial"

    def __init__(
        self,
        array_shape: Union[tuple[str, ...], str],
        coords: Optional[dict[Hashable, Any]] = None,
        encoding: Optional[dict[str, Any]] = None,
        attributes: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Convert array into xarray object.

        Can use `.like` to record from a reference data object.

        Args:
            array_shape (Union[tuple[str, ...], str]):
                Order / naming of dimensions of incoming array. Can be str split by ' '.
                Special name is 'variable' corresponding to variables in a dataset.
                That dim will be split into variables
            coords (Optional[dict[Hashable, Any]], optional):
                Coordinates to set xarray object with, not all have to be given.
                Cannot be a tuple. As 'variable' is special in `array_shape`, 'variable' in coords
                names the variables.
                Defaults to None.
            encoding (Optional[dict[str, Any]], optional):
                Encoding to set, can be variable, or dimension. Defaults to None.
            attributes (Optional[dict[str, Any]], optional):
                Attributes to set. Can use `__dataset__` if dataset to update dataset attrs Defaults to None.

        Examples:
            ## Like
            ```python
                import edit.data
                import edit.pipeline
                import numpy as np

                sample = edit.data.archive.ERA5.sample()('2000-01-01T00')
                converter = edit.pipeline.operations.numpy.conversion.ToXarray.like(sample)
                converter.apply(np.ones((1, 1, 721, 1440)))
            ```

            ## Manually
            ```python
                import edit.pipeline
                import numpy as np

                converter = edit.pipeline.operations.numpy.conversion.ToXarray(
                    'time latitude longitude',
                    coords = {'latitude': np.arange(-90, 90, 0.25), 'longitude': np.arange(-180, 180, 0.25)}
                )
                converter.apply(np.ones((1, 721, 1440)))
            ```
        """
        super().__init__(split_tuples=True, recognised_types={"apply": np.ndarray, "undo": (xr.DataArray, xr.Dataset)})
        self.record_initialisation()

        self._array_shape = array_shape.split(" ") if isinstance(array_shape, str) else array_shape
        self._coords = coords or {}
        self._coords.update(kwargs)  # type: ignore

        self._encoding = encoding or {}
        self._attributes = attributes or {}

    def apply_func(self, sample: np.ndarray):
        """
        Convert np array to xarray
        """
        if "variable" not in self._array_shape:
            xr_obj = xr.DataArray(sample, coords=self._coords, dims=self._array_shape, attrs=self._attributes)
            xr_obj = edit.data.transform.attributes.SetAttributes(self._attributes, apply_on="dataarray")(xr_obj)  # type: ignore

        else:
            array_shape = list(self._array_shape)
            var_index = array_shape.index("variable")
            array_shape.pop(var_index)
            data_vars = {}

            ds_coords = dict(self._coords)
            ds_coords.pop("variable", None)

            ds_attrs = dict(self._attributes)
            dataset_attribute = ds_attrs.pop("__dataset__", {})

            xarray_coord = xr.Coordinates(ds_coords)

            for var_i, var in enumerate(np.take(sample, i, var_index) for i in range(sample.shape[var_index])):
                var_name = self._coords["variable"][var_i] if "variable" in self._coords else str(var_i)
                data_vars[var_name] = xr.DataArray(
                    var, coords=xarray_coord, dims=array_shape, name=var_name, attrs=ds_attrs.pop(var_name, {})
                )

            xr_obj = xr.Dataset(data_vars=data_vars, coords=ds_coords)
            xr_obj = edit.data.transform.attributes.SetAttributes(dataset_attribute, apply_on="dataset")(xr_obj)  # type: ignore

        xr_obj = edit.data.transform.attributes.SetEncoding(self._encoding)(xr_obj)  # type: ignore
        xr_obj = edit.data.transform.attributes.SetAttributes(
            self._attributes, apply_on="per_variable" if isinstance(xr_obj, xr.Dataset) else "dataarray"
        )(
            xr_obj
        )  # type: ignore
        return xr_obj

    def undo_func(self, sample: Union[xr.DataArray, xr.Dataset]):
        """
        Convert xarray to numpy array
        """
        if isinstance(sample, xr.DataArray):
            return sample.to_numpy()
        if isinstance(sample, xr.Dataset):
            return np.stack([sample[var].to_numpy() for var in sample], axis=0)

    @classmethod
    def like(
        cls, reference_dataset: Union[xr.DataArray, xr.Dataset], drop_coords: Optional[list[str]] = None
    ) -> "ToXarray":
        """
        Get `ToXarray` Operation setup from a reference dataset

        Args:
            reference_dataset (Union[xr.DataArray, xr.Dataset]):
                Reference dataset to use to setup converter.

        Returns:
            (ToXarray):
                Converter setup to convert like `reference_dataset`.
        """
        drop_coords = [drop_coords] if isinstance(drop_coords, str) else drop_coords

        if isinstance(reference_dataset, xr.DataArray):
            array_shape = tuple(map(str, reference_dataset.dims))
            coords = {key: list(val.values) for key, val in reference_dataset.coords.items()}
            encoding = reference_dataset.encoding
            attributes = reference_dataset.attrs

        else:
            array_shape = ("variable", *tuple(map(str, reference_dataset[list(reference_dataset.data_vars)[0]].dims)))
            coords = {key: val.values.tolist() for key, val in reference_dataset.coords.items()}
            encoding = reference_dataset.encoding
            attributes = reference_dataset.encoding

            var_names = []
            for var in reference_dataset:
                var_names.append(var)
                encoding[var] = reference_dataset[var].encoding
                attributes[var] = {
                    key: val for key, val in reference_dataset[var].attrs.items() if key not in ATTRIBUTES_IGNORE
                }

            attributes["__dataset__"] = {
                key: val for key, val in reference_dataset.attrs.items() if key not in ATTRIBUTES_IGNORE
            }
            coords["variable"] = var_names

        for coord in reference_dataset.coords:
            encoding[coord] = reference_dataset[coord].encoding
            attributes[coord] = reference_dataset[coord].attrs

        if drop_coords:
            tuple(coords.pop(key) for key in drop_coords)
        return ToXarray(array_shape, coords, encoding, attributes)


class ToDask(Operation):
    """
    Numpy -> dask Converter
    """

    _override_interface = "Serial"

    def __init__(self, chunks: int | str | tuple[int, str] = "auto"):
        super().__init__(split_tuples=True, recursively_split_tuples=True, recognised_types={"apply": np.ndarray})
        self.record_initialisation()
        self._chunks = chunks

    def apply_func(self, sample: np.ndarray):
        import dask.array as da

        return da.from_array(sample)

    def undo_func(self, sample):
        return sample.compute()
