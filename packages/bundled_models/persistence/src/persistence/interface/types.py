"""
Common data array/set transformations supported by the persistence model, the main usecase is to map
a function to each data variable independently. This is a common pattern as more often than not we
wouldn't be intermixing variables in basic pre-processing steps.

TODO: this should be somewhere more common
"""

from typing import Union, Generic
from collections.abc import Callable
from enum import StrEnum, auto
import xarray as xr
import numpy as np
import numpy.typing as npt

PetDataArrayLike = Union[xr.DataArray, xr.Dataset, npt.ArrayLike]


class PetInputDataType(StrEnum):
    XR_DATAARRAY = "xr_dataarray"
    XR_DATASET = "xr_dataset"
    NP_ARRAY = "np_array"
    UNKNOWN = auto()


class PetDataset:
    def __init__(
        self,
        arraylike: PetDataArrayLike,
        dummy_varname="_dummyvarname",  # used for xarray dataarrays and numpy arrays
        dimnames: list[str] = None,  # used only for numpy arrays
    ):
        """
        Takes a PetDataArrayLike and converts it to a PetDataset which is compatible with the
        `map_each_var` computation.

        `dimnames` is only relevant for numpy - and only if using name-based indexing for retrieving
        e.g. time dimension
        """
        self.raw_type = PetInputDataType.UNKNOWN
        self.ds = self.from_arrlike(arraylike, dummy_varname, dimnames)
        self.return_raw_result = True

    def with_return_raw_result(self, return_raw_result: bool = True):
        """
        Optionally set this to return raw array from `map_each_var`

        NOTE: this is a special purpose function. It is useful when multiple operations that take in
        PetDataArrayLike are chained. In which case self.return_raw_result = False will have some
        slight performance benefit, otherwise you'd have to do:

            ```
            pd1 = PetDataset(arr)
            res1 = pd1.map_each_var(fn1)
            pd2 = PetDataset(res1)  # each of this call incurs a overhead.
            res2 = pd2.map_each_var(fn2)
            ```

        Instead, setting `with_return_raw_result(False)` we can chain methods:

            ```
            pet_ds = PetDataset(arr)
            # no over head since the return type of each method is already a PetDataset
            result = pet_ds.map_each_var(fn1).map_each_var(fn2)...
            ```

        Finally we can set:

            ```
            raw_result =
                pet_ds.map_each_var(fn1)
                    .map_each_var(fn2)
                    ...
                    .with_return_raw_result()
                    .map_each_var(final_fn)
            ```

        if we explicitly need the raw result at the end.

        The default (True) is always to return the original array type. This would be the case for
        most one-off computations.
        """
        self.return_raw_result = return_raw_result

    def from_np_array(
        self, arraylike: npt.ArrayLike, dummy_varname, dimnames
    ) -> xr.Dataset:
        self.raw_type = PetInputDataType.NP_ARRAY
        return self.from_xr_dataarray(
            xr.DataArray(arraylike, dims=dimnames), dummy_varname
        )

    def from_xr_dataarray(self, arraylike: xr.DataArray, dummy_varname) -> xr.Dataset:
        self.raw_type = PetInputDataType.XR_DATAARRAY
        return xr.Dataset({dummy_varname: arraylike})

    def from_xr_dataset(self, arraylike: xr.Dataset) -> xr.Dataset:
        self.raw_type = PetInputDataType.XR_DATASET
        return arraylike

    def from_arrlike(self, arraylike, dummy_varname, dimnames) -> xr.Dataset:
        # Order is important here, For example:
        # xr.DataArray may be a npt.ArrayLike, but not the other way around. If we swap the order,
        # the xr.DataArray constructor will never be reached.

        msg_type_error = """
            The provided data does not have a supported array type, supported array types are:
            xr.DataArray, xr.Dataset and np.ndarray.
        """

        if isinstance(arraylike, xr.Dataset):
            return self.from_xr_dataset(arraylike)

        if isinstance(arraylike, xr.DataArray):
            return self.from_xr_dataarray(arraylike, dummy_varname)

        if isinstance(arraylike, (np.ndarray, list, tuple)):
            arraylike = np.asarray(arraylike)  # force convert just in case
            return self.from_np_array(arraylike, dummy_varname, dimnames)

        # unsupported type
        raise TypeError(msg_type_error)

    def map_each_var(
        self,
        _fn: Callable[[xr.DataArray, ...], xr.DataArray],
        *_fn_args,
        **_fn_kwargs,
    ) -> PetDataArrayLike:
        """
        Applies a function over each data array in the dataset. The return type will be dataset.

        The return type of each function operation itself will be per variable (dataarray).

        Only functions that have common structure associated to the variables in the Dataset will
        work properly.

        IMPORTANT: global attributes and special variables may not be preserved. This operation is
        destructive and for intermediate computation purposes only.

        Args:
            _fn: takes a DataArray as its first input arg and produces a DataArray as output
            _fn_args: additional positional arguments to provide to _fn
            _fn_kwargs: additional keyword arguments to provide to _fn
        """
        errmsg_badinputtype = "PetDataset.map_each_var: invalid input type detected"
        errmsg_singlearrayret = (
            "PetDataset.map_each_var: Expect function to return a single xr.DataArray"
        )

        if self.raw_type == PetInputDataType.UNKNOWN:
            raise RuntimeError(errmsg_badinputtype)

        dict_res = {}

        for k_var, v_da in self.ds.data_vars.items():
            # sense check
            assert isinstance(v_da, xr.DataArray)

            da_res = _fn(v_da, *_fn_args, **_fn_kwargs)

            if not isinstance(da_res, xr.DataArray):
                raise RuntimeError(errmsg_singlearrayret)

            dict_res[k_var] = da_res

        ds_res = xr.Dataset(dict_res)

        if self.return_raw_result:
            return self._raw_result(ds_res)

        # return upgraded dataset by default
        return ds_res

    def _raw_result(self, ds: xr.Dataset) -> PetDataArrayLike:
        """
        Converts a result back into the original data structure. Down-converting is a lot safer and
        so less checks required.

        NOTE: the returned datatype may have dummy names attached, as such these results are for
        intermediate computation purposes only, not for operational outputs.
        """
        if self.raw_type == PetDataArrayLike.UNKNOWN:
            # this should not happen - _raw_result should not be called externally
            raise RuntimeError("PetDataset._raw_result: Invalid raw type encountered")
        elif self.raw_type == PetDataArrayLike.XR_DATASET:
            # nothing to do
            return ds
        elif self.raw_type == PetDataArrayLike.XR_DATAARRAY:
            # extract the dataarray
            return ds[self._dummyvarname]
        elif self.raw_type == PetDataArrayLike.NP_ARRAY:
            # extract the numpy array - note this may force a memory load.
            return ds[self._dummyvarname].values
