import concurrent.futures
import multiprocessing
from enum import StrEnum, auto
from dataclasses import dataclass, field
from collections.abc import Callable
from contextlib import contextmanager
from typing import Union, Generator
from collections import namedtuple

import numpy as np
import xarray as xr

from persistence.types import PetDataArrayLike
from persistence.methods._impute import SimpleImpute
from persistence.methods._median import _median_of_three_numpy
from persistence.interface._metadata import PersistenceMetadata
from persistence.interface._method import PersistenceMethod
from persistence.interface._chunker import (
    PersistenceDataChunk,
    PersistenceChunker,
    PersistenceChunkInfo,
)
from persistence.interface._backend import PersistenceBackendType


ChunkResult = namedtuple("ChunkResult", ["array", "slice_dims"])


@dataclass
class PersistenceComputePool:
    """
    Generates a compute pool and uses the given chunk genarator along with the configured method to
    perform the computations.

    Joins the chunks back together at completion according to the FIFO order.

    Computation here happens at a lower structural level (numpy or chosen system backend).

    ---

    Algorithm (see `compute_chunks`):

        1. retrieve chunks (numpy arrays)
        2. perform compute on each chunk depending on the persistence method
        3. join numy arrays -> will be of the form

            for i in nd-index:

                arr[x0, x1, x2, ..., t, ...]
                = arr[x0, x1, x2, ...]
                = slab

                OR

                arr[x0, x1, t, x2, ...]
                = arr[x0, x1, 1, x2, ...]
                = slab

           here, x0, x1, x2 are the multi-indices that are incremented when filling in the slabs.

           Because the persistence methods all reduce the time index to a cardinality of 1, both of
           these scenarios are equally efficient.
        4. use the stored data-array information (shapes/dimnames)

    ---

    Further, to reiterate the assumption, in persistence methods chunks are loaded lazily, but
    evaluated eagerly, in otherwords the computation itself should not use `dask`. And loading is
    forced to be synchronous e.g.

        load chunk 1 ---> compute [worker 1]
                                    | finish compute
        *>>> load chunk 2 ---> compute [worker 2]
                                    |
            *>>> load chunk 3 ---> compute [worker 3]
                                    |---> at this point, we should only have:
                                         - two chunks in memory with multiple time indices
                                         - one "result" chunk with the reduced time dimension

        *>>> the time taken to load a chunk into memory

    Keep the above in mind when running this program, as it may help to debug issues.
    Any scheduling/wait time implementation is out of scope here, and in fact is an anti-pattern.

    (This does not mean scheduling cannot be used - it just needs to be used at a higher level and
    at a distributed compute level - NOT at a single node compute level)

    ---

    Important:

    - As per the rest of the persistence structures, the time dimension existing is crucial, and the
      time dimension is what is aggregated over, and therfore not chunked. It is instead simpler to
      act on, and chunk the embarassingly parallel independent dimensions (e.g. spatial dimensions).

    - Persistence computation is single-variate, it may in the future infer something from the
      dimensionality, but it may not infer information from other variables.

    - In other words, coordinate information may be considered, but not other variables in a
      provided dataset. Therefore, the absolute highest level structure returnable by this
      computation a DataArray.

    - The reason for this is that multi-variate persistence models are an anti-pattern, since
      persistence models inherently shouldn't do any inference, physics, or _parametric_ statistical
      learning. Unparamaterized methods, i.e. methods that do NOT use knowledge of what the
      coordinates or other variables represent - other than the trivial inference that they are
      different dimensions and have a certain shape, are okay.

    ---

    Future considerations:

    - There could be methods in the future that aggregate based on neighbouring dimensions, in such
      a scenario, the computation is still parameterless, but the methods could derive additional
      statistical patterns and "state" parameters that could improve performance. This may cause
      some non-determinism based on how chunks are chosen.

    - However, as long as these filters are semi bounded - e.g. "9 parameter savitzky golay filter",
      then there is a guarantee that despte how large the chunks are the maximum number of
      neighbouring parameters used in any "smarts" is 9 - spatially this could be a convolutional
      3x3 grid for example doing some smoothing or noise inference. And therefore, maintain some
      level of determinism as long as the chunk sizes don't fall below this criteria.

    - Regardless, `PersistenceMetadata` and `PersistenceChunkInfo` are easily serialisable
      structures that can be logged as part as experiments.

    - For now the only independent parameter that is known by the algorithms, is the time dimension.
    """

    chunk_generator: Generator[PersistenceDataChunk]  # the chunks used for computation
    chunk_info: PersistenceChunkInfo
    metadata: PersistenceMetadata

    @staticmethod
    def _job_wrapper(chunk: PersistenceDataChunk) -> ChunkResult:
        """
        This wrapper needs to be static, as we may not want the state info of
        this class to propagate.
        """
        return ChunkResult(
            array=PersistenceCompute(chunk.arr_chunk, chunk.metadata).compute(),
            slice_dims=chunk.slice_dims,
        )

    def map_and_join_chunks(self) -> xr.DataArray:
        """
        1. Send chunks to workers
        2. Each worker runs the jobwrapper which invokes the configured persistence method
        3. Join the resulting list of numpy results along the time dimension
        4. Re-insert dimension names from chunk_info

        TODO: this should only be called via a main guard or entrypoint

        Calling forkserver preload and early inheriting any modules that may be forked is a
        desirable way to call this, if multi-platform compatiblity is needed:

        e.g.

        if __name__ == "__main__":
            ctx = multiprocessing.get_context("fork_server")
            ctx.set_forkserver_preload(["module_name", "__main__"])
            args = parse_args(...)
            generator = build_generator(args)

            with concurrent.futures.ProcessPoolExecutor(..., mp_context=ctx) as exec:
                res = exec.map(fn, iter(generator))
                # do stuff with result
        """
        # compute result shape by suppressing the time dimension
        shape_res = [
            v if i != self.metadata.idx_time_dim else 1
            for i, v in enumerate(self.chunk_info.shape_full)
        ]
        arr_res = np.empty(shape_res)

        if self.metadata.num_workers <= 1:
            # loop through instead
            for chunk in iter(self.chunk_generator):
                arr_res_chunk = PersistenceComputePool._job_wrapper(chunk)
                arr_res[chunk.slice_dims] = arr_res_chunk
        else:
            # dispatch chunks to workers
            # TODO: forkserver does/may not work with windows/mac, unless main-guarded
            with concurrent.futures.ProcessPoolExecutor(
                self.metadata.num_workers,
                mp_context=multiprocessing.get_context("forkserver"),
            ) as pp_exec:
                results = pp_exec.map(
                    PersistenceComputePool._job_wrapper, iter(self.chunk_generator)
                )
                for res_chunk in iter(results):
                    arr_res[*res_chunk.slice_dims] = res_chunk.array

        da_res = xr.DataArray(arr_res, dims=self.chunk_info.dim_names)

        return da_res


# TODO: the variable references are not right - need to use self.metadata
@dataclass
class PersistenceCompute:
    arr: PetDataArrayLike
    metadata: PersistenceMetadata

    def _method_impl(self, arr: np.ndarray) -> np.ndarray:
        match self.metadata.backend:
            case PersistenceBackendType.NUMPY:
                return self._method_impl_numpy(arr)
            case PersistenceBackendType.NUMBA:
                return self._method_impl_numba(arr)
            case PersistenceBackendType.RUST:
                return self._method_impl_rust(arr)
            case _:
                raise NotImplementedError("PersistenceCompute: Unknown backend")

    def _method_impl_numpy(self, arr: np.ndarray) -> np.ndarray:
        match self.metadata.method:
            case PersistenceMethod.MEDIAN_OF_THREE:
                return _median_of_three_numpy(arr, self.metadata.idx_time_dim)
            case PersistenceMethod.MOST_RECENT:
                raise NotImplementedError("TODO")
            case _:
                raise NotImplementedError(
                    f"PersistenceCompute: compute method {self.method} has not been implemented"
                )

    def _method_impl_numba(self, arr: np.ndarray) -> np.ndarray:
        raise NotImplementedError("numba backend is not supported")

    def _method_impl_rust(self, arr: np.ndarray) -> np.ndarray:
        raise NotImplementedError("rust backend is not supported")

    def _slice_time(self, arr: np.ndarray) -> np.ndarray:
        """
        Further slices the data chunk into a smaller chunk required for the computation (usually
        after imputation.
        """
        # slice out data required for the computation
        len_time_max = arr.shape[self.metadata.idx_time_dim]
        len_time_cmp = self.metadata.len_time_compute()
        arr_sliced = np.take(
            arr,
            range(len_time_max - len_time_cmp, len_time_max),
            axis=self.metadata.idx_time_dim,
        )

        return arr_sliced

    def _impute(self, arr: np.ndarray) -> np.ndarray:
        # default to pass-through
        arr_imputed = arr

        if self.metadata.do_impute:
            imputer = SimpleImpute(arr)
            arr_imputed = imputer.impute_mean()

        return arr_imputed

    def compute(self) -> np.ndarray:
        # check backend support
        self.metadata.backend.check_support()

        # slice: to num_lookback indices
        arr_sliced: np.ndarray = self._slice_time(self.arr)

        # impute: fill missing values
        arr_imputed: np.ndarray = self._impute(arr_sliced)

        # compute: using specified persistence method and preprocessed array
        arr_persist: np.ndarray = self._method_impl(arr_imputed)

        return arr_persist
