"""
Runs persistence model on the data loaded from the pipeline. Chunks the input data from the pipeline.

TODO: use threads instead of processes.

ANTIPATTERNS (for developers):

    - do not chunk over time, on-the-fly (except for specific exceptions). Data is expected to
      arrive already pre-bounded to a specific dimensional extent. Any chunking here is done on top
      of that

    - do not assume the use of threads will "just work (TM)" if you have to, force threads to 1 if
      its an issue.

    - do not share data between threads when it is not required. If you have to share data you MUST
      include barriers appropriately to prevent race conditions and deadlocks.

    - do not assume this will be called as a library (but can be if the OS allows it and its been
      tested sufficiently).

FUTUREWORK:

    - Add the ability to bypass python completely for data loading. (see examples for a zmq example)

        - Current architecture expects data to be lazily loaded from python but eagerly computed by
          the backend, which may still be python or could be something like rust or C.

        - The target alternative or toggle is for this to be inverted in a way that the data loading
          itself is done by the backend, allowing for even better control over the processing.

        - Persistence computation is relatively isolated enough from "frameworks" to be a perfect
          candidate to do this.
"""

import xarray as xr

from persistence.interface import (
    PetDataArrayLike,
    PersistenceComputePool,
    PersistenceBackendType,
    PersistenceMethod,
    PersistenceMetadata,
    PersistenceChunker,
    PersistenceChunkInfo,
    PetDataset,
)

from persistence.config.dask import _set_synchronous_dask


def predict(
    arr: PetDataArrayLike,
    idx_time_dim: int,
    num_workers: int = None,
    num_chunks: int = None,
    method: PersistenceMethod | str = PersistenceMethod.MEDIAN_OF_THREE,
    simple_impute: bool = True,
    backend_type: PersistenceBackendType = PersistenceBackendType.NUMPY,
) -> PetDataArrayLike:
    """
    Calculate the persistence of historical observations, to be used as a baseline for other models.

    Persistence methods essentially compute either:

        a. reduce an array with multiple time indices into 1 time index, given the input with
           multiple time indices (the number of time indices required, depends on the perisistence
           method). ---> single time index

        b. A stochastic signal that has the maximum likelihood (depending on method) of representing
           the data at the leadtime given the short amount of contextual history. E.g. this could be
           the starting context using a. followed by some behaviour inferred from day cycles inferred
           from the historical data. ---> multi-time indices, maybe autoregressive

        Only a. is currently supported.

    What persistence tries to answer is the following:

        Given some trivial, human comprehendable methods, am "I" - this program - able to apply the
        method(s) according to the user configuration on some limited amount of historical data to
        produce output that is competitive (speed, memory usage, accuracy, skill etc.) to the model
        that I'm compared against.

        Because if the answer is "yes I can match this complex algorithm, then that invalidates the
        need for the complex algorithm, especially since persistence is explainable and bounded to
        the observations by definition.

        If the answer is "no" then the follow up is, how does this compare with other competitive
        models, which essentially paves grounds for verfication and ranking models.


    The general idea is that we are transforming a set of user requirements and a nd-dataarray into
    a time reduced (single time index) nd-dataarray if n > 1 (otherwise we'd just get back a single
    scalar). In this process we would also be doing chunking, multiprocessing, and offloading to
    a different compute backend, if requested. By default no data splicing occurs and the backend is
    chosen to be numpy.

    The above is repeated for each "variable" in the input data structure independently, where the
    concept of a "variable" only applies in the case that the input is a `xr.Dataset` _or_ if the
    underling `xr.DataArray` has a "name". The results are recomposed back into the original data
    structure with/or without variables - depending.

    (C, M, D_(TxN), I) -> D_(T'xN)

    where:
        D = data provided - usually observations
            (must include time dimension, may have multiple dimensions)
        C = chunk strategy (index, number of chunks)
            (or none if doing it all in one go)
        M = persistence method
            (defaults to most recent observation)
        I = simple imputation of missing values
            (optional)
        T = time dimension
        T' = forecast time/lead time
        N = other dimensions
        D_(T'xN) = data collapsed to persistence output

    Use imputation only if data is sparse and predictable.

    Args:

        arr (array-like) - required:
            ArrayLike - supports numpy and xarray

        idx_time (int) - required:
            the dimension for time index

        num_workers (int):
            number of workers to use for processing persistence, defaults to number of cpus.

        num_chunks (int):
            number of chunks to use, defaults to `min(num_cpu, len(chunk_dimension))`

        method (str | StrEnum):
            The method to use to compute persistence. see `PersistenceMethod`.
            Supports:
                - "median_of_three"
                - "most_recent"

        simple_impute (bool):
            defaults to True. Set to False if nan needs to be preserved.
            NOTE: methods that require multiple non-nan datapoints to function may be forced to nan.

        backend_type (str | StrEnum):
            see `PersistenceBackendType`. The backend compute engine to use.
            Supports:
                - "numpy"

    Returns:

        an array (PetDataArrayLike) matched to the same specific input type in
        (PetDataArrayLike), i.e. output is guaranteed to have the same type as
        the input array.

        FUTUREWORK:

            Optionally also return and/or cache a stochastic signal (autoregressive function) that
            can be applied onto the persistence output (if the given method supports it). This
            allows for persistence guided by some simple derived trend (like day cycles).

            Again, its important that this stochastic trend isn't derived using complicated methods,
            and hence the user cannot provide this signal - it has to be pre-derived and cached by
            one of the persistence methods dynamically.
    """
    # force it to EnumStr - auto raises error if not compatible.
    if isinstance(method, str):
        method = PersistenceMethod(method)
    if isinstance(backend_type, str):
        backend_type = PersistenceBackendType(backend_type)

    # Force to sync dask as early as possible
    with _set_synchronous_dask():
        # lift structure to dataset representation (higher order)
        # structural order (highest to lowest)
        # - xr.Dataset
        # - xr.DataArray
        # - np.ndarray
        pet_ds = PetDataset(arr)

        # construct metadata
        metadata = PersistenceMetadata(
            idx_time_dim=idx_time_dim,
            method=method,
            num_workers=num_workers,
            num_chunks_desired=num_chunks,
            do_impute=simple_impute,
            backend=backend_type,
        )

        # apply function on each variable and destruct result
        # destructurize ONLYIF original array was lower order
        arr_result = pet_ds.map_each_var(_predict_single_var, metadata)

        # safety capture for dev/test
        assert isinstance(arr_result, type(arr))

        return arr_result


def _predict_single_var(
    da: xr.DataArray,
    metadata: PersistenceMetadata,
) -> xr.DataArray:
    """
    Computes persistence for a single data array, has the same interface as _compute_persistence
    except that the first argument is a data array.

    input: dataarray -> chunk -> impute -> compute persistence -> merge chunks -> dataarray :output
    """
    # --- simple chunk strategy (split) ---
    # build chunker struct
    chunker = PersistenceChunker(da=da, metadata=metadata)
    # this would have been filled up post-init or an error would have been raised.
    chunk_info = chunker.chunk_info
    # lazy - returns generator only.
    chunk_generator = chunker.generate_chunks()

    # --- launch compute pool and run method against chunks (apply and join)---
    # build compute struct
    # - this registers things from the metadata such as method and backend etc.
    # - uses chunk info to determine how to re-join the chunks.
    worker_pool = PersistenceComputePool(chunk_generator, chunk_info, metadata)
    da_result = worker_pool.map_and_join_chunks()

    return da_result


if __name__ == "__main__":
    raise NotImplementedError("TODO - standalone call")
