"""
Runs persistence model on the data loaded from the pipeline. Chunks the input data from the pipeline
and uses multiprocessing (if specified to do so).

Persistence potentially needs to be computed on the fly. Depending on the persistence method, and
model it is being compared against, the computation may require ingestion of a reasonable amount of
historical data.

The common use-case is to offload the data loading to something at a higher level (pet-pipeline).

This module can't control the loading process, instead what it controls is the way in which the
chunks are indexed, so that they can be _processed_ (CPU not IO) efficiently.

Examples of what can be done:
    - choice of backend (e.g. numba/rust etc., defaulting to numpy) - wip currently only numpy is
      supported

    - choice of number of chunks and workers for python to slice data into multiple workers.
      (embarassingly parallel)

    - choice of persistence method

    - flexiblity in how the input array/slab is provided, currently supports:
        - numpy array (<--- almost any hypercube datastructure can be converted to this)
        - xarray dataset
        - xarray dataarray

CAUTION:

    Due to the way data is stored and loaded, multiprocessing may sometimes be necessary but should
    be used with caution. Some tips, when in doubt just set the workers to 1, but you may still
    chunk the data if required due to memory issues.

    Again, the chunking here is not to do with loading, its to do with efficient processing.
    Assumedly the data is already chunked as it is loaded via some other framework. The chunking
    applied here is on top of that to further sub-slice things to take into account the need to
    ingest a large amount of data for aggregation computations.

ANTIPATTERNS (for developers):

    - do not chunk over time (except for specific exceptions)

    - do not use external multiprocessing/threading like dask

    - do not use multiprocessing IF the compute backend already does it efficiently, UNLESS we are
      IO bound.

    - do not use threading. IO bound issues should be resolved at a higher level because persistence
      methods (currently) have no control over how the data is loaded - actually this is the same
      for everything in PET that delegates data loading to the pipeline.

    - do not implemnent methods with heavy parametric statical inference or methods that are aware
      of the "meaning" of orthogonal dimensions in the hypercube other than "time".

    - do not do any overly clever chunk/worker optimization - this is the user's responsiblity

    - do not assume this will be called as a library (but can be if the OS allows it and its been
      tested sufficiently).

IMPORTANT:

    The "proper" way to run this module is a standalone process/script. But it _may_ work as part of
    a script/pipeline _if_ the underling OS supports it. See the executor pool defined in
    `interface._compute` and the main guard at the bottom.

FUTUREWORK:

    - Add the ability to bypass python completely for data loading.

        - Current architecture expects data to be lazily loaded from python but eagerly computed by
          the backend, which may still be python or could be something like rust or C.

        - The target alternative or toggle is for this to be inverted in a way that the data loading
          itself is done by the backend, allowing for even better control over the processing.

        - Persistence computation is relatively isolated enough from "frameworks" to be a perfect
          candidate to do this.
"""

from persistence.interface import (
    PetDataArrayLike,
    PersistenceComputePool,
    PersistenceBackend,
    PersistenceMethod,
    PetDataset,
)


def predict(
    arr: PetDataArrayLike,
    idx_time_dim: int,
    num_workers: int = None,
    num_chunks: int = None,
    method: PersistenceMethod | str = PersistenceMethod.MEDIAN_OF_THREE,
    simple_impute: bool = True,
    backend_type: PersistenceBackendType = PersistenceBackendType.NUMPY,
) -> pet_persist.PetDataArrayLike:
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
    if isinstance(method, str):
        # force it to EnumStr - auto raises error if not compatible.
        method = PersistenceMethod(method)

    # --- DEPRECATED ---
    # TODO: remove with_return_raw_result from PetDataset, there's no reason to
    # keep the lifted structure when the caller likely only requires the
    # original structure back.
    # pet_ds = pet_persist.PetDataset(arr).with_return_raw_result(return_raw_result)
    # ---

    # lift structure to dataset representation (higher order)
    # structural order (highest to lowest)
    # - xr.Dataset
    # - xr.DataArray
    # - np.ndarray
    pet_ds = PetDataset(arr)

    raise NotImplementedError("TODO: map to persistence metadata")

    metadta = PersistenceMetadata(...)

    # apply function (ALWAYS) and destruct result (ONLYIF original array was lower order)
    arr_result = pet_ds.map_each_var(
        _predict_single_var,
        metadata,
    )

    # safety capture for dev/test
    assert type(arr) == type(arr_result)

    return arr_result


# TODO: make this ingest PersistenceMetadata instead...
def _predict_single_var(
    da: xr.DataArray,
    idx_time_dim: int,
    num_chunks: int = None,
    method: PersistenceMethod = PersistenceMethod.MEDIAN_OF_THREE,
    simple_impute: bool = True,
):
    """
    Computes persistence for a single data array, has the same interface as _compute_persistence
    except that the first argument is a data array.
    """
    # create metadata

    # input dataarray -> chunk -> impute -> compute persistence -> merge chunks
    chunker = PersistenceChunker(
        da_lazy=da,
        method=method,
        num_chunks=num_chunks,
        idx_time_dim=idx_time,
    )

    # TODO: worker pool
    # TODO: work chain i.e. slice -> impute -> compute
    # TODO: merge result
    raise NotImplementedError("TODO - some missing parts")


if __name__ == "__main__":
    raise NotImplementedError("TODO - standalone call")
