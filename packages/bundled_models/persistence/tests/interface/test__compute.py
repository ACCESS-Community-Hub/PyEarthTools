"""
Tests various compute methods and backends at a high level. The focus is on structural preservation
of the various computations that are dispatched into multiprocessing workers. Also ensuring correct
mapping to the method/backend given the user input.

NOTE: this only does a very basic test of the method itself. Actual implementation and computational
accuracy of the method, and any edge cases are tested elsewhere.
"""

import numpy as np
import xarray as xr
import functools

from persistence.interface._backend import PersistenceBackendType
from persistence.interface._chunker import PersistenceChunker
from persistence.interface._compute import PersistenceCompute, PersistenceComputePool
from persistence.interface._metadata import PersistenceMetadata
from persistence.interface._method import PersistenceMethod


def _compute_single(
    method: PersistenceMethod,
    backend: PersistenceBackendType,
    random=False,  # defaults to "arange" i.e. value = 1-d index reshaped into nd-array
    shape_input=(4, 5, 2, 6, 10),
    numchunks=21,
    time_index=3,
) -> (PersistenceMetadata, np.ndarray, np.ndarray):
    """
    Helper function to create example data for a single computation.

    Useful for comparison of single workers vs pools, for various persistence methods and backends

    Returns references to:
        - metadata
        - input array (np.ndarray)
        - output array (np.ndarray)
    """
    # repeatability - re-seed rng state and bind it to `rng` variable
    rng = np.random.default_rng(seed=42)

    # derive array shape
    shape_input = list(shape_input)
    total_size = functools.reduce(lambda x, y: x * y, shape_input)

    # choose whether to use linear increments (essentially the equivilent 1d index as the value or a
    # random number as the value
    arr_in = None
    if random:
        arr_in = np.arange(total_size).reshape(shape_input)
    else:
        arr_in = rng.random(shape_input)

    # specify metadata (mocked user input)
    metadata = PersistenceMetadata(
        idx_time_dim=time_index,
        method=method,
        num_chunks_desired=numchunks,
        do_impute=True,
        backend=backend,
    )

    # compute output
    pc = PersistenceCompute(arr=arr_in, metadata=metadata)
    arr_out = pc.compute()

    # expect the array shape to be the same except for time dimension which should be reduced to 1
    expect_shape = [
        s if i != metadata.idx_time_dim else 1 for i, s in enumerate(arr_in.shape)
    ]

    # simple shape assert
    assert expect_shape == list(arr_out.shape)
    # return meta information for further tests in caller
    return metadata, arr_in, arr_out


def _compute_pool(
    method: PersistenceMethod,
    backend: PersistenceBackendType,
    _fn_compute_single=_compute_single,
    *_fn_extra_args,
    **_fn_extra_kwargs,
) -> (PersistenceMetadata, xr.DataArray, xr.DataArray):
    """
    Same as _compute_single but for xarrays and using chunked pools.

    Cheats a bit by using _compute_single as a default to avoid repetition for basic tests.

    Returns references to:
        - metadata
        - input array (xr.DataArray)
        - output array (xr.DataArray)
    """
    metadata, arr_in, arr_out = _fn_compute_single(
        method, backend, *_fn_extra_args, **_fn_extra_kwargs
    )

    # upgrade to data arrays with dummy names, except for the time index which will be 't'
    dim_names = [
        "x" + str(i) if i != metadata.idx_time_dim else "t"
        for i in range(len(arr_in.shape))
    ]

    # upgrade to dataarray
    da_in = xr.DataArray(arr_in, dims=dim_names)

    # chunk generator
    chunker = PersistenceChunker(da=da_in, metadata=metadata)

    # propagate information to compute pool
    pcp = PersistenceComputePool(
        chunk_generator=chunker.generate_chunks(),
        chunk_info=chunker.chunk_info,
        metadata=metadata,
    )

    # compute and retrieve chunks (joined back into data array)
    da_out = pcp.map_and_join_chunks()

    # expect the array shape to be the same except for time dimension which should be reduced to 1
    expect_shape = [
        s if i != metadata.idx_time_dim else 1 for i, s in enumerate(arr_in.shape)
    ]

    # simple shape assert
    assert list(da_out.shape) == expect_shape
    # dimnames should not have changed - NOTE: this may regress if xarray decides to deprecate dims
    # in favour of sizes, in which case we should be extracting the "keys" as an ordered tuple.
    assert dim_names == list(da_out.dims)
    # single worker and pool should have the same values
    assert np.allclose(da_out.values, arr_out)
    # return meta information for further tests in caller
    return metadata, da_in, da_out


def test_compute_medianofthree_workerpool_numpy():
    """
    method:  median of three
    backend: numpy

    expect lookback of 6 used for imputation (default)
    expect lookback of 3 used for median of three computation (definition)
    expect dimension shape to be preserved and only the time dimension to be reduced to 1
    expect dimension names to be mapped to the right shape
    expected array can be easily constructed using a manual equivilent numpy operation e.g.:
        1. create a range of numbers
        2. compute median the trivial way over the axis
        3. sense check a few cherrypicked numbers
        4. compare the output against the output of the worker pool
        5. repeat the above, but for a random array (in which case 3. is not necessary - and in fact
           cannot be done deterministically)

    Most of the same above strategy can be repeated for most of the other tests.

    ([numpy array], metadata) -> xarray dataarray
    """
    # values = 1-d index
    _, da_in, da_out = _compute_pool(
        PersistenceMethod.MEDIAN_OF_THREE,
        PersistenceBackendType.NUMPY,
    )

    # cherry picked tests (TODO)

    # values = random (TODO)


def test_compute_mostrecent_workerpool_numpy():
    """
    Sense check for most recent computation method
    """
    pass


def test_no_impute_workerpool_numpy():
    """
    Check when imputation is disabled - should preserve nans
    """
    pass


def test_compute_backend_supported():
    """
    Sense check for supported backends - should succeed

    NOTE: individual backend support themselves are done in tests of form <test>_<backend>
    e.g. test_compute_medianofthree_workerpool_numpy tests the median of three computation on the
    `numpy` backend pool
    """
    pass


def test_compute_backend_unsupported():
    """
    Sense check for unsupported backends - should error out
    """
    pass
