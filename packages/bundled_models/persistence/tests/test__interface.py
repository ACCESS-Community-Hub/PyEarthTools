"""
Basic suite of tests that make sure that the interface objects work as expected.
"""

import numpy as np
import xarray as xr
import persistence as pet_persist


def test_persistence_method_obj():
    """
    Basic test to check object creation: PersistenceMethod
    """
    persistence_mostrecent = pet_persist.PersistenceMethod.MOST_RECENT
    persistence_median = pet_persist.PersistenceMethod.MEDIAN_OF_THREE

    # sense checks - mostrecent
    assert persistence_mostrecent.num_time_indices_required() == 1
    assert persistence_mostrecent.min_lookback() == 2
    assert persistence_mostrecent.min_lookback(3) == 3  # 3 * 1

    # sense checks - median
    assert persistence_median.num_time_indices_required() == 3
    assert persistence_median.min_lookback() == 6
    assert persistence_median.min_lookback(50) == 150  # 3 * 50


def test_persistence_data_chunk_obj():
    arr_chunk = np.random.randint(0, 10, (2, 5, 8))
    persistence_method = pet_persist.PersistenceMethod.MOST_RECENT
    idx_time: int = 1  # len = 5

    metadata = pet_persist.PersistenceMetadata(
        idx_time_dim=idx_time,
        method=persistence_method,
    )

    datachunk = pet_persist.PersistenceDataChunk(
        arr_chunk=arr_chunk,
        metadata=metadata,
    )

    assert datachunk.arr_chunk.shape.index(5) == datachunk.metadata.idx_time_dim
    assert datachunk.metadata.method.min_lookback() == 2


def test_persistence_chunker_obj():
    """
    Basic test to check object creation: PersistenceChunker
    """
    da = xr.DataArray(
        np.random.randint(0, 10, (2, 5, 8)),
        dims=["x0", "time", "x2"],
    )
    idx_time: int = 1  # len = 5
    num_chunks: int = 4  # each chunk is 2x5x2
    persistence_method = pet_persist.PersistenceMethod.MOST_RECENT
    metadata = pet_persist.PersistenceMetadata(
        idx_time_dim=idx_time,
        method=persistence_method,
        num_chunks=num_chunks,
    )
    chunker = pet_persist.PersistenceChunker(
        da=da,
        metadata=metadata,
    )

    # sense checks
    assert da.shape.index(5) == chunker.metadata.idx_time_dim
    assert chunker.metadata.num_chunks == 4
    assert chunker.metadata.method.num_time_indices_required() == 1


def test_chunker_multi_index_increment():
    """
    Tests the scenario in the docstrings for mult index increment

    i.e.
        shape = (2, 4, 10, 2)
        chunk_size = 47 (or increment size)

    Also does a double increment and a manual isel on the dataarray to make sure the sizes are as
    expected.

    For this particular purpose we shall include a dummy dimension - time and it should be ignored.

        (2, 4, 5*, 10, 2)

    * time dimension

    as per the doc string example we expect giving a start index of all zeros and a increment (chunk
    size) of 47, the next index we should receive is:

        (0, 2, 5*, 3, 1)
    """
    da = xr.DataArray(
        np.random.randint(0, 10, (2, 4, 5, 10, 2)),
        dims=["x0", "x1", "time", "x3", "x4"],
    )
    idx_time: int = 2
    chunk_size: int = 47

    # NOTE: num_chunks is a dummy and not used since we want to explicitly test "47"
    # still we set it abnormally high here to check that it is clipped to the data cardinality
    # appropriately.
    num_chunks: int = 999

    persistence_method = pet_persist.PersistenceMethod.MOST_RECENT
    metadata = pet_persist.PersistenceMetadata(
        idx_time_dim=idx_time,
        method=persistence_method,
        num_chunks=999,
    )
    chunker = pet_persist.PersistenceChunker(
        da=da,
        metadata=metadata,
    )

    assert chunker.metadata.num_chunks == 2 * 4 * 10 * 2

    start_index = (0, 0, 0, 0, 0)
    end_index = chunker.increment_multi_index(start_index, chunk_size)

    assert end_index == [0, 2, 5, 3, 1]

    # check slicing
    np_start_index = np.asarray(list(start_index))
    np_end_index = np.asarray(end_index) + 1

    # assert xarray dataarray dims returns a tuple (since tuples are ordered sets)
    assert isinstance(da.dims, tuple)

    dim_names = list(da.dims)
    multi_slice = {
        dim_names[i]: slice(v[0], v[1], 1)
        for v, i in enumerate(zip(np_start_index, np_end_index))
    }
    da_slice = da.isel(**multi_slice)
    da_slice.shape


def test_chunker_multi_index_increment_with_single_dim():
    """
    Tests multi index increment for the case where there is only a single dimension This should
    return the entire array back as-is since there can only be one dimension in this case and that
    dimension cannot be chunked - i.e. time
    """
    pass


def test_chunker_multi_index_increment_unit_cardinality():
    """
    Tests multi index increment for the case where there are multiple indices but the indices all
    have a cardinality of 1 => we can only have one chunk, regardless of what we set num_chunks to.
    """
    # set num_chunks to 10 arbitrarily

    # chunks should be trimmed to min(10, np.prod(all_dims_except_time) => 1) = 1
    pass
