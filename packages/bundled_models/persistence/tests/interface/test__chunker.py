import functools
import xarray as xr
import numpy as np

import persistence.interface._chunker as _chunker
import persistence.interface._metadata as _metadata
import persistence.interface._method as _method

_pcr = _chunker.PersistenceChunker
_pci = _chunker.PersistenceChunkInfo
_pdc = _chunker.PersistenceDataChunk
_pma = _metadata.PersistenceMetadata
_pmd = _method.PersistenceMethod


def test_generate_chunks_default():
    """
    default chunk count is 1, i.e. no chunks or the entire dataset is a single chunk, this should
    give the same result as ..._single_large_chunk.

    This is a separate test because the default may change, but we still want to retain the test
    below for a single large chunk
    """


def test_generate_chunks_common_usecases():
    """
    common usecases for chunking

    Assume a reasonable number of dimensions for this test.
    (3, 8, 10*, 5, 4)

    10* => is the time dimension and should be ignored by the chunking strategy.

    total size = 3 * 8 * 5 * 4 = 480

    we test the following chunk sizes:
    - chunk start index = 3, chunksize = 4,   chunkshape = (1, 1, 10, 1, 4)
    - chunk start index = 1, chunksize = 20,  chunkshape = (1, 1, 10, 5, 4)
    - chunk start index = 0, chunksize = 160, chunkshape = (1, 8, 10, 5, 4)

    the desired chunks that can result in the above results are:
    - 4   >= chunksize > 1,  120 <= numchunks < 480, choose 479 arbitrarily
    - 20  >= chunksize > 4,   24 <= numchunks < 120, choose 24 arbitrarily
    - 160 >= chunksize > 20,   3 <= numchunks < 24,  choose 11 arbitrarily

    NOTE:
        The first two cases above are intentionally edge cases and sit at the boundaries.
        More edge cases such as:
            - intentionally bad settings of chunks,
            - impact of chunking along the first/last index,
            - the position of the time index,
            - testing defaults,
        are covered in other tests.
    """
    arr_shape = [3, 8, 10, 5, 4]
    arr_shape_notime = [v if i != 2 else 1 for i, v in enumerate(arr_shape)]
    size_total = functools.reduce(lambda x, y: x * y, arr_shape_notime)
    num_chunks = [479, 24, 11]
    # with MEDIAN_OF_THREE we expect 2 * 3 = 6 indices for time
    method = _pmd.MEDIAN_OF_THREE
    exp_result = [
        (3, 4, [1, 1, 6, 1, 4]),
        (1, 20, [1, 1, 6, 5, 4]),
        (0, 160, [1, 8, 6, 5, 4]),
    ]
    idx_time_dim = 2
    test_data = xr.DataArray(np.ones(arr_shape), dims=["x0", "x1", "t", "x2", "x3"])

    for i, nchk in enumerate(num_chunks):
        metadata = _pma(
            idx_time_dim=idx_time_dim, num_chunks_desired=nchk, method=method
        )
        chunker = _pcr(da=test_data, metadata=metadata)
        assert chunker.chunk_info.lsi_chunk == exp_result[i][0]
        assert chunker.chunk_info.size_chunk == exp_result[i][1]
        assert chunker.chunk_info.num_chunks == size_total // exp_result[i][1]
        for data_chunk in chunker.generate_chunks():
            assert list(data_chunk.arr_chunk.shape) == exp_result[i][2]


def test_generate_chunks_single_large_chunk():
    """
    explicitly set chunk sizes = 1
    """
    pass


def test_generate_chunks_each_element_is_a_chunk():
    """
    exlicitly set num_chunks = total size
    """
    pass


def test_generate_chunks_edge_cases():
    """
    - desired num chunks is less than 1
    - desired num chunks is greater than the max supported chunk size
    """
    pass


def test_chunk_caculation_single_worker():
    """
    basic test of multiprocessing pool processing the generated chunks, but with a single worker.
    This should work in most setups.

    TODO: copy the notes below to the compute pool - this is a temporary location

    NOTE: chunking only saves memory if num_chunks > num_workers. And that too only during
    processing since we only load a fraction of the input array at a given time.

    NOTE: regardless, the final array will be joined in-memory, this is unavoidable unless each
    worker writes straight to disk - which is out of scope. So the minimum memory usage will always
    be greater than the size of the entire hypercube for a single time instance (persistence returns
    1 time point)

    """


# TODO:
# --- optional tests that are run only if the system can handle it ---
# @pytest.mark.skipif(
#     mem < "1GiB", reason="system memory is not large enough to run test"
# )
# def test_chunking_large_data_large_chunks():
#     """
#     skip if system does not have enough memory
#     """
#     pass
#
#
# def test_multiprocessing_pool_ingest():
#     """
#     skip if system only has a single worker
#     """
#     pass
