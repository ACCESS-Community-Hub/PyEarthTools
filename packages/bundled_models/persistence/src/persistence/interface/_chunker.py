from dataclasses import dataclass
import copy
import math
import numpy as np
import xarray as xr
import functools

from typing import Generator

from persistence.interface._metadata import PersistenceMetadata
from persistence.interface.types import PetDataArrayLike

# ---
# 1000 chunks is more than enough for most usecases. Persistence methods should not be using large
# amounts of historical data, and therefore should not need heavy chunking for data to fit in
# memory.
#
# If memory is an issue, this needs to be solved at a higher level where properties of the chunk
# strategy at the storage level are known and data can be optimally bounded (spatially or otherwise)
# before reaching the persistence chunker.
#
# Further the minimum memory usage is lower bounded by an entire single time slice of the of the
# data being processed since that is the output, and also is affected by the number of parallel
# workers used.
#
# Increasing chunk counts past a certain certain amount is therefore counter-productive.
_MAX_NUM_CHUNKS = 1000
# ---


@dataclass
class PersistenceChunkInfo:
    # ---
    # least significant chunk index (fastest varying), most significant is 0, indices are
    # incremented from least significant (fast) to most significant (slow)
    lsi_chunk: int
    # ---
    num_chunks: int
    size_chunk: int
    dim_names: list[str]
    shape_full: list[int]


@dataclass
class PersistenceDataChunk:
    """
    The reason this is a class is that, there could be more useful info here in the future such as
    start/end slices, and the chunk identifier, but for now its just a shallow wrapper and
    effectively a type alias.
    """

    arr_chunk: np.ndarray

    # chunks are calculated independently and in different workers so a reference
    # to the metadata is convenient. This is a small over-head.
    metadata: PersistenceMetadata

    # list containing slices of each dimension that make up the chunk
    slice_dims: list[slice]

    # reduced dimensions expected from the output (reduced)
    slice_dims_reduced: list[slice]


@dataclass
class PersistenceChunker:
    """
    The persistence chunker chunks a xarray dataarray and relays them using a generator (lazy).

    Important:

        This is not a general purpose chunker. It is tailored for persistence and has a critical
        assumption that the time dimension will not be chunked during the computation (it may still
        be chunked in storage - this is fine). The chunking strategy is also intentionally
        simplistic and greedy.

        Depending on the method we could require 1 historical entry or 200. Therefore, there is no
        "optimal" choice of chunks and workers here, since the data is not guaranteed to be stored
        optimally for every choice of persistence method. The reason why persistence is so much
        different to other models, is because we aren't storing any weights everything is done
        on-the-fly.

        Hence, if there are issues with memory, the solution should be at a higher level where the
        chunking strategy of the stored data is known, and appropriately bounded or alternatively
        prepared offline with a storage strategy conducive to persistence calculations, BEFORE being
        passed into this chunker. This may introduce a storage burden, but is imperative for any
        sort of baseline model that cannot rely on stored weights, to function.

    The chunking algorithm is as follows:

        Divide the total size (product of the data shape) by the desired number of chunks (rounded
        up, min chunk size = 1). This is the desired chunk size.

        Working backwards from the fastest varying index/axis/dimension (len - 1), find if the
        desired chunk size is greater tha the product of the cardinality any slower varying indices.
        (natural element = 1) e.g.

        product[len - 1] = 1
        product[len - 2] = shape[len - 1] * product[len - 1]
        product[len - 3] = shape[len - 2] * product[len - 2]
        ...

        If the chunk size is smaller than the product, stop. Create a marker at this index - call it
        the "stop" index (i.e. the most significant index used for the chunk size calculation). The
        product at the given iteration is the _actual_ chunk size.

        Then, for all indices that are more significant the "stop" index, increment it as a multi
        index ring to find the start and end indices of the hyperslab.

        In other words, the chunks are designed in such a way that indices that are faster varying
        than the "stop" index are always at their cardinality (max size), and slower varying indices
        are incremented and used for selection. Increments are over the fastest of the slowest
        varying index (i.e. fastest most significant index).

        Note: the time index is a special case and should be ignored.

        Note: the most significant index is the slowest varying index and the least significant
              index is the fastest varying index. i.e.

                x[i0,...]  v.s. x[...,iN] => i0 is slow varying, iN is fast varying.

        Note: It is *usually* more efficient to to increment chunks by the slower varying indices -
              as this *usually* guarentees that the chunks are contiguous in memory (C-style). But
              for updating individual values in a chunk the opposite is true. i.e. traversing chunks
              v.s. traversing elements. Here we want the former for chunking, and the latter for
              computation. Which is why we chunk with slower varying indices and compute with faster
              varying indices (with whatever backend of choice).

        Note: The reason why dask isn't used (or at least forced into synchronous mode),
              is because its configuration in PET (but possibly in general) is hard to pin down.

        Note: we could have used numpy.nditer with a external loop, but we would like to keep the
              structure of the array and not flatten it. Further, we are only dealing with a max of
              1000 so any benefit would be minimal.

        FUTUREWORK: The loaders should present options to use direct mechanisms to load particular
        types of data rather than xarray. For now this class has no control over the data loader.
    """

    da: xr.DataArray
    metadata: PersistenceMetadata
    chunk_info: PersistenceChunkInfo | None = None

    # TODO:
    #   add data shape as an explicit input, as even da.shape may trigger a computation depending on
    #   the underlying storage type.

    @staticmethod
    def _b10_to_mi(b10: int, mi_size: list[int]) -> list[int]:
        """
        Given:
            1. a base10 (integer) representation of the product of a multiindex
            2. a list of the cardinality of each index (size of each index)
        convert the base10 representation of a multiindex back to a multiindex.
        """
        assert b10 >= 0
        assert all([x is not None and x >= 0 for x in mi_size])

        rem = b10  # set remainder to the orignal base10 value

        # incrementing the most significant shifts the hyperslab by the product of the size of every
        # other index after it. This is a running product that is the "base" of a given multi index.
        # the least significant index will have a base of 1.
        mi_sizeshift = mi_size[1:] + [1]
        prod = functools.reduce(lambda x, y: x * y, mi_sizeshift)

        num_idx = len(mi_size)  # number of indices
        mi = [None for i in range(num_idx)]  # initialize multi index to return

        for i, s in enumerate(mi_sizeshift):
            # calculate quotient/remainder
            quo, rem = divmod(rem, prod)

            # update multi-index forwards (most-significant first)
            mi[i] = quo

            # update product by reverting the most recent size (i.e. divide) the minimum product
            # must be one.
            prod = max(prod // s, 1)

        assert all([x is not None and x >= 0 for x in mi])
        assert len(mi) == len(mi_size)
        return mi

    @staticmethod
    def _mi_to_b10(mi: list[int], mi_size: list[int]) -> int:
        """
        Given:
            1. a list of indices (for each dimension)
            2. a list of the cardinality of each index (size of each index)
        convert the multiindex (1.) into a base10 (integer) representation.
        """
        assert len(mi) == len(mi_size)
        assert all([x is not None and x >= 0 for x in mi])
        assert all([x is not None and x >= 0 for x in mi_size])

        prodscan = 1  # running accumulation of product
        b10 = 0  # calculated using prodsum

        # need to reverse arrays since least significant needs to be computed first
        for i, v in enumerate(zip(mi[::-1], mi_size[::-1])):
            ix, s = v
            b10 += ix * prodscan
            # update product with latest size
            prodscan *= s

        assert b10 >= 0
        return b10

    @staticmethod
    def _inc_mi(mi: list[int], mi_size: list[int], inc=1) -> list[int]:
        """
        Increments a multindex by 1, note: this is the inefficient way, but it doesn't need to be
        efficient - chunk sizes are hard capped to 1000. Note: the fastest varying index (last
        index) is incremented first since that minimizes cache misses.

        Algorithm:

          Convert multi index to base10, then
          add 1 to base10 value (or inc if specified) - trivial increment, then
          convert back to multiindex
        """
        assert inc > 0
        assert len(mi) == len(mi_size)
        assert all([x is not None and x >= 0 for x in mi])
        assert all([x is not None and x >= 0 for x in mi_size])

        fn_b10 = functools.partial(PersistenceChunker._mi_to_b10, mi_size=mi_size)
        fn_b10_inv = functools.partial(PersistenceChunker._b10_to_mi, mi_size=mi_size)
        mi_next = fn_b10_inv(fn_b10(mi) + inc)

        if mi_next[0] >= mi_size[0]:
            raise OverflowError(
                f"PersistenceChunker: increment multindex - overflow {mi} + {inc} goes past the"
                f" maximum sizes: {mi_size}."
            )

        assert all(
            [x is not None and x >= 0 and x < s for x, s in zip(mi_next, mi_size)]
        )
        return mi_next

    @staticmethod
    def _compute_chunkinfo_greedy(
        desired_numchunks: int,
        mi_size: list[int],
        dim_names: list[str],
    ) -> PersistenceChunkInfo:
        """
        This is a greedy chunksize calculation, because it prefers having entire dimensions as part
        of a chunk rather than partial extents in a dimension. Although this is the only chunking
        strategy that will be conceivably used in the near future.

        Returns a structure (PersistenceChunkInfo) containing
            1. actual chunk size
            2. actual chunk count
            3. the position (least significant) of the first index that should be be used for
               incrementing chunks (using multi-indexing)
            4. dimension names (passed through)
        """
        assert desired_numchunks >= 1

        if isinstance(mi_size, tuple):
            mi_size = list(mi_size)

        total_size = functools.reduce(lambda x, y: x * y, mi_size)
        desired_chunksize = int(max(1, math.ceil(total_size / desired_numchunks)))

        num_idx = len(mi_size)
        prodsize = 1
        actual_chunksize = None
        first_chunkindex = None

        for i, s in enumerate(mi_size[::-1]):
            if prodsize >= desired_chunksize and s != 1:
                first_chunkindex = num_idx - i - 1
                actual_chunksize = prodsize
                break
            prodsize *= s

        # single chunk
        if first_chunkindex is None or actual_chunksize is None:
            actual_chunksize = prodsize
            actual_numchunks = 1
            first_chunkindex = 0

        actual_numchunks = total_size // actual_chunksize

        assert actual_chunksize >= desired_chunksize
        assert actual_numchunks <= desired_numchunks

        return PersistenceChunkInfo(
            num_chunks=actual_numchunks,
            size_chunk=actual_chunksize,
            lsi_chunk=first_chunkindex,
            dim_names=dim_names,
            shape_full=mi_size,
        )

    def __post_init__(self):
        # safety: don't want assume sets or dict keys because they may be unordered (depending on
        # the version of python). However, most likely, dict is okay as long as we don't support
        # python<=3.7
        assert isinstance(self.da.dims, tuple) or isinstance(self.da.dims, list)

        # check for chunks
        if (
            self.metadata.num_chunks_desired < 1
            or self.metadata.num_chunks_desired > _MAX_NUM_CHUNKS
        ):
            err_msg = f"specified num chunks is invalid, valid range: 0 < num chunks <= {_MAX_NUM_CHUNKS}"
            raise ValueError(err_msg)

        # ---
        # Suppress time index for calculations.
        #
        # NOTE:
        #
        #   Expanding an array by one dimension with a dimensionality 1, for example, has no impact
        #   on the chunk size, since the retraction operation of squeezing out the dimension, of
        #   size 1, also does not affect chunk size. Therefore, to suppress a dimension we set its
        #   size to 1 or drop it. Forcing to 0 is not right here, since that'd result in a empty array.
        #
        #   Since we want to preserve structure, we can't drop it so our only remaining option is to
        #   force the size to 1.
        shape_notime = list(self.da.shape)
        shape_notime[self.metadata.idx_time_dim] = 1
        # ---

        self.chunk_info = self._compute_chunkinfo_greedy(
            self.metadata.num_chunks_desired,
            shape_notime,
            self.da.dims,
        )

        # check that the input data shape has enough time indices to support the persistence
        # calculation (including preprocessing).
        len_time_max = self.da.shape[self.metadata.idx_time_dim]
        len_time_prp = self.metadata.len_time_preprocess()
        if len_time_prp > len_time_max:
            raise ValueError(
                "PersistenceChunker: input DataArray does have enough time indices for this"
                " persistence method."
            )

    def _get_dim_slices(self, mi: list[int]) -> dict[str, slice]:
        """
        maps slices to dimension names.

        1. slices time based on required number of historical data for imputation/persistence
           calculations.

           NOTE:

               This is an added safety, since it is expected that something higher level would have
               sliced this by now. But, in case the data-array points (lazily) to the entire history
               (for example), this slicing makes certain that the data that is loaded into memory is
               still reasonably bounded.

        2. slices other indices based on required chunk sizes
        """
        assert self.chunk_info is not None and self.chunk_info.lsi_chunk is not None
        assert all([x is not None and x >= 0 for x in mi])

        dict_slice_dims = {}
        len_time_max = self.da.shape[self.metadata.idx_time_dim]
        len_time_prp = self.metadata.len_time_preprocess()
        # this is static for all chunks
        slice_time = slice(len_time_max - len_time_prp, len_time_max)

        for idx, name in enumerate(self.da.dims):
            dim_size = self.da.shape[idx]

            # time dimension => use special time slicing
            if idx == self.metadata.idx_time_dim:
                # assert time dimension name is stored correctly - random safety check
                assert name == self.chunk_info.dim_names[self.metadata.idx_time_dim]
                dict_slice_dims[name] = slice_time

            # multi-indexer dimension => 1^m slice => incremental chunk of size 1
            elif idx < self.chunk_info.lsi_chunk + 1:
                dict_slice_dims[name] = slice(mi[idx], mi[idx] + 1)

            # chunk dimension => N_i^(n-m) slice => use the entire dimension as a chunk (N_i)
            else:
                dict_slice_dims[name] = slice(0, dim_size)

        assert all(n in dict_slice_dims for n in self.chunk_info.dim_names)
        return dict_slice_dims

    def generate_chunks(self) -> Generator[PersistenceDataChunk]:
        """
        Evaluate chunks by loading each chunk into memory, the chunks are lazily loaded but eagerly
        evaluated in memory in the backend. Chunks should ideally be contiguous in memory. (Except
        for time).

        This generator generally would be fed into a multiprocessing worker pool in conjunction with
        a method to process each chunk.
        """
        # chunksize = 1, early return
        if (
            self.chunk_info.num_chunks == 1
            or self.chunk_info.size_chunk >= self.da.size
        ):
            # select everything for both input and result
            slice_dims = [slice(None)] * len(self.da.shape)
            slice_dims_reduced = slice_dims
            yield PersistenceDataChunk(
                self.da, self.metadata, slice_dims, slice_dims_reduced
            )
            return

        # TODO: add a fast return for the special case when time is the only dimension.
        shape_notime = list(self.da.shape)
        shape_notime[self.metadata.idx_time_dim] = 1
        shape_notime_trimmed = shape_notime[: (self.chunk_info.lsi_chunk + 1)]
        mi_inc = [0 for _ in shape_notime_trimmed]

        for _ in range(self.chunk_info.num_chunks):
            dict_slice_dims = self._get_dim_slices(mi_inc)
            arr_chunk = self.da.isel(dict_slice_dims)

            # pass chunk to caller
            slice_dims = list(dict_slice_dims.values())
            slice_dims_reduced = copy.deepcopy(slice_dims)
            slice_dims_reduced[self.metadata.idx_time_dim] = slice(None, None, None)
            yield PersistenceDataChunk(
                arr_chunk,
                self.metadata,
                slice_dims,
                slice_dims_reduced,
            )

            # increment index and break if overflow is detected.
            try:
                mi_inc = self._inc_mi(mi_inc, mi_size=shape_notime_trimmed)
            except OverflowError:
                return
