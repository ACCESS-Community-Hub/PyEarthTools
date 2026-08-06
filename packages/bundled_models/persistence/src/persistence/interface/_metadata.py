from dataclasses import dataclass, field
from multiprocessing import cpu_count
from persistence.interface._backend import PersistenceBackendType
from persistence.interface._method import (
    PersistenceMethod,
    _DEFAULT_PERSISTENCE_SPARSITY_MULTIPLIER,
)


@dataclass
class PersistenceMetadata:
    """
    Reference to common data that is passed around during persistence computations.
    """

    idx_time_dim: int  # index of time dimension
    method: PersistenceMethod  # persistence method to use

    # --- (kw)args with defaults ---
    # IMPORTANT: These are essentially tuning parameters that affect performance. The defaults are
    # usually okay, but they need to be considered carefully for certain systems with limited
    # computational power.
    num_workers: int = field(default_factory=cpu_count)

    # ---
    # NOTE:
    #
    #   A hyperslab/cube is bound by orthogonal hyperplanes, each with its surface parallel to
    #   a unique axis or dimension. In our case a hyperslab is a chunk.
    #
    #   The above constraint simplifies retrieval of chunks, without needing to flatten or change
    #   the underlying data structure. On the other hand, the constraint makes it harder to
    #   accomodate every possible chunk size/count.
    #
    #   Therefore, the number of chunks requested by the user is a desire, not a guarentee.
    #   The actual chunksize is computed at runtime, and depends on the data shape.
    #
    #   The runtime algorithm must abide by the constraints of hyperslab selection while choosing a
    #   chunk size that is close to the desired chunk size.
    num_chunks_desired: int = 1
    # ---

    do_impute: bool = True
    backend: PersistenceBackendType = PersistenceBackendType.NUMPY

    # ---
    # multiplier to determine how much data to load, essentially
    #
    #   S * N, where,
    #   N = Minimum amount of data required for computing a method
    #   S = this multiplier.
    #
    # The default is conservatively set at 2 so that it is capable of treating missing values, while
    # not overzealously loading things into memory.
    #
    # If a dataset does not have missing values this can be set to 1, to minimize the load on memory.
    #
    # On the other hand some datasets may need a much larger sparsity multiplier as they are mostly
    # sparse - this can be useful when values from historical observations quite far into the past
    # can still be useful for persistence.
    sparsity_multiplier: int = _DEFAULT_PERSISTENCE_SPARSITY_MULTIPLIER
    # ---

    def len_time_preprocess(self) -> int:
        """
        number of historical time indices required for preprocessing, e.g. imputation to fill
        missing values.

        This is used during the chunking and pre-processing phase.
        """
        _len = int(self.method.min_lookback(self.sparsity_multiplier))
        assert _len >= 1
        return _len

    def len_time_compute(self) -> int:
        """
        number of historical time indices required for the persistence computation.

        This is used during the compute phase.
        """
        _len = int(self.method.num_time_indices_required())
        # safety: this must always be smaller than or equal to the pre-processing length
        assert _len <= self.len_time_preprocess()
        assert _len >= 1
        return _len
