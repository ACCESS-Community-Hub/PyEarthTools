from enum import StrEnum, auto

# 50% sparsity is reasonable, though some data like precipitation may be more sparse than this
_DEFAULT_PERSISTENCE_SPARSITY_MULTIPLIER = 2


class PersistenceMethod(StrEnum):
    """
    Methods to use for persistence.

    MEDIAN_OF_THREE:
        computes the median of the three most recent observations.

    MOST_RECENT:
        uses the most-recent value as persistence.

    Additionally, num_lookback is used to determine how many indices in the past are required from a
    dataslab in order to compute a persistence method.

    This is determined by the actual number of indices required multiplied by a sparsity factor to
    account for missing values. Missing values will optionally be imputed.
    """

    MOST_RECENT = "most_recent"
    MEDIAN_OF_THREE = "median_of_three"
    UNKNOWN = auto()

    def num_time_indices_required(self) -> int:
        """
        number of time indices required for computing a particular method
        """
        match self:
            case PersistenceMethod.MOST_RECENT:
                return 1
            case PersistenceMethod.MEDIAN_OF_THREE:
                return 3
            case _:
                raise NotImplementedError(
                    "PersistenceMethod: Invalid persistence method."
                )

    def min_lookback(
        self, sparsity_multiplier=_DEFAULT_PERSISTENCE_SPARSITY_MULTIPLIER
    ) -> int:
        """
        The minimum amount of lookback required to compute the corresponding metric.
        By default we assume a 50% sparsity and require at least double the number of values
        required for the compuation.
        """
        if sparsity_multiplier < 1:
            raise ValueError("PersistenceMethod: Sparsity multiplier must be >= 1")

        return int(self.num_time_indices_required() * sparsity_multiplier)
