import math

import numpy as np

from dset.training.data.templates import (
    DataIterationOperator,
    DataIterator,
    SequentialIterator,
)


@SequentialIterator
class DropNan(DataIterationOperator):
    """
    Drop any data with nans when iterating.
    """

    def __iter__(self):
        for data in self.iterator:
            if isinstance(data, tuple):
                if all(np.isnan(d).any() for d in data):
                    continue
            else:
                if np.isnan(data).any():
                    continue
            yield data


@SequentialIterator
class DropValue(DataIterationOperator):
    """
    Drop Data containing a value abover a percentage
    """

    def __init__(self, iterator: DataIterator, value: float, percentage: float) -> None:
        """
        Drop Data if number of elements equal to value are greater than percentage when iterating.

        When using __getitem__ do nothing.


        Parameters
        ----------
        iterator
            Iterator
        search_value
            Value to search for
        percentage
            Percentage of which an exceedance drops data
        """
        super().__init__(iterator)

        self.function = (
            lambda x: ((np.count_nonzero(x == value) / math.prod(x.shape)) * 100)
            > percentage
        )
        self.__doc__ = f"Drop data containing more than {percentage}% of {value}"

    def __iter__(self):
        for data in self.iterator:
            if isinstance(data, tuple):
                if all(self.function(d) for d in data):
                    continue
            else:
                if self.function(data):
                    continue
            yield data
