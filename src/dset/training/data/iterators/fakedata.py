"""
Fake Data Iterator to be used to test model performance without any data loading slowing things down.
"""

import warnings
import functools
import numpy as np

from dset.training.data.sequential import SequentialIterator
from dset.training.data.templates import DataIterator, DataIndex, OperatorIndex, DataInterface

@SequentialIterator
class FakeData(DataIterator):
    """
    Fake Data Iterator.

    Used to remove Data loading times for model testing
    """
    def __init__(self, index: DataInterface | OperatorIndex | DataIndex | tuple, num_iterations: int = None, catch: tuple[Exception] | Exception = None) -> None:
        """Setup Fake Data Loader

        `index` can be another iterator in which to infer shape from, or explicit shape

        Args:
            index (DataInterface | OperatorIndex | DataIndex | tuple): Iterator to get shape of data from, or explicit shape definition
            num_iterations (int, optional): Manual Number of iterations, so that `[set_iterable][dset.training.data.templates.DataIterator.set_iterable] doesn't need to be used. Defaults to None.
            catch (tuple[Exception] | Exception, optional): Exceptions to catch. Defaults to None.
        """
        super().__init__(index, catch)
        self.num_iterations = num_iterations

        self._data = None

        warnings.warn("Using FakeData Iterator. No data can be trusted or used for model training.", UserWarning)

        self.fake_data = True

    def set_fake(self, state: bool):
        """Set whether fake data is used or not

        Args:
            state (bool): Generate Fake Data
        
        Raises:
            TypeError: If self.index is tuple, as true data cannot be returned
        """
        if isinstance(self.index, tuple):
            raise TypeError(f"Cannot change fake data state if `index` is tuple")
        self.fake_data = state

    def _generate_data(self, shape: tuple) -> np.ndarray | tuple[np.ndarray]:
        """Generate Fake Data of given shape

        Args:
            shape (tuple): Shape of data to generate, can be tuple with shape or tuple of tuples with shape

        Returns:
            np.ndarray | tuple[np.ndarray]: Fake data
        """
        if isinstance(shape, tuple) and isinstance(shape[0], tuple):
            return tuple(self._generate_data(shp) for shp in shape)
        else:
            return np.zeros(shape)

    def _find_shape(self, data : tuple | np.ndarray) -> tuple:
        """Find shape of incoming data

        Args:
            data (tuple | np.ndarray): Either numpy array or tuple of numpy arrays

        Raises:
            TypeError: If type cannot be understood

        Returns:
            tuple: Shape of data
        """
        if isinstance(data, tuple):
            return tuple(self._find_shape(d) for d in data)
        elif isinstance(data, np.ndarray):
            return data.shape
        elif hasattr(data, 'shape'):
            return data.shape
        raise TypeError(f"Unable to get shape of object {data!r}")

    #@functools.lru_cache(1)
    def _get_fake_data(self) -> np.ndarray:
        """Get and Cache Fake Data

        Returns:
            np.ndarray: Generated Fake Data
        """
        if self._data:
            return self._data
            
        if isinstance(self.index, tuple):
            self._data = self._generate_data(self.index)
            return self._data

        for data_sample in self.index:
            break
        shape = self._find_shape(data_sample)
        self._data = self._generate_data(shape)
        return self._data


    @functools.wraps(DataIterator.set_iterable)
    def set_iterable(self, *args, **kwargs):
        super().set_iterable(*args, **kwargs)
        if hasattr(self.index, 'set_iterable'):
            self.index.set_iterable(*args, **kwargs)

    def __iter__(self) -> tuple | np.ndarray: 
        """Iterate over data

        If `self.fake_data` is True, generate and yield fake data up to either `num_iterations` or length of `set_iterable`

        Yields:
            tuple | np.ndarray: Data

        Raises:
            RuntimeError: If no iterating bounds set
        """
        if not self.set_iterable and not self._start:
            raise RuntimeError(f"In order to iterate `num_iterations` must be given or `set_iterable` run")

        if self.num_iterations and self.fake_data:
            for _ in range(self.num_iterations):
                yield self._get_fake_data()
        else: 
           for i in self.index:
                yield i

    def __getitem__(self, idx):        
        if self.fake_data:
            return self._get_fake_data()
        if isinstance(self.index, tuple):
            raise IndexError(f"As shape was given instead of an index, this cannot be indexed.")
        return super().__getitem__(idx)

    def __getattr__(self, key):
        if isinstance(self.index, tuple):
            raise AttributeError(f"As shape was given instead of an index, no further attributes can be retrieved.")
        return super().__getattr__(key)
    