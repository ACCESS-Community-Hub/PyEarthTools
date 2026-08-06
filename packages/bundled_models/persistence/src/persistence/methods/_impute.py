"""
This module handles imputation of missing data using very simple techniques.

Only mean is currently supported.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SimpleImpute:
    arr: np.ndarray

    def impute_mean(self) -> np.ndarray:
        """
        To keep the imputation representative of the data but yet simple we can do a simple
        mean interpolation over the data slab.

        NOTE: This is non-deterministic depending on the data chunking strategy.
        """
        nanmask = np.isnan(self.arr)
        if not nanmask.any() or nanmask.all():
            # if nothing is missing or everything is missing, return the original array as-is
            return self.arr
        else:
            # otherwise, replace missing values with the mean of the slab
            # NOTE: the following flattens the array by default if axis isn't specified
            fillval = np.nanmean(self.arr)
            arr_imputed = np.where(nanmask, fillval, self.arr)
            return arr_imputed
