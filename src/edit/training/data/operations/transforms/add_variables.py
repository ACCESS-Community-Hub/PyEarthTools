import numpy as np
import xarray as xr
from edit.data import Transform


def time_of_year(method: str):
    ''' Transform to add TOY var to data '''
    class AddTimeOfYear(Transform):

        def apply(self, ds: xr.Dataset):

            if method == 'dayofyear':
                value = (np.cos(ds.time.dt.dayofyear.item()*np.pi/(366/2)) + 1)/2

            if method == 'monthofyear':
                value = (np.cos(ds.time.dt.date.item().month*np.pi/6 ) + 1)/2
            
            value = value * np.ones([len(ds[dim]) for dim in list(ds.dims)])
            ds[method] = (ds.dims, value)

            ds = ds.transpose(*list(ds.dims))
            return ds
    
    return AddTimeOfYear()