"""
MASS Access

Provides a generic class to assist in getting data from MASS
"""

from __future__ import annotations
from abc import abstractmethod, ABCMeta
import warnings

from functools import cached_property

import xarray as xr
import tempfile
from pathlib import Path
import uuid

from edit.data.indexes import BaseCacheIndex
from edit.data import IndexWarning, DataNotFoundError
from edit.data.transform import Transform, TransformCollection


class MASS(BaseCacheIndex, metaclass = ABCMeta):
    """
    Core `MASS` to be implemented by any data class accessing MASS.
    Subclasses from the `BaseCacheIndex` to provide automated generation, and saving if a `cache` is specified.
    
    A subclass must also inherit from other indexes to provide the more complex indexing functions, usually this will be `edit.data.indexes.ArchiveIndex`.

    By default, all data is cached to a temporary directory.

    A child class must implement `._mass_filepath`
    """
    _interim_temp_dir = None
    
    def __init__(
        self, 
        *args, 
        pre_save_transforms: Transform | TransformCollection = TransformCollection(), 
        **kwargs
        ):
        """
        Setup MASS Indexer
        
        All other kwargs go up to `BaseCacheIndex`.
        
        Args:
            pre_save_transforms (Transform | TransformCollection, optional):
                Transformations to be applied after conversion to netcdf, but before saving to cache.
                Defaults to TransformCollection().
            
        """
        kwargs["cache"] = kwargs.pop("cache", "temp")
        kwargs["pattern"] = kwargs.pop("pattern", "TemporalExpandedDateVariable")

        super().__init__(*args, **kwargs)
        
        if not hasattr(self, "catalog"):
            self.make_catalog()
        self._save_catalog(self.catalog, 'index')
        self.pre_save_transforms = TransformCollection() + pre_save_transforms
        
    @property
    def _interim_dir(self):
        """Interim directory to save pp files and select files"""
        
        if self._interim_temp_dir is not None:
            return self._interim_temp_dir.name
        
        dir = str(Path(self.cache)) if self.cache is not None else None
        self._interim_temp_dir = tempfile.TemporaryDirectory(dir = dir, prefix = '._download_')
        Path(self._interim_temp_dir.name).mkdir(exist_ok = True, parents = True)
        return self._interim_temp_dir.name
    
    @_interim_dir.deleter
    def _interim_dir(self):
        if self._interim_temp_dir is None:
            return
        self._interim_temp_dir.cleanup()
        del self._interim_temp_dir

    def _iris_post(self, iris_data):
        """
        Processing required after loading the iris data
        """
        ## Provide an implementation in the child class
        return iris_data
    
    def _convert_to_netcdf(self, iris_temp_path) -> str:
        """
        Convert an iris data file on disk to netcdf
        
        Returns:
            (str):
                Path to netcdf object. Will be stored in `_interim_dir,
                and deleted when this object is.            
        """
        import iris
        try:
            iris.FUTURE.save_split_attrs = True
        except Exception:
            pass
        
        temp = Path(self._interim_dir) / (str(uuid.uuid4().hex) + '.nc')
        # issue with runaway disk usage
        iris.save(self._iris_post(iris.load(iris_temp_path)), temp)
        return temp
        
    @abstractmethod
    def _mass_filepath(self, querytime: str) -> str | tuple | dict[str, str | tuple[str, dict]] | list[str | tuple[str, dict]]:
        """
        Get filepath on mass for data needed by the user
                
        If value is tuple, or element in iterable is tuple, first element is path, second is select.
        If not tuple, select wont be used, and a fullpath will be needed
        """
        pass
    
    
    def _format_select(self, select_args: dict | None) -> str:
        """
        Get mass formatted select arguments
        
        Key, values of select args, if val is tuple, will be stringified 'correctly'.
        """        
        if select_args is None or len(select_args) == 0:
            return None
        
        if isinstance(select_args, str):
            return select_args
        
        for key, val in select_args.items():
            if isinstance(val, tuple):
                if len(val) == 1:
                    val = val[0]
                else:
                    val = f"({','.join(map(str, val))})"
            select_args[key] = val

        return f"begin\n" + '\n'.join(f"{key}={value}" for key,value in select_args.items()) +"\nend\n"
        
    
    def _retrieve_from_mass(self, filepath: str, select_values: str | None) -> xr.Dataset:
        """
        Retrieve data from mass
        
        Args:
            filepath (str):
                Massfile path to pull from
            select_values (str | None):
                Fully formatted select str for mass,
                If None, uses get which will require a full path to be given in `filepath`
            
        """
        import subprocess
        
        download_extension = str(filepath).split('.')[-1]
        iris_temp_path = Path(self._interim_dir) / (str(uuid.uuid4().hex) + f".{download_extension}")
        
        if select_values is not None:
            select_file = open(Path(self._interim_dir) / f'select_{str(uuid.uuid4().hex)}.txt', 'w')
            select_file.write(select_values)
            select_file.close()

            command = f'moo select -f {select_file.name} {filepath} {iris_temp_path}'
        else:
            command = f'moo get -f {filepath} {iris_temp_path}'
                    
        output = subprocess.run(command.split(' '), check = False, capture_output=True)
        try:
            output.check_returncode()
            if 'nc' in download_extension:
                return iris_temp_path
            return self._convert_to_netcdf(iris_temp_path)   
        
        except subprocess.CalledProcessError:
            pass
        raise IndexError(f"Retrieval from MASS raised an error. Ran {command!r}. \nstderr:\n{str(output.stderr)}")
                                                             

    
    def _generate(self, *args, **kwargs) -> xr.Dataset:
        """
        Generate data from MASS
        
        Uses `_retrieve_from_mass` to get paths and select file.
        
        Returns:
            (xr.Dataset):
                Generated dataset to be cached to disk.
        """
        mass_path: str | dict | list = self._mass_filepath(*args, **kwargs)
        
        retrieve = lambda path, select : self._retrieve_from_mass(path, self._format_select(select))
        open_xarray = lambda paths: xr.open_mfdataset(paths, preprocess = self.pre_save_transforms)
        split_select = lambda user_input: user_input if isinstance(user_input, tuple) else (user_input, None)
        
        if isinstance(mass_path, str):
            return self.pre_save_transforms(xr.open_dataset(retrieve(mass_path, None)))
        if isinstance(mass_path, list):
            return open_xarray([retrieve(*split_select(mass_path[i])) for i in len(mass_path)])
        if isinstance(mass_path, dict):
            return open_xarray([retrieve(*split_select(mass_path[k])) for k in list(mass_path.keys())])
        
        raise TypeError(f"Cannot parse filepaths: {mass_path!r}. Should be str, list or dictionary.")
                
    def __del__(self):
        del self._interim_dir