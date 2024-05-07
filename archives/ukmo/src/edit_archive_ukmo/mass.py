"""
MASS Access
"""

from __future__ import annotations
from abc import abstractmethod, ABCMeta
import warnings

from functools import cached_property

import xarray as xr
import tempfile
from pathlib import Path
import uuid

from edit.data.indexes import CachingIndex
from edit.data import IndexWarning, DataNotFoundError


class MASS(CachingIndex, metaclass = ABCMeta):
    """
    Core `MASS` to be implemented by any data class accessing MASS.
    Subclasses from the `CachingIndex` to provide automated generation, and saving if a `cache` is specified.

    By default, all data is cached to a temporary directory.

    A child class must implement `._mass_filepath`
    """

    def __init__(self, *args, **kwargs):
        kwargs["cache"] = kwargs.pop("cache", "temp")
        kwargs["pattern"] = kwargs.pop("pattern", "TemporalExpandedDateVariable")
        
        super().__init__(*args, **kwargs)
        if not hasattr(self, "catalog"):
            self.make_catalog()
        self._save_catalog(self.catalog, 'index')
        
    @cached_property
    def _interim_dir(self):
        if self.cache is None:
            return tempfile.TemporaryDirectory()
        return self.cache 
        return tempfile.TemporaryDirectory(dir = str(Path(self.cache))) # to be moved back when done testing

    def _iris_post(self, iris_data):
        """
        Processing required after loading the iris data
        """
        ## Provide an implementation in the child class
        return iris_data
    
    def _convert_to_xarray(self, iris_temp_path) -> xr.Dataset:
        """
        Convert an iris data file on disk to an xarray object
        """
        import iris
        
        temp = Path(self._interim_dir) / (str(uuid.uuid4().hex) + '.nc') #tempfile.TemporaryFile(suffix = '.nc', dir = self._interim_dir.name).name
        # issue with runaway disk usage
        iris.save(self._iris_post(iris.load(iris_temp_path)), temp)
        return xr.open_dataset(temp)
        
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
        
        Key, values of select args, if val is tuple, will be stringified.
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
                    val = f"({','.join(val)})"
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

        iris_temp_path = Path(self._interim_dir) / (str(uuid.uuid4().hex) + '.pp')
        
        if select_values is not None:
            select_file = open(Path(self._interim_dir) / f'select_{str(uuid.uuid4().hex)}.txt', 'w')
            select_file.write(select_values)
            select_file.close()

            command = f'moo select -f {select_file.name} {filepath} {iris_temp_path}'
        else:
            command = f'moo get -f {filepath} {iris_temp_path}'
                    
        subprocess.run(command.split(' '), check = True, capture_output=True)
        return self._convert_to_xarray(iris_temp_path)   
    
    def _generate(self, *args, **kwargs):
        """
        Generate data from mass
        """
        mass_path: str | dict | list = self._mass_filepath(*args, **kwargs)
        
        retrieve = lambda path, select : self._retrieve_from_mass(path, self._format_select(select))
        split_select = lambda user_input: user_input if isinstance(user_input, tuple) else (user_input, None)
        
        if isinstance(mass_path, str):
            return retrieve(mass_path, None)
        elif isinstance(mass_path, list):
            return xr.merge([retrieve(*split_select(mass_path[i])) for i in range(len(mass_path))])
        elif isinstance(mass_path, dict):
            return xr.merge([retrieve(*split_select(mass_path[k])) for k in mass_path.keys()])
        else:
            raise TypeError('No')
            
        return xr_obj
    
    def __del__(self):
        pass
        # self._interim_dir.cleanup()