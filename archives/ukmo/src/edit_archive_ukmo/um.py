"""
UnifiedModel Indexer

- Retrieve data from MASS
- Either forecast or 0 step
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal
import math

from edit.data import EDITDatetime, TimeDelta, transform
from edit.data.exceptions import DataNotFoundError
from edit.data.indexes import ArchiveIndex, ForecastIndex, decorators
from edit.data.transform import Transform, TransformCollection
from edit.data.archive import register_archive

from edit.utils.decorators import classproperty

from edit_archive_ukmo.mass import MASS
from edit_archive_ukmo._stash import S_SPEC_STASH, M_SPEC_STASH
from edit_archive_ukmo._processed import PROCESSED_VARIABLES

STASH_CONVERSION = {**S_SPEC_STASH, **M_SPEC_STASH}

VALID_LEVELS = [None, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50, 70, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 925, 950, 975, 1000]


def convert_variable_names(iris_cubelist):
    inverted_stash = {value: key for key,value in STASH_CONVERSION.items()}

    for cube in iris_cubelist: # iterate over all cubes
        # stashcode is an iris object 
        stashcode = cube.metadata.attributes['STASH']
        original_name = cube.name()
        # lbuser3 gets the digits after the last 0, equlivilent to
        # the moo select argument (in _stash.py)
        if stashcode.lbuser3() not in inverted_stash:
            continue
        new_name = inverted_stash[stashcode.lbuser3()]
        cube.rename(new_name)
        cube.attributes['original_name'] = original_name
    return iris_cubelist


class BaseUMModel(MASS):

    
    def _iris_post(self, iris_cubelist):
        ## convert iris data based on stash codes to names as given in STASH_CONVERSION
        return convert_variable_names(iris_cubelist)
    
    def _select_args(self, querytime: str, forecast_leadtime: int = 0, spec: str = 'm') -> dict[str, str]:
        """
        Create select arguments for MASS call        
        """
        cycle_time = EDITDatetime(querytime)
        cycle_time += TimeDelta((forecast_leadtime, 'hours'))
        
        cycle_t1 = cycle_time.strftime('%Y/%m/%d %H:%M')
        stashs = tuple(str(STASH_CONVERSION[x]) for x in self.variables)
            
        file_leadtime = math.ceil(forecast_leadtime / 3)*3

        select_dict = dict(
            pp_file = f"'prod{spec}_op_gl-{'mn' if spec == 's' else 'up'}_{EDITDatetime(querytime).strftime('%Y%m%d_%H')}_{file_leadtime:03d}.pp'",
            stash=stashs,
            T1=f"[{{{cycle_t1}}}..{{{cycle_t1}}}]",
        )
        if self.levels is not None:
            select_dict['lblev'] = tuple(self.levels)
        return select_dict
    
    
@register_archive('UM')
class UnifiedModel(BaseUMModel, ArchiveIndex):
    """Unified Model from MASS"""
   
    @classproperty
    def analysis(cls) -> 'UnifiedModel':
        """Property to allow Forecast class init"""
        return UnifiedModel
    
    @classproperty
    def forecast(cls) -> 'UnifiedModelForecast':
        """Property to allow Forecast class init"""
        return UnifiedModelForecast
    
    @classproperty
    def processed(cls) -> 'UnifiedModelForecast':
        """Property to get processed index"""
        return UnifiedModelProcessed
    
    @classproperty
    def processed_forecast(cls) -> 'UnifiedModelProcessedForecast':
        """"""
        return UnifiedModelProcessedForecast
    
    @property
    def _desc_(self):
        # Anything you want appearing at the top of the index
        return {
            "singleline": "Unified Model",
            "range": "?-current",
            "Documentation": "",
        }
      
    @decorators.alias_arguments(
        # Aliases for arguments in the init
        variables=["variable"],
        levels=["level"],
    )
    @decorators.check_arguments(
        # Checking of arguments, can be list, or module path to txt file
        variables=list(STASH_CONVERSION.keys()),
        levels=VALID_LEVELS,
    )
    def __init__(
        self,
        variables: list[str] | str,
        *,
        levels: int | list[int] | None = None,
        transforms: Transform | TransformCollection = TransformCollection(),
        **kwargs,
    ):
        """
        Setup UM Indexer

        Args:
            variables (list[str] | str):
                Data variables to retrieve
            levels (int | list[int] | None):
                Level values
            transforms (Transform | TransformCollection, optional): 
                Base Transforms to apply.
                Defaults to TransformCollection().
        """
        self.make_catalog() # Create the repr

        variables = [variables] if isinstance(variables, str) else variables
        
        # Ensure the pattern is made correctly
        kwargs['pattern_kwargs'] = kwargs.pop('pattern_kwargs', {}) 
        kwargs['pattern_kwargs']['variables'] = variables
            
        self.variables = variables
        
        # Setup base transforms to be applied anytime data is retrieved
        base_transforms = TransformCollection()
        base_transforms += transform.variables.variable_trim(variables)
        # Remove extra coordinates
        base_transforms += transform.coordinates.drop(['height', 'forecast_period', 'forecast_reference_time'], ignore_missing=True)

        # Transforms to be applied before saving the data
        pre_save_transforms = TransformCollection()
        pre_save_transforms += transform.dimensions.expand('time')
        pre_save_transforms += transform.variables.drop('latitude_longitude')
        pre_save_transforms += kwargs.pop('pre_save_transforms', {})

        # Select on levels
        levels = [levels] if isinstance(levels, int) else levels
        self.levels = levels

        if levels is not None:
            base_transforms += transform.coordinates.select(
                {coord: levels for coord in ["level"]}, ignore_missing=True
            )

        super().__init__(
            transforms=base_transforms + transforms,
            data_interval=(6, 'hour'),
            pre_save_transforms = pre_save_transforms,
            **kwargs,
        )
        
        

    def _mass_filepath(
        self,
        querytime: str | EDITDatetime,
    ) ->  dict[str, tuple[str, dict]]:
        """
        Get the file paths on MASS to retrieve from, and the associated select args
        """
        UM_HOME = self.ROOT_DIRECTORIES["UM"]

        paths = {}
        
        querytime = EDITDatetime(querytime)
        
        for variable in self.variables:
            spec = 'm' if variable in M_SPEC_STASH else 's'
            var_path = Path(UM_HOME.format(spec = spec)) / str(querytime.year) 
            paths[spec] = (str(var_path) + '.pp', self._select_args(querytime, spec = spec))
            
        return paths

@register_archive('UMForecast')
class UnifiedModelForecast(BaseUMModel, ForecastIndex):
    """Unified Model Forecast from MASS
    
    Allows for retrieval of forecast data
    
    """
    @classproperty
    def processed(cls) -> 'UnifiedModelProcessedForecast':
        """Property to get processed index"""
        return UnifiedModelProcessedForecast
    
    @property
    def _desc_(self):
        # Anything you want appearing at the top of the index
        return {
            "singleline": "Unified Model Forecast",
            "range": "?-current",
            "Documentation": "",
        }

    @decorators.alias_arguments(
        # Aliases for arguments in the init
        variables=["variable"],
        levels=["level"],
    )
    @decorators.check_arguments(
        # Checking of arguments, can be list, or module path to txt file
        variables=list(STASH_CONVERSION.keys()),
        levels=VALID_LEVELS,
    )
    def __init__(
        self,
        variables: list[str] | str,
        forecast_leadtime: int | tuple[int, int],
        *,
        levels: int | list[int] | None = None,
        transforms: Transform | TransformCollection = TransformCollection(),
        **kwargs,
    ):
        """
        Setup UM Forecast Indexer

        Args:
            variables (list[str] | str):
                Data variables to retrieve
            forecast_leadtime (int | tuple[int, int]):
                Leadtime to get, if int, is final time.
                If tuple, is start and end of range
            levels (int | list[int] | None):
                Level values
            transforms (Transform | TransformCollection, optional): 
                Base Transforms to apply.
                Defaults to TransformCollection().
        """
        self.make_catalog() # Create the repr

        variables = [variables] if isinstance(variables, str) else variables
        
        DEFAULT_INTERVAL = 3
        
        if isinstance(forecast_leadtime, int):
            forecast_leadtime = (0, forecast_leadtime, DEFAULT_INTERVAL)
        elif isinstance(forecast_leadtime, (list, tuple)):
            if len(forecast_leadtime) == 1:
                forecast_leadtime = (0, forecast_leadtime[0], DEFAULT_INTERVAL)
            elif len(forecast_leadtime) == 2:
                forecast_leadtime = (*forecast_leadtime, DEFAULT_INTERVAL)
            elif len(forecast_leadtime) == 3:
                forecast_leadtime = (*forecast_leadtime,)
            else:
                raise ValueError(f"Cannot parse `forecast_leadtime` of {forecast_leadtime!r}")
        else:
            raise TypeError(f"Cannot parse `forecast_leadtime` of {forecast_leadtime!r}")
            
        self.forecast_leadtime = forecast_leadtime
        
        # Ensure the pattern is made correctly
        kwargs['pattern_kwargs'] = kwargs.pop('pattern_kwargs', {}) 
        kwargs['pattern_kwargs']['variables'] = variables
            
        self.variables = variables
        
        # Setup base transforms to be applied anytime data is retrieved
        base_transforms = TransformCollection()
        base_transforms += transform.variables.variable_trim(variables)
        # Remove extra coordinates
        base_transforms += transform.coordinates.drop(['height', 'forecast_reference_time'], ignore_missing=True)

        # Transforms to be applied before saving the data
        pre_save_transforms = TransformCollection()
        pre_save_transforms += transform.dimensions.expand(['time', 'forecast_period'])
        pre_save_transforms += transform.variables.drop('latitude_longitude')

        # Select on levels
        levels = [levels] if isinstance(levels, int) else levels
        self.levels = levels

        if levels is not None:
            base_transforms += transform.coordinates.select(
                {coord: levels for coord in ["level"]}, ignore_missing=True
            )

        super().__init__(
            transforms=base_transforms + transforms,
            data_interval=(6, 'hour'),
            pre_save_transforms = pre_save_transforms,
            **kwargs,
        )    
  
    def _mass_filepath(
        self,
        querytime: str | EDITDatetime,
    ) ->  dict[str, tuple[str, dict]]:
        """
        Get the file paths on MASS to retrieve from, and the associated select args
        
        Will be one per spec and leadtime
        """
        UM_HOME = self.ROOT_DIRECTORIES["UM"]

        paths = {}
        
        querytime = EDITDatetime(querytime)
        
        leadtimes = tuple(range(*self.forecast_leadtime))

        for variable in self.variables:
            for leadtime in leadtimes: 
                spec = 'm' if variable in M_SPEC_STASH else 's'
                var_path = Path(UM_HOME.format(spec = spec)) / str(querytime.year)
                paths[f"{spec}_{leadtime}"] = (str(var_path) + '.pp', self._select_args(querytime, leadtime, spec = spec))
        return paths

    
class UnifiedModelProcessed(BaseUMModel, ArchiveIndex):
    """Unified Model from MASS"""
    
    @classproperty
    def forecast(cls) -> 'UnifiedModelProcessedForecast':
        """Property to allow Forecast class init"""
        return UnifiedModelProcessedForecast
    
    @property
    def _desc_(self):
        # Anything you want appearing at the top of the index
        return {
            "singleline": "Unified Model",
            "range": "?-current",
            "Documentation": "",
        }

    @decorators.alias_arguments(
        # Aliases for arguments in the init
        variables=["variable"],
        levels=["level"],
    )     
      
    @decorators.alias_arguments(
        # Aliases for arguments in the init
        variables=["variable"],
        levels=["level"],
    )
    @decorators.check_arguments(
        # Checking of arguments, can be list, or module path to txt file
        variables=list(PROCESSED_VARIABLES),
        levels=VALID_LEVELS,
    )
    def __init__(
        self,
        variables: list[str] | str,
        *,
        levels: int | list[int] | None = None,
        transforms: Transform | TransformCollection = TransformCollection(),
        **kwargs,
    ):
        """
        Setup UM Processed Indexer

        Args:
            variables (list[str] | str):
                Data variables to retrieve
            levels (int | list[int] | None):
                Level values
            transforms (Transform | TransformCollection, optional): 
                Base Transforms to apply.
                Defaults to TransformCollection().
        """
        self.make_catalog() # Create the repr

        variables = [variables] if isinstance(variables, str) else variables
        
        # Ensure the pattern is made correctly
        kwargs['pattern_kwargs'] = kwargs.pop('pattern_kwargs', {}) 
        kwargs['pattern_kwargs']['variables'] = variables
            
        self.variables = variables
        
        # Setup base transforms to be applied anytime data is retrieved
        base_transforms = TransformCollection()
        base_transforms += transform.variables.variable_trim(variables)
        # Remove extra coordinates
        base_transforms += transform.coordinates.drop(['bnds', 'height', 'forecast_period', 'forecast_reference_time'], ignore_missing=True)

        # Transforms to be applied before saving the data
        pre_save_transforms = TransformCollection()
        pre_save_transforms += transform.dimensions.expand('time')
        pre_save_transforms += transform.variables.drop(['latitude_longitude', 'longitude_bnds', 'latitude_bnds'])
        pre_save_transforms += kwargs.pop('pre_save_transforms', {})

        # Select on levels
        levels = [levels] if isinstance(levels, int) else levels
        self.levels = levels

        if levels is not None:
            base_transforms += transform.coordinates.select(
                {coord: levels for coord in ["level"]}, ignore_missing=True
            )

        super().__init__(
            transforms=base_transforms + transforms,
            data_interval=(6, 'hour'),
            pre_save_transforms = pre_save_transforms,
            **kwargs,
        )
        
        

    def _mass_filepath(
        self,
        querytime: str | EDITDatetime,
    ) ->  dict[str, tuple[str, dict]]:
        """
        Get the file paths on MASS to retrieve from, and the associated select args
        """
        UM_HOME = self.ROOT_DIRECTORIES["UMProcessed"]

        paths = {}
        'moose:/opfc/atm/global/lev1/20240402T1200Z.nc.file/20240408T1800Z-PT0150H00M-precipitation_accumulation-PT06H.nc'
        querytime = EDITDatetime(querytime)
        for variable in self.variables:
            time_str = querytime.strftime('%Y%m%dT%H00Z')
            var_path = Path(UM_HOME) / f"{time_str}.nc.file" / f"{time_str}-PT0000H00M-{variable}"
            paths[variable] = str(var_path) + '.nc'
            
        return paths
    
class UnifiedModelProcessedForecast(BaseUMModel, ForecastIndex):
    """Unified Model from MASS"""

    
    @property
    def _desc_(self):
        # Anything you want appearing at the top of the index
        return {
            "singleline": "Unified Model",
            "range": "?-current",
            "Documentation": "",
        }

    @decorators.alias_arguments(
        # Aliases for arguments in the init
        variables=["variable"],
        levels=["level"],
    )     
      
    @decorators.alias_arguments(
        # Aliases for arguments in the init
        variables=["variable"],
        levels=["level"],
    )
    @decorators.check_arguments(
        # Checking of arguments, can be list, or module path to txt file
        variables=list(PROCESSED_VARIABLES),
        levels=VALID_LEVELS,
    )
    def __init__(
        self,
        variables: list[str] | str,
        forecast_leadtime: int | tuple[int, int],
        *,
        levels: int | list[int] | None = None,
        transforms: Transform | TransformCollection = TransformCollection(),
        **kwargs,
    ):
        """
        Setup UM Forecast Indexer

        Args:
            variables (list[str] | str):
                Data variables to retrieve
            forecast_leadtime (int | tuple[int, int]):
                Leadtime to get, if int, is final time.
                If tuple, is start and end of range
            levels (int | list[int] | None):
                Level values
            transforms (Transform | TransformCollection, optional): 
                Base Transforms to apply.
                Defaults to TransformCollection().
        """
        self.make_catalog() # Create the repr

        variables = [variables] if isinstance(variables, str) else variables
        
        DEFAULT_INTERVAL = 3
        
        if isinstance(forecast_leadtime, int):
            forecast_leadtime = (0, forecast_leadtime, DEFAULT_INTERVAL)
        elif isinstance(forecast_leadtime, (list, tuple)):
            if len(forecast_leadtime) == 1:
                forecast_leadtime = (0, forecast_leadtime[0], DEFAULT_INTERVAL)
            elif len(forecast_leadtime) == 2:
                forecast_leadtime = (*forecast_leadtime, DEFAULT_INTERVAL)
            elif len(forecast_leadtime) == 3:
                forecast_leadtime = (*forecast_leadtime,)
            else:
                raise ValueError(f"Cannot parse `forecast_leadtime` of {forecast_leadtime!r}")
        else:
            raise TypeError(f"Cannot parse `forecast_leadtime` of {forecast_leadtime!r}")
            
        self.forecast_leadtime = forecast_leadtime
        
        # Ensure the pattern is made correctly
        kwargs['pattern_kwargs'] = kwargs.pop('pattern_kwargs', {}) 
        kwargs['pattern_kwargs']['variables'] = variables
            
        self.variables = variables
        
        # Setup base transforms to be applied anytime data is retrieved
        base_transforms = TransformCollection()
        base_transforms += transform.variables.variable_trim(variables)
        # Remove extra coordinates
        base_transforms += transform.coordinates.drop(['height', 'forecast_reference_time'], ignore_missing=True)

        # Transforms to be applied before saving the data
        pre_save_transforms = TransformCollection()
        pre_save_transforms += transform.dimensions.expand(['time', 'forecast_period'])
        pre_save_transforms += transform.variables.drop('latitude_longitude')

        # Select on levels
        levels = [levels] if isinstance(levels, int) else levels
        self.levels = levels

        if levels is not None:
            base_transforms += transform.coordinates.select(
                {coord: levels for coord in ["level"]}, ignore_missing=True
            )

        super().__init__(
            transforms=base_transforms + transforms,
            data_interval=(6, 'hour'),
            pre_save_transforms = pre_save_transforms,
            **kwargs,
        )   
        
        

    def _mass_filepath(
        self,
        querytime: str | EDITDatetime,
    ) ->  dict[str, tuple[str, dict]]:
        """
        Get the file paths on MASS to retrieve from, and the associated select args
        """
        UM_HOME = self.ROOT_DIRECTORIES["UMProcessed"]

        paths = {}
        leadtimes = tuple(range(*self.forecast_leadtime))

        querytime = EDITDatetime(querytime)
        for variable in self.variables:
            for leadtime in leadtimes: 
                time_str = querytime.strftime('%Y%m%dT%H00Z')
                leadtime_str = (querytime + TimeDelta((leadtime, 'hours'))).strftime('%Y%m%dT%H00Z')
                
                var_path = Path(UM_HOME) / f"{time_str}.nc.file" / f"{leadtime_str}-PT0{leadtime:03d}H00M-{variable}"
            paths[variable] = str(var_path) + '.nc'
            
        return paths