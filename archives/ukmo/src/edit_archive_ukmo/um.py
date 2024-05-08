"""
UnifiedModel Indexer

- Retrieve data from MASS
- Either forecast or 0 step
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal


from edit.data import EDITDatetime, TimeDelta, transform
from edit.data.exceptions import DataNotFoundError
from edit.data.indexes import ArchiveIndex, ForecastIndex, decorators
from edit.data.transform import Transform, TransformCollection
from edit.data.archive import register_archive

from edit_archive_ukmo.mass import MASS
from edit_archive_ukmo._stash import STASH_CONVERSION

M_SPEC_VARIABLES = ['specific_humidity']
VALID_LEVELS = [None, 50,100,150,200,250,300,400,500,600,700,850,925,1000]


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

@register_archive('UM')
class UnifiedModel(MASS, ArchiveIndex):
    """Unified Model from MASS"""

    @property
    def _desc_(self):
        # Anything you want appearing at the top of the index
        return {
            "singleline": "Unified Model",
            "range": "?-current",
            "Documentation": "",
        }
    
    @staticmethod
    def forecast(*args, **kwargs) -> 'UnifiedModelForecast':
        """Property to allow Forecast class init"""
        return UnifiedModelForecast(*args, **kwargs)

    @decorators.alias_arguments(
        # Aliases for arguments in the init
        variables=["variable"],
        levels=["level"],
    )
    def __new__(cls, *args, **kwargs):
        if 'forecast_leadtime' in kwargs:
            cls = UnifiedModelForecast
        return super().__new__(cls)
        
      
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
        
    def _iris_post(self, iris_cubelist):
        ## convert iris data based on stash codes to names as given in STASH_CONVERSION
        return convert_variable_names(iris_cubelist)
    
    
    def _select_args(self, querytime: str, spec: str = 'm') -> dict[str, str]:
        """
        Create select arguments for MASS call        
        """
        cycle_t1 = EDITDatetime(querytime).strftime('%Y/%m/%d %H:%M')
        stashs = tuple(str(STASH_CONVERSION[x]) for x in self.variables)
            
        select_dict = dict(
            pp_file = f"'prods_op_gl-mn_{EDITDatetime(querytime).strftime('%Y%m%d_%H')}_000.pp'",
            stash=stashs,
            T1=f"[{{{cycle_t1}}}..{{{cycle_t1}}}]",
        )
        if self.levels is not None:
            select_dict['lblev'] = tuple(self.levels)
        return select_dict

        

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
            spec = 'm' if variable in M_SPEC_VARIABLES else 's'
            var_path = Path(UM_HOME.format(spec = spec)) / str(querytime.year)  #/ OTHER_THINGS
            paths[spec] = (str(var_path) + '.pp', self._select_args(querytime, spec))
            
        return paths

@register_archive('UMForecast')
class UnifiedModelForecast(MASS, ForecastIndex):
    """Unified Model Forecast from MASS
    
    Allows for retrieval of forecast data
    
    """

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
        self.forecast_leadtime = forecast_leadtime
        
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
        
    def _iris_post(self, iris_cubelist):
        ## convert iris data based on stash codes to names as given in STASH_CONVERSION
        return convert_variable_names(iris_cubelist)
    
    
    def _select_args(self, querytime: str, forecast_leadtime: int, spec: str = 'm') -> dict[str, str]:
        """
        Create select arguments for MASS call        
        """
        cycle_time = EDITDatetime(querytime)
        cycle_time += TimeDelta((forecast_leadtime, 'hours'))
        
        cycle_t1 = cycle_time.strftime('%Y/%m/%d %H:%M')
        stashs = tuple(str(STASH_CONVERSION[x]) for x in self.variables)
            
        select_dict = dict(
            pp_file = f"'prods_op_gl-mn_{EDITDatetime(querytime).strftime('%Y%m%d_%H')}_{forecast_leadtime:03d}.pp'",
            stash=stashs,
            T1=f"[{{{cycle_t1}}}..{{{cycle_t1}}}]",
        )
        if self.levels is not None:
            select_dict['lblev'] = tuple(self.levels)
        return select_dict

        

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
        
        leadtimes = (0,)
        if isinstance(self.forecast_leadtime, int):
            leadtimes = tuple(range(0, self.forecast_leadtime, 3))
        elif isinstance(self.forecast_leadtime, tuple):
            leadtimes = tuple(range(self.forecast_leadtime[0], self.forecast_leadtime[1], 3))
        else:
            raise TypeError(f"Cannot parse forecast_leadtime of {forecast_leadtime!r}")
            
        for variable in self.variables:
            for leadtime in leadtimes:
                spec = 'm' if variable in M_SPEC_VARIABLES else 's'
                var_path = Path(UM_HOME.format(spec = spec)) / str(querytime.year)  #/ OTHER_THINGS
                paths[f"{spec}_{leadtime}"] = (str(var_path) + '.pp', self._select_args(querytime, leadtime, spec))

        return paths
