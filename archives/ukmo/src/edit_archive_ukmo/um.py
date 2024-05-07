"""
UnifiedModel
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal


from edit.data import EDITDatetime, transform
from edit.data.exceptions import DataNotFoundError
from edit.data.indexes import ArchiveIndex, decorators
from edit.data.transform import Transform, TransformCollection
from edit.data.archive import register_archive

from edit_archive_ukmo.mass import MASS
from edit_archive_ukmo._stash import STASH_CONVERSION

M_SPEC_VARIABLES = ['specific_humidity']
VALID_LEVELS = [None, 50,100,150,200,250,300,400,500,600,700,850,925,1000]

@register_archive('UM')
class UnifiedModel(MASS):
    """Unified Model"""

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

        # Select on levels
        levels = [levels] if isinstance(levels, int) else levels
        self.levels = levels

        if levels is not None:
            base_transforms += transform.coordinates.select(
                {coord: levels for coord in ["level"]}, ignore_missing=True
            )

        super().__init__(
            transforms=base_transforms + transforms,
            data_interval=(1, 'hour'),
            **kwargs,
        )
        
    def _iris_post(self, iris_cubelist):
        ## convert iris data based on stash codes to names as given in STASH_CONVERSION
        inverted_stash = {value: key for key,value in STASH_CONVERSION.items()}
        
        for cube in iris_cubelist: # iterate over all cubes
            # stashcode is an iris object 
            stashcode = cube.metadata.attributes['STASH']
            original_name = cube.name()
            # lbuser3 gets the digits after the last 0, equlivilent to
            # the moo select argument (in _stash.py)
            new_name = inverted_stash[stashcode.lbuser3()]
            cube.rename(new_name)
            print(cube.attributes)
            cube.attributes['orginal_name'] = original_name
            print(cube.attributes)
        return iris_cubelist
    
    
    def _select_args(self, querytime: str, spec: str = 'm') -> dict[str, str]:
        #TODO add level
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
        Get the file paths on mass to retrieve from
        
        As `_select_args` is specified, does not need to be full path.
        
                
        """
        UM_HOME = self.ROOT_DIRECTORIES["UM"]

        paths = {}
        
        querytime = EDITDatetime(querytime)
        
        for variable in self.variables:
            spec = 'm' if variable in M_SPEC_VARIABLES else 's'
            var_path = Path(UM_HOME.format(spec = spec)) / str(querytime.year)  #/ OTHER_THINGS
            paths[spec] = (str(var_path) + '.pp', self._select_args(querytime, spec))
            
        return paths
