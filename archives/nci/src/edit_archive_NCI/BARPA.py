"""
Bureau of Meteorology Atmospheric Regional Projections for Australia (BARPA)
"""

from __future__ import annotations
from pathlib import Path

from edit.data import EDITDatetime, transform, DataNotFoundError
from edit.data.indexes import ArchiveIndex, decorators, VariableDefault
from edit.data.transform import Transform, TransformCollection
from edit.data.archive import register_archive

from edit_archive_NCI.utilities import check_project

"""
Structure order

order:
  - activity
  - product
  - domain
  - institution
  - gcm_model_name
  - experiment_name
  - ensemble_member
  - rcm_model_name
  - version
  - frequency
  - variable

"""

BARPA_DIR_STRUCTURE = "{activity}/{product}/{domain}/{institution}/{gcm_model_name}/{experiment_name}/{ensemble_member}/{rcm_model_name}/{version}/{frequency}/"

@register_archive('BARPA')
class BARPA(ArchiveIndex):
    """Index into Bureau of Meteorology Atmospheric Regional Projections for Australia"""

    @decorators.alias_arguments(variables=["variable"])
    @decorators.check_arguments(struc="edit_archive_NCI.structure.BARPA.struc")
    def __init__(
        self,
        variables: list[str] | str,
        gcm_model_name: str,
        frequency: str,
        experiment_name: str = VariableDefault,
        *,
        activity: str = VariableDefault,
        product: str = VariableDefault,
        domain: str = VariableDefault,
        institution: str = VariableDefault,
        ensemble_member: str = VariableDefault,
        rcm_model_name: str = VariableDefault,
        version: str = VariableDefault,
        transforms: Transform | TransformCollection = TransformCollection(),
    ):
        """
        Bureau of Meteorology Atmospheric Regional Projections for Australia (BARPA)

        High resolution Climate simulation in the Australia Region.

        All arguments with VariableDefault as default might not have to be given,
        if depending on the structure only one option is available, that will be picked.
        Otherwise an error will be raised.

        Args:
            variables (list[str] | str): 
                Variables to retireve
            gcm_model_name (str): 
                Global Coupled Model
            frequency (str): 
                Temporal Frequency
            experiment_name (str, optional): 
                Experiment Name. Defaults to VariableDefault.
            activity (str, optional): 
                Defaults to VariableDefault.
            product (str, optional): 
                Defaults to VariableDefault.
            domain (str, optional): 
                Defaults to VariableDefault.
            institution (str, optional): 
                Defaults to VariableDefault.
            ensemble_member (str, optional): 
                Defaults to VariableDefault.
            rcm_model_name (str, optional): 
                Defaults to VariableDefault.
            version (str, optional): 
                Defaults to VariableDefault.
            transforms (Transform | TransformCollection, optional): 
                Transforms to apply to the data. Defaults to TransformCollection().
        """        
        self.make_catalog()
        check_project(project_code='ia39')
        
        variables = [variables] if isinstance(variables, str) else variables
        self.variables = variables

        self.dir = Path(BARPA_DIR_STRUCTURE.format(**locals()))

        super().__init__(transforms)

    def filesystem(
        self,
        querytime: str | EDITDatetime,
    ) -> Path:
        BARPA_HOME = Path(self.ROOT_DIRECTORIES["BARPA"])

        discovered_paths = {}

        querytime = EDITDatetime(querytime).at_resolution("year")

        for variable in self.variables:
            dir_path = BARPA_HOME / self.dir / variable
            paths = list(dir_path.glob(f"*{querytime.year}01-{querytime.year}12*.nc"))
            if len(paths) == 0:
                raise DataNotFoundError(f"Could not find data at {dir_path} at time {querytime}")
            discovered_paths[variable] = paths[0]
        return discovered_paths
