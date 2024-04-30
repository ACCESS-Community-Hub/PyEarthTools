"""
Ocean Modelling and Analysis Prediction System
"""

from __future__ import annotations

import datetime
from glob import glob
from pathlib import Path
from typing import Any, Literal


from edit.data import EDITDatetime, transform

from edit.data.exceptions import DataNotFoundError
from edit.data.indexes import ArchiveIndex, decorators
from edit.data.transform import Transform, TransformCollection
from edit.data.archive import register_archive

from edit_archive_NCI.utilities import check_project


OceanMaps_TYPES = ["analysis", "forecast"]
OceanMaps_SUBVAR = ["ocean_an00", "ocean_an01", "ocean_an02", "ocean_an_ensemble"]


OceanMaps_REGEX = "{ROOT_DIR}/version_{version}/{datatype}/{variable}/"
OceanMaps_SUBVAR_FORMAT = "{sub_var}_{date}12_{variable}*"

@register_archive('OceanMaps')
class OceanMaps(ArchiveIndex):
    """Index into Ocean Modelling and Analysis Prediction System"""

    @property
    def _desc_(self):
        return {
            "singleline": "Ocean Modelling and Analysis Prediction System",
            "Resolution": "1 day",
        }

    @decorators.alias_arguments(variables=["variable"])
    @decorators.check_arguments(
        datatype=OceanMaps_TYPES,
        sub_var=OceanMaps_SUBVAR,
        variables="edit_archive_NCI.variables.OceanMaps.{datatype}.valid",
    )
    def __init__(
        self,
        variables: list[str] | str,
        datatype: Literal[OceanMaps_TYPES],
        sub_var: Literal[OceanMaps_SUBVAR],
        *,
        depth_value: Any = None,
        version: str = "3.4",
        transforms: Transform | TransformCollection = TransformCollection(),
    ):
        """
        Setup BRAN Indexer

        Args:
            variables (list[str] | str):
                Data variables to retrieve
            datatype (Literal[OceanMaps_TYPES]):
                Data type to retrieve, must be ["analysis", "forecast"]
            sub_var (Literal[OceanMaps_SUBVAR]):
                OceanMaps identifier, must be "ocean_an00", "ocean_an01", "ocean_an02", "ocean_an_ensemble"]
            depth_value (Any, optional):
                Depth value to select if data contains levels. Defaults to None.
            version (str, optional):
                OceanMaps Version. Defaults to "3.4".
            transforms (Transform | TransformCollection, optional):
                Base Transforms to apply. Defaults to TransformCollection().
        """
        self.make_catalog()
        check_project(project_code='rr6')

        variables = [variables] if isinstance(variables, str) else variables
        self.variables = variables

        self.datatype = datatype
        self.sub_var = sub_var

        self.version = version

        base_transform = transform.variables.variable_trim(variables)

        self.depth_value = depth_value
        if depth_value is not None:
            base_transform += transform.coordinates.select(
                {coord: depth_value for coord in ["st_ocean"]}, ignore_missing=True
            )
        super().__init__(transforms=base_transform + transforms, data_interval=(1, "D"))

    def filesystem(
        self,
        basetime: str | datetime.datetime | EDITDatetime,
    ) -> Path:
        OceanMaps_HOME = self.ROOT_DIRECTORIES["OceanMaps"]

        paths = {}

        basetime = EDITDatetime(str(basetime))
        basetime.set_components(["hour", "minute", "second"], False)
        basetime -= datetime.timedelta(days=1)

        for variable in self.variables:
            basepath = OceanMaps_REGEX.format(
                ROOT_DIR=OceanMaps_HOME,
                version=self.version,
                datatype=self.datatype,
                variable=variable,
            )
            var_path = Path(basepath) / OceanMaps_SUBVAR_FORMAT.format(
                sub_var=self.sub_var,
                date=str(basetime).replace("-", ""),
                variable=variable,
            )

            for file in glob(str(var_path)):
                file = Path(file)
                if file.exists():
                    paths[variable] = file
                    break
            else:
                raise DataNotFoundError(
                    f"Unable to find data for: basetime: {basetime}, variables: {variable} at {var_path}"
                )

        return paths
