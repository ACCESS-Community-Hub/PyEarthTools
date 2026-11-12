# Copyright Commonwealth of Australia, Bureau of Meteorology 2026.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pathlib

import xarray

import pyearthtools
import pyearthtools.data

from pyearthtools.data import Petdt
from pyearthtools.data.indexes import ArchiveIndex
from pyearthtools.data.transforms import Transform, TransformCollection
from pyearthtools.data.archive import register_archive


@register_archive("MOGlobal10km", sample_kwargs=dict(variables=["temperature_on_pressure_levels"]))
class MOGlobal10km(ArchiveIndex):
    """
    Data Accessor for accessing Met Office data through the AWS Sustainable Data initiative.
    """

    DATASET_NAME = 'Met Office Global Deterministic 10km forecast'
    MO_GLOBAL_AWS_DIR_TEMPLATE = "{root_dir}/{vt_str}"
    MO_GLOBAL_FNAME_TEMPLATE = "{vt_str}-PT0000H00M-{var_name}.nc"
    LICENSE_NAME='CC BY-SA'
    LICENSE_URL='https://creativecommons.org/licenses/by-sa/4.0/deed.en'

    def __init__(
        self,
        variables: list[str],
        license_ok: bool = False,
        *,
        transforms: Transform | TransformCollection | None = None,
    ):
        """
        Init function for Met Office Global 10km model accessor class.
        """
        
        if not license_ok:
            print('LICENSE WARNING: You have not accepted the license by passing in the license_OK=True '
                  'flag to the object constrcutor to indicate you accept the data license. This ' 
                  f'{MOGlobal10km.DATASET_NAME} data is under British Crown Copyright, licensed for use under '
                  f'the {MOGlobal10km.LICENSE_NAME} license. Please read the license terms at '
                  f'{MOGlobal10km.LICENSE_URL} to check your use of the data is in accordance with the data license. '
                  'Pass the license_ok=True argument to the object constructor to not see this message in future' )
        else:
            print(f'Data license accepted for {MOGlobal10km.DATASET_NAME}')

        self._variables = variables
        self._open_args = {
            "engine": "h5netcdf",  #
            "storage_options": {"anon": True},
        }
        self._mf_args = {"concat_dim": "time", "combine": "nested"}

        super_transforms = (
            TransformCollection(
                [
                    pyearthtools.data.transforms.variables.Trim(self._variables),
                ]
            )
            + transforms
        )

        # call the base class
        super().__init__(
            transforms=super_transforms,
        )
        self.record_initialisation()

    def filesystem(
        self,
        querytime: str | Petdt,
    ) -> pathlib.Path | dict[str, str | pathlib.Path]:

        querytime = Petdt(querytime)
        paths = []
        vt_template = "{dt.year:04d}{dt.month:02d}{dt.day:02d}T{dt.hour:02d}{dt.minute:02d}Z"
        _fname_template = "{vt_str}-PT0000H00M-{var_name}.nc"
        root_dir = self.ROOT_DIRECTORIES["MOGlobal10km_AWS"]

        for var_name in self._variables:
            try:
                current_fname = MOGlobal10km.MO_GLOBAL_FNAME_TEMPLATE.format(
                    vt_str=vt_template.format(dt=querytime), var_name=var_name
                )
                current_dir = MOGlobal10km.MO_GLOBAL_AWS_DIR_TEMPLATE.format(
                    root_dir=root_dir, vt_str=vt_template.format(dt=querytime)
                )
                current_path = f"{current_dir}/{current_fname}"
                paths += [current_path]
            except KeyError:
                print(f"No data for var {var_name}")

        print(paths)
        return paths

    def load(self, *args, **kwargs):
        ds = xarray.merge([xarray.open_dataset(path1, **self._open_args) for path1 in args[0]])
        return ds

        return ds

    def _desc_(self):
        return {
            "singleline": "Met Office 10km Global Determinsitic Forecast Analysis ",
            "range": "June 2026",
            "Documentation": "https://registry.opendata.aws/met-office-global-deterministic/",
        }
