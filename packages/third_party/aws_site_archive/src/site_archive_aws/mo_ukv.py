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


@register_archive("MOUKV", sample_kwargs=dict(variables=["temperature_on_pressure_levels"]))
class MOUKV(ArchiveIndex):
    """
    Data Accessor for accessing Met Office UKV model data through the AWS Sustainable Data initiative.
    Info page: https://registry.opendata.aws/met-office-uk-deterministic/
    """

    DATASET_NAME = 'Met Office UK Deterministic (UKV) 2km forecast'
    MO_UKV_AWS_DIR_TEMPLATE = "{root_dir}/{vt_str}"
    MO_UKV_FNAME_TEMPLATE = "{vt_str}-PT0000H00M-{var_name}.nc"
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
        Init function for Met Office UKV model data accessor.
        """

        if not license_ok:
            print('LICENSE WARNING: You have not accepted the license by passing in the license_OK=True '
                  f'flag to the object constrcutor to indicate you accept the data license. This {MOUKV.DATASET_NAME} '
                  'data is under British Crown Copyright, licensed for use under '
                  f'the {MOUKV.LICENSE_NAME} license. Please read the license terms at {MOUKV.LICENSE_URL} to check '
                  'your use of the data is in accordance with the data license. Pass the license_ok=True argument '
                  'to the object constructor to not see this message in future' )
        else:
            print(f'Data license accepted for {MOUKV.DATASET_NAME}')
                  
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

        if not self._variables:
            raise pyearthtools.data.DataNotFoundError("No variables specified to load.")

        querytime = Petdt(querytime)
        paths = []

        vt_template = "{dt.year:04d}{dt.month:02d}{dt.day:02d}T{dt.hour:02d}{dt.minute:02d}Z"
        root_dir = self.ROOT_DIRECTORIES["MOUKV_AWS"]
        for var_name in self._variables:
            try:
                current_fname = MOUKV.MO_UKV_FNAME_TEMPLATE.format(
                    vt_str=vt_template.format(dt=querytime), var_name=var_name
                )
                current_dir = MOUKV.MO_UKV_AWS_DIR_TEMPLATE.format(
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

    def _desc_(self):
        return {
            "singleline": "Met Office UKV Forecast Analysis ",
            "range": "June 2026",
            "Documentation": "https://registry.opendata.aws/met-office-uk-deterministic/",
        }
