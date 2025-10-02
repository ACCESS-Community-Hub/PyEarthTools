# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
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

"""
Met Office UKV (subset)
"""

from __future__ import annotations

from pathlib import Path

import pyearthtools.data

from pyearthtools.data import Petdt
from pyearthtools.data.exceptions import DataNotFoundError
from pyearthtools.data.indexes import ArchiveIndex, decorators
from pyearthtools.data.transforms import Transform, TransformCollection
from pyearthtools.data.archive import register_archive

from site_archive_met_office.utilities import cached_iterdir, postprocess_dataset


MOUKV_RESOLUTION = (6, "hour")
MOUKV_RENAME = {"grid_latitude": "latitude", "grid_longitude": "longitude"}


@register_archive("MOUKV", sample_kwargs=dict(variable="2t"))
class MOUKV(ArchiveIndex):
    """MOUKV (subset)"""

    @property
    def _desc_(self):
        return {
            "singleline": "Met Office UKV (subset)",
            "range": "2018",
            "Documentation": "https://www.metoffice.gov.uk/binaries/content/assets/metofficegovuk/pdf/data/ukv_model_data_sheet_2019.pdf",
        }

    @decorators.alias_arguments(
        level_value=["pressure"],
        variables=["variable"],
        product=["resolution"],
    )
    @decorators.variable_modifications(variable_keyword="variables", remove_variables=False)
    @decorators.deprecated_arguments(
        level="`level` is deprecated. Simply provide the variables, `level` will be autofound."
    )
    def __init__(
        self,
        variables: list[str] | str,
        *,
        level_value: int | float | list[int | float] | tuple[list | int, ...] | None = None,
        transforms: Transform | TransformCollection | None = None,
    ):
        """
        Setup MOUKV Low-Res Indexer

        Args:
            variables (list[str] | str):
                Data variables to retrieve
            resolution (Literal[MOUKV_RESOLUTION], optional):
                Resolution of data, must be one of 'monthly-averaged','monthly-averaged-by-hour', 'reanalysis'.
                Defaults to 'reanalysis'.
            level_value: (int, optional):
                Level value to select if data contains levels. Defaults to None.
            transforms (Transform | TransformCollection, optional):
                Base Transforms to apply.
                Defaults to TransformCollection().
        """

        variables = [variables] if isinstance(variables, str) else variables
        self.variables = variables
        self.resolution = MOUKV_RESOLUTION
        self.level_value = level_value
        base_transform = pyearthtools.data.transforms.variables.Trim(variables) + (transforms or TransformCollection())

        base_transform += pyearthtools.data.transforms.attributes.Rename(MOUKV_RENAME)

        if level_value:
            base_transform += pyearthtools.data.transforms.coordinates.Select(
                {coord: level_value for coord in ["level"]}, ignore_missing=True
            )

        super().__init__(
            transforms=base_transform + (transforms or TransformCollection()),
            data_interval=MOUKV_RESOLUTION,
        )
        self.record_initialisation()

    def filesystem(
        self,
        querytime: str | Petdt,
    ) -> Path | dict[str, str | Path]:
        MOGLOBAL_HOME = self.ROOT_DIRECTORIES["MOUKV"]

        paths = {}
        querytime = Petdt(querytime)

        # Format the querytime as a date
        query_date = querytime.strftime("%Y%m%d")

        # Extract model initialization time (e.g., "00", "06")
        # TODO: Default to "00" if not specified - I think Petdt adds Txx when not specified for all time resolution steps.
        model_time = querytime.strftime("%H")

        # Search for files matching the query date and model initialization time
        files_in_dir = cached_iterdir(Path(MOGLOBAL_HOME))
        relevant_files = [
            filename for filename in files_in_dir if query_date in str(filename) and f"_{model_time}_" in str(filename)
        ]

        # Debugging print statements
        # print(f'Number of files in directory: {len(files_in_dir)}')
        # print("Query date:", query_date)
        # print("Query time:", querytime)
        # print("Model time:", model_time)
        # print("Matching files:", relevant_files)

        if not relevant_files:
            raise DataNotFoundError(f"Unable to find data for: basetime: {querytime} at {MOGLOBAL_HOME}")

        # Map the relevant files to their paths
        for filename in relevant_files:
            paths[str(filename)] = Path(MOGLOBAL_HOME) / filename

        return paths

    # Override the __getitem__ method to apply postprocessing
    def __getitem__(self, key):
        ds = super().__getitem__(key)
        ds = postprocess_dataset(ds)
        return ds

    # Do we need this?
    @property
    def _import(self):
        """module to import when this class is used"""
        return "pyearthtools.site_archive_met_office.MOUKV"
