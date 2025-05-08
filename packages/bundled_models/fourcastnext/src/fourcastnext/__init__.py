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

from fourcastnext.lightning_model import FourCastNextLM
from fourcastnext import registered_model

from pyearthtools.pipeline.operation import Operation
import xarray as xr

__version__ = "0.1.0"

# TODO: Come up with a more elegant regridding approach that ideally can also
# handle any resolution of grid, and so not be limited specifically to this model
# and the resolution of ERA5 full res


class CropToRectangle(Operation):
    """Cut with Bounding box"""

    def __init__(self, warn=True):
        """
        Default ERA5 is 721x1440. FourCastNeXt needs to be able to use 2x2 kernels, so needs an even number grid
        dimension. For now, this class just disposes of the surplus pixels. In future the cropping strategy
        from the paper could be implemented, or a complex regrid could be performed to resample to an even grid
        """
        super().__init__()
        self.record_initialisation()

    def apply_func(self, dataset: xr.Dataset):
        subset_dataset = dataset.isel(
            latitude=slice(0, -1),
        )

        return subset_dataset

    def undo_func(self, dataset_to_undo, **kwargs):

        # Just return the cropped data, cannot 'undo' this one
        return dataset_to_undo
