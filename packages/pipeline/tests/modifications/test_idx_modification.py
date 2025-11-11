# Copyright Commonwealth of Australia, Bureau of Meteorology 2025.
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

import pytest

from pyearthtools.pipeline.modifications import TemporalWindow
from pyearthtools.pipeline import Pipeline
from pyearthtools.data.time import TimeDelta
from pyearthtools.data import Petdt, Index


@pytest.mark.parametrize("merge_method", (None, sum))
def test_temporal_window(merge_method):
    """Test temporal window."""

    # instantiate temporal window with merge method
    temporal_window_step = TemporalWindow(
        prior_indexes=[-2, -1],
        posterior_indexes=[0],
        timedelta=TimeDelta((1, "day")),
        merge_method=merge_method,
    )

    # instantiate pipeline with temporal window
    class test_data_accessor(Index):
        _data = {
            Petdt("2025-01-01"): 1,
            Petdt("2025-01-02"): 2,
            Petdt("2025-01-03"): 3,
        }

        def get(self, time):
            return self._data[time]

    pipeline = Pipeline(test_data_accessor(), temporal_window_step)
    result = pipeline["2025-01-03"]
    if merge_method is None:
        assert result == ([1, 2], [3])
    elif merge_method is sum:
        assert result == (3, 3)
