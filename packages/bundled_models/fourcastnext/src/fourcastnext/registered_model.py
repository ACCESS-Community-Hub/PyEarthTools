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

from __future__ import annotations

from typing import Any
from pathlib import Path
import logging
import math

import pyearthtools.zoo

import fourcastnext


CONFIG_PATH = Path(__file__, "../configs/").resolve()
LOG = logging.getLogger("pyearthtools.zoo.fourcastnext")


@pyearthtools.zoo.register("Development/FourCastNextRM", exists="ignore")
class FourCastNextRM(pyearthtools.zoo.BaseForecastModel):
    """
    FourCastNeXt was originally developed by FourCastNeXt ([Guo et al. 2024](https://doi.org/10.48550/arXiv.2401.05584)) 

    This class provides the underlying architecture as a registered model within the framework,
    so that it can be trained according to whatever data and resolution may be of interest.

    Users need to train their own model weights.

    

    Arguments:
        lead_time (int | str | pyearthtools.data.TimeDelta):
            Lead time to predict to. If int will be given as hours.
            Separate delta notation by -.
        interval (int):
            Data interval in hours. Defaults to 6.
        ckpt_path (str, optional):
            Override for weights path
    """

    _name = "Development/FourCastNextRM"
    _times = [-6]
    _download_paths = []

    def __init__(
        self,
        *,
        pipeline_name: str = None,
        pipeline=None,
        output: str | Path,
        lead_time: int | str,
        ckpt_path: str | None = None,
        interval: int = 6,
        lightning_model_params={},
        **kwargs,
    ) -> None:
        """
        Create FourCastNeXt Model

        Args:
            pipeline_name: Pipeline name to use
            output: Output location
            lead_time: Lead time of forecast (hours).
            interval: Data interval in hours. Defaults to 6.
            ckpt_path: Override for weights path
        """
        self.lead_time = pyearthtools.zoo.utils.delta_conversion(lead_time, "hour")
        if ckpt_path:
            self._redownload_each_time = True
            self._download_paths = [(ckpt_path, "weights.ckpt")]  # type: ignore

        self.ckpt_path = ckpt_path
        self.lightning_model_params = lightning_model_params
        self.lightning_model = fourcastnext.FourCastNextLM(self.lightning_model_params)

        self.interval = interval

        super().__init__(pipeline_name=pipeline_name, pipeline=pipeline, output=output, **kwargs)

    def load(self, **kwargs) -> tuple[Any, dict[str, Any]]:
        """Load model

        Returns:
            (tuple[Any, dict[str, Any]]):
                Predictor, index kwargs
        """

        model_kwargs = dict(
            data_interval=(self.interval, "hours"),
            prediction_function="recurrent",
            prediction_config=dict(
                steps=math.ceil(self.lead_time // self.interval),
                verbose=True,
            ),
            **kwargs,
        )
        import pyearthtools.training

        model_wrapper = pyearthtools.training.wrapper.lightning.Predict(self.lightning_model, self.pipeline)

        model_wrapper.load(self.ckpt_path)

        wrapper = pyearthtools.training.wrapper.predict.TimeSeriesAutoRecurrent(
            model_wrapper, interval=f"{self.interval} hours"
        )

        return wrapper, model_kwargs
