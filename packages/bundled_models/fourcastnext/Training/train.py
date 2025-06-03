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

import logging

import hydra
from hydra.utils import instantiate

from omegaconf import OmegaConf

from pathlib import Path


import sys

sys.path.append(str((Path(__file__).parent / "./../src").resolve()))

logger = logging.getLogger(__name__)
logging.getLogger("cfgrib").setLevel(logging.ERROR)
logging.getLogger("matplotlib").setLevel(logging.ERROR)


@hydra.main(config_path="./configs", config_name="config", version_base=None)
def train(cfg):
    print(OmegaConf.to_yaml(cfg))

    import pyearthtools.training
    import pyearthtools.pipeline

    splits = {
        "train_split": pyearthtools.pipeline.iterators.DateRange(*cfg.data.splits.train),
        "valid_split": pyearthtools.pipeline.iterators.DateRange(*cfg.data.splits.valid),
    }

    if cfg.data.splits.random:
        seed = cfg.data.splits.get("random_seed", 42)
        splits = {
            "train_split": pyearthtools.pipeline.iterators.Randomise(splits["train_split"], seed),
            "valid_split": pyearthtools.pipeline.iterators.Randomise(splits["valid_split"], None),
        }

    pipelines = None
    try:
        pipelines = OmegaConf.to_object(cfg.data.pipelines)
    except ValueError:
        pipelines = cfg.data.pipelines
    datamodule = pyearthtools.training.data.lightning.PipelineLightningDataModule(
        pipelines,  # type: ignore
        **splits,
        **cfg.data.module,
    )

    # TODO: The print function caused an exception within library code
    # print(datamodule)

    # cfg.model.model_params.update(cfg.data.model_updates)
    model = instantiate(cfg.model)

    trainer = pyearthtools.training.lightning.Train(
        model,
        datamodule,
        path=cfg.path,
        **OmegaConf.to_object(cfg.trainer),  # type: ignore
    )

    if cfg.fit or input("Fit? (y/n): ").lower() == "y":
        trainer.fit()


if __name__ == "__main__":
    train()
