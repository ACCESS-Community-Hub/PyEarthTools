import hydra
from omegaconf import OmegaConf
import site_archive_nci

config_path = (
    "/g/data/kd24/tjl/src/PyEarthTools/packages/bundled_models/fourcastnext/Training/limited_variables_early_stopping"
)

initialsed = hydra.initialize_config_dir(version_base=None, config_dir=config_path)
cfg = hydra.compose(config_name="limited_vars_early_stop.yaml")
# print(cfg)

import pyearthtools.training
import pyearthtools.pipeline

splits = {
    "train_split": pyearthtools.pipeline.iterators.DateRange(*cfg.data.splits.train),
    "valid_split": pyearthtools.pipeline.iterators.DateRange(*cfg.data.splits.valid),
}

pipelines = None

datamodule = pyearthtools.training.data.lightning.PipelineLightningDataModule(
    "/g/data/kd24/tjl/src/PyEarthTools/packages/bundled_models/fourcastnext/Training/pipelines/foo.pipe",  # type: ignore
    **splits,
    **cfg.data.module,
)

model = hydra.utils.instantiate(cfg.model)

trainer = pyearthtools.training.lightning.Train(
    model,
    datamodule,
    path=cfg.path,
    trainer_kwargs={"num_sanity_val_steps": 0},
    **OmegaConf.to_object(cfg.trainer),  # type: ignore
)

# This takes 10 plus minutes just to get started, comprising
# - Sanity Checking
# - Sanity Checking Data Loader
# - Just sitting there for another 10+ minutes with no explanation
trainer.fit()
