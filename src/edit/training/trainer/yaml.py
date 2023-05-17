"""
Allow a Trainer Configuration to be saved and loaded from a yaml file


### Info

| Key | SubKey | Description |
| -------| --- | ---------------- |
| model | | Model Configuration |
| model |.Source | Set to Model location on Python PATH | 
| model | .* | Any Keyword arguments to be passed to the model |
| | |
| data  |  | Data Pipeline |
| data  | .Source | Data Pipeline Config |
| data  | .Ranges | Data Iteration Ranges. `train_data` & `valid_data` |
| | |
| trainer | | Training Configuration |
| trainer | .* | Any Keyword arguments to be passed to the trainer |

!!! Example
    ```yaml
    model:
        Source: 'Models.Architecture'
        model_params:
            img_size: 256
            in_channels: 6
            out_channels: 6

    data:
        Source:
            archive.ERA5:
                variables: '2t'
                level : 'single'
            iterators.TemporalInterface:
                samples : [6,6]
                sample_interval : 10
            iterators.Iterator:
                catch: ['edit.data.DataNotFoundError', 'ValueError', 'OSError']
            operations.filters.DropAllNan: {}
            operations.PatchingDataIndex:
                kernel_size: [256,256]
            operations.filters.DropValue:
                value: 0
                percentage: 80
            operations.filters.DropNan: {}
            operations.values.ForceNormalised: {}
            operations.values.FillNa: {}
            operations.reshape.Squish: {axis: -4}
            loaders.PytorchIterable: {}

        Ranges:
            train_data:
                start: '2021-01-01T00:00'
                end: '2022-01-01'
                interval: 60
            valid_data:
                start: '2022-01-01T00:00'
                end: '2022-04-01'
                interval: 10
    trainer:
        root_dir: ''
        num_workers: 12
        strategy: 'ddp'
        accelerator: "gpu"
        logger: 'tensorboard'
        max_epochs: 100
        batch_size: 64

    ```
"""
from __future__ import annotations

from pathlib import Path
import importlib

import yaml

# import torch

from edit.training import data
from edit.training.models import networks
from edit.training.trainer.pytorch.trainer import EDITLightningTrainer
from edit.training.trainer.xgboost.trainer import EDITXGBoostTrainer

from edit.training.data.utils import get_callable

TRAINER_ASSIGNMENT = {
    EDITLightningTrainer : ['pytorch', 'lightning'],
    EDITXGBoostTrainer : ['xgboost'],
}

def flip_dict(dict):
    return_dict = {}
    for k, v in dict.items():
        for i in v:
            return_dict[i] = k
    return return_dict

def from_yaml(yaml_file: str, **kwargs) -> EDITLightningTrainer:
    """Load and create trainer from Yaml Config

    !!! Warning
        See above for information regarding keys 

    Args:
        yaml_file (str): 
            Path to yaml config
        **kwargs (dict, optional):
            All passed into trainer config
    Returns:
        (EDITTrainerWrapper): 
            Loaded Trainer
    """    
    with open(yaml_file, "r") as file:
        config = dict(yaml.safe_load(file))

    if not "order" in config["data"]["Source"]:
        config["data"]["Source"].update(order=list(config["data"]["Source"].keys()))

    data_iterator = lambda: data.from_dict(dict(config["data"]["Source"]))

    # if 'accelerator' in kwargs and kwargs['accelerator'] == 'auto':
    #     kwargs['accelerator'] = 'gpu' if torch.cuda.is_available() else 'cpu'

    config["trainer"].update(**kwargs)

    if "root_dir" in config["trainer"]:
        config['trainer']['path'] = config["trainer"].pop('root_dir')
        
    if "path" in config["trainer"]:
        if "%auto%" in config["trainer"]['path']:
            config["trainer"]['path'] = Path(config["trainer"]['path'].replace("%auto%","")) / '/'.join(Path(yaml_file).with_suffix('').parts[1:])

        Path(config["trainer"]["path"]).mkdir(exist_ok=True, parents=True)
        with open(Path(config["trainer"]["path"]) / "config.yaml", "w") as file:
            yaml.dump(config, file)

    train_data = data_iterator()
    train_data.set_iterable(**config["data"]["Ranges"]["train_data"])

    if "valid_data" in config["data"]["Ranges"]:
        valid_data = data_iterator()
        valid_data.set_iterable(**config["data"]["Ranges"]["valid_data"])
    else:
        valid_data = None

    model_name = config["model"].pop("Source")
    #try:
    try:
        model = get_callable(model_name)
    except (ImportError, ModuleNotFoundError):
        raise ImportError(f"Could not find model: {model_name}")
    #except (AttributeError, ModuleNotFoundError):
        #if hasattr(networks, model_name):
            #model = getattr(networks, model_name)
        #else:
            #model = get_callable("edit.training.models.networks." + model_name)

    model = model(**config["model"])

    trainer_class = EDITLightningTrainer
    trainer_dict = flip_dict(TRAINER_ASSIGNMENT)

    if 'type' in config['trainer']:
        trainer_type = config['trainer'].pop('type')
        if trainer_type in trainer_dict:
            trainer_class = trainer_dict[trainer_type]
        else:
            try:
                trainer_class = get_callable(trainer_type)
            except (ImportError, ModuleNotFoundError):
                raise KeyError(f"Could not find trainer: {trainer_type}")

        if trainer_class is None:
            raise KeyError(f"Trainer type {trainer_type} not recognised. Use {trainer_dict.keys()} or import path.")

    return trainer_class(
        model=model,
        train_data=train_data,
        valid_data=valid_data,
        path=config["trainer"].pop("path", None),
        **config["trainer"]
    )
