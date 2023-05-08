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
from edit.training.trainer.trainer import EDITTrainerWrapper

from edit.training.data.utils import get_callable


def from_yaml(yaml_file: str, **kwargs) -> EDITTrainerWrapper:
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
        if "%auto%" in config["trainer"]['root_dir']:
            config["trainer"]['root_dir'] = Path(config["trainer"]['root_dir'].replace("%auto%","")) / '/'.join(Path(yaml_file).with_suffix('').parts[1:])

        Path(config["trainer"]["root_dir"]).mkdir(exist_ok=True, parents=True)
        with open(Path(config["trainer"]["root_dir"]) / "config.yaml", "w") as file:
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
    model = get_callable(model_name)
    #except (AttributeError, ModuleNotFoundError):
        #if hasattr(networks, model_name):
            #model = getattr(networks, model_name)
        #else:
            #model = get_callable("edit.training.models.networks." + model_name)

    model = model(**config["model"])

    return EDITTrainerWrapper(
        model=model,
        train_data=train_data,
        valid_data=valid_data,
        path=config["trainer"].pop("root_dir", None),
        **config["trainer"]
    )
