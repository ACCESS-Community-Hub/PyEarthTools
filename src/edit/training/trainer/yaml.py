# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty 
# of any kind including, but not limited to, fitness for a particular 
# purpose. The user assumes the entire risk as to the use and 
# performance of the software. In no event shall the copyright holder 
# be held liable for any claim, damages or other liability arising 
# from the use of the software.

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
from collections import OrderedDict

import yaml
import re

from edit.training import trainer

import edit.pipeline
from edit.utils.imports import dynamic_import

TRAINER_ASSIGNMENT = OrderedDict()
if hasattr(trainer, "EDITLightningTrainer"):
    TRAINER_ASSIGNMENT[trainer.lightning.Training] = ["pytorch", "lightning"]
if hasattr(trainer, "EDITXGBoostTrainer"):
    TRAINER_ASSIGNMENT[trainer.EDITXGBoostTrainer] = ["xgboost"]


def flip_dict(dict: dict) -> dict:
    """Flip a dictionary of lists"""
    return_dict = OrderedDict()
    for k, v in dict.items():
        for i in v:
            return_dict[i] = k
    return return_dict


def from_yaml(config: str | Path | dict, **kwargs):
    """Load and create trainer from dictionary config or yaml file

    !!! Warning
        See above for information regarding keys

    Args:
        config (str):
            Path to yaml config or dictionary
        **kwargs (dict, optional):
            All passed into trainer config
    Returns:
        (EDITTrainer):
            Loaded Trainer
    """
    yaml_file = None
    if not isinstance(config, dict):
        yaml_file = config
        with open(config, "r") as file:
            config = dict(yaml.safe_load(file))

    if not "order" in config["data"]["Source"]:
        config["data"]["Source"].update(order=list(config["data"]["Source"].keys()))

    data_iterator = lambda: edit.pipeline.from_dict(dict(config["data"]["Source"]))

    # if 'accelerator' in kwargs and kwargs['accelerator'] == 'auto':
    #     kwargs['accelerator'] = 'gpu' if torch.cuda.is_available() else 'cpu'

    config["trainer"].update(**kwargs)

    if "root_dir" in config["trainer"]:
        config["trainer"]["path"] = config["trainer"].pop("root_dir")

    if "path" in config["trainer"]:
        auto_match = re.search(r"%auto.*%", config["trainer"]["path"])
        if auto_match:
            auto_match = auto_match[0]
            auto_parts: list[str] = auto_match.replace("%", "").split("_")
            if not yaml_file:
                raise ValueError(f"Cannot fill %auto% if config file path not given.")
            parts = Path(yaml_file).absolute().with_suffix("").parts

            if len(auto_parts) == 2:
                neg = False
                if "-" in auto_parts[-1]:
                    auto_parts[-1] = auto_parts[-1].replace("-", "")
                    neg = True
                if auto_parts[-1].isdigit():
                    parts = parts[int(auto_parts[-1]) * (-1 if neg else 1) :]
                elif auto_parts[-1] in parts:
                    parts = parts[parts.index(auto_parts[-1]) + 1 :]
                else:
                    raise KeyError(f"Cannot parse {auto_match}, Path was {Path(yaml_file).absolute()}")

            config["trainer"]["path"] = str(Path(config["trainer"]["path"].replace(auto_match, "")) / "/".join(parts))

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

    if "model" not in config:
        raise KeyError(f"model could not be found in config. Ensure a model definition exists.")
    model_name = config["model"].pop("Source")
    # try:
    try:
        model = dynamic_import(model_name)
    except (ImportError, ModuleNotFoundError):
        raise ImportError(f"Could not find model: {model_name}")

    model = model(**config["model"])

    trainer_dict = flip_dict(TRAINER_ASSIGNMENT)
    if len(TRAINER_ASSIGNMENT.keys()) == 0:
        raise ImportError(f"Could not find any trainer to use, they were unable to be imported.")

    trainer_class = flip_dict(TRAINER_ASSIGNMENT)[list(flip_dict(TRAINER_ASSIGNMENT).keys())[0]]

    if "type" in config["trainer"]:
        trainer_type = config["trainer"].pop("type")
        if trainer_type in trainer_dict:
            trainer_class = trainer_dict[trainer_type]
        else:
            try:
                trainer_class = dynamic_import(trainer_type)
            except (ImportError, ModuleNotFoundError):
                raise KeyError(f"Could not find trainer: {trainer_type}")

        if trainer_class is None:
            raise KeyError(f"Trainer type {trainer_type} not recognised. Use {trainer_dict.keys()} or import path.")

    return trainer_class(
        model=model,
        train_data=train_data,
        valid_data=valid_data,
        path=config["trainer"].pop("path", None),
        **config["trainer"],
    )
