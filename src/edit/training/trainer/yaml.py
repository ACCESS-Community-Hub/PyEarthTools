from pathlib import Path
import importlib

import yaml
import copy

from edit.training import data
from edit.training.models import networks
from edit.training.trainer.trainer import EDITTrainerWrapper


def get_callable(module: str):
    """
    Provide dynamic import capability

    Parameters
    ----------
        module
            String of path the module, either module or specific function/class

    Returns
    -------
        Specified module or function
    """
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError:
        module = module.split(".")
        return getattr(get_callable(".".join(module[:-1])), module[-1])
    except ValueError as e:
        raise ModuleNotFoundError("End of module definition reached")


def from_yaml(yaml_file: str, **kwargs) -> EDITTrainerWrapper:
    """
    Load and create trainer from Yaml Config

    Parameters
    ----------
        yaml_file
            Path to yaml config

        **kwargs
            All passed into trainer config

    Returns
    -------
        EDITTrainerWrapper
    """
    with open(yaml_file, "r") as file:
        config = dict(yaml.safe_load(file))

    if not "order" in config["data"]["Source"]:
        config["data"]["Source"].update(order=list(config["data"]["Source"].keys()))

    data_iterator = lambda: data.from_dict(dict(config["data"]["Source"]))

    config["trainer"].update(**kwargs)

    if "root_dir" in config["trainer"]:
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
    try:
        model = get_callable(model_name)
    except (AttributeError, ModuleNotFoundError):
        if hasattr(networks, model_name):
            model = getattr(networks, model_name)
        else:
            model = get_callable("edit.training.models.networks." + model_name)

    model = model(**config["model"])

    return EDITTrainerWrapper(
        model,
        config["trainer"].pop("root_dir", None),
        train_data,
        valid_data,
        **config["trainer"]
    )
