import importlib

import yaml

from dset.training import data
from dset.training.trainer.trainer import DSETTrainerWrapper


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


def load_from_yaml(yaml_file: str, **kwargs):
    """
    Load and create trainer from Yaml Config
    """
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    data_iterator = lambda: data.from_dict(config["data"]["Source"])

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
        model = get_callable('dset.training.models.networks.' + model_name)

    model = model(**config["model"])
    trainer_config = config["trainer"]
    trainer_config.update(**kwargs)

    return DSETTrainerWrapper(
        model,
        config["trainer"].pop("root_dir"),
        train_data,
        valid_data,
        **trainer_config
    )


