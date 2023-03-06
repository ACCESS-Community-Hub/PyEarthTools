import importlib

import click
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


def load_from_yaml(yaml_file: str):
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

    return DSETTrainerWrapper(
        model,
        config["trainer"].pop("root_dir"),
        train_data,
        valid_data,
        **config["trainer"]
    )


@click.group(name="Trainer From Yaml")
def entry_point():
    pass


@entry_point.command(name="fit")
@click.argument(
    "yaml_file",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True
    ),
)
def fit(yaml_file):
    """
    From Yaml Config Fit Model
    """
    trainer = load_from_yaml(yaml_file)
    trainer.fit()


@entry_point.command(name="predict")
@click.argument(
    "yaml_file",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True
    ),
)
@click.argument(
    "checkpoint",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True
    ),
)
@click.argument("index", type=str)
@click.argument("save_file", type=click.Path())
def predict(yaml_file, checkpoint, index, save_file):
    """
    Using Yaml Config & Checkpoint, predict at index
    """
    trainer = load_from_yaml(yaml_file)
    trainer.load(checkpoint)

    predictions = trainer.predict(index, undo=True)
    predictions[-1].to_netcdf(save_file)


if __name__ == "__main__":
    entry_point()
