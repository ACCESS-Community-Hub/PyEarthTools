"""
EDIT Trainer Commands
"""
import click

from edit.training.data.context import PatchingUpdate
from edit.training.trainer.yaml import from_yaml


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
@click.option(
    "--resume",
    type=bool,
    default=True,
)
def fit(yaml_file: str | click.Path, resume: bool):
    """From Yaml Config, fit model.

    Args:
        yaml_file (str): Path to yaml config
        resume (bool): Use existing model
    """
    trainer = from_yaml(yaml_file)
    trainer.fit(resume=resume)


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
@click.option("--stride_size", type=int)
def predict(
    yaml_file: str | click.Path,
    checkpoint: str | click.Path,
    index: str,
    save_file: str | click.Path,
    stride_size: int = None,
):
    """Using Yaml Config & Checkpoint, predict at index

    Args:
        yaml_file (str | click.Path): Path to taml config
        checkpoint (str | click.Path): Path to model checkpoint
        index (str): Index to predict at
        save_file (str | click.Path): Where to save prediction
        stride_size (int, optional): Update to stride size. Defaults to None.
    """
    trainer = from_yaml(yaml_file)
    trainer.load(checkpoint)

    with PatchingUpdate(trainer, stride_size=stride_size):
        predictions = trainer.predict(index, undo=True)
    predictions[-1].to_netcdf(save_file)


if __name__ == "__main__":
    entry_point()
