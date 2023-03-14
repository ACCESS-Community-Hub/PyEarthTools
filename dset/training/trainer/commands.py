import click

from dset.training.data.context import PatchingUpdate
from dset.training.trainer.yaml import load_from_yaml


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
@click.option("--stride_size", type=int)
def predict(yaml_file, checkpoint, index, save_file, stride_size=None):
    """
    Using Yaml Config & Checkpoint, predict at index
    """
    trainer = load_from_yaml(yaml_file)
    trainer.load(checkpoint)

    with PatchingUpdate(trainer, stride_size=stride_size):
        predictions = trainer.predict(index, undo=True)
    predictions[-1].to_netcdf(save_file)


if __name__ == "__main__":
    entry_point()
