"""
EDIT Trainer Commands
"""
import click
import yaml


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
    from edit.training.trainer.yaml import from_yaml

    trainer = from_yaml(yaml_file)
    trainer.fit(resume=resume)


@entry_point.command(name="predict")
@click.argument(
    "yaml_file",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True
    ),
)
@click.argument("index", type=str)
@click.argument("save_file", type=click.Path())
@click.option(
    "--checkpoint",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True
    ),
    default=None,
)
@click.option("--stride_size", type=int, default=None)
@click.option("--recurrence", type=int, default=None)
def predict(
    yaml_file: str | click.Path,
    index: str,
    save_file: str | click.Path,
    checkpoint: str | click.Path = None,
    stride_size: int = None,
    recurrence: int = None,
):
    """Using Yaml Config & Checkpoint, predict at index

    Args:
        yaml_file (str | click.Path): Path to yaml config
        index (str): Index to predict at
        save_file (str | click.Path): Where to save prediction
        checkpoint (str | click.Path, optional): Path to model checkpoint
        stride_size (int, optional): Update to stride size. Defaults to None.
        recurrence (int, optional): Times to recur. Defaults to None.
    """
    from edit.training.data.context import PatchingUpdate
    from edit.training.trainer.yaml import from_yaml

    trainer = from_yaml(yaml_file)

    resume = True
    if checkpoint is not None:
        trainer.load(checkpoint)
        resume = False

    with PatchingUpdate(trainer, stride_size=stride_size):
        if not recurrence:
            predictions = trainer.predict(index, resume=resume, undo=True)[-1]
        else:
            predictions = trainer.predict_recurrent(
                index, undo=True, resume=resume, recurrence=recurrence
            )

    predictions.to_netcdf(save_file)


if __name__ == "__main__":
    entry_point()
