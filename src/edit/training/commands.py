"""
EDIT Trainer Commands
"""
from __future__ import annotations

import click
import sys


@click.group(name="Trainer From Yaml")
def entry_point():
    pass


@entry_point.command(
    name="fit",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    ),
)
@click.pass_context
@click.argument(
    "yaml_file",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True
    ),
)
@click.option(
    "--load/--new",
    type=bool,
    default=True,
    help = 'Load model or not'
)
@click.option(
    "--paths",
    type= list,
    default = [],
    help = 'Paths to add to PythonPath'
)
def fit(ctx, yaml_file: str | click.Path, load: bool, paths: list[str]):
    """
    From Yaml Config, fit model.

    yaml_file (str): Path to yaml config
    """
    from edit.training.trainer.yaml import from_yaml

    for path in paths:
        sys.path.append(path)

    d = dict()
    if len(ctx.args) > 1:
        for i in range(0, len(ctx.args), 2):
            if not str(ctx.args[i]).startswith("--"):
                raise KeyError(
                    f"{ctx.args[i]} is an invalid kwarg, ensure it starts with '--'"
                )
            d[str(ctx.args[i]).replace("--", "")] = (
                int(ctx.args[i + 1]) if ctx.args[i + 1].isdigit() else ctx.args[i + 1]
            )

    extra_kwargs = d

    trainer = from_yaml(yaml_file, **extra_kwargs)
    trainer.fit(load=load)


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
    from edit.pipeline.context import PatchingUpdate
    from edit.training.trainer.yaml import from_yaml

    trainer = from_yaml(yaml_file)

    resume = True
    if checkpoint is not None:
        trainer.load(checkpoint)
        resume = False

    with PatchingUpdate(trainer, stride_size=stride_size):
        if not recurrence:
            predictions = trainer.predict(index, load=resume, undo=True)[-1]
        else:
            predictions = trainer.predict_recurrent(
                index, undo=True, load=resume, recurrence=recurrence
            )

    predictions.to_netcdf(save_file)


if __name__ == "__main__":
    entry_point()
