# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

from __future__ import annotations

from pathlib import Path

from pyearthtools.data.indexes import FileSystemIndex


def save(
    plot,
    callback: FileSystemIndex,
    *args,
    save_kwargs: dict = {},
    **kwargs,
):
    """Save plot objects"""
    path = callback.search(*args, **kwargs)
    if not isinstance(path, (str, Path)):
        raise NotImplementedError(f"Cannot handle saving with paths as {type(path)}")
    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    suffix = path.suffix

    if hasattr(plot, "fig"):
        plot = plot.fig
    if not hasattr(plot, "savefig"):
        raise TypeError(f"Unable to determine how to save {type(plot)}")

    plot.savefig(path, **save_kwargs)

    return path
