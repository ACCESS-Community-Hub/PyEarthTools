# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

"""
Command Line Interface for `edit.data`
"""

from __future__ import annotations
from pathlib import Path

import click


@click.group(name="EDIT Data")
def entry_point():
    pass


@entry_point.group(name="geographic")
def geographic():
    """Commands related to `edit.data.static.geographic`"""
    pass


@geographic.command(name="setup")
@click.option("--verbose/--quiet", type=bool, default=False)
def setup(verbose):
    """Download all geographic static files"""
    import edit.data

    if edit.data.static.geographic._download_all(verbose=verbose):
        print(f"Successfully downloaded all files")
    else:
        print(f"Failed to download all files")


def split_dictionary(dictionary: dict[str, dict] = {}, **kwargs) -> list[list[str]]:
    list_of_keys = [[*list(dictionary.keys()), *list(kwargs.keys())]]

    pass_keys = {}

    for _, v in {**dictionary, **kwargs}.items():
        if isinstance(v, dict):
            for key, val in v.items():
                pass_keys[key] = val
        elif isinstance(v, str):
            pass_keys[v] = {}
        elif isinstance(v, list):
            for a in v:
                pass_keys[a] = {}
    if len(pass_keys) > 0:
        response = split_dictionary(**pass_keys)
    else:
        response = []

    for resp in response:
        list_of_keys.append(resp)
    return list_of_keys


@entry_point.command(name="structure")
@click.argument("top", type=click.Path())
@click.option(
    "--blacklisted",
    "-b",
    type=str,
    multiple=True,
    default=[],
    help="Folder names to exclude.",
)
@click.option(
    "--save",
    type=click.Path(),
    default=None,
    help="Save location, if not given print out.",
)
@click.option("--verbose/--quiet", type=bool, default=False)
def create_structure(top, blacklisted, save, verbose):
    """
    Generate a structure file for use in argument checking

    User must specify the order of the layers

    \b
    Args:
        top: Path
            Location to generate structure for
    """
    from edit.data.indexes.utilities.structure import structure
    import yaml

    structure_dict: dict[str, dict | list] = {}
    structure_d: dict[str, dict[str, Any] | list[str]] = structure(top, blacklisted=blacklisted, verbose=verbose)  # type: ignore

    response = input(f"Would you like to specify the order? (Yes/No): ")
    order = []
    if "y" in response.lower():
        for level in split_dictionary(structure_d):  # type: ignore
            level_str = level if len(level) < 5 else [*level[0:4], "...", *level[-4:-1]]
            order.append(input(f"What is the name of level: {level_str}?: "))
    else:
        if Path(save).exists():
            order = yaml.safe_load(open(save, "r"))["order"]
        else:
            print("In order to use this within `edit`, you will need to specify order in the structure.")
            order = ["USER_INPUT_HERE"]

    structure_dict["order"] = order
    structure_dict.update(structure_d)

    if save is not None:
        with open(save, "w") as outfile:
            yaml.dump(structure_dict, outfile, default_flow_style=False)
    else:
        print(structure_dict)


if __name__ == "__main__":
    entry_point()
