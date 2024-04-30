# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty 
# of any kind including, but not limited to, fitness for a particular 
# purpose. The user assumes the entire risk as to the use and 
# performance of the software. In no event shall the copyright holder 
# be held liable for any claim, damages or other liability arising 
# from the use of the software.

"""
Structured Archive Template

Usage:

See template-archive/src/edit_archive_template/structure/template.struc for the full 'data' structure.
This defines the available arguments.

>>> StructuredTemplate('bom', product_3 = 'g')
# As there is only one option for `product_2` it is auto filled in.

>>> StructuredTemplate('ukmo', product_3 = 'experimental')
# This will raise an error as `product_2` is ill defined.
"""

from __future__ import annotations

from pathlib import Path

from edit.data import EDITDatetime, transform
from edit.data.indexes import ArchiveIndex, decorators

from edit.data.indexes import VariableDefault, VARIABLE_DEFAULT

from edit.data.transform import Transform, TransformCollection
from edit.data.archive import register_archive


@register_archive("StructuredTemplate")
class StructuredTemplate(ArchiveIndex):
    """Structured Template for `edit.data` archives."""

    @property
    def _desc_(self):
        return {
            "singleline": "Structured Template",
            "WARNING": "This is just a template, do not use.",
        }

    # Create aliases
    # As some parameters are called different things by different people, an alias is helpful
    @decorators.alias_arguments(
        product_1=["agency"],
    )

    # Check the arguments
    # Force the user to use a set of valid arguments
    @decorators.check_arguments("edit_archive_template.structure.template.struc")
    def __init__(
        self,
        product_1: str | VARIABLE_DEFAULT = VariableDefault,
        product_2: str | VARIABLE_DEFAULT = VariableDefault,
        product_3: str | VARIABLE_DEFAULT = VariableDefault,
        *,
        transforms: Transform | TransformCollection = TransformCollection(),
    ):
        """
        Setup Template Index

        Args:
            product_* (Literal[VALID_PRODUCTS]):
                Example args, will be set a value if not given if only one option, otherwise will force an input.
            transforms (Transform | TransformCollection, optional):
                Base Transforms to apply.
                Defaults to TransformCollection().
        """
        self.make_catalog()

        # Set other init args
        self.products = (product_1, product_2, product_3)

        # Add the neccessary transforms to the data
        base_transform = TransformCollection()
        base_transform += transform.variables.rename_variables(test="new_variable")

        # Set the interval of the data
        interval = (1, "hour")

        super().__init__(
            transforms=base_transform + transforms,
            data_interval=interval,
        )

    def filesystem(
        self,
        querytime: str | EDITDatetime,
    ) -> Path | dict[str, str | Path]:
        """
        The tricky bit of the code,

        Take the init args, and direct `edit` to the files that are relevant.

        Can use glob, path names or anything you want.

        Can return a single path or a dictionary assigning variables to paths
        """

        return {}
