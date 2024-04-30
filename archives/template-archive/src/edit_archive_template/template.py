# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty 
# of any kind including, but not limited to, fitness for a particular 
# purpose. The user assumes the entire risk as to the use and 
# performance of the software. In no event shall the copyright holder 
# be held liable for any claim, damages or other liability arising 
# from the use of the software.

"""
Base Archive Template

Usage:

See template-archive/src/edit_archive_template/variables/template/* for the valid variables.
This defines the available arguments.

>>> BaseTemplate('var_1')
# Is a valid variable

>>> BaseTemplate('var_3')
# Will raise an error as `var_3` is invalid unless `product` == `other_product`

"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from edit.data import EDITDatetime, transform
from edit.data.exceptions import DataNotFoundError
from edit.data.indexes import ArchiveIndex, decorators
from edit.data.transform import Transform, TransformCollection
from edit.data.archive import register_archive

VALID_PRODUCTS = ["test", "other_product"]


@register_archive("BaseTemplate")
class BaseTemplate(ArchiveIndex):
    """BaseTemplate for `edit.data` archives."""

    @property
    def _desc_(self):
        return {
            "singleline": "Template",
            "WARNING": "This is just a template, do not use.",
        }

    # Create aliases
    # As some parameters are called different things by different people, an alias is helpful
    @decorators.alias_arguments(
        variables=["variable", "var"],
    )

    # Check the arguments
    # Force the user to use a set of valid arguments from a validity file
    # Can use args within the path
    # See `structured.py` for a struc example
    @decorators.check_arguments(
        product=VALID_PRODUCTS,
        variables="edit_archive_template.variables.template.{product}.valid",
    )
    def __init__(
        self,
        variables: list[str] | str,
        *,
        product: Literal[VALID_PRODUCTS] = "test",  # type: ignore
        transforms: Transform | TransformCollection = TransformCollection(),
    ):
        """
        Setup Template Index

        Args:
            variables (list[str] | str):
                Data variables to retrieve
            product (Literal[VALID_PRODUCTS]):
                Product of data to retrieve
            transforms (Transform | TransformCollection, optional):
                Base Transforms to apply.
                Defaults to TransformCollection().
        """
        self.make_catalog()

        # Force variables to a list
        variables = [variables] if isinstance(variables, str) else variables
        self.variables = variables

        # Set other init args
        self.product = product

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

        DATA_HOME_PATH = ""

        paths = {}

        querytime = EDITDatetime(querytime)

        for variable in self.variables:
            var_path = Path(DATA_HOME_PATH) / variable / str(querytime.year)

            relevant_path = var_path / querytime.replace(day=1).strftime("%Y%m%d")

            if relevant_path.exists():
                paths[variable] = relevant_path
                continue

            raise DataNotFoundError(
                f"Unable to find data for: basetime: {querytime}, variables: {variable} at {var_path}"
            )
        return paths
