# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty 
# of any kind including, but not limited to, fitness for a particular 
# purpose. The user assumes the entire risk as to the use and 
# performance of the software. In no event shall the copyright holder 
# be held liable for any claim, damages or other liability arising 
# from the use of the software.

from __future__ import annotations
from collections import OrderedDict

import functools
import yaml

import edit.data
from edit.data.catalog import Catalog, CatalogEntry

import edit.utils
from edit.utils import parsing


class CatalogMixin:
    def make_catalog(
        self,
        name: str | None = None,
        ignore_variables=[],
        override: bool = False,
    ):
        """
        Make a [CatalogEntry][edit.data.CatalogEntry] from this classes initalisation arguments
        at `.catalog`.

        !!! Warning
            Requires `super().__init__()` to have been called prior

        Args:
            name (str | None, optiona):
                Override for name of catalog. Defaults to None
            ignore_variables (list, optional):
                Variables to ignore when making CatalogEntry. Defaults to [].
        """
        # if hasattr(self, "catalog") and not override:
        # return

        init_args = OrderedDict(parsing.get_initialise_args(ignore=ignore_variables))
        # try:
        #     init_args = utilities.parsing.get_initialise_args(ignore=ignore_variables)
        # except KeyError as e:
        #     warnings.warn(f"Catalog could not be made due to {e}", UserWarning)
        #     return
        from edit.data.transform.transform import Transform, TransformCollection

        for key, val in init_args.items():
            if isinstance(val, (Transform, TransformCollection)):
                init_args[key] = edit.data.transform.utils.parse_transforms(val)

        if hasattr(self, 'catalog'):
            prior_args = self.catalog._kwargs
            prior_args.update(init_args)
            init_args = prior_args

        self.catalog = CatalogEntry(
            self.__class__,
            **init_args,
            name=name or self.__class__.__name__,
            class_path=getattr(self, "_registered_path", None),
        )

    @functools.wraps(CatalogEntry.save)
    def save_index(self, *args, **kwargs):
        return self.catalog.save(*args, **kwargs)

    @staticmethod
    @functools.wraps(Catalog.load)
    def load_index(*args, **kwargs):
        return Catalog.load(*args, **kwargs)

    def __add__(self, other):
        """
        Add indexes together to create a Catalog of indexes
        """
        if not hasattr(other, "catalog"):
            return NotImplemented
        try:
            other.catalog
        except NotImplementedError:
            return NotImplemented
        return Catalog(self, other)


CATALOGUED_NAME = "edit.catalogued"


# define the representer, responsible for serialization
def Catalog_representer(dumper, data):
    type_data = type(data)
    return dumper.represent_mapping(
        f"!{str(type_data.__module__).replace('edit', CATALOGUED_NAME)}.{type_data.__name__}",
        data.catalog.to_dict()["kwargs"],
    )


def Catalog_constructer(loader, tag_suffix: str, node):
    tag_suffix = tag_suffix.replace(".catalogued", "")
    return edit.utils.imports.dynamic_import("edit" + tag_suffix)(**loader.construct_mapping(node))


yaml.add_multi_representer(CatalogMixin, Catalog_representer)
yaml.add_multi_constructor(f"!{CATALOGUED_NAME}", Catalog_constructer)
