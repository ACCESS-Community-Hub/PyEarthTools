"""
Data processing tools
"""
DEFAULT_FORMAT_SUBSET: str = "...HW"

DEFAULT_FORMAT_PATCH_ORGANISE: str = "P...HW"
DEFAULT_FORMAT_PATCH: str = "RP...HW"
DEFAULT_FORMAT_PATCH_AFTER: str = "...HW"

from . import patches, reorder, subset
from .tesselator import Tesselator
