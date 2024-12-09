# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

from __future__ import annotations

CLASS_NAME_TO_TRIM = "pyearthtools.data"


def format_class_name(class_to_find: object) -> list[str]:
    """
    Format class name for use in normalisation caching

    Args:
        class_to_find (str): Class to find name for

    Returns:
        list[str]: Components of class name
    """
    class_str = str(class_to_find.__class__).split("'")[1]
    class_str = class_str.replace(CLASS_NAME_TO_TRIM, "")
    class_str_list = class_str.strip().split(".")

    if "" in class_str_list:
        class_str_list.remove("")
    return []
