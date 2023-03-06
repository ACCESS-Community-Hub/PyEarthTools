"""
Reorder Data
"""


import numpy as np


def setup_formats(format_1: str, format_2: str) -> tuple[str, str]:
    """
    Convert Non fully defined format into fully defined format
    Inserts unaccounted for dimensions into variable location

    One format must be fully defined

    Args:
    ----------
        format_1: First Format
        format_2: First Format

    Raise:
    ---------
        ValueError: If both format are not fully defined

    Returns:
    ----------
        format_1, format_2 now defined

    Examples:
    ----------
        setup_formats('RPTCHW', 'RP...HW')
        'RPTCHW', 'RPTCHW'

        setup_formats('TCRPHW', 'RP...HW')
        'TCRPHW', 'RPTCHW'

        setup_formats('RP...HW', 'TCRPHW')
        'RPTCHW', 'TCRPHW'
    """
    if "..." in format_1:
        if "..." in format_2:
            raise ValueError(
                f"At least one format must be fully defined, '{format_1}, {format_2}' is invalid"
            )
        return tuple(np.flip(setup_formats(format_2, format_1)))

    format_edit_1 = format_1
    for char in format_edit_1:
        if char in format_2:
            format_edit_1 = format_edit_1.replace(char, "")
    format_2 = format_2.replace("...", format_edit_1)

    return format_1, format_2


def reorder(data: np.ndarray, current_format: str, new_format: str) -> np.ndarray:
    """
    Reorder data given string current format and new format

    Uses numpy moveaxis for axis changes.

    Args:
    -------
        data: The data to move axis of
        current_format: Current Format with letters notating axis
                e.g. "TCHW" - Time, Channels, Height, Width
        new_format: New Format with letters notating axis
                e.g. "CTHW" - Channels, Time, Height, Width

    Returns:
    -------
        Data with moved axis

    Examples:
    -------
        >>> x = np.zeros((3, 4, 5))
        >>> reorder(x, "TAC", "CAT").shape
        (5,4,3)

        >>> reorder(x, "TAC", "CTA").shape
        (5,3,4)
    """
    if current_format == new_format:
        return data

    current_format, new_format = setup_formats(current_format, new_format)

    current_indexes = np.arange(0, len(current_format), 1)

    new_indexes = []

    for entry in current_format:
        for i, compare in enumerate(new_format):
            if entry == compare:
                new_indexes.append(i)

    data = np.moveaxis(data, current_indexes, new_indexes)
    return data


def move_to_end(data, current_format: str, to_move: str) -> tuple:
    """
    Given data format, move given axis to the end

    Args:
    ----------
        data: Data to reorder
        current_format: Format of Data
        to_move: Characters to Move, must be in current_format

    Returns:
    ----------
        Current_format reordered, Reordered data

    Examples:
    ----------
        >>> x = np.zeros((3, 4, 5))
        >>> move_to_end(x, "TAC", "T").shape
        (4,5,3)

        >>> move_to_end(x, "TAC", "TA").shape
        (5,3,4)

        >>> move_to_end(x, "TAC", "AT").shape
        (5,4,3)

    """
    new_format = current_format
    for element in to_move:
        if element not in current_format:
            raise KeyError(f"{element} not in {current_format}")

        new_format = new_format.replace(element, "")
        new_format += element

    return new_format, reorder(data, current_format, new_format)
