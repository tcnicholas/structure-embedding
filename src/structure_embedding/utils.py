from __future__ import annotations

from difflib import get_close_matches
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt


def uniform_repr(
    object_name: str,
    *positional_args: Any,
    max_width: int = 60,
    stringify: bool = True,
    indent_size: int = 2,
    **keyword_args: Any,
) -> str:
    """
    Generates a uniform string representation of an object, supporting both
    positional and keyword arguments.

    Parameters
    ----------
    object_name
        The name of the object to represent.
    positional_args
        The positional arguments to represent.
    max_width
        The maximum width of the string representation before wrapping.
    stringify
        Whether to wrap string values in quotes.
    indent_size
        The number of spaces to use for indentation.
    keyword_args
        The keyword arguments to represent.

    Returns
    -------
    The string representation of the object.

    Notes
    -----
    - Shamelessly copied from the `uniform_repr` function in the `graph-pes`
    package (https://github.com/jla-gardner/graph-pes). Go check it out!
    """

    def format_value(value: Any) -> str:
        """
        Converts a value to a string, optionally wrapping strings in quotes.
        """

        if isinstance(value, str) and stringify:
            return f'"{value}"'

        return str(value)

    # format positional and keyword arguments
    components = [format_value(arg) for arg in positional_args]
    components += [
        f"{key}={format_value(value)}" for key, value in keyword_args.items()
    ]

    # construct a single-line representation
    single_line_repr = f"{object_name}({', '.join(components)})"
    if len(single_line_repr) < max_width and "\n" not in single_line_repr:
        return single_line_repr

    # if exceeding max width, format as a multi-line representation.
    def indent(text: str) -> str:
        """Indents text with a specified number of spaces."""
        indentation = " " * indent_size
        return "\n".join(f"{indentation}{line}" for line in text.split("\n"))

    # build multi-line representation
    multi_line_repr = f"{object_name}(\n"
    multi_line_repr += ",\n".join(indent(component) for component in components)
    multi_line_repr += "\n)"

    return multi_line_repr


def closest_key_error(
    key: str,
    keys: Sequence[str],
) -> None:
    """
    Raise a KeyError suggesting the closest key if available.

    Parameters
    ----------
    key
        The key that was not found.
    keys
        The keys to search for a close.

    Raises
    ------
    If no close key is found, raise a KeyError with a suggestion.
    """

    matches = get_close_matches(key, keys)

    if matches:
        raise KeyError(f"Key '{key}' not found. Did you mean: {matches[0]}?")
    else:
        raise KeyError(f"Key '{key}' not found.")


def rotate_data(
    data: npt.NDArray[np.floating], degrees: float = 90.0
) -> npt.NDArray[np.floating]:
    """
    Rotate 2D points by a given number of degrees counter-clockwise.

    Parameters
    ----------
    data
        2D array of points to rotate.
    degrees
        Number of degrees to rotate the points counter-clockwise.

    Returns
    -------
    The rotated points.
    """

    radians = np.deg2rad(degrees)
    cos_angle = np.cos(radians)
    sin_angle = np.sin(radians)

    rotation_matrix = np.array(
        [[cos_angle, -sin_angle], [sin_angle, cos_angle]]
    )

    return data @ rotation_matrix.T
