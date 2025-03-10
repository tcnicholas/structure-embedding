from __future__ import annotations

from difflib import get_close_matches
from typing import Any


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
    """

    def format_value(value: Any) -> str:
        """
        Converts a value to a string, optionally wrapping strings in quotes.
        """
        if isinstance(value, str) and stringify:
            return f'"{value}"'
        return str(value)

    # Format positional and keyword arguments
    components = [format_value(arg) for arg in positional_args]
    components += [
        f"{key}={format_value(value)}" 
        for key, value in keyword_args.items()
    ]

    # Construct a single-line representation
    single_line_repr = f"{object_name}({', '.join(components)})"
    if len(single_line_repr) < max_width and "\n" not in single_line_repr:
        return single_line_repr

    # If exceeding max width, format as a multi-line representation.
    def indent(text: str) -> str:
        """Indents text with a specified number of spaces."""
        indentation = " " * indent_size
        return "\n".join(f"{indentation}{line}" for line in text.split("\n"))

    # Build multi-line representation
    multi_line_repr = f"{object_name}(\n"
    multi_line_repr += ",\n".join(indent(component) for component in components)
    multi_line_repr += "\n)"

    return multi_line_repr


def closest_key_error(key, keys):
    """
    Raise a KeyError suggesting the closest key if available.
    """
    matches = get_close_matches(key, keys)
    if matches:
        raise KeyError(f"Key '{key}' not found. Did you mean: {matches[0]}?")
    else:
        raise KeyError(f"Key '{key}' not found.")