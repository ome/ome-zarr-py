"""Simple conversion helpers."""

from typing import List


def int_to_rgba(v: int) -> List[float]:
    """Get rgba (0-1) e.g. (1, 0.5, 0, 1) from integer.
    >>> print(int_to_rgba(0))
    [0.0, 0.0, 0.0, 0.0]
    >>> print([round(x, 3) for x in int_to_rgba(100100)])
    [0.0, 0.004, 0.529, 0.016]
    """
    return [x / 255 for x in v.to_bytes(4, signed=True, byteorder="big")]


def int_to_rgba_255(v: int) -> List[int]:
    """Get rgba (0-255) from integer.
    >>> print(int_to_rgba_255(0))
    [0, 0, 0, 0]
    >>> print([round(x, 3) for x in int_to_rgba_255(100100)])
    [0, 1, 135, 4]
    """
    return [x for x in v.to_bytes(4, signed=True, byteorder="big")]


def rgba_to_int(r: int, g: int, b: int, a: int) -> int:
    """Use int.from_bytes to convert a color tuple.

    >>> print(rgba_to_int(0, 0, 0, 0))
    0
    >>> print(rgba_to_int(0, 1, 135, 4))
    100100
    """
    return int.from_bytes([r, g, b, a], byteorder="big", signed=True)
