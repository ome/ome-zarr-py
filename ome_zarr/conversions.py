from typing import List


def int_to_rgba(v: int) -> List[float]:
    """Get rgba (0-1) e.g. (1, 0.5, 0, 1) from integer"""
    return [x / 255 for x in v.to_bytes(4, signed=True, byteorder="big")]


def rgba_to_int(r: int, g: int, b: int, a: int) -> int:
    return int.from_bytes([r, g, b, a], byteorder="big", signed=True)
