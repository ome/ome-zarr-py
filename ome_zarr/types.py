"""Definition of complex types for use elsewhere."""

from typing import Any, Callable, Union

LayerData = Union[tuple[Any], tuple[Any, dict], tuple[Any, dict, str]]

PathLike = Union[str, list[str]]

ReaderFunction = Callable[[PathLike], list[LayerData]]

JSONDict = dict[str, Any]
