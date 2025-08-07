"""Definition of complex types for use elsewhere."""

from collections.abc import Callable
from typing import Any

LayerData = tuple[Any] | tuple[Any, dict] | tuple[Any, dict, str]

PathLike = str | list[str]

ReaderFunction = Callable[[PathLike], list[LayerData]]

JSONDict = dict[str, Any]
