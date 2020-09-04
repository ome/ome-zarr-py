"""Definition of complex types for use elsewhere."""

from typing import Any, Callable, Dict, List, Tuple, Union

LayerData = Union[Tuple[Any], Tuple[Any, Dict], Tuple[Any, Dict, str]]

PathLike = Union[str, List[str]]

ReaderFunction = Callable[[PathLike], List[LayerData]]

JSONDict = Dict[str, Any]
