"""Axes class for validating and transforming axes
"""
from typing import Any, Dict, List, Union

from .format import CurrentFormat, Format

KNOWN_AXES = {"x": "space", "y": "space", "z": "space", "c": "channel", "t": "time"}


class Axes:
    def __init__(
        self,
        axes: Union[List[str], List[Dict[str, str]]],
        fmt: Format = CurrentFormat(),
    ) -> None:
        """
        Constructor, transforms axes and validates

        Raises ValueError if not valid
        """
        if axes is not None:
            self.axes = self._axes_to_dicts(axes)
        elif fmt.version in ("0.1", "0.2"):
            # strictly 5D
            self.axes = self._axes_to_dicts(["t", "c", "z", "y", "x"])
        self.fmt = fmt
        self.validate()

    def validate(self) -> None:
        """Raises ValueError if not valid"""
        if self.fmt.version in ("0.1", "0.2"):
            return

        # check names (only enforced for version 0.3)
        if self.fmt.version == "0.3":
            self._validate_03()
            return

        self._validate_axes_types()

    def to_list(
        self, fmt: Format = CurrentFormat()
    ) -> Union[List[str], List[Dict[str, str]]]:
        if fmt.version == "0.3":
            return self._get_names()
        return self.axes

    @staticmethod
    def _axes_to_dicts(
        axes: Union[List[str], List[Dict[str, str]]]
    ) -> List[Dict[str, str]]:
        """Returns a list of axis dicts with name and type"""
        axes_dicts = []
        for axis in axes:
            if isinstance(axis, str):
                axis_dict = {"name": axis}
                if axis in KNOWN_AXES:
                    axis_dict["type"] = KNOWN_AXES[axis]
                axes_dicts.append(axis_dict)
            else:
                axes_dicts.append(axis)
        return axes_dicts

    def _validate_axes_types(self) -> None:
        """
        Validate the axes types according to the spec, version 0.4+
        """
        axes_types = [axis.get("type") for axis in self.axes]
        known_types = list(KNOWN_AXES.values())
        unknown_types = [atype for atype in axes_types if atype not in known_types]
        if len(unknown_types) > 1:
            raise ValueError(
                "Too many unknown axes types. 1 allowed, found: %s" % unknown_types
            )

        def _last_index(item: str, item_list: List[Any]) -> int:
            return max(loc for loc, val in enumerate(item_list) if val == item)

        if "time" in axes_types and _last_index("time", axes_types) > 0:
            raise ValueError("'time' axis must be first dimension only")

        if axes_types.count("channel") > 1:
            raise ValueError("Only 1 axis can be type 'channel'")

        if "channel" in axes_types and _last_index(
            "channel", axes_types
        ) > axes_types.index("space"):
            raise ValueError("'space' axes must come after 'channel'")

    def _get_names(self) -> List[str]:
        """Returns a list of axis names"""
        axes_names = []
        for axis in self.axes:
            if "name" not in axis:
                raise ValueError("Axis Dict %s has no 'name'" % axis)
            axes_names.append(axis["name"])
        return axes_names

    def _validate_03(self) -> None:

        val_axes = tuple(self._get_names())
        if len(val_axes) == 2:
            if val_axes != ("y", "x"):
                raise ValueError(f"2D data must have axes ('y', 'x') {val_axes}")
        elif len(val_axes) == 3:
            if val_axes not in [("z", "y", "x"), ("c", "y", "x"), ("t", "y", "x")]:
                raise ValueError(
                    "3D data must have axes ('z', 'y', 'x') or ('c', 'y', 'x')"
                    " or ('t', 'y', 'x'), not %s" % (val_axes,)
                )
        elif len(val_axes) == 4:
            if val_axes not in [
                ("t", "z", "y", "x"),
                ("c", "z", "y", "x"),
                ("t", "c", "y", "x"),
            ]:
                raise ValueError("4D data must have axes tzyx or czyx or tcyx")
        else:
            if val_axes != ("t", "c", "z", "y", "x"):
                raise ValueError("5D data must have axes ('t', 'c', 'z', 'y', 'x')")
