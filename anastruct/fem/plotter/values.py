import math
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

from .element import (
    plot_values_axial_force,
    plot_values_bending_moment,
    plot_values_deflection,
    plot_values_element,
    plot_values_shear_force,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from anastruct.fem.system import SystemElements


def det_scaling_factor(max_unit: float, max_val_structure: float) -> float:
    """Determines the scaling factor for the plotting values.

    Args:
        max_unit (float): Maximum value of the unit to be plotted
        max_val_structure (float): Maximum value of the structure coordinates

    Returns:
        float: Scaling factor
    """
    if math.isclose(max_unit, 0):
        return 0.1
    return 0.15 * max_val_structure / max_unit


class PlottingValues:
    def __init__(self, system: "SystemElements", mesh: int):
        """Class for determining the plotting values of the elements.

        Args:
            system (SystemElements): System of elements
            mesh (int): Number of points to plot
        """
        self.system: "SystemElements" = system
        self.mesh: int = max(3, mesh)
        # used for scaling the plotting values.
        self._max_val_structure: Optional[float] = None

    @property
    def max_val_structure(self) -> float:
        """Maximum value of the structures dimensions.

        Returns:
            float: Maximum value of the structures dimensions.
        """
        if self._max_val_structure is None:
            self._max_val_structure = max(
                map(
                    lambda el: max(
                        el.vertex_1.x, el.vertex_2.x, el.vertex_1.y, el.vertex_2.y
                    ),
                    self.system.element_map.values(),
                )
            )
        return self._max_val_structure

    def displacements(
        self, factor: Optional[float], linear: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Determines the plotting values for displacements

        Args:
            factor (Optional[float]): Factor by which to multiply the plotting values perpendicular to the elements axis
            linear (bool): If True, the bending in between the elements is determined.

        Returns:
            Tuple[np.ndarray, np.ndarray]: x and y values
        """
        if factor is None:
            # needed to determine the scaling factor
            max_displacement = max(
                map(
                    lambda el: (
                        np.sqrt(
                            (el.max_extension or 0) ** 2
                            + (el.max_total_deflection or 0) ** 2
                        )
                    ),
                    self.system.element_map.values(),
                )
            )
            factor = det_scaling_factor(max_displacement, self.max_val_structure)
        xy = np.hstack(
            [
                plot_values_deflection(el, factor, linear)
                for el in self.system.element_map.values()
            ]
        )
        return xy[0, :], xy[1, :]

    def bending_moment(self, factor: Optional[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Determines the plotting values for bending moment

        Args:
            factor (Optional[float]): Factor by which to multiply the plotting values perpendicular to the elements axis

        Returns:
            Tuple[np.ndarray, np.ndarray]: x and y values
        """
        if factor is None:
            # maximum moment determined by comparing the node's moments and the sagging moments.
            max_moment = max(
                map(
                    lambda el: max(
                        abs(el.node_1.Tz),
                        abs(el.node_2.Tz),
                        abs(((el.all_qp_load[0] + el.all_qp_load[1]) / 16) * el.l**2),
                    ),
                    self.system.element_map.values(),
                )
            )
            factor = det_scaling_factor(max_moment, self.max_val_structure)

        assert self.system.element_map[1].bending_moment is not None
        n = len(self.system.element_map[1].bending_moment)
        xy = np.hstack(
            [
                plot_values_bending_moment(el, factor, n)
                for el in self.system.element_map.values()
            ]
        )
        return xy[0, :], xy[1, :]

    def axial_force(self, factor: Optional[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Determines the plotting values for axial force

        Returns:
            Tuple[np.ndarray, np.ndarray]: x and y values
        """
        if factor is None:
            max_force = max(
                map(
                    lambda el: max(
                        abs(el.N_1 or 0.0),
                        abs(el.N_2 or 0.0),
                        abs(((el.all_qn_load[0] + el.all_qn_load[1]) / 2) * el.l),
                    ),
                    self.system.element_map.values(),
                )
            )
            factor = det_scaling_factor(max_force, self.max_val_structure)
        assert self.system.element_map[1].axial_force is not None
        n = len(self.system.element_map[1].axial_force)
        xy = np.hstack(
            [
                plot_values_axial_force(el, factor, n)
                for el in self.system.element_map.values()
            ]
        )
        return xy[0, :], xy[1, :]

    def shear_force(self, factor: Optional[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Determines the plotting values for shear force

        Args:
            factor (Optional[float]): Factor by which to multiply the plotting values perpendicular to the elements axis

        Returns:
            Tuple[np.ndarray, np.ndarray]: x and y values
        """
        if factor is None:
            max_force = max(
                map(
                    lambda el: np.max(
                        np.abs(el.shear_force if el.shear_force is not None else [0.0])
                    ),
                    self.system.element_map.values(),
                )
            )
            factor = det_scaling_factor(max_force, self.max_val_structure)
        xy = np.hstack(
            [
                plot_values_shear_force(el, factor)
                for el in self.system.element_map.values()
            ]
        )
        return xy[0, :], xy[1, :]

    def structure(self) -> Tuple[np.ndarray, np.ndarray]:
        """Determines the plotting values for the structure itself

        Returns:
            Tuple[np.ndarray, np.ndarray]: x and y values
        """
        xy = np.hstack(
            [plot_values_element(el) for el in self.system.element_map.values()]
        )
        return xy[0, :], xy[1, :]
