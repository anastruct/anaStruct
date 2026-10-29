import copy
import math
import pprint
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Union

from anastruct.basic import arg_to_list

if TYPE_CHECKING:
    from anastruct.fem.system import SystemElements


class LoadCase:
    """
    Group different loads in a load case
    """

    def __init__(self, name: str):
        """Create a load case

        Args:
            name (str): Name of the load case
        """
        self.name: str = name
        self.spec: dict = {}
        self.c: int = 0

    def q_load(
        self,
        q: Union[float, Sequence[float]],
        element_id: Union[int, Sequence[int]],
        direction: Union[str, Sequence[str]] = "element",
        rotation: Optional[Union[float, Sequence[float]]] = None,
        q_perp: Optional[Union[float, Sequence[float]]] = None,
    ) -> None:
        """
        Apply a q-load to an element.

        :param element_id: (int/ list) representing the element ID
        :param q: (flt) value of the q-load
        :param direction: (str) "element", "x", "y", "parallel"
        """
        self.c += 1
        self.spec[f"q_load-{self.c}"] = {
            "q": q,
            "element_id": element_id,
            "direction": direction,
            "rotation": rotation,
            "q_perp": q_perp,
        }

    def point_load(
        self,
        node_id: Union[int, Sequence[int]],
        Fx: Union[float, Sequence[float]] = 0,
        Fy: Union[float, Sequence[float]] = 0,
        rotation: Union[float, Sequence[float]] = 0,
    ) -> None:
        """
        Apply a point load to a node.

        :param node_id: (int/ list) Nodes ID.
        :param Fx: (flt/ list) Force in global x direction.
        :param Fy: (flt/ list) Force in global x direction.
        :param rotation: (flt/ list) Rotate the force clockwise. Rotation is in degrees.
        """
        self.c += 1
        self.spec[f"point_load-{self.c}"] = {
            "node_id": node_id,
            "Fx": Fx,
            "Fy": Fy,
            "rotation": rotation,
        }

    def moment_load(
        self, node_id: Union[int, Sequence[int]], Tz: Union[float, Sequence[float]]
    ) -> None:
        """
        Apply a moment on a node.

        :param node_id: (int/ list) Nodes ID.
        :param Tz: (flt/ list) Moments acting on the node.
        """
        self.c += 1
        self.spec[f"moment_load-{self.c}"] = {"node_id": node_id, "Tz": Tz}

    def dead_load(
        self, element_id: Union[int, Sequence[int]], g: Union[float, Sequence[float]]
    ) -> None:
        """
        Apply a dead load in kN/m on elements.

        :param element_id: (int/ list) representing the element ID
        :param g: (flt/ list) Weight per meter. [kN/m] / [N/m]
        """
        self.c += 1
        self.spec[f"dead_load-{self.c}"] = {"element_id": element_id, "g": g}

    def imposed_displacement(
        self,
        node_id: int,
        ux: Optional[float] = None,
        uy: Optional[float] = None,
        phi_z: Optional[float] = None,
    ) -> None:
        """
        Apply an imposed displacement (support settlement) and/or rotation to a node.

        The imposed values are multiplied by the load factor of the load combination,
        so a settlement belongs to a single load case and is not counted again in the
        other cases.

        :param node_id: (int) representing the node ID
        :param ux: (flt) Imposed displacement in the global x direction
        :param uy: (flt) Imposed displacement in the global y direction
        :param phi_z: (flt) Imposed rotation about the global z-axis in radians
        """
        self.c += 1
        self.spec[f"imposed_displacement-{self.c}"] = {
            "node_id": node_id,
            "ux": ux,
            "uy": uy,
            "phi_z": phi_z,
        }

    def thermal_load(
        self,
        element_id: Union[int, Sequence[int]],
        delta_t: Union[float, Sequence[float]],
        alpha: Union[float, Sequence[float]],
        delta_t_gradient: Optional[Union[float, Sequence[float]]] = None,
        h: Optional[Union[float, Sequence[float]]] = None,
    ) -> None:
        """
        Apply a thermal load to an element.

        :param element_id: (int/ list) representing the element ID
        :param delta_t: (flt/ list) Uniform temperature change of the element
        :param alpha: (flt/ list) Coefficient of thermal expansion of the element material
        :param delta_t_gradient: (flt/ list) Temperature difference over the section depth.
                                 Requires `h`.
        :param h: (flt/ list) Section depth over which `delta_t_gradient` acts
        """
        self.c += 1
        self.spec[f"thermal_load-{self.c}"] = {
            "element_id": element_id,
            "delta_t": delta_t,
            "alpha": alpha,
            "delta_t_gradient": delta_t_gradient,
            "h": h,
        }

    def __str__(self) -> str:
        return f"Loadcase {self.name}:\n" + pprint.pformat(self.spec)


def _clean(value: float) -> float:
    """Zero for a value at rounding level (the decomposition of a load in an angle)"""
    return 0.0 if abs(value) < 1e-12 else value


def _add_loads(
    target: "SystemElements", case: "SystemElements", base: "SystemElements"
) -> None:
    """Add the loads that a load case adds to the base system to the loads of `target`

    The loads of a load case replace the loads of the base system at the same node or
    element, so the added loads are those of the case minus those of the base system. The
    values are the stored ones, which contain the load factor of the case and the
    orientation of the y-axis.

    Args:
        target (SystemElements): System with the loads of the earlier load cases
        case (SystemElements): Copy of the base system with the load case applied
        base (SystemElements): The base system
    """
    for node_id, (fx, fy) in case.loads_point.items():
        base_fx, base_fy = base.loads_point.get(node_id, (0.0, 0.0))
        target_fx, target_fy = target.loads_point.get(node_id, (0.0, 0.0))
        target.loads_point[node_id] = (
            target_fx + fx - base_fx,
            target_fy + fy - base_fy,
        )

    for node_id, moment in case.loads_moment.items():
        target.loads_moment[node_id] = (
            target.loads_moment.get(node_id, 0.0)
            + moment
            - base.loads_moment.get(node_id, 0.0)
        )

    for element_id, element in case.element_map.items():
        base_element = base.element_map[element_id]
        target_element = target.element_map[element_id]

        # a dead load of the case is added to the self-weight of the element
        added_dead_load = element.dead_load - base_element.dead_load
        if added_dead_load != 0:
            target_element.dead_load += added_dead_load
            target.loads_dead_load.add(element_id)

        # distributed loads, perpendicular and parallel to the element; the self-weight
        # is part of both properties, so it is counted once, in the base system
        # (a dead load of the case was added above, so it is not part of the added q-loads)
        dead_perp = added_dead_load * math.cos(target_element.angle)
        dead_par = -added_dead_load * math.sin(target_element.angle)
        added_perp = [
            _clean(a - b - dead_perp)
            for a, b in zip(element.all_qp_load, base_element.all_qp_load)
        ]
        added_par = [
            _clean(a - b - dead_par)
            for a, b in zip(element.all_qn_load, base_element.all_qn_load)
        ]
        if any(value != 0 for value in added_perp + added_par):
            if target_element.q_load == (0.0, 0.0) and target_element.q_perp_load == (
                0.0,
                0.0,
            ):
                # the first distributed load on this element: keep its decomposition
                target_element.q_load = element.q_load
                target_element.q_perp_load = element.q_perp_load
                target_element.q_angle = element.q_angle
                target_element.q_direction = element.q_direction
            else:
                angle = target_element.angle
                dead_load = target_element.dead_load
                total_perp = [
                    a + b for a, b in zip(target_element.all_qp_load, added_perp)
                ]
                total_par = [
                    a + b for a, b in zip(target_element.all_qn_load, added_par)
                ]
                # the sum, as a load in the direction of the element:
                # all_qp_load = q_perp + g * cos, all_qn_load = -q - g * sin
                target_element.q_angle = angle
                target_element.q_direction = "angle"
                # (values at rounding level, from the decomposition in an angle, are zero)
                target_element.q_perp_load = (
                    _clean(total_perp[0] - dead_load * math.cos(angle)),
                    _clean(total_perp[1] - dead_load * math.cos(angle)),
                )
                target_element.q_load = (
                    _clean(-(total_par[0] + dead_load * math.sin(angle))),
                    _clean(-(total_par[1] + dead_load * math.sin(angle))),
                )
            target.loads_q[element_id] = case.loads_q.get(
                element_id, target.loads_q.get(element_id, [(0.0, 0.0), (0.0, 0.0)])
            )
            target.plotter.max_q = max(target.plotter.max_q, case.plotter.max_q)

    for element_id, (strain, curvature) in case.loads_temperature.items():
        base_strain, base_curvature = base.loads_temperature.get(element_id, (0.0, 0.0))
        target_strain, target_curvature = target.loads_temperature.get(
            element_id, (0.0, 0.0)
        )
        temperature_load = (
            target_strain + strain - base_strain,
            target_curvature + curvature - base_curvature,
        )
        target.loads_temperature[element_id] = temperature_load
        target.element_map[element_id].temperature_load = temperature_load

    for node_id, displacements in case.loads_displacement.items():
        base_displacements = base.loads_displacement.get(node_id, (None, None, None))
        target_displacements = target.loads_displacement.get(
            node_id, (None, None, None)
        )
        combined = []
        for value, base_value, target_value in zip(
            displacements, base_displacements, target_displacements
        ):
            if value is None:
                combined.append(target_value)
            else:
                combined.append((target_value or 0.0) + value - (base_value or 0.0))
        target.loads_displacement[node_id] = (combined[0], combined[1], combined[2])


class LoadCombination:
    def __init__(self, name: str):
        self.name: str = name
        self.spec: dict = {}

    def add_load_case(
        self,
        lc: Union[LoadCase, Sequence[LoadCase]],
        factor: Union[float, Sequence[float]],
    ) -> None:
        """
        Add a load case to the load combination.

        :param lc: (:class:`anastruct.fem.util.LoadCase`)
        :param factor: (flt) Multiply all the loads in this LoadCase with this factor.
        """
        if isinstance(lc, LoadCase):
            n = 1
        else:
            n = len(lc)
        lc = arg_to_list(lc, n)
        factor = arg_to_list(factor, n)
        for i, lci in enumerate(lc):
            self.spec[lci.name] = [lci, factor[i]]

    def solve(
        self,
        system: "SystemElements",
        force_linear: bool = False,
        verbosity: int = 0,
        max_iter: int = 200,
        geometrical_non_linear: bool = False,
        **kwargs: Any,
    ) -> Dict[str, "SystemElements"]:
        """
        Evaluate the Load Combination.

        :param system: (:class:`anastruct.fem.system.SystemElements`) Structure to apply loads on.
        :param force_linear: (bool) Force a linear calculation. Even when the system has non
                             linear nodes.
        :param verbosity: (int) 0: Log calculation outputs. 1: silence.
        :param max_iter: (int) Maximum allowed iterations.
        :param geometrical_non_linear: (bool) Calculate second order effects and determine the
                                       buckling factor.
        :return: (ResultObject)

        The results of the load cases are the results of every load case on its own; the
        loads of the system itself are part of each of them, and a load of the case at the same
        node or element replaces them. The 'combination' is the result of one calculation of the
        system with the loads of all the load cases, each multiplied by its factor, added to the
        loads of the system itself, which count once. This is the same as the sum of
        the load cases for a linear calculation, and the only correct one if the calculation is
        not linear (second order effects, or non linear nodes without `force_linear`): the
        combination then has its own buckling factor, for the combined loads.

        Development **kwargs (passed to the solve of every load case and of the combination):
            :param naked: (bool) Whether or not to run the solve function without doing
                          post processing.
            :param discretize_kwargs: When doing a geometric non linear analysis you can reduce or
                                      increase the number of elements created that are used for
                                      determining the buckling_factor
        """

        results = {}
        for lc, factor in self.spec.values():
            ss = copy.deepcopy(system)
            ss._invalidate_solution()  # also when the system has been solved before

            ss.load_factor = factor
            ss.apply_load_case(lc)
            ss.solve(
                force_linear, verbosity, max_iter, geometrical_non_linear, **kwargs
            )
            results[lc.name] = ss

        # the combination is one calculation with the loads of all the load cases: the
        # results of non linear calculations can not be superposed, and for a linear one it
        # is the same, while the loads of the system itself count once
        ss_combined = copy.deepcopy(system)
        ss_combined._invalidate_solution()
        # every load case is applied to the system without its loads, so that the loads of the
        # case are added to the loads of the system (in the result of a load case on its own
        # they replace the loads of the system at the same node or element)
        unloaded = copy.deepcopy(system)
        unloaded._invalidate_solution()
        unloaded.remove_loads(dead_load=True)
        for lc, factor in self.spec.values():
            ss = copy.deepcopy(unloaded)
            ss.load_factor = factor
            ss.apply_load_case(lc)
            _add_loads(ss_combined, ss, unloaded)
        ss_combined.solve(
            force_linear, verbosity, max_iter, geometrical_non_linear, **kwargs
        )
        results["combination"] = ss_combined
        return results
