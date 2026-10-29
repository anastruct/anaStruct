import math
from typing import TYPE_CHECKING, List, Tuple

import numpy as np

from anastruct.basic import FEMException
from anastruct.fem.system_components.util import (
    describe_unrestrained_dof,
    dof_index,
)
from anastruct.fem.elements import det_axial, det_moment, det_shear

if TYPE_CHECKING:
    from anastruct._types import AxisNumber
    from anastruct.fem.elements import Element
    from anastruct.fem.system import SystemElements


def set_force_vector(
    system: "SystemElements", force_list: List[Tuple[int, "AxisNumber", float]]
) -> np.ndarray:
    """Set a force vector on a node within the system

    Args:
        system (SystemElements): System to which the force vector is applied
        force_list (List[Tuple[int, AxisNumber, float]]): List of tuples containing
            the node id, direction (1 = x, 2 = y, 3 = z), and force magnitude

    Returns:
        np.ndarray: System force vector with the applied forces on the nodes
    """
    assert system.system_force_vector is not None
    for id_, direction, force in force_list:
        system.system_force_vector[dof_index(system, id_) + direction - 1] += force

    return system.system_force_vector


def prep_matrix_forces(system: "SystemElements") -> None:
    """Prepare the system force vector for the matrix assembly

    Args:
        system (SystemElements): System to which the force vector is applied
    """
    system.system_force_vector = np.zeros(len(system.node_map) * 3)
    apply_perpendicular_q_load(system)
    apply_parallel_qn_load(system)
    apply_point_load(system)
    apply_moment_load(system)
    apply_temperature_load(system)


def apply_temperature_load(system: "SystemElements") -> None:
    """Apply a temperature load to the system

    The equivalent nodal forces are derived from the free thermal eigen strains of
    the element: e0 = [alpha * delta_t * l, -kappa * l / 2, -kappa * l / 2], with
    kappa = alpha * delta_t_gradient / h the free thermal curvature. The sign of the
    curvature follows from the kinematic matrix: the relative end rotations are positive
    for a curvature that bends the element towards its right-hand side (looking from
    its first to its second node), which is the side of the colder face. A positive
    gradient is a warmer face on the left (the upper face of a horizontal element), so
    the free end of a cantilever bends towards the colder face. The fixed-end
    forces are obtained with the element's own kinematic and condensed constitutive
    matrices, so (internal) hinges and springs at the element ends are accounted for
    automatically.

    Args:
        system (SystemElements): System to which the temperature load is applied
    """
    for element_id, (strain, curvature) in system.loads_temperature.items():
        element = system.element_map[element_id]
        if strain == 0 and curvature == 0:
            continue

        e0 = np.array(
            [strain * element.l, -curvature * element.l / 2, -curvature * element.l / 2]
        )
        if element.type == "truss":
            e0[1] = e0[2] = 0.0

        primary_force = element.kinematic_matrix.T @ element.constitutive_matrix @ e0

        element.element_primary_force_vector -= primary_force

        # Set force vector
        assert system.system_force_vector is not None
        for node_id, part in (
            (element.node_id1, primary_force[0:3]),
            (element.node_id2, primary_force[3:]),
        ):
            start = dof_index(system, node_id)
            system.system_force_vector[start : start + 3] += part


def apply_moment_load(system: "SystemElements") -> None:
    """Apply a moment load to the system

    Args:
        system (SystemElements): System to which the moment load is applied
    """
    for node_id, Tz in system.loads_moment.items():
        if Tz != 0 and system.node_map[node_id].hinge:
            raise FEMException(
                "Invalid parameters",
                f"A moment load at node {node_id} can not be carried: the node is an internal "
                "hinge (no element transfers a moment there). Place the moment load at a node "
                "where an element is rigidly connected.",
            )
        if Tz != 0 and all(
            element.type == "truss" for element in system.node_element_map[node_id]
        ):
            raise FEMException(
                "Invalid parameters",
                f"A moment load at node {node_id} can not be carried: only truss elements, "
                "which have no bending stiffness, are connected to the node.",
            )
        set_force_vector(system, [(node_id, 3, Tz)])


def apply_point_load(system: "SystemElements") -> None:
    """Apply a point load to the system

    Args:
        system (SystemElements): System to which the point load is applied
    """
    for node_id in system.loads_point:
        Fx, Fy = system.loads_point[node_id]
        if node_id in system.inclined_roll:
            # The degrees of freedom of a node with an inclined roller support are in the
            # system of the support, rotated by delta with respect to the global system
            # (u_global = R(delta) u_support, see the kinematic matrix), so the load
            # is given in that system as well: f_support = R(delta)^T f_global.
            delta = system.inclined_roll[node_id]
            cos_d, sin_d = math.cos(delta), math.sin(delta)
            Fx, Fy = cos_d * Fx + sin_d * Fy, -sin_d * Fx + cos_d * Fy
        # system force vector.
        set_force_vector(
            system,
            [
                (node_id, 1, Fx),
                (node_id, 2, Fy),
            ],
        )


def apply_perpendicular_q_load(system: "SystemElements") -> None:
    """Apply a perpendicular q load to the system

    Args:
        system (SystemElements): System to which the perpendicular q load is applied
    """
    for element_id in system.loads_dead_load:
        element = system.element_map[element_id]
        qi_perpendicular = element.all_qp_load[0]
        q_perpendicular = element.all_qp_load[1]
        if q_perpendicular == 0 and qi_perpendicular == 0:
            continue

        kl, kr = element.end_rotational_stiffness

        # minus because of systems positive rotation
        left_moment = det_moment(
            kl, kr, qi_perpendicular, q_perpendicular, 0, element.EI, element.l
        )
        right_moment = -det_moment(
            kl, kr, qi_perpendicular, q_perpendicular, element.l, element.EI, element.l
        )
        rleft = det_shear(
            kl, kr, qi_perpendicular, q_perpendicular, 0, element.EI, element.l
        )
        rright = -det_shear(
            kl, kr, qi_perpendicular, q_perpendicular, element.l, element.EI, element.l
        )

        rleft_x = rleft * math.sin(element.a1)
        rright_x = rright * math.sin(element.a2)

        rleft_y = rleft * math.cos(element.a1)
        rright_y = rright * math.cos(element.a2)

        if element.type == "truss":
            left_moment = 0
            right_moment = 0

        primary_force = np.array(
            [rleft_x, rleft_y, left_moment, rright_x, rright_y, right_moment]
        )
        element.element_primary_force_vector -= primary_force

        # Set force vector
        assert system.system_force_vector is not None
        for node_id, part in (
            (element.node_id1, primary_force[0:3]),
            (element.node_id2, primary_force[3:]),
        ):
            start = dof_index(system, node_id)
            system.system_force_vector[start : start + 3] += part


def apply_parallel_qn_load(system: "SystemElements") -> None:
    """Apply a parallel qn load to the system

    Args:
        system (SystemElements): System to which the parallel qn load is applied
    """
    for element_id in system.loads_dead_load:
        element = system.element_map[element_id]
        qni_parallel = element.all_qn_load[0]
        qn_parallel = element.all_qn_load[1]
        if qn_parallel == 0 and qni_parallel == 0:
            continue

        # minus because of systems positive rotation
        rleft = -det_axial(qni_parallel, qn_parallel, 0, element.EA, element.l)
        rright = det_axial(qni_parallel, qn_parallel, element.l, element.EA, element.l)

        rleft_x = -rleft * math.cos(element.a1)
        rright_x = -rright * math.cos(element.a2)

        rleft_y = rleft * math.sin(element.a1)
        rright_y = rright * math.sin(element.a2)

        element.element_primary_force_vector[0] -= rleft_x
        element.element_primary_force_vector[1] -= rleft_y
        element.element_primary_force_vector[3] -= rright_x
        element.element_primary_force_vector[4] -= rright_y

        set_force_vector(
            system,
            [
                (element.node_1.id, 2, rleft_y),
                (element.node_2.id, 2, rright_y),
                (element.node_1.id, 1, rleft_x),
                (element.node_2.id, 1, rright_x),
            ],
        )


def dead_load(system: "SystemElements", g: float, element_id: int) -> None:
    """Apply a dead load self-weight to an element in the system

    Args:
        system (SystemElements): System to which the dead load is applied
        g (float): Magnitude of the dead load self-weight
        element_id (int): Element id to which the dead load is applied
    """
    system.loads_dead_load.add(element_id)
    system.element_map[element_id].dead_load = g


def assemble_system_matrix(
    system: "SystemElements", validate: bool = False, geometric_matrix: bool = False
) -> None:
    """Assemble the system matrix
    Shape of the matrix = n nodes * n d.o.f. = n * 3

    Args:
        system (SystemElements): System to be prepared
        validate (bool, optional): Whether or not to validate the system. Defaults to False.
        geometric_matrix (bool, optional): Whether or not to include the current geometric matrix. Defaults to False.
    """
    system._remainder_indexes = []
    if not geometric_matrix:
        shape = len(system.node_map) * 3
        system.shape_system_matrix = shape
        system.system_matrix = np.zeros((shape, shape))

    assert system.system_matrix is not None
    for (node_id, translation), K in system.system_spring_map.items():
        #  first index is row, second is column
        matrix_index = dof_index(system, node_id) + translation - 1
        system.system_matrix[matrix_index][matrix_index] += K

    # Determine the elements location in the stiffness matrix.
    # system matrix [K]
    #
    # [fx 1] [K        |  \ node 1 starts at row 1
    # |Fy 1] | K       |  /
    # |Tz 1] |  K      | /
    # |fx 2] |   K     |  \ node 2 starts at row 4
    # |Fy 2] |    K    |  /
    # |Tz 2] |     K   | /
    # |fx 3] |      K  |  \ node 3 starts at row 7
    # |Fy 3] |       K |  /
    # [Tz 3] [        K] /
    #
    #         n   n  n
    #         o   o  o
    #         d   d  d
    #         e   e  e
    #         1   2  3
    #
    # thus with appending numbers in the system matrix: column = row

    # Iterate over the actual elements rather than assuming the element ids are
    # a contiguous 1..N sequence: remove_element (e.g. via insert_node) can leave
    # gaps in element_map, and assembly is an order-independent accumulation into
    # the system matrix indexed by node id (see #308).
    for element in system.element_map.values():
        element_matrix = element.stiffness_matrix

        # n1 and n2 are starting indexes of the rows and the columns for node 1 and node 2
        n1 = dof_index(system, element.node_1.id)
        n2 = dof_index(system, element.node_2.id)
        system.system_matrix[n1 : n1 + 3, n1 : n1 + 3] += element_matrix[0:3, :3]
        system.system_matrix[n1 : n1 + 3, n2 : n2 + 3] += element_matrix[0:3, 3:]

        system.system_matrix[n2 : n2 + 3, n1 : n1 + 3] += element_matrix[3:6, :3]
        system.system_matrix[n2 : n2 + 3, n2 : n2 + 3] += element_matrix[3:6, 3:]

    # returns True if symmetrical.
    if validate:
        assert np.allclose((system.system_matrix.transpose()), system.system_matrix)


def set_displacement_vector(
    system: "SystemElements", nodes_list: List[Tuple[int, "AxisNumber"]]
) -> np.ndarray:
    """Set the displacement vector for the system

    Args:
        system (SystemElements): System to which the displacement vector is applied
        nodes_list (List[Tuple[int, AxisNumber]]): List of tuples containing
            the node id and the direction (1 = x, 2 = y, 3 = z)

    Raises:
        IndexError: This often occurs if you set supports before the all the elements are
            modelled. First finish the model.

    Returns:
        np.ndarray: System displacement vector with the applied displacements on the nodes
    """
    if system.system_displacement_vector is None:
        system.system_displacement_vector = np.ones(len(system.node_map) * 3) * np.nan

    for i in nodes_list:
        index = dof_index(system, i[0]) + i[1] - 1

        try:
            system.system_displacement_vector[index] = 0
        except IndexError as e:
            raise IndexError(
                e,
                "This often occurs if you set supports before the all the elements are modelled. "
                "First finish the model.",
            ) from e
    return system.system_displacement_vector


def process_conditions(system: "SystemElements") -> None:
    """Process the conditions of the system

    Args:
        system (SystemElements): System to be processed
    """
    indexes = []
    system._remainder_indexes = []
    # remove the unsolvable values from the matrix and vectors
    assert system.shape_system_matrix is not None
    assert system.system_displacement_vector is not None
    # The restrained degrees of freedom are determined once, in process_supports. The
    # displacement vector can not be used again after a solve: it then holds the
    # solution, and a degree of freedom that is exactly 0 (e.g. the transverse
    # displacements of a column under an axial load) is not a restrained one.
    restrained = system._restrained_dofs
    if restrained is None:
        restrained = system.system_displacement_vector == 0
    for i in range(system.shape_system_matrix):
        if restrained[i] or i in system._prescribed_displacements:
            indexes.append(i)
        else:
            system._remainder_indexes.append(i)

    system.system_displacement_vector = np.delete(
        system.system_displacement_vector, indexes, 0
    )
    assert system.system_force_vector is not None
    assert system.system_matrix is not None
    system.reduced_force_vector = np.delete(system.system_force_vector, indexes, 0)

    # correct the reduced force vector for imposed (non-zero) displacements:
    # K_ff * u_f = F_f - K_fr * u_r
    if system._prescribed_displacements:
        prescribed_indexes = sorted(system._prescribed_displacements.keys())
        prescribed_values = np.array(
            [system._prescribed_displacements[i] for i in prescribed_indexes]
        )
        system.reduced_force_vector -= (
            system.system_matrix[np.ix_(system._remainder_indexes, prescribed_indexes)]
            @ prescribed_values
        )

    system.reduced_system_matrix = np.delete(system.system_matrix, indexes, 0)
    system.reduced_system_matrix = np.delete(system.reduced_system_matrix, indexes, 1)


def process_supports(system: "SystemElements") -> None:
    """Process the supports of the system

    Args:
        system (SystemElements): System to be processed
    """
    if system.system_displacement_vector is None:
        # also a model without any support (which is unstable, but not an error here)
        system.system_displacement_vector = np.ones(len(system.node_map) * 3) * np.nan

    for node in system.supports_hinged:
        set_displacement_vector(system, [(node.id, 1), (node.id, 2)])

    for i, supports_rolli in enumerate(system.supports_roll):
        if not system.supports_roll_rotate[i]:
            set_displacement_vector(
                system,
                [
                    (supports_rolli.id, system.supports_roll_direction[i]),
                    (supports_rolli.id, 3),
                ],
            )
        else:
            set_displacement_vector(
                system,
                [(supports_rolli.id, system.supports_roll_direction[i])],
            )

    for node in system.supports_rotational:
        set_displacement_vector(system, [(node.id, 3)])

    for node in system.internal_hinges:
        set_displacement_vector(system, [(node.id, 3)])
        for el in system.node_element_map[node.id]:
            # the node copies of the post-processing hold the hinge status of an earlier solve
            el.refresh_nodes(system.node_map)
            el.compile_constitutive_matrix()
            el.compile_stiffness_matrix()

    for node in system.supports_fixed:
        set_displacement_vector(system, [(node.id, 1), (node.id, 2), (node.id, 3)])

    for node, roll in system.supports_spring_x:
        if not roll:
            set_displacement_vector(system, [(node.id, 2)])

    for node, roll in system.supports_spring_y:
        if not roll:
            set_displacement_vector(system, [(node.id, 1)])

    for node, roll in system.supports_spring_z:
        if not roll:
            set_displacement_vector(system, [(node.id, 1), (node.id, 2)])

    for node_id, angle in system.inclined_roll.items():
        for el in system.node_element_map[node_id]:
            if el.node_1.id == node_id:
                el.a1 = el.angle + angle
            elif el.node_2.id == node_id:
                el.a2 = el.angle + angle

            el.compile_kinematic_matrix()
            el.compile_stiffness_matrix()

    # imposed (non-zero) displacements and rotations on restrained degrees of freedom
    assert system.system_displacement_vector is not None
    for node_id, (ux, uy, phi_z) in system.loads_displacement.items():
        if node_id in system.inclined_roll:
            raise FEMException(
                "Invalid parameters",
                describe_unrestrained_dof(
                    system, node_id, 1 if ux is not None else 2 if uy is not None else 3
                ),
            )
        if phi_z is not None and any(
            hinge.id == node_id for hinge in system.internal_hinges
        ):
            raise FEMException(
                "Invalid parameters",
                f"A rotation cannot be imposed on internal hinge {node_id}.",
            )
        for direction, displacement in ((1, ux), (2, uy), (3, phi_z)):
            if displacement is None or displacement == 0:
                continue
            index = dof_index(system, node_id) + direction - 1
            if np.isnan(system.system_displacement_vector[index]):
                raise FEMException(
                    "StabilityError",
                    describe_unrestrained_dof(system, node_id, direction),
                )
            system.system_displacement_vector[index] = displacement
            system._prescribed_displacements[index] = displacement

    # the restrained degrees of freedom (value 0, or an imposed displacement)
    system._restrained_dofs = system.system_displacement_vector == 0
    for index in system._prescribed_displacements:
        system._restrained_dofs[index] = True
