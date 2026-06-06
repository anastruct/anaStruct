from typing import TYPE_CHECKING, Optional, Sequence, Tuple, Union

import numpy as np

from anastruct.basic import FEMException, angle_x_axis
from anastruct.fem.node import Node
from anastruct.vertex import Vertex

if TYPE_CHECKING:
    from anastruct._types import VertexLike
    from anastruct.fem.system import MpType, Spring, SystemElements


def check_internal_hinges(system: "SystemElements", node_id: int) -> None:
    """Identify internal hinges, set their hinge status

    Args:
        system (SystemElements): System in which the node is located
        node_id (int): Id of the node in the system
    """

    node = system.node_map[node_id]

    hinges = []
    for el_id in node.elements:
        el = system.element_map[el_id]
        assert el.springs is not None
        if node_id == el.node_id1 and 1 in el.springs and el.springs[1] == 0:
            hinges.append(1)
        elif node_id == el.node_id2 and 2 in el.springs and el.springs[2] == 0:
            hinges.append(1)
        else:
            hinges.append(0)

    # If at least one element is connected
    # and no more than one element is rigidly connected
    if (1 < len(hinges) <= 1 + sum(hinges)) or (len(hinges) == 1 and sum(hinges) == 1):
        system.internal_hinges.append(node)

    if node in system.internal_hinges:
        node.hinge = True

        # If the elements aren't already all hinged, then set them to be
        if len(hinges) - sum(hinges) >= 1:
            for el_id in node.elements:
                el = system.element_map[el_id]
                assert el.springs is not None
                if node_id == el.node_id1 and not (
                    1 in el.springs and el.springs[1] == 0
                ):
                    el.springs.update({1: 0})
                if node_id == el.node_id2 and not (
                    2 in el.springs and el.springs[2] == 0
                ):
                    el.springs.update({2: 0})
                system.element_map[el_id].springs = el.springs


def append_node_id(
    system: "SystemElements",
    point_1: Vertex,
    point_2: Vertex,
    node_id1: int,
    node_id2: int,
) -> None:
    """Append the node id to the system if it is not already in the system

    Args:
        system (SystemElements): System in which the node is located
        point_1 (Vertex): Vertex of the first point
        point_2 (Vertex): Vertex of the second point
        node_id1 (int): Node id of the first point
        node_id2 (int): Node id of the second point
    """
    if node_id1 not in system.node_map:
        system.node_map[node_id1] = Node(node_id1, vertex=point_1)
    if node_id2 not in system.node_map:
        system.node_map[node_id2] = Node(node_id2, vertex=point_2)


def remove_node_id(
    system: "SystemElements",
    node_id: int,
) -> None:
    """Remove the node id from the system

    Note that this function does NOT check if the node is still used in the system. It is
    intended to be used as a helper function to `remove_element()` and others, not to be
    used directly by users.

    Args:
        system (SystemElements): System in which the node is located
        node_id (int): Node id of the node
    """
    node = system.node_map[node_id]
    system._vertices.pop(system.node_map[node_id].vertex)
    system.node_map.pop(node_id)
    if node_id in system.loads_point:
        system.loads_point.pop(node_id)
    if node_id in system.loads_moment:
        system.loads_moment.pop(node_id)
    if node in system.supports_fixed:
        system.supports_fixed.remove(node)
    if node in system.supports_hinged:
        system.supports_hinged.remove(node)
    if node in system.supports_rotational:
        system.supports_rotational.remove(node)
    if node in system.internal_hinges:
        system.internal_hinges.remove(node)
    if node in system.supports_roll:
        ind = system.supports_roll.index(node)
        system.supports_roll.remove(node)
        system.supports_roll_direction.pop(ind)
        system.supports_roll_rotate.pop(ind)
    if node_id in system.inclined_roll:
        system.inclined_roll.pop(node_id)
    if node in [item[0] for item in system.supports_spring_x]:
        ind = [item[0] for item in system.supports_spring_x].index(node)
        system.supports_spring_x.pop(ind)
    if node in [item[0] for item in system.supports_spring_y]:
        ind = [item[0] for item in system.supports_spring_y].index(node)
        system.supports_spring_y.pop(ind)
    if node in [item[0] for item in system.supports_spring_z]:
        ind = [item[0] for item in system.supports_spring_z].index(node)
        system.supports_spring_z.pop(ind)
    if node_id in [item[0] for item in system.supports_spring_args]:
        ind = [item[0] for item in system.supports_spring_args].index(node_id)
        system.supports_spring_args.pop(ind)


def det_vertices(
    system: "SystemElements",
    location_list: Union[
        "VertexLike",
        Sequence["VertexLike"],
    ],
) -> Tuple[Vertex, Vertex]:
    """Determine the vertices of a location list

    Args:
        system (SystemElements): System in which the nodes are located
        location_list (Union[ Vertex, Sequence[Vertex], Sequence[int], Sequence[float], Sequence[Sequence[float]], ]):
            List of one or two locations

    Raises:
        FEMException: Raised when the location list is not a list of two points, a list of
            two coordinates, or a list of two lists of two coordinates.

    Returns:
        Tuple[Vertex, Vertex]: Tuple of two vertices
    """
    if isinstance(location_list, Vertex):
        point_1 = system._previous_point
        point_2 = location_list
    elif isinstance(location_list, Sequence) and len(location_list) == 1:
        point_1 = system._previous_point
        point_2 = Vertex(location_list[0])
    elif (
        isinstance(location_list, Sequence)
        and len(location_list) == 2
        and isinstance(location_list[0], (float, int, np.number))
        and isinstance(location_list[1], (float, int, np.number))
    ):
        point_1 = system._previous_point
        point_2 = Vertex(location_list[0], location_list[1])
    elif isinstance(location_list, Sequence) and len(location_list) == 2:
        point_1 = Vertex(location_list[0])
        point_2 = Vertex(location_list[1])
    else:
        raise FEMException(
            "Flawed inputs",
            "The location_list should be a list of two points, a list of two coordinates, "
            + "or a list of two lists of two coordinates.",
        )
    system._previous_point = point_2

    return point_1, point_2


def det_node_ids(
    system: "SystemElements", point_1: Vertex, point_2: Vertex
) -> Tuple[int, int]:
    """Determine the node ids of two points

    Args:
        system (SystemElements): System in which the nodes are located
        point_1 (Vertex): First point
        point_2 (Vertex): Second point

    Returns:
        Tuple[int, int]: Tuple of two node ids
    """
    node_ids = []
    for p in (point_1, point_2):
        # k = str(p)
        k = p
        if k in system._vertices:
            node_id = system._vertices[k]
        else:
            node_id = len(system._vertices) + 1
            system._vertices[k] = node_id
        node_ids.append(node_id)
    return node_ids[0], node_ids[1]


def add_node(
    system: "SystemElements", point: Vertex, node_id: Optional[int] = None
) -> int:
    """Add a node, optionally with a specific ID, without adding an element

    Args:
        system (SystemElements): System in which the nodes are located
        point (Vertex): Location of the node
        node_id (Optional[int], optional): node_id to assign to the node. Defaults to None,
            which means to use the first available node_id automatically.

    Raises:
        FEMException: Raised when the location is already assigned to a different node id.
        FEMException: Raised when the node id is already assigned to a different location.

    Returns:
        int: The node id of the added (or existing) node
    """
    if point in system._vertices:
        if node_id is not None:
            existing_node_id = system._vertices[point]
            if existing_node_id != node_id:
                raise FEMException(
                    "Flawed inputs",
                    f"Location {point} is already assigned to node id {existing_node_id}, "
                    f"cannot assign to node id {node_id}.",
                )
        return existing_node_id

    if node_id is None:
        node_id = max(system.node_map.keys(), default=0) + 1
    elif node_id in system.node_map and system.node_map[node_id].vertex != point:
        raise FEMException(
            "Flawed inputs",
            f"Node id {node_id} is already assigned to a different location.",
        )

    system._vertices[point] = node_id
    system.node_map[node_id] = Node(node_id, vertex=point)
    return node_id


def support_check(system: "SystemElements", node_id: int) -> None:
    """Check if the node is a hinge

    Args:
        system (SystemElements): System in which the node is located
        node_id (int): Node id of the node

    Raises:
        FEMException: Raised when the node is a hinge
    """
    if system.node_map[node_id].hinge:
        raise FEMException(
            "Flawed inputs",
            "You cannot add a rotation-restraining support to a hinged node.",
        )


def force_elements_orientation(
    point_1: Vertex,
    point_2: Vertex,
    node_id1: int,
    node_id2: int,
    spring: Optional["Spring"],
    mp: Optional["MpType"],
) -> Tuple[
    Vertex,
    Vertex,
    int,
    int,
    Optional["Spring"],
    Optional["MpType"],
    float,
]:
    """Force the elements to be in the first and the last quadrant of the unity circle.
    Meaning the first node is always left and the last node is always right. Or they
    are both on one vertical line.

    The angle of the element will thus always be between -90 till +90 degrees.

    Args:
        point_1 (Vertex): First point of the element
        point_2 (Vertex): Second point of the element
        node_id1 (int): Node id of the first point
        node_id2 (int): Node id of the second point
        spring (Optional[Spring]): Any spring releases of the element
        mp (Optional[MpType]): Any maximum plastic moments of the element

    Returns:
        Tuple[ Vertex, Vertex, int, int, Optional[Spring], Optional[MpType], float ]:
            Tuple of the first point, second point, first node id, second node id, spring releases,
            maximum plastic moments, and the angle of the element
    """
    # determine the angle of the element with the global x-axis
    delta_x = point_2.x - point_1.x
    delta_y = point_2.y - point_1.y  # minus sign to work with an opposite y-axis
    angle = angle_x_axis(delta_x, delta_y)

    if delta_x < 0:
        # switch points
        point_1, point_2 = point_2, point_1
        node_id1, node_id2 = node_id2, node_id1

        angle = angle_x_axis(-delta_x, -delta_y)

        if spring is not None:
            assert isinstance(
                spring, dict
            ), "The spring parameter should be a dictionary."
            if 1 in spring and 2 in spring:
                spring[1], spring[2] = spring[2], spring[1]
            elif 1 in spring:
                spring[2] = spring.pop(1)
            elif 2 in spring:
                spring[1] = spring.pop(2)

        if mp is not None:
            if 1 in mp and 2 in mp:
                mp[1], mp[2] = mp[2], mp[1]
            elif 1 in mp:
                mp[2] = mp.pop(1)
            elif 2 in mp:
                mp[1] = mp.pop(2)
    return point_1, point_2, node_id1, node_id2, spring, mp, angle
