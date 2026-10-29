from typing import TYPE_CHECKING, Literal, Optional, Sequence, Tuple, Union

import numpy as np

from anastruct.basic import FEMException, angle_x_axis
from anastruct.fem.node import Node
from anastruct.vertex import Vertex

if TYPE_CHECKING:
    from anastruct._types import VertexLike
    from anastruct.fem.system import MpType, Spring, SystemElements


def dof_index(system: "SystemElements", node_id: int) -> int:
    """Index of the first degree of freedom of a node in the system matrix and vectors

    The nodes are numbered 1, 2, ... in the order they are added, but removing an element can
    remove a node and leave a gap in the ids. The degrees of freedom of the remaining nodes
    follow each other in the order of their ids, three per node.

    Args:
        system (SystemElements): System in which the node is located
        node_id (int): Id of the node

    Returns:
        int: Index of the displacement in x direction of the node (the next two are y and the
            rotation)
    """
    if system._node_index is None:
        system._node_index = {
            id_: position for position, id_ in enumerate(sorted(system.node_map))
        }
    return system._node_index[node_id] * 3


def remove_internal_hinge(system: "SystemElements", node_id: int) -> None:
    """Remove a node from the internal hinges

    Args:
        system (SystemElements): System in which the node is located
        node_id (int): Id of the node
    """
    system.internal_hinges.pop(system._internal_hinge_index.pop(node_id))
    system._internal_hinge_index = {
        hinge.id: position for position, hinge in enumerate(system.internal_hinges)
    }


def check_internal_hinges(system: "SystemElements", node_id: int) -> None:
    """Identify internal hinges, set their hinge status

    A node is an internal hinge when it is declared as one (`add_internal_hinge`), or
    when every element that meets in the node has a hinge (`{1: 0}` or `{2: 0}`) at that
    node, so nothing can transfer a moment. The rotation of the node is then
    restrained and all the elements are released at the node.

    If one or more elements are rigidly connected in the node, they share the rotation
    of the node, which is solved like any other degree of freedom (a moment load or a
    rotational support at the node is carried by the rigid elements), and only the
    elements with a hinge are released, through their own springs. A node with one
    rigid element and released elements therefore behaves as a hinge for the rigid
    element too, as long as the node carries no moment load or rotational support.

    The status follows from the springs as given by the user, which are never
    modified here, and it is determined again every time, so it does not depend on
    the order in which the elements were added.

    Args:
        system (SystemElements): System in which the node is located
        node_id (int): Id of the node in the system
    """

    node = system.node_map[node_id]

    # the elements per node of the system (node.elements is emptied by the
    # post-processing, which replaces the node objects)
    released = []
    for el in system.node_element_map.get(node_id, []):
        end: Literal[1, 2] = 1 if node_id == el.node_id1 else 2
        released.append((el.springs or {}).get(end) == 0)

    is_hinge = node_id in system._explicit_internal_hinges or (
        len(released) > 0 and all(released)
    )

    # match by node id: the node objects in the node map are replaced by the
    # post-processing, so an object from an earlier solve is not the current one
    index = system._internal_hinge_index
    if is_hinge:
        if node_id in index:
            system.internal_hinges[index[node_id]] = node
        else:
            index[node_id] = len(system.internal_hinges)
            system.internal_hinges.append(node)
    elif node_id in index:
        remove_internal_hinge(system, node_id)
    node.hinge = is_hinge


def describe_unrestrained_dof(
    system: "SystemElements", node_id: int, direction: int
) -> str:
    """Explain why an imposed displacement of a node can not be applied

    A displacement can only be imposed in a direction that a support restrains: in a free
    direction the node moves as the structure requires, so its displacement is a result, not
    a condition. This is the case for the free direction of a roller (sliding support), the
    rotation of a hinged support, a spring support (which yields), and above all for an
    inclined roller, which restrains only the translation normal to its free direction while
    the imposed ux and uy are in the global directions.

    Args:
        system (SystemElements): System in which the node is located
        node_id (int): Id of the node
        direction (int): The degree of freedom: 1 = ux, 2 = uy, 3 = phi_z

    Returns:
        str: The explanation, with what the support at the node does restrain
    """
    names = {1: "ux", 2: "uy", 3: "phi_z"}
    translations = {1: "x", 2: "y"}
    name = names[direction]
    message = f"An imposed {name} at node {node_id} is not possible: "

    if node_id in system.inclined_roll:
        # the free direction has the angle `angle`, the restrained one is normal to it
        angle = -float(np.degrees(system.inclined_roll[node_id]))
        return (
            message
            + f"the node has an inclined roller support. It is free to move along the "
            f"direction of {angle:g} degrees and restrains only the translation in the "
            f"direction of {angle + 90:g} degrees, normal to it. A displacement can only be "
            "imposed in a direction that a support restrains; the global ux and uy have a "
            "component along the free direction, which can not be imposed (the node moves "
            "there as the structure requires)."
        )

    for node, direction_i, rotate in zip(
        system.supports_roll,
        system.supports_roll_direction,
        system.supports_roll_rotate,
    ):
        if node.id == node_id:
            free = translations[1 if direction_i == 2 else 2]
            return (
                message
                + f"the node has a roller support that is free to move in the {free} "
                f"direction (it restrains the {translations[direction_i]} translation"
                + (" and the rotation" if not rotate else "")
                + "). A displacement can only be imposed in a direction that a support "
                "restrains."
            )

    if any(node.id == node_id for node in system.supports_hinged):
        return (
            message
            + "the node has a hinged support, which restrains the translations but not "
            "the rotation. A displacement can only be imposed in a direction that a "
            "support restrains."
        )

    if any(node.id == node_id for node in system.supports_rotational):
        return (
            message
            + "the node has a rotational support, which restrains only the rotation. A "
            "displacement can only be imposed in a direction that a support restrains."
        )

    for supports in (
        system.supports_spring_x,
        system.supports_spring_y,
        system.supports_spring_z,
    ):
        if any(node.id == node_id for node, _ in supports):
            return (
                message
                + "the node has a spring support, which yields instead of restraining. A "
                "displacement can only be imposed in a direction that a rigid support "
                "restrains."
            )

    return (
        message
        + "no support restrains that degree of freedom of the node. A displacement can "
        "only be imposed in a direction that a support restrains (fixed, hinged, roller or "
        "rotational support), and the support must be added to the system."
    )


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
    system._node_index = None


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
    system._vertices.pop(system.node_map[node_id].vertex)
    system.node_map.pop(node_id)
    system._node_index = None
    if node_id in system.loads_point:
        system.loads_point.pop(node_id)
    if node_id in system.loads_moment:
        system.loads_moment.pop(node_id)
    # (the nodes of the supports are compared by id: after a solve the node objects of the
    # node map are replaced by copies with the results)
    for supports in (
        system.supports_fixed,
        system.supports_hinged,
        system.supports_rotational,
    ):
        supports[:] = [n for n in supports if n.id != node_id]
    if node_id in system._internal_hinge_index:
        remove_internal_hinge(system, node_id)
    system._explicit_internal_hinges.discard(node_id)
    for ind in reversed(
        [i for i, n in enumerate(system.supports_roll) if n.id == node_id]
    ):
        system.supports_roll.pop(ind)
        system.supports_roll_direction.pop(ind)
        system.supports_roll_rotate.pop(ind)
    if node_id in system.inclined_roll:
        system.inclined_roll.pop(node_id)
    for springs in (
        system.supports_spring_x,
        system.supports_spring_y,
        system.supports_spring_z,
    ):
        springs[:] = [item for item in springs if item[0].id != node_id]
    system.supports_spring_args[:] = [
        item for item in system.supports_spring_args if item[0] != node_id
    ]
    for key in [key for key in system.system_spring_map if key[0] == node_id]:
        system.system_spring_map.pop(key)
    system.loads_displacement.pop(node_id, None)


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
            # (the ids have gaps after remove_element, so the count would reuse an id)
            node_id = (
                max(
                    max(system.node_map, default=0),
                    max(system._vertices.values(), default=0),
                )
                + 1
            )
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
        existing_node_id = system._vertices[point]
        if node_id is not None:
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
    system._node_index = None
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
            # do not modify the dictionary of the caller, it may be shared
            spring = dict(spring)
            if 1 in spring and 2 in spring:
                spring[1], spring[2] = spring[2], spring[1]
            elif 1 in spring:
                spring[2] = spring.pop(1)
            elif 2 in spring:
                spring[1] = spring.pop(2)

        if mp is not None:
            mp = dict(mp)
            if 1 in mp and 2 in mp:
                mp[1], mp[2] = mp[2], mp[1]
            elif 1 in mp:
                mp[2] = mp.pop(1)
            elif 2 in mp:
                mp[1] = mp.pop(2)
    return point_1, point_2, node_id1, node_id2, spring, mp, angle
