import collections.abc
import copy
import math
import re
from typing import (
    TYPE_CHECKING,
    Any,
    Collection,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    overload,
)

import numpy as np

from anastruct.basic import FEMException, arg_to_list
from anastruct.fem import plotter, system_components
from anastruct.fem.elements import Element
from anastruct.fem.postprocess import SystemLevel as post_sl
from anastruct.fem.util.load import LoadCase
from anastruct.sectionbase import properties
from anastruct.vertex import Vertex, vertex_range

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from anastruct._types import (
        AxisNumber,
        Dimension,
        LoadDirection,
        MpType,
        Spring,
        SupportDirection,
        VertexLike,
    )
    from anastruct.fem.node import Node


class SystemElements:
    """
    Modelling any structure starts with an object of this class.

    Attributes:
        EA: Standard axial stiffness of elements, default=15,000
        EI: Standard bending stiffness of elements, default=5,000
        figsize: (tpl) Matplotlibs standard figure size
        element_map: (dict) Keys are the element ids, values are the element objects
        node_map: (dict) Keys are the node ids, values are the node objects.
        node_element_map: (dict) maps node ids to element objects.
        loads_point: (dict) Maps node ids to point loads.
        loads_q: (dict) Maps element ids to q-loads.
        loads_moment: (dict) Maps node ids to moment loads.
        loads_dead_load: (set) Element ids that have a dead load applied.

    Methods:
        add_element: Add a new element to the structure.
        add_element_grid: Add multiple elements based upon sequential x and y coordinates.
        add_sequential_elements: Add multiple elements based upon any number of sequential points.
        add_multiple_elements: Add multiple elements defined by the first and the last point.
        add_truss_element: Add a new truss element (an element that only has axial force) to the structure.
        add_support_hinged: Add a hinged support to a node.
        add_support_fixed: Add a fixed support to a node.
        add_support_roll: Add a roll support to a node.
        add_support_rotational: Add a rotational support to a node.
        add_support_spring: Add a spring support to a node.
        insert_node: Insert a node into an existing structure.
        solve: Compute the results of current model.
        validate: Validate the current model.
    """

    def __init__(
        self,
        figsize: Optional[Tuple[float, float]] = (12, 8),
        EA: float = 15e3,
        EI: float = 5e3,
        load_factor: float = 1.0,
        mesh: int = 50,
        invert_y_loads: bool = True,
    ):
        """Create a new structure

        Args:
            figsize (Tuple[float, float], optional): Figure size to generate. Defaults to (12, 8).
            EA (float, optional): Axial stiffness. Defaults to 15e3.
            EI (float, optional): Bending stiffness. Defaults to 5e3.
            load_factor (float, optional): Load factor by which to multiply all loads. Defaults to 1.0.
            mesh (int, optional): Number of mesh elements, only used for plotting. Defaults to 50.
            invert_y_loads (bool, optional): Whether to invert the y-direction of the loads, such that a positive
                Fy load will be in the direction of gravity. Defaults to True.
        """
        # init object
        self.post_processor = post_sl(self)
        self.plotter = plotter.Plotter(self, mesh)
        self.plot_values = plotter.PlottingValues(self, mesh)

        # standard values if none provided
        self.EA = EA
        self.EI = EI
        self.figsize = figsize
        # whether to invert the y-direction of the loads
        self.orientation_cs = -1 if invert_y_loads else 1

        # structure system
        self.element_map: Dict[int, Element] = (
            {}
        )  # maps element ids to the Element objects.
        self.node_map: Dict[int, Node] = (  # pylint: disable=used-before-assignment
            {}
        )  # maps node ids to the Node objects.
        self.node_element_map: Dict[int, List[Element]] = (
            {}
        )  # maps node ids to Element objects
        # keys matrix index (for both row and columns), values K, are processed
        # assemble_system_matrix
        self.system_spring_map: Dict[int, float] = {}

        # list of indexes that remain after conditions are applied
        self._remainder_indexes: List[int] = []

        # keep track of the nodes of the supports
        self.supports_fixed: List[Node] = []
        self.supports_hinged: List[Node] = []
        self.supports_rotational: List[Node] = []
        self.internal_hinges: List[Node] = []
        self.supports_roll: List[Node] = []
        self.supports_spring_x: List[Tuple[Node, bool]] = []
        self.supports_spring_y: List[Tuple[Node, bool]] = []
        self.supports_spring_z: List[Tuple[Node, bool]] = []
        self.supports_roll_direction: List[Literal[1, 2]] = []
        self.inclined_roll: Dict[int, float] = (
            {}
        )  # map node ids to inclination angle relative to global x-axis.
        self.supports_roll_rotate: List[bool] = []

        # save tuples of the arguments for copying purposes.
        self.supports_spring_args: List[tuple] = []

        # keep track of the loads
        self.loads_point: Dict[int, Tuple[float, float]] = (
            {}
        )  # node ids with a point loads {node_id: (x, y)}
        self.loads_q: Dict[int, List[Tuple[float, float]]] = (
            {}
        )  # element ids with a q-loadad
        self.loads_moment: Dict[int, float] = {}
        self.loads_dead_load: Set[int] = (
            set()
        )  # element ids with q-load due to dead load

        # results
        self.reaction_forces: Dict[int, Node] = {}  # node objects
        self.non_linear = False
        self.non_linear_elements: Dict[int, Dict[Literal[1, 2], float]] = (
            {}
        )  # keys are element ids, values are dicts: {node_index: max moment capacity}
        self.buckling_factor: Optional[float] = None

        # previous point of element
        self._previous_point = Vertex(0, 0)
        self.load_factor = load_factor

        # Objects state
        self.count = 0
        self.system_matrix: Optional[np.ndarray] = None
        self.system_force_vector: Optional[np.ndarray] = None
        self.system_displacement_vector: Optional[np.ndarray] = None
        self.shape_system_matrix: Optional[int] = (
            None  # actually is the size of the square system matrix
        )
        self.reduced_force_vector: Optional[np.ndarray] = None
        self.reduced_system_matrix: Optional[np.ndarray] = None
        self._vertices: Dict[Vertex, int] = {}  # maps vertices to node ids

    @property
    def id_last_element(self) -> int:
        """ID of the last element added to the structure

        Returns:
            int: ID of the last element added to the structure
        """
        return max(self.element_map.keys())

    @property
    def id_last_node(self) -> int:
        """ID of the last node added to the structure

        Returns:
            int: ID of the last node added to the structure
        """
        return max(self.node_map.keys())

    def add_sequential_elements(
        self,
        location: Sequence["VertexLike"],
        EA: Optional[Union[List[float], np.ndarray, float]] = None,
        EI: Optional[Union[List[float], np.ndarray, float]] = None,
        g: Optional[Union[List[float], np.ndarray, float]] = None,
        mp: Optional["MpType"] = None,
        spring: Optional["Spring"] = None,
        **kwargs: dict,
    ) -> None:
        """Add multiple elements based upon any number of sequential points.

        Args:
            location (Sequence[VertexLike]): Sequence of points
                that define the elements.
            EA (Optional[Union[List[float], np.ndarray, float]], optional): Axial stiffnesses. Defaults to None.
            EI (Optional[Union[List[float], np.ndarray, float]], optional): Bending stiffnesses. Defaults to None.
            g (Optional[Union[List[float], np.ndarray, float]], optional): Self-weights. Defaults to None.
            mp (Optional[MpType], optional): Maximum plastic moment capacities for all elements. Defaults to None.
            spring (Optional[Spring], optional): Springs for all elements. Defaults to None.

        Raises:
            FEMException: The mp parameter should be a dictionary.
        """

        location = [
            Vertex(*loc) if isinstance(loc, Sequence) else loc for loc in location
        ]
        length = np.ones(len(location))
        if EA is None:
            EA_arr = length * self.EA
        elif isinstance(EA, (float, int)):
            EA_arr = length * EA
        elif isinstance(EA, (list, np.ndarray)):
            EA_arr = np.array(EA)
        else:
            raise FEMException(
                "Wrong parameters", "EA should be a float, list or numpy array."
            )
        if EI is None:
            EI_arr = length * self.EI
        elif isinstance(EI, (float, int)):
            EI_arr = length * EI
        elif isinstance(EI, (list, np.ndarray)):
            EI_arr = np.array(EI)
        else:
            raise FEMException(
                "Wrong parameters", "EI should be a float, list or numpy array."
            )
        if g is None:
            g_arr = length * 0.0
        elif isinstance(g, (float, int)):
            g_arr = length * float(g)
        elif isinstance(g, (list, np.ndarray)):
            g_arr = np.asarray(g, dtype=float)
        else:
            raise FEMException(
                "Wrong parameters", "g should be a float, list or numpy array."
            )

        for i in range(len(location) - 1):
            self.add_element(
                [location[i], location[i + 1]],
                EA_arr[i],
                EI_arr[i],
                g_arr[i],
                mp,
                spring,
                **kwargs,
            )

    def add_element_grid(
        self,
        x: Union[List[float], np.ndarray],
        y: Union[List[float], np.ndarray],
        EA: Optional[Union[List[float], np.ndarray, float]] = None,
        EI: Optional[Union[List[float], np.ndarray, float]] = None,
        g: Optional[Union[List[float], np.ndarray, float]] = None,
        mp: Optional["MpType"] = None,
        spring: Optional["Spring"] = None,
        **kwargs: dict,
    ) -> None:
        """Add multiple elements based upon sequential x and y coordinates.

        Args:
            x (Union[List[float], np.ndarray]): X coordinates of the grid
            y (Union[List[float], np.ndarray]): Y coordinates of the grid
            EA (Optional[Union[List[float], np.ndarray, float]], optional): Axial stiffnesses. Defaults to None.
            EI (Optional[Union[List[float], np.ndarray, float]], optional): Bending stiffnesses. Defaults to None.
            g (Optional[Union[List[float], np.ndarray, float]], optional): Self-weights. Defaults to None.
            mp (Optional[MpType], optional): Maximum plastic moment capacities for all elements. Defaults to None.
            spring (Optional[Spring], optional): Springs for all elements. Defaults to None.

        Raises:
            FEMException: x and y should have the same length.
        """
        # This is the only function which uses separate x and y coordinates, rather than a list of vertices.
        # I've left it here for backwards compatibility, but it should be deprecated in the future.
        if len(x) != len(y):
            raise FEMException(
                "Wrong parameters", "x and y should have the same length."
            )
        vertices = [Vertex(x[i], y[i]) for i in range(len(x))]
        self.add_sequential_elements(vertices, EA, EI, g, mp, spring, **kwargs)

    def add_truss_element(
        self,
        location: Union[Sequence["VertexLike"], "VertexLike"],
        EA: Optional[float] = None,
        **kwargs: dict,
    ) -> int:
        """Add a new truss element (an element that only has axial force) to the structure
        Example:
            .. code-block:: python

                   location=[[x, y], [x, y]]
                   location=[Vertex, Vertex]
                   location=[x, y]
                   location=Vertex

        Args:
            location (Union[Sequence[VertexLike], VertexLike]):
                The two nodes of the element or the next node of the element.
            EA (Optional[float], optional): Axial stiffness of the new element. Defaults to None.

        Returns:
            int: ID of the new element
        """
        return self.add_element(
            location,
            EA,
            EI=None,
            g=0,
            mp=None,
            sprint=None,
            element_type="truss",
            **kwargs,
        )

    def add_element(
        self,
        location: Union[Sequence["VertexLike"], "VertexLike"],
        EA: Optional[float] = None,
        EI: Optional[float] = None,
        g: float = 0,
        mp: Optional["MpType"] = None,
        spring: Optional["Spring"] = None,
        **kwargs: Any,
    ) -> int:
        """Add a new general element (an element with axial and lateral force) to the structure
        Example:
            .. code-block:: python

                location=[[x, y], [x, y]]
                location=[Vertex, Vertex]
                location=[x, y]
                location=Vertex

                mp={1: 210e3, 2: 180e3}

                spring={1: k, 2: k}

                # Set a hinged node:
                spring={1: 0}

        Args:
            location (Union[Sequence[VertexLike], VertexLike]):
                The two nodes of the element or the next node of the element
            EA (Optional[float], optional): Axial stiffness of the new element. Defaults to None.
            EI (Optional[float], optional): Bending stiffness of the new element. Defaults to None.
            g (float, optional): Self-weight of the new element. Defaults to 0.
            mp (Optional[MpType], optional): Maximum plastic moment of each end node. Keys are integers representing
                the nodes. Values are the bending moment capacity. Defaults to None.
            spring (Optional[Spring], optional): Rotational spring or hinge (k=0) of each end node.
                Keys are integers representing the nodes. Values are the bending moment capacity. Defaults to None.

        Optional Keyword Args:
            element_type (ElementType): "general" (axial and lateral force) or "truss" (axial force only)
            steelsection (str): Steel section name like IPE 300
            orient (OrientAxis): Steel section axis for moment of inertia - 'y' and 'z' possible
            b (float): Width of generic rectangle section
            h (float): Height of generic rectangle section
            d (float): Diameter of generic circle section
            sw (bool): If true self weight of section is considered as dead load
            E (float): Modulus of elasticity for section material
            gamma (float): Weight of section material per volume unit. [kN/m3] / [N/m3]s

        Returns:
            int: ID of the new element
        """

        if mp is None:
            mp = {}
        if spring is None:
            spring = {}

        element_type = kwargs.get("element_type", "general")

        EA = self.EA if EA is None else EA
        EI = self.EI if EI is None else EI

        section_name = ""
        # change EA EI and g if steel section specified
        if "steelsection" in kwargs:
            section_name, EA, EI, g = properties.steel_section_properties(**kwargs)
        # change EA EI and g if rectangle section specified
        if "h" in kwargs:
            section_name, EA, EI, g = properties.rectangle_properties(**kwargs)
        # change EA EI and g if circle section specified
        if "d" in kwargs:
            section_name, EA, EI, g = properties.circle_properties(**kwargs)

        if element_type == "truss":
            EI = 1e-14

        # add the element number
        self.count += 1

        point_1, point_2 = system_components.util.det_vertices(self, location)
        node_id1, node_id2 = system_components.util.det_node_ids(self, point_1, point_2)

        (
            point_1,
            point_2,
            node_id1,
            node_id2,
            spring,
            mp,
            angle,
        ) = system_components.util.force_elements_orientation(
            point_1, point_2, node_id1, node_id2, spring, mp
        )

        system_components.util.append_node_id(
            self, point_1, point_2, node_id1, node_id2
        )

        # add element
        element = Element(
            id_=self.count,
            EA=EA,
            EI=EI,
            l=(point_2 - point_1).modulus(),
            angle=angle,
            vertex_1=point_1,
            vertex_2=point_2,
            type_=element_type,
            spring=spring,
            section_name=section_name,
        )
        element.node_id1 = node_id1
        element.node_id2 = node_id2
        element.node_map = {
            node_id1: self.node_map[node_id1],
            node_id2: self.node_map[node_id2],
        }

        self.element_map[self.count] = element

        for node in (node_id1, node_id2):
            if node in self.node_element_map:
                self.node_element_map[node].append(element)
            else:
                self.node_element_map[node] = [element]

        # Register the elements per node
        for node_id in (node_id1, node_id2):
            self.node_map[node_id].elements[element.id] = element
            system_components.util.check_internal_hinges(self, node_id)

        assert mp is not None
        if len(mp) > 0:
            assert isinstance(mp, dict), "The mp parameter should be a dictionary."
            self.non_linear_elements[element.id] = mp
            self.non_linear = True
        system_components.assembly.dead_load(self, g, element.id)

        return self.count

    def remove_element(self, element_id: int) -> None:
        """Remove an element from the structure.

        Args:
            element_id (int): ID of the element to remove
        """
        element = self.element_map[element_id]
        node_id1 = element.node_id1
        node_id2 = element.node_id2

        # Remove element object itself from node maps
        for node_id in (node_id1, node_id2):
            self.node_element_map[node_id].remove(element)
            if len(self.node_element_map[node_id]) == 0:
                self.node_element_map.pop(node_id)
            self.node_map[node_id].elements.pop(element_id)
            # Check if node is now orphaned, and remove it and its loads if so
            if len(self.node_map[node_id].elements) == 0:
                system_components.util.remove_node_id(self, node_id)

        # Remove element_id
        self.element_map.pop(element_id)
        if element_id in self.loads_q:
            self.loads_q.pop(element_id)
        if element_id in self.loads_dead_load:
            self.loads_dead_load.remove(element_id)
        if element_id in self.non_linear_elements:
            self.non_linear_elements.pop(element_id)

    def add_multiple_elements(
        self,
        location: Union[Sequence["VertexLike"], "VertexLike"],
        n: Optional[int] = None,
        dl: Optional[float] = None,
        EA: Optional[float] = None,
        EI: Optional[float] = None,
        g: float = 0,
        mp: Optional["MpType"] = None,
        spring: Optional["Spring"] = None,
        **kwargs: Any,
    ) -> List[int]:
        """Add multiple elements defined by the first and the last point.

        Example:

            .. code-block:: python

                last={'EA': 1e3, 'mp': 290}

        Args:
            location (Union[Sequence[VertexLike], VertexLike]):
                The two nodes of the element or the next node of the element.
            n (Optional[int], optional): Number of elements to add between the first and last nodes. Defaults to None.
            dl (Optional[float], optional): Length of sub-elements to add between the first and last nodes.
                Length will be rounded down if necessary such that all sub-elements will have the same length.
                Defaults to None.
            EA (Optional[float], optional): Axial stiffness. Defaults to None.
            EI (Optional[float], optional): Bending stiffness. Defaults to None.
            g (float, optional): Self-weight. Defaults to 0.
            mp (Optional[MpType], optional): Maximum plastic moment capacity at ends of element. Defaults to None.
            spring (Optional[Spring], optional): Rotational springs or hinges (k=0) at ends of element.
                Defaults to None.

        Optional Keyword Args:
            element_type (ElementType): "general" (axial and lateral force) or "truss" (axial force only)
            first (dict): Different arguments for the first element
            last (dict): Different arguments for the last element
            steelsection (str): Steel section name like IPE 300
            orient (OrientAxis): Steel section axis for moment of inertia - 'y' and 'z' possible
            b (float): Width of generic rectangle section
            h (float): Height of generic rectangle section
            d (float): Diameter of generic circle section
            sw (bool): If true self weight of section is considered as dead load
            E (float): Modulus of elasticity for section material
            gamma (float): Weight of section material per volume unit. [kN/m3] / [N/m3]s

        Raises:
            FEMException: One, and only one, of n and dl should be passed as argument.

        Returns:
            List[int]: IDs of the new elements
        """

        if mp is None:
            mp = {}
        if spring is None:
            spring = {}

        first = kwargs.get("first", {})
        last = kwargs.get("last", {})
        element_type = kwargs.get("element_type", "general")

        for el in (first, last):
            if "EA" not in el:
                el["EA"] = EA
            if "EI" not in el:
                el["EI"] = EI
            if "g" not in el:
                el["g"] = g
            if "mp" not in el:
                el["mp"] = mp
            if "spring" not in el:
                el["spring"] = spring
            if "element_type" not in el:
                el["element_type"] = element_type

        point_1, point_2 = system_components.util.det_vertices(self, location)
        length = (point_2 - point_1).modulus()
        direction = (point_2 - point_1).unit()

        if dl is None and n is not None:
            var_n = int(np.ceil(n))
            lengths = np.linspace(start=0, stop=length, num=var_n + 1)
        elif dl is not None and n is None:
            var_n = int(np.ceil(length / dl) - 1)
            lengths = np.linspace(start=0, stop=var_n * dl, num=var_n + 1)
            lengths = np.append(lengths, length)
        else:
            raise FEMException(
                "Wrong parameters",
                "One, and only one, of n and dl should be passed as argument.",
            )

        point = point_1 + direction * lengths[1]
        elements = [
            self.add_element(
                [point_1, point],
                first["EA"],
                first["EI"],
                first["g"],
                first["mp"],
                first["spring"],
                element_type=first["element_type"],
                **kwargs,
            )
        ]

        if var_n > 1:
            for i in range(2, var_n):
                point = point_1 + direction * lengths[i]
                elements.append(
                    self.add_element(
                        point,
                        EA,
                        EI,
                        g,
                        mp,
                        spring,
                        element_type=element_type,
                        **kwargs,
                    )
                )

            elements.append(
                self.add_element(
                    point_2,
                    last["EA"],
                    last["EI"],
                    last["g"],
                    last["mp"],
                    last["spring"],
                    element_type=last["element_type"],
                    **kwargs,
                )
            )
        return elements

    def insert_node(
        self,
        element_id: int,
        location: Optional["VertexLike"] = None,
        factor: Optional[float] = None,
    ) -> dict[str, int]:
        """Insert a node into an existing structure.
        This can be done by adding a new Vertex at any given location, or by setting
        a factor of the elements length. E.g. if you want a node at 40% of the elements
        length, you pass factor = 0.4.

        Args:
            element_id (int): Id number of the element in which you want to insert the node
            location (Optional[VertexLike], optional): Location in which to insert the node.
                Defaults to None.
            factor (Optional[float], optional): Fraction of distance from start to end of elmeent on which to
                divide the element. Must be between 0 and 1. Defaults to None.

        Returns:
            dict[str, int]: Dictionary with keys:
                'new_node_id': ID of the newly created node
                'new_element_id1': ID of the first new element created
                'new_element_id2': ID of the second new element created
                'old_element_id': ID of the old element that was split
        """
        element_id_to_split = _negative_index_to_id(element_id, self.element_map)
        element_to_split = self.element_map[element_id_to_split]

        # Determine vertex of the new node to insert
        if factor is not None and location is None:
            if factor < 0 or factor > 1:
                raise FEMException(
                    "Invalid factor parameter",
                    f"Factor should be between 0 and 1, but is {factor}",
                )
            location_vertex = Vertex(
                factor * (element_to_split.vertex_2 - element_to_split.vertex_1)
                + element_to_split.vertex_1
            )
        elif factor is None and location is not None:
            assert location is not None
            location_vertex = Vertex(location)
            length1 = (location_vertex - element_to_split.vertex_1).modulus()
            length2 = (element_to_split.vertex_2 - location_vertex).modulus()
            factor = length1 / (length1 + length2)
        else:
            raise FEMException(
                "Invalid parameters",
                "Either factor or location - but not both - must be passed as argument.",
            )

        # Determine parameters of the new elements
        vertex_start = element_to_split.vertex_1
        vertex_end = element_to_split.vertex_2
        mp = (
            self.non_linear_elements[element_id_to_split]
            if element_id_to_split in self.non_linear_elements
            else {}
        )
        mp1: "MpType" = {}
        mp2: "MpType" = {}
        spring1: "Spring" = {}
        spring2: "Spring" = {}
        if len(mp) != 0:
            if 1 in mp:
                mp1 = {1: mp[1]}
            if 2 in mp:
                mp2 = {2: mp[2]}
        if element_to_split.springs is not None:
            if 1 in element_to_split.springs:
                spring1 = {1: element_to_split.springs[1]}
            if 2 in element_to_split.springs:
                spring2 = {2: element_to_split.springs[2]}

        # Add the new elements
        element_id1 = self.add_element(
            [vertex_start, location_vertex],
            EA=element_to_split.EA,
            EI=element_to_split.EI,
            g=element_to_split.dead_load,
            mp=mp1,
            spring=spring1,
        )
        element_id2 = self.add_element(
            [location_vertex, vertex_end],
            EA=element_to_split.EA,
            EI=element_to_split.EI,
            g=element_to_split.dead_load,
            mp=mp2,
            spring=spring2,
        )

        # Copy the q-loads from the old element to the new elements
        if element_id_to_split in self.loads_q:
            q_load_start = element_to_split.q_load[0]
            q_load_end = element_to_split.q_load[1]
            q_perp_load_start = element_to_split.q_perp_load[0]
            q_perp_load_end = element_to_split.q_perp_load[1]
            location_q_load = factor * (q_load_end - q_load_start) + q_load_start
            location_q_perp_load = (
                factor * (q_perp_load_end - q_perp_load_start) + q_perp_load_start
            )
            assert element_to_split.q_angle is not None
            self.q_load(
                q=[-q_load_start, -location_q_load],
                element_id=element_id1,
                rotation=np.degrees(element_to_split.q_angle),
                q_perp=[q_perp_load_start, location_q_perp_load],
            )
            self.q_load(
                q=[-location_q_load, -q_load_end],
                element_id=element_id2,
                rotation=np.degrees(element_to_split.q_angle),
                q_perp=[location_q_perp_load, q_perp_load_end],
            )

        # Remove the old element from everywhere it's referenced
        self.remove_element(element_id_to_split)

        return {
            "new_node_id": self.id_last_node,
            "new_element_id1": element_id1,
            "new_element_id2": element_id2,
            "old_element_id": element_id_to_split,
        }

    def insert_node_old(
        self,
        element_id: int,
        location: Optional["VertexLike"] = None,
        factor: Optional[float] = None,
    ) -> None:
        """Insert a node into an existing structure.
        This can be done by adding a new Vertex at any given location, or by setting
        a factor of the elements length. E.g. if you want a node at 40% of the elements
        length, you pass factor = 0.4.

        Note: this method completely rebuilds the SystemElements object and is therefore
        slower then building a model with `add_element` methods.

        Args:
            element_id (int): Id number of the element in which you want to insert the node
            location (Optional[VertexLike], optional): Location in which to insert the node.
                Defaults to None.
            factor (Optional[float], optional): Fraction of distance from start to end of elmeent on which to
                divide the element. Must be between 0 and 1. Defaults to None.
        """

        ss = SystemElements(
            EA=self.EA, EI=self.EI, load_factor=self.load_factor, mesh=self.plotter.mesh
        )
        element_id = _negative_index_to_id(element_id, self.element_map)

        for element in self.element_map.values():
            g = self.element_map[element.id].dead_load
            mp = (
                self.non_linear_elements[element.id]
                if element.id in self.non_linear_elements
                else {}
            )
            if element_id == element.id:
                if factor is not None and location is None:
                    if factor < 0 or factor > 1:
                        raise FEMException(
                            "Invalid factor parameter",
                            f"Factor should be between 0 and 1, but is {factor}",
                        )
                    location_vertex = Vertex(
                        factor * (element.vertex_2 - element.vertex_1)
                        + element.vertex_1
                    )
                elif factor is None and location is not None:
                    assert location is not None
                    location_vertex = Vertex(location)
                else:
                    raise FEMException(
                        "Invalid parameters",
                        "Either factor or location - but not both - must be passed as argument.",
                    )

                mp1: "MpType" = {}
                mp2: "MpType" = {}
                spring1: "Spring" = {}
                spring2: "Spring" = {}
                if len(mp) != 0:
                    if 1 in mp:
                        mp1 = {1: mp[1]}
                    if 2 in mp:
                        mp2 = {2: mp[2]}
                if element.springs is not None:
                    if 1 in element.springs:
                        spring1 = {1: element.springs[1]}
                    if 2 in element.springs:
                        spring2 = {2: element.springs[2]}

                ss.add_element(
                    [element.vertex_1, location_vertex],
                    EA=element.EA,
                    EI=element.EI,
                    g=g,
                    mp=mp1,
                    spring=spring1,
                )
                ss.add_element(
                    [location_vertex, element.vertex_2],
                    EA=element.EA,
                    EI=element.EI,
                    g=g,
                    mp=mp2,
                    spring=spring2,
                )

            else:
                ss.add_element(
                    [element.vertex_1, element.vertex_2],
                    EA=element.EA,
                    EI=element.EI,
                    g=g,
                    mp=mp,
                    spring=element.springs,
                )
        self.__dict__ = ss.__dict__.copy()

    def solve(
        self,
        force_linear: bool = False,
        verbosity: int = 0,
        max_iter: int = 200,
        geometrical_non_linear: int = False,
        **kwargs: Any,
    ) -> np.ndarray:
        """Compute the results of current model.

        Args:
            force_linear (bool, optional): Force a linear calculation, even when the system has non-linear nodes.
                Defaults to False.
            verbosity (int, optional): Log calculation outputs (0), or silence (1). Defaults to 0.
            max_iter (int, optional): Maximum allowed iterations. Defaults to 200.
            geometrical_non_linear (int, optional): Calculate second order effects and determine the buckling factor.
                Defaults to False.

        Optional Keyword Args:
            naked (bool): Whether or not to run the solve function without doing post processing.
            discretize_kwargs (dict): When doing a geometric non linear analysis you can reduce or
                                      increase the number of elements created that are used for
                                      determining the buckling_factor

        Raises:
            FEMException: The eigenvalues of the stiffness matrix are non zero, which indicates an unstable structure.
                Check your support conditions

        Returns:
            np.ndarray: Displacements vector.
        """

        for node_id in self.node_map:
            system_components.util.check_internal_hinges(self, node_id)

        if self.system_displacement_vector is None:
            system_components.assembly.process_supports(self)
            assert self.system_displacement_vector is not None

        naked = kwargs.get("naked", False)

        if not naked:
            if not self.validate():
                if all(
                    "general" in element.type for element in self.element_map.values()
                ):
                    raise FEMException(
                        "StabilityError",
                        "The eigenvalues of the stiffness matrix are non zero, "
                        "which indicates a instable structure. "
                        "Check your support conditions",
                    )

        # (Re)set force vectors
        for el in self.element_map.values():
            el.reset()
        system_components.assembly.prep_matrix_forces(self)
        assert (
            self.system_force_vector is not None
        ), "There are no forces on the structure"

        if self.non_linear and not force_linear:
            return system_components.solver.stiffness_adaptation(
                self, verbosity, max_iter
            )

        system_components.assembly.assemble_system_matrix(self)
        if geometrical_non_linear:
            discretize_kwargs = kwargs.get("discretize_kwargs", None)
            self.buckling_factor = system_components.solver.geometrically_non_linear(
                self,
                verbosity,
                return_buckling_factor=True,
                discretize_kwargs=discretize_kwargs,
            )
            return self.system_displacement_vector

        system_components.assembly.process_conditions(self)

        # solution of the reduced system (reduced due to support conditions)
        assert self.reduced_system_matrix is not None
        assert self.reduced_force_vector is not None
        reduced_displacement_vector = np.linalg.solve(
            self.reduced_system_matrix, self.reduced_force_vector
        )

        # add the solution of the reduced system in the complete system displacement vector
        assert self.shape_system_matrix is not None
        self.system_displacement_vector = np.zeros(self.shape_system_matrix)
        np.put(
            self.system_displacement_vector,
            self._remainder_indexes,
            reduced_displacement_vector,
        )

        # determine the displacement vector of the elements
        for el in self.element_map.values():
            index_node_1 = (el.node_1.id - 1) * 3
            index_node_2 = (el.node_2.id - 1) * 3

            # node 1 ux, uy, phi
            el.element_displacement_vector[:3] = self.system_displacement_vector[
                index_node_1 : index_node_1 + 3
            ]
            # node 2 ux, uy, phi
            el.element_displacement_vector[3:] = self.system_displacement_vector[
                index_node_2 : index_node_2 + 3
            ]
            el.determine_force_vector()

        if not naked:
            # determining the node results in post processing class
            self.post_processor.node_results_elements()
            self.post_processor.node_results_system()
            self.post_processor.reaction_forces()
            self.post_processor.element_results()

            # check the values in the displacement vector for extreme values, indicating a
            # flawed calculation
            assert np.any(self.system_displacement_vector < 1e6), (
                "The displacements of the structure exceed 1e6. "
                "Check your support conditions,"
                "or your elements Young's modulus"
            )

        return self.system_displacement_vector

    def validate(self, min_eigen: float = 1e-9) -> bool:
        """Validate the stability of the stiffness matrix.

        Args:
            min_eigen (float, optional): Minimum value of the eigenvalues of the stiffness matrix. This value
                should be close to zero. Defaults to 1e-9.

        Returns:
            bool: True if the structure is stable, False if not.
        """

        ss = copy.copy(self)
        system_components.assembly.prep_matrix_forces(ss)
        assert ss.system_force_vector is not None
        assert (
            np.abs(ss.system_force_vector).sum() != 0
        ), "There are no forces on the structure"
        ss._remainder_indexes = []
        system_components.assembly.assemble_system_matrix(ss)

        system_components.assembly.process_conditions(ss)

        assert ss.reduced_system_matrix is not None
        w, _ = np.linalg.eig(ss.reduced_system_matrix)
        return bool(np.all(w > min_eigen))

    def add_support_hinged(self, node_id: Union[int, Sequence[int]]) -> None:
        """Model a hinged support at a given node.

        Args:
            node_id (Union[int, Sequence[int]]): Represents the nodes ID
        """
        if not isinstance(node_id, collections.abc.Iterable):
            node_id = [node_id]

        for id_ in node_id:
            id_ = _negative_index_to_id(id_, self.node_map.keys())

            # add the support to the support list for the plotter
            self.supports_hinged.append(self.node_map[id_])

    def add_support_rotational(self, node_id: Union[int, Sequence[int]]) -> None:
        """Model a rotational support at a given node.

        Args:
            node_id (Union[int, Sequence[int]]): Represents the nodes ID
        """
        if not isinstance(node_id, collections.abc.Iterable):
            node_id = [node_id]

        for id_ in node_id:
            id_ = _negative_index_to_id(id_, self.node_map.keys())

            # add the support to the support list for the plotter
            self.supports_rotational.append(self.node_map[id_])

    def add_internal_hinge(self, node_id: Union[int, Sequence[int]]) -> None:
        """Model a internal hinge at a given node.
        This may alternatively be done by setting a spring restraint to zero (`{1: 0}` or `{2: 0}`).
        The effect is the same, though this function may be easier to use.

        Args:
            node_id (Union[int, Sequence[int]]): Represents the nodes ID
        """
        if not isinstance(node_id, collections.abc.Iterable):
            node_id = [node_id]

        for id_ in node_id:
            id_ = _negative_index_to_id(id_, self.node_map.keys())

            # add the support to the support list for the plotter
            self.internal_hinges.append(self.node_map[id_])

    def add_support_roll(
        self,
        node_id: Union[Sequence[int], int],
        direction: Union[Sequence["SupportDirection"], "SupportDirection"] = "x",
        angle: Union[Sequence[Optional[float]], Optional[float]] = None,
        rotate: Union[Sequence[bool], bool] = True,
    ) -> None:
        """Add a rolling support at a given node.

        Args:
            node_id (Union[Sequence[int], int]): Represents the nodes ID
            direction (Union[Sequence[SupportDirection], SupportDirection], optional): Represents the direction
                that is free ("x", "y", "1", or "2"). Defaults to "x".
            angle (Union[Sequence[Optional[float]], Optional[float]], optional): Angle in degrees relative to
                global x-axis. If angle is given, the support will be inclined. Defaults to None.
            rotate (Union[Sequence[bool], bool], optional): If set to False, rotation at the roller will also
                be restrianed. Defaults to True.

        Raises:
            FEMException: Invalid direction, if the direction parameter is invalid
        """
        if not isinstance(node_id, collections.abc.Iterable):
            node_id = [node_id]
        if not isinstance(direction, collections.abc.Iterable):
            direction = [direction]
        if not isinstance(angle, collections.abc.Iterable):
            angle = [angle]
        if not isinstance(rotate, collections.abc.Iterable):
            rotate = [rotate]

        assert len(node_id) == len(direction) == len(angle) == len(rotate)

        for id_, direction_, angle_, rotate_ in zip(node_id, direction, angle, rotate):
            id_ = _negative_index_to_id(id_, self.node_map.keys())

            if direction_ in ("x", "2", 2):
                direction_i: Literal[1, 2] = 2
            elif direction_ in ("y", "1", 1):
                direction_i = 1
            else:
                raise FEMException(
                    "Invalid direction",
                    f"Direction should be 'x', 'y', '1' or '2', but is {direction_}",
                )

            if angle_ is not None:
                direction_i = 2
                self.inclined_roll[id_] = float(np.radians(-angle_))

            # add the support to the support list for the plotter
            self.supports_roll.append(self.node_map[id_])
            self.supports_roll_direction.append(direction_i)
            self.supports_roll_rotate.append(rotate_)

    def add_support_fixed(
        self,
        node_id: Union[Sequence[int], int],
    ) -> None:
        """Add a fixed support at a given node.

        Args:
            node_id (Union[Sequence[int], int]): Represents the nodes ID
        """
        if not isinstance(node_id, collections.abc.Iterable):
            node_id = [
                node_id,
            ]

        for id_ in node_id:
            id_ = _negative_index_to_id(id_, self.node_map.keys())
            system_components.util.support_check(self, id_)

            # add the support to the support list for the plotter
            self.supports_fixed.append(self.node_map[id_])

    def add_support_spring(
        self,
        node_id: Union[Sequence[int], int],
        translation: Union[Sequence["AxisNumber"], "AxisNumber"],
        k: Union[Sequence[float], float],
        roll: Union[Sequence[bool], bool] = False,
    ) -> None:
        """Add a spring support at a given node.

        Args:
            node_id (Union[Sequence[int], int]): Represents the nodes ID
            translation (Union[Sequence[AxisNumber], AxisNumber]): Represents the prevented translation or rotation.
                1 = translation in x, 2 = translation in y, 3 = rotation about z
            k (Union[Sequence[float], float]): Stiffness of the spring
            roll (Union[Sequence[bool], bool], optional): If set to True, only the translation of the
                spring is controlled. Defaults to False.
        """
        self.supports_spring_args.append((node_id, translation, k, roll))
        # The stiffness of the spring is added in the system matrix at the location that
        # represents the node and the displacement.
        if not isinstance(node_id, collections.abc.Iterable):
            node_id = (node_id,)
        if not isinstance(translation, collections.abc.Iterable):
            translation = (translation,)
        if not isinstance(k, collections.abc.Iterable):
            k = (k,)
        if not isinstance(roll, collections.abc.Iterable):
            roll = (roll,)

        assert len(node_id) == len(translation) == len(k) == len(roll)

        for id_, translation_, k_, roll_ in zip(node_id, translation, k, roll):
            id_ = _negative_index_to_id(id_, self.node_map.keys())
            system_components.util.support_check(self, id_)

            # determine the location in the system matrix
            # row and column are the same
            matrix_index = (id_ - 1) * 3 + translation_ - 1

            self.system_spring_map[matrix_index] = k_

            # add the support to the support list for the plotter
            if translation_ == 1:
                self.supports_spring_x.append((self.node_map[id_], roll_))
            elif translation_ == 2:
                self.supports_spring_y.append((self.node_map[id_], roll_))
            elif translation_ == 3:
                self.supports_spring_z.append((self.node_map[id_], roll_))
            else:
                raise FEMException(
                    "Invalid translation",
                    f"Translation should be 1, 2 or 3, but is {translation_}",
                )

    def q_load(
        self,
        q: Union[float, Sequence[float]],
        element_id: Union[int, Sequence[int]],
        direction: Union["LoadDirection", Sequence["LoadDirection"]] = "element",
        rotation: Optional[Union[float, Sequence[float]]] = None,
        q_perp: Optional[Union[float, Sequence[float]]] = None,
    ) -> None:
        """Apply a q-load (distributed load) to an element.

        Args:
            q (Union[float, Sequence[float]]): Value of the q-load
            element_id (Union[int, Sequence[int]]): The element ID to which to apply the load
            direction (Union[&quot;LoadDirection&quot;, Sequence[&quot;LoadDirection&quot;]], optional):
                "element", "x", "y", "parallel", or "perpendicular". Defaults to "element".
            rotation (Optional[Union[float, Sequence[float]]], optional): Rotate the force clockwise.
                Rotation is in degrees. Defaults to None.
            q_perp (Optional[Union[float, Sequence[float]]], optional): Value of any q-load perpendicular
                to the indicated direction/rotatione. Defaults to None.

        Raises:
            FEMException: _description_
        """
        q_arr: Sequence[Sequence[float]]
        q_perp_arr: Sequence[Sequence[float]]
        if isinstance(q, Sequence):
            q_arr = [q]
        elif isinstance(q, (int, float)):
            q_arr = [[q, q]]
        if q_perp is None:
            q_perp_arr = [[0, 0]]
        elif isinstance(q_perp, Sequence):
            q_perp_arr = [q_perp]
        elif isinstance(q_perp, (int, float)):
            q_perp_arr = [[q_perp, q_perp]]

        direction_flag = rotation is None

        n_elems = len(element_id) if isinstance(element_id, Sequence) else 1
        element_id = arg_to_list(element_id, n_elems)
        direction = arg_to_list(direction, n_elems)
        rotation = arg_to_list(rotation, n_elems)
        q_arr = arg_to_list(q_arr, n_elems)
        q_perp_arr = arg_to_list(q_perp_arr, n_elems)

        for i, element_idi in enumerate(element_id):
            id_ = _negative_index_to_id(element_idi, self.element_map.keys())
            self.plotter.max_q = max(
                self.plotter.max_q,
                (q_arr[i][0] ** 2 + q_perp_arr[i][0] ** 2) ** 0.5,
                (q_arr[i][1] ** 2 + q_perp_arr[i][1] ** 2) ** 0.5,
            )

            if direction_flag:
                if direction[i] == "x":
                    rotation[i] = 0
                elif direction[i] == "y":
                    rotation[i] = np.pi / 2
                elif direction[i] == "parallel":
                    rotation[i] = self.element_map[element_id[i]].angle
                elif direction[i] in ("element", "perpendicular"):
                    rotation[i] = np.pi / 2 + self.element_map[element_id[i]].angle
                else:
                    raise FEMException(
                        "Invalid direction parameter",
                        "Direction should be 'x', 'y', 'parallel', 'perpendicular' or 'element'",
                    )
            else:
                rotation[i] = math.radians(rotation[i])
                direction[i] = "angle"

            cos = math.cos(rotation[i])
            sin = math.sin(rotation[i])
            self.loads_q[id_] = [
                (
                    (q_perp_arr[i][0] * cos + q_arr[i][0] * sin) * self.load_factor,
                    (q_arr[i][0] * self.orientation_cs * cos + q_perp_arr[i][0] * sin)
                    * self.load_factor,
                ),
                (
                    (q_perp_arr[i][1] * cos + q_arr[i][1] * sin) * self.load_factor,
                    (q_arr[i][1] * self.orientation_cs * cos + q_perp_arr[i][1] * sin)
                    * self.load_factor,
                ),
            ]
            el = self.element_map[id_]
            el.q_load = (
                self.orientation_cs * self.load_factor * q_arr[i][0],
                self.orientation_cs * self.load_factor * q_arr[i][1],
            )
            el.q_perp_load = (
                q_perp_arr[i][0] * self.load_factor,
                q_perp_arr[i][1] * self.load_factor,
            )
            el.q_direction = direction[i]
            el.q_angle = rotation[i]

    def point_load(
        self,
        node_id: Union[int, Sequence[int]],
        Fx: Union[float, Sequence[float]] = 0.0,
        Fy: Union[float, Sequence[float]] = 0.0,
        rotation: Union[float, Sequence[float]] = 0.0,
        Fz: Optional[Union[float, Sequence[float]]] = None,
    ) -> None:
        """Apply a point load to a node.

        Args:
            node_id (Union[int, Sequence[int]]): The node ID to which to apply the load
            Fx (Union[float, Sequence[float]], optional): Force in the global X direction. Defaults to 0.0.
            Fy (Union[float, Sequence[float]], optional): Force in the global Y direction. Defaults to 0.0.
            rotation (Union[float, Sequence[float]], optional): Rotate the force clockwise by the given
                angle in degrees. Defaults to 0.0.

        Raises:
            FEMException: Point loads may not be placed at the location of inclined roller supports
        """
        if isinstance(Fy, (int, float)) and Fy == 0.0 and Fz is not None:
            Fy = Fz  # for backwards compatibility with old y/z axes behaviour
        n = len(node_id) if isinstance(node_id, Sequence) else 1
        node_id = arg_to_list(node_id, n)
        Fx = arg_to_list(Fx, n)
        Fy = arg_to_list(Fy, n)
        rotation = arg_to_list(rotation, n)

        for i, node_idi in enumerate(node_id):
            id_ = _negative_index_to_id(node_idi, self.node_map.keys())
            if (
                id_ in self.inclined_roll
                and np.mod(self.inclined_roll[id_], np.pi / 2) != 0
            ):
                raise FEMException(
                    "StabilityError",
                    "Point loads may not be placed at the location of inclined roller supports",
                )
            self.plotter.max_system_point_load = max(
                self.plotter.max_system_point_load, (Fx[i] ** 2 + Fy[i] ** 2) ** 0.5
            )
            cos = math.cos(math.radians(rotation[i]))
            sin = math.sin(math.radians(rotation[i]))
            self.loads_point[id_] = (
                (Fx[i] * cos + Fy[i] * sin) * self.load_factor,
                (Fy[i] * self.orientation_cs * cos + Fx[i] * sin) * self.load_factor,
            )

    def moment_load(
        self,
        node_id: Union[int, Sequence[int]],
        Tz: Union[float, Sequence[float]] = 0.0,
        Ty: Optional[Union[float, Sequence[float]]] = None,
    ) -> None:
        """Apply a moment load to a node.

        Args:
            node_id (Union[int, Sequence[int]]): The node ID to which to apply the load
            Tz (Union[float, Sequence[float]]): Moment load (about the global Y direction) to apply
        """
        if isinstance(Tz, (int, float)) and Tz == 0.0 and Ty is not None:
            Tz = Ty  # for backwards compatibility with old y/z axes behaviour
        n = len(node_id) if isinstance(node_id, Sequence) else 1
        node_id = arg_to_list(node_id, n)
        Tz = arg_to_list(Tz, n)

        for i, node_idi in enumerate(node_id):
            id_ = _negative_index_to_id(node_idi, self.node_map.keys())
            self.loads_moment[id_] = Tz[i] * self.load_factor

    @overload
    def show_structure(
        self,
        verbosity: int,
        scale: float,
        offset: Tuple[float, float],
        figsize: Optional[Tuple[float, float]],
        show: bool,
        supports: bool,
        values_only: Literal[True],
        annotations: bool,
    ) -> Tuple[np.ndarray, np.ndarray]: ...

    @overload
    def show_structure(
        self,
        verbosity: int,
        scale: float,
        offset: Tuple[float, float],
        figsize: Optional[Tuple[float, float]],
        show: Literal[False],
        supports: bool,
        values_only: Literal[False],
        annotations: bool,
    ) -> "Figure": ...

    @overload
    def show_structure(
        self,
        verbosity: int = 0,
        scale: float = 1.0,
        offset: Tuple[float, float] = (0, 0),
        figsize: Optional[Tuple[float, float]] = None,
        show: Literal[True] = True,
        supports: bool = True,
        values_only: Literal[False] = False,
        annotations: bool = False,
    ) -> None: ...

    def show_structure(
        self,
        verbosity: int = 0,
        scale: float = 1.0,
        offset: Tuple[float, float] = (0, 0),
        figsize: Optional[Tuple[float, float]] = None,
        show: bool = True,
        supports: bool = True,
        values_only: bool = False,
        annotations: bool = False,
    ) -> Union[Tuple[np.ndarray, np.ndarray], Optional["Figure"]]:
        """Plot the structure.

        Args:
            verbosity (int, optional): 0: All information, 1: Suppress information. Defaults to 0.
            scale (float, optional): Scale of the plot. Defaults to 1.0.
            offset (Tuple[float, float], optional): Offset the plots location on the figure. Defaults to (0, 0).
            figsize (Optional[Tuple[float, float]], optional): Change the figure size. Defaults to None.
            show (bool, optional): Plot the result or return a figure. Defaults to True.
            supports (bool, optional): Plot the supports. Defaults to True.
            values_only (bool, optional): Return the values that would be plotted as tuple containing
                two arrays: (x, y). Defaults to False.
            annotations (bool, optional): if True, structure annotations are plotted. It includes section name.

        Returns:
            Figure: If show is False, return a figure.
        """
        figsize = self.figsize if figsize is None else figsize
        if values_only:
            return self.plot_values.structure()
        return self.plotter.plot_structure(
            figsize, verbosity, show, supports, scale, offset, annotations=annotations
        )

    def change_plot_colors(self, plot_colors: Dict) -> None:
        """
        Calls the change_plot_colors method of the plotter object

        Args:
            colors (Dict): A dictionary containing plot components and colors
            as key-value pairs.
        """
        self.plotter.change_plot_colors(plot_colors)

    @overload
    def show_bending_moment(
        self,
        factor: Optional[float],
        verbosity: int,
        scale: float,
        offset: Tuple[float, float],
        figsize: Optional[Tuple[float, float]],
        show: bool,
        values_only: Literal[True],
    ) -> Tuple[np.ndarray, np.ndarray]: ...

    @overload
    def show_bending_moment(
        self,
        factor: Optional[float],
        verbosity: int,
        scale: float,
        offset: Tuple[float, float],
        figsize: Optional[Tuple[float, float]],
        show: Literal[False],
        values_only: Literal[False],
    ) -> "Figure": ...

    @overload
    def show_bending_moment(
        self,
        factor: Optional[float] = None,
        verbosity: int = 0,
        scale: float = 1.0,
        offset: Tuple[float, float] = (0, 0),
        figsize: Optional[Tuple[float, float]] = None,
        show: Literal[True] = True,
        values_only: Literal[False] = False,
    ) -> None: ...

    def show_bending_moment(
        self,
        factor: Optional[float] = None,
        verbosity: int = 0,
        scale: float = 1,
        offset: Tuple[float, float] = (0, 0),
        figsize: Optional[Tuple[float, float]] = None,
        show: bool = True,
        values_only: bool = False,
    ) -> Union[Tuple[np.ndarray, np.ndarray], Optional["Figure"]]:
        """Plot the bending moment.

        Args:
            factor (Optional[float], optional): Influence the plotting scale. Defaults to None.
            verbosity (int, optional):  0: All information, 1: Suppress information. Defaults to 0.
            scale (float, optional): Scale of the plot. Defaults to 1.
            offset (Tuple[float, float], optional): Offset the plots location on the figure. Defaults to (0, 0).
            figsize (Optional[Tuple[float, float]], optional): Change the figure size. Defaults to None.
            show (bool, optional): Plot the result or return a figure. Defaults to True.
            values_only (bool, optional): Return the values that would be plotted as tuple containing
                two arrays: (x, y). Defaults to False.

        Returns:
            Figure: If show is False, return a figure.
        """

        if values_only:
            return self.plot_values.bending_moment(factor)
        figsize = self.figsize if figsize is None else figsize
        return self.plotter.bending_moment(
            factor, figsize, verbosity, scale, offset, show
        )

    @overload
    def show_axial_force(
        self,
        factor: Optional[float],
        verbosity: int,
        scale: float,
        offset: Tuple[float, float],
        figsize: Optional[Tuple[float, float]],
        show: bool,
        values_only: Literal[True],
    ) -> Tuple[np.ndarray, np.ndarray]: ...

    @overload
    def show_axial_force(
        self,
        factor: Optional[float],
        verbosity: int,
        scale: float,
        offset: Tuple[float, float],
        figsize: Optional[Tuple[float, float]],
        show: Literal[False],
        values_only: Literal[False],
    ) -> "Figure": ...

    @overload
    def show_axial_force(
        self,
        factor: Optional[float] = None,
        verbosity: int = 0,
        scale: float = 1.0,
        offset: Tuple[float, float] = (0, 0),
        figsize: Optional[Tuple[float, float]] = None,
        show: Literal[True] = True,
        values_only: Literal[False] = False,
    ) -> None: ...

    def show_axial_force(
        self,
        factor: Optional[float] = None,
        verbosity: int = 0,
        scale: float = 1,
        offset: Tuple[float, float] = (0, 0),
        figsize: Optional[Tuple[float, float]] = None,
        show: bool = True,
        values_only: bool = False,
    ) -> Union[Tuple[np.ndarray, np.ndarray], Optional["Figure"]]:
        """Plot the axial force.

        Args:
            factor (Optional[float], optional): Influence the plotting scale. Defaults to None.
            verbosity (int, optional):  0: All information, 1: Suppress information. Defaults to 0.
            scale (float, optional): Scale of the plot. Defaults to 1.
            offset (Tuple[float, float], optional): Offset the plots location on the figure. Defaults to (0, 0).
            figsize (Optional[Tuple[float, float]], optional): Change the figure size. Defaults to None.
            show (bool, optional): Plot the result or return a figure. Defaults to True.
            values_only (bool, optional): Return the values that would be plotted as tuple containing
                two arrays: (x, y). Defaults to False.

        Returns:
            Figure: If show is False, return a figure.
        """
        if values_only:
            return self.plot_values.axial_force(factor)
        figsize = self.figsize if figsize is None else figsize
        return self.plotter.axial_force(factor, figsize, verbosity, scale, offset, show)

    @overload
    def show_shear_force(
        self,
        factor: Optional[float],
        verbosity: int,
        scale: float,
        offset: Tuple[float, float],
        figsize: Optional[Tuple[float, float]],
        show: bool,
        values_only: Literal[True],
    ) -> Tuple[np.ndarray, np.ndarray]: ...

    @overload
    def show_shear_force(
        self,
        factor: Optional[float],
        verbosity: int,
        scale: float,
        offset: Tuple[float, float],
        figsize: Optional[Tuple[float, float]],
        show: Literal[False],
        values_only: Literal[False],
    ) -> "Figure": ...

    @overload
    def show_shear_force(
        self,
        factor: Optional[float] = None,
        verbosity: int = 0,
        scale: float = 1.0,
        offset: Tuple[float, float] = (0, 0),
        figsize: Optional[Tuple[float, float]] = None,
        show: Literal[True] = True,
        values_only: Literal[False] = False,
    ) -> None: ...

    def show_shear_force(
        self,
        factor: Optional[float] = None,
        verbosity: int = 0,
        scale: float = 1,
        offset: Tuple[float, float] = (0, 0),
        figsize: Optional[Tuple[float, float]] = None,
        show: bool = True,
        values_only: bool = False,
    ) -> Union[Tuple[np.ndarray, np.ndarray], Optional["Figure"]]:
        """Plot the shear force.

        Args:
            factor (Optional[float], optional): Influence the plotting scale. Defaults to None.
            verbosity (int, optional):  0: All information, 1: Suppress information. Defaults to 0.
            scale (float, optional): Scale of the plot. Defaults to 1.
            offset (Tuple[float, float], optional): Offset the plots location on the figure. Defaults to (0, 0).
            figsize (Optional[Tuple[float, float]], optional): Change the figure size. Defaults to None.
            show (bool, optional): Plot the result or return a figure. Defaults to True.
            values_only (bool, optional): Return the values that would be plotted as tuple containing
                two arrays: (x, y). Defaults to False.

        Returns:
            Figure: If show is False, return a figure.
        """
        if values_only:
            return self.plot_values.shear_force(factor)
        figsize = self.figsize if figsize is None else figsize
        return self.plotter.shear_force(factor, figsize, verbosity, scale, offset, show)

    @overload
    def show_reaction_force(
        self,
        verbosity: int,
        scale: float,
        offset: Tuple[float, float],
        figsize: Optional[Tuple[float, float]],
        show: Literal[False],
    ) -> "Figure": ...

    @overload
    def show_reaction_force(
        self,
        verbosity: int = 0,
        scale: float = 1.0,
        offset: Tuple[float, float] = (0, 0),
        figsize: Optional[Tuple[float, float]] = None,
        show: Literal[True] = True,
    ) -> None: ...

    def show_reaction_force(
        self,
        verbosity: int = 0,
        scale: float = 1,
        offset: Tuple[float, float] = (0, 0),
        figsize: Optional[Tuple[float, float]] = None,
        show: bool = True,
    ) -> Optional["Figure"]:
        """Plot the reaction force.

        Args:
            verbosity (int, optional):  0: All information, 1: Suppress information. Defaults to 0.
            scale (float, optional): Scale of the plot. Defaults to 1.
            offset (Tuple[float, float], optional): Offset the plots location on the figure. Defaults to (0, 0).
            figsize (Optional[Tuple[float, float]], optional): Change the figure size. Defaults to None.
            show (bool, optional): Plot the result or return a figure. Defaults to True.

        Returns:
            Figure: If show is False, return a figure.
        """
        figsize = self.figsize if figsize is None else figsize
        return self.plotter.reaction_force(figsize, verbosity, scale, offset, show)

    @overload
    def show_displacement(
        self,
        factor: Optional[float],
        verbosity: int,
        scale: float,
        offset: Tuple[float, float],
        figsize: Optional[Tuple[float, float]],
        show: bool,
        linear: bool,
        values_only: Literal[True],
    ) -> Tuple[np.ndarray, np.ndarray]: ...

    @overload
    def show_displacement(
        self,
        factor: Optional[float],
        verbosity: int,
        scale: float,
        offset: Tuple[float, float],
        figsize: Optional[Tuple[float, float]],
        show: Literal[False],
        linear: bool,
        values_only: Literal[False],
    ) -> "Figure": ...

    @overload
    def show_displacement(
        self,
        factor: Optional[float] = None,
        verbosity: int = 0,
        scale: float = 1.0,
        offset: Tuple[float, float] = (0, 0),
        figsize: Optional[Tuple[float, float]] = None,
        show: Literal[True] = True,
        linear: bool = False,
        values_only: Literal[False] = False,
    ) -> None: ...

    def show_displacement(
        self,
        factor: Optional[float] = None,
        verbosity: int = 0,
        scale: float = 1,
        offset: Tuple[float, float] = (0, 0),
        figsize: Optional[Tuple[float, float]] = None,
        show: bool = True,
        linear: bool = False,
        values_only: bool = False,
    ) -> Union[Tuple[np.ndarray, np.ndarray], Optional["Figure"]]:
        """Plot the displacement.

        Args:
            factor (Optional[float], optional): Influence the plotting scale. Defaults to None.
            verbosity (int, optional):  0: All information, 1: Suppress information. Defaults to 0.
            scale (float, optional): Scale of the plot. Defaults to 1.
            offset (Tuple[float, float], optional): Offset the plots location on the figure. Defaults to (0, 0).
            figsize (Optional[Tuple[float, float]], optional): Change the figure size. Defaults to None.
            show (bool, optional): Plot the result or return a figure. Defaults to True.
            linear (bool, optional): Don't evaluate the displacement values in between the elements. Defaults to False.
            values_only (bool, optional): Return the values that would be plotted as tuple containing
                two arrays: (x, y). Defaults to False.

        Returns:
            Figure: If show is False, return a figure.
        """
        if values_only:
            return self.plot_values.displacements(factor, linear)
        figsize = self.figsize if figsize is None else figsize
        return self.plotter.displacements(
            factor, figsize, verbosity, scale, offset, show, linear
        )

    @overload
    def show_results(
        self,
        verbosity: int,
        scale: float,
        offset: Tuple[float, float],
        figsize: Optional[Tuple[float, float]],
        show: Literal[False],
    ) -> "Figure": ...

    @overload
    def show_results(
        self,
        verbosity: int = 0,
        scale: float = 1.0,
        offset: Tuple[float, float] = (0, 0),
        figsize: Optional[Tuple[float, float]] = None,
        show: Literal[True] = True,
    ) -> None: ...

    def show_results(
        self,
        verbosity: int = 0,
        scale: float = 1,
        offset: Tuple[float, float] = (0, 0),
        figsize: Optional[Tuple[float, float]] = None,
        show: bool = True,
    ) -> Optional["Figure"]:
        """Plot all the results in one window.

        Args:
            verbosity (int, optional): 0: All information, 1: Suppress information. Defaults to 0.
            scale (float, optional): Scale of the plot. Defaults to 1.
            offset (Tuple[float, float], optional): Offset the plots location on the figure. Defaults to (0, 0).
            figsize (Optional[Tuple[float, float]], optional): Change the figure size. Defaults to None.
            show (bool, optional): Plot the result or return a figure. Defaults to True.

        Returns:
            Figure: If show is False, return a figure.
        """
        figsize = self.figsize if figsize is None else figsize
        return self.plotter.results_plot(figsize, verbosity, scale, offset, show)

    @overload
    def get_node_results_system(
        self, node_id: None = None
    ) -> List[Dict[str, Union[int, float]]]: ...

    @overload
    def get_node_results_system(self, node_id: int) -> Dict[str, Union[int, float]]: ...

    def get_node_results_system(
        self, node_id: Optional[int] = None
    ) -> Union[List[Dict[str, Union[int, float]]], Dict[str, Union[int, float]]]:
        """Get the node results. These are the opposite of the forces and displacements
        working on the elements and may seem counter intuitive.

        Args:
            node_id (int, optional): The node's ID. If node_id == None or 0, the results of all nodes are returned.
                Defaults to None.

        Returns:
            Union[ List[Dict[str, Union[int, float]]], Dict[str, Union[int, float]] ]:
                If node_id == 0, returns a list containing tuples with the results:
                [(id, Fx, Fy, Tz, ux, uy, phi_z), (id, Fx, Fy...), () .. ]
                If node_id > 0, returns a dict with the results:
                {"id": id, "Fx": Fx, "Fy": Fy, "Tz": Tz, "ux": ux, "uy": uy, "phi_z": phi_z}
        """
        result_list = []
        if node_id is not None and node_id != 0:
            node_id = _negative_index_to_id(node_id, self.node_map)
            node = self.node_map[node_id]
            return {
                "id": node.id,
                "Fx": node.Fx,
                "Fy": node.Fy_neg,
                "Tz": node.Tz,
                "ux": node.ux,
                "uy": -node.uy,
                "phi_z": node.phi_z,
            }
        for node in self.node_map.values():
            result_list.append(
                {
                    "id": node.id,
                    "Fx": node.Fx,
                    "Fy": node.Fy_neg,
                    "Tz": node.Tz,
                    "ux": node.ux,
                    "uy": -node.uy,
                    "phi_z": node.phi_z,
                }
            )
        return result_list

    @overload
    def get_node_displacements(self, node_id: None) -> List[Dict[str, Any]]: ...

    @overload
    def get_node_displacements(self, node_id: int) -> Dict[str, Any]: ...

    def get_node_displacements(
        self, node_id: Optional[int] = None
    ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """Get the node displacements.

        Args:
            node_id (int, optional): The node's ID. If node_id == None or 0, the results of all nodes are returned.
                Defaults to None.

        Returns:
            Union[List[Dict[str, Any]], Dict[str, Any]]: If node_id == 0, returns a list containing
                tuples with the results: [(id, ux, uy, phi_z), (id, ux, uy, phi_z),  ... (id, ux, uy, phi_z) ]
                If node_id > 0, returns a dict with the results: {"id": id, "ux": ux, "uy": uy, "phi_z": phi_z}
        """
        result_list = []
        if node_id is not None and node_id != 0:
            node_id = _negative_index_to_id(node_id, self.node_map)
            node = self.node_map[node_id]
            return {
                "id": node.id,
                "ux": -node.ux,
                "uy": node.uy,  # - * -  = +
                "phi_z": node.phi_z,
            }
        for node in self.node_map.values():
            result_list.append(
                {
                    "id": node.id,
                    "ux": -node.ux,
                    "uy": node.uy,  # - * -  = +
                    "phi_z": node.phi_z,
                }
            )
        return result_list

    @overload
    def get_element_results(
        self, element_id: None, verbose: bool
    ) -> List[Dict[str, Any]]: ...

    @overload
    def get_element_results(self, element_id: int, verbose: bool) -> Dict[str, Any]: ...

    def get_element_results(
        self, element_id: Optional[int] = None, verbose: bool = False
    ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """Get the element results.

        Args:
            element_id (int, optional): The element's ID. If element_id == None or 0, the results of all elements
                are returned. Defaults to None.
            verbose (bool, optional): If set to True, then numerical results for the deflection and the bending
                moment are also returned. Defaults to False.

        Returns:
            Union[List[Dict[str, Any]], Dict[str, Any]]: If element_id == 0,
                returns a list containing dicts with the results: [{"id": id, "length": length, "alpha": alpha,
                    "umax": umax, "umin": umin, "u": u, "wmax": wmax, "wmin": wmin, "w": w, "Mmin": Mmin, "Mmax": Mmax,
                    "M": M, "Qmin": Qmin, "Qmax": Qmax, "Q": Q, "Nmin": Nmin, "Nmax": Nmax, "N": N, "q": q}, ... ]
                If element_id > 0, returns a dict with the results: {"id": id, "length": length, "alpha": alpha,
                    "umax": umax, "umin": umin, "u": u, "wmax": wmax, "wmin": wmin, "w": w, "Mmin": Mmin, "Mmax": Mmax,
                    "M": M, "Qmin": Qmin, "Qmax": Qmax, "Q": Q, "Nmin": Nmin, "Nmax": Nmax, "N": N, "q": q}
        """
        if element_id is not None and element_id != 0:
            element_id = _negative_index_to_id(element_id, self.element_map)
            el = self.element_map[element_id]

            assert el.extension is not None
            assert el.axial_force is not None
            assert el.total_deflection is not None

            if el.type == "truss":
                return {
                    "id": el.id,
                    "length": el.l,
                    "alpha": el.angle,
                    "umax": np.max(el.extension),
                    "umin": np.min(el.extension),
                    "u": el.extension if verbose else None,
                    "wtotmax": np.max(el.total_deflection),
                    "wtotmin": np.min(el.total_deflection),
                    "wtot": el.total_deflection if verbose else None,
                    "Nmin": np.min(el.axial_force),
                    "Nmax": np.max(el.axial_force),
                    "N": el.axial_force if verbose else None,
                }
            assert el.deflection is not None
            assert el.shear_force is not None
            assert el.bending_moment is not None

            return {
                "id": el.id,
                "length": el.l,
                "alpha": el.angle,
                "umax": np.max(el.extension),
                "umin": np.min(el.extension),
                "u": el.extension if verbose else None,
                "wmax": np.min(el.deflection),
                "wmin": np.max(el.deflection),
                "w": el.deflection if verbose else None,
                "wtotmax": np.min(el.total_deflection),
                "wtotmin": np.max(el.total_deflection),
                "wtot": el.total_deflection if verbose else None,
                "Mmin": np.min(el.bending_moment),
                "Mmax": np.max(el.bending_moment),
                "M": el.bending_moment if verbose else None,
                "Qmin": np.min(el.shear_force),
                "Qmax": np.max(el.shear_force),
                "Q": el.shear_force if verbose else None,
                "Nmin": np.min(el.axial_force),
                "Nmax": np.max(el.axial_force),
                "N": el.axial_force if verbose else None,
                "q": el.q_load,
            }
        result_list = []
        for el in self.element_map.values():
            assert el.extension is not None
            assert el.axial_force is not None
            assert el.total_deflection is not None

            if el.type == "truss":
                result_list.append(
                    {
                        "id": el.id,
                        "length": el.l,
                        "alpha": el.angle,
                        "umax": np.max(el.extension),
                        "umin": np.min(el.extension),
                        "u": el.extension if verbose else None,
                        "wtotmax": np.min(el.total_deflection),
                        "wtotmin": np.max(el.total_deflection),
                        "wtot": el.total_deflection if verbose else None,
                        "Nmin": np.min(el.axial_force),
                        "Nmax": np.max(el.axial_force),
                        "N": el.axial_force if verbose else None,
                    }
                )

            else:
                assert el.deflection is not None
                assert el.shear_force is not None
                assert el.bending_moment is not None

                result_list.append(
                    {
                        "id": el.id,
                        "length": el.l,
                        "alpha": el.angle,
                        "umax": np.max(el.extension),
                        "umin": np.min(el.extension),
                        "u": el.extension if verbose else None,
                        "wmax": np.min(el.deflection),
                        "wmin": np.max(el.deflection),
                        "w": el.deflection if verbose else None,
                        "wtotmax": np.min(el.total_deflection),
                        "wtotmin": np.max(el.total_deflection),
                        "wtot": el.total_deflection if verbose else None,
                        "Mmin": np.min(el.bending_moment),
                        "Mmax": np.max(el.bending_moment),
                        "M": el.bending_moment if verbose else None,
                        "Qmin": np.min(el.shear_force),
                        "Qmax": np.max(el.shear_force),
                        "Q": el.shear_force if verbose else None,
                        "Nmin": np.min(el.axial_force),
                        "Nmax": np.max(el.axial_force),
                        "N": el.axial_force if verbose else None,
                        "q": el.q_load,
                    }
                )
        return result_list

    @overload
    def get_element_result_range(
        self,
        unit: Literal["shear", "moment", "axial"],
        minmax: Literal["min", "max", "abs"],
    ) -> List[float]: ...

    @overload
    def get_element_result_range(
        self, unit: Literal["shear", "moment", "axial"], minmax: Literal["both"]
    ) -> List[Tuple[float, float]]: ...

    def get_element_result_range(
        self,
        unit: Literal["shear", "moment", "axial"],
        minmax: Literal["min", "max", "abs", "both"] = "abs",
    ) -> Union[List[float], List[Tuple[float, float]]]:
        """Get the element results. Returns a list with the min/max results of each element for a certain unit.

        Args:
            unit (str): "shear", "moment", or "axial"
            minmax (str), optional: "min", "max", "abs", or "both". Defaults to "abs".
                Note that "both" returns a tuple with two values: (min, max)

        Raises:
            NotImplementedError: If the unit is not implemented.

        Returns:
            List[Union[float, Tuple[float]]]: List with min and/or max results of each element for a certain unit.
        """

        @overload
        def minmax_array(
            array: np.ndarray, minmax: Literal["min", "max", "abs"]
        ) -> float: ...

        @overload
        def minmax_array(
            array: np.ndarray, minmax: Literal["both"]
        ) -> Tuple[float, float]: ...

        def minmax_array(
            array: np.ndarray, minmax: Literal["min", "max", "abs", "both"] = minmax
        ) -> Union[float, Tuple[float, float]]:
            assert len(array) > 0
            # technically, np.amin/np.amax returns 'Any', hence the type: ignore's
            if minmax == "min":
                return np.amin(array)  # type: ignore
            if minmax == "max":
                return np.amax(array)  # type: ignore
            if minmax == "abs":
                return np.amax(np.abs(array))  # type: ignore
            if minmax == "both":
                return (np.amin(array), np.amax(array))
            raise NotImplementedError("minmax must be 'min', 'max', 'abs', or 'both'")

        if unit == "shear":
            return [
                minmax_array(el.shear_force)  # type: ignore
                for el in self.element_map.values()
                if el.shear_force is not None
            ]
        if unit == "moment":
            return [
                minmax_array(el.bending_moment)  # type: ignore
                for el in self.element_map.values()
                if el.bending_moment is not None
            ]
        if unit == "axial":
            return [
                minmax_array(el.axial_force)  # type: ignore
                for el in self.element_map.values()
                if el.axial_force is not None
            ]
        raise NotImplementedError("unit must be 'shear', 'moment', or 'axial'")

    def get_node_result_range(self, unit: Literal["ux", "uy", "phi_z"]) -> List[float]:
        """Get the node results. Returns a list with the node results for a certain unit.

        Args:
            unit (str): "uy", "ux", or "phi_z"

        Raises:
            NotImplementedError: If the unit is not implemented.

        Returns:
            List[float]: List with the node results for a certain unit.
        """
        if unit == "uy":
            return [node.uy for node in self.node_map.values()]  # - * -  = +
        if unit == "ux":
            return [-node.ux for node in self.node_map.values()]
        if unit == "phi_z":
            return [node.phi_z for node in self.node_map.values()]
        raise NotImplementedError

    def find_node_id(
        self, vertex: Union[Vertex, Sequence[float]], tolerance: float = 1e-9
    ) -> Optional[int]:
        """Find the ID of a certain location.

        Args:
            vertex (Union[Vertex, Sequence[float]]): Vertex_xz, [x, y], (x, y)
            tolerance (float): Tolerance for matching existing node locations (length units). Defaults to 1e-9.
        Raises:
            TypeError: vertex must be a list, tuple or Vertex

        Returns:
            Optional[int]: id of the node at the location of the vertex
        """
        vertex_v = Vertex(vertex)
        try:
            return next(
                filter(
                    lambda x: math.isclose(x.vertex.x, vertex_v.x, abs_tol=tolerance)
                    and math.isclose(x.vertex.y, vertex_v.y, abs_tol=tolerance),
                    self.node_map.values(),
                )
            ).id
        except StopIteration:
            return None

    def nodes_range(
        self, dimension: "Dimension"
    ) -> List[Union[float, Tuple[float, float], None]]:
        """Retrieve a list with coordinates x or y.

        Args:
            dimension (str): "both", 'x' or 'y'

        Returns:
            List[Union[float, Tuple[float, float], None]]: List with coordinates x or y
        """
        return list(
            map(
                lambda x: (
                    x.vertex.x
                    if dimension == "x"
                    else (
                        x.vertex.y
                        if dimension == "y"
                        else (
                            x.vertex.y_neg
                            if dimension == "y_neg"
                            else (
                                (x.vertex.x, x.vertex.y)
                                if dimension == "both"
                                else None
                            )
                        )
                    )
                ),
                self.node_map.values(),
            )
        )

    def nearest_node(
        self, dimension: "Dimension", val: Union[float, Sequence[float]]
    ) -> Union[int, None]:
        """Retrieve the nearest node ID.

        Args:
            dimension (str): "both", 'x', 'y' or 'z'
            val (Union[float, Sequence[float]]): Value of the dimension.

        Returns:
            Union[int, None]: ID of the node.
        """
        if dimension == "both" and isinstance(val, Sequence):
            return self.node_map[
                int(
                    np.argmin(
                        np.sqrt(
                            (np.array(self.nodes_range("x")) - val[0]) ** 2
                            + (np.array(self.nodes_range("y_neg")) - val[1]) ** 2
                        )
                    )
                )
            ].id
        return self.node_map[
            int(np.argmin(np.abs(np.array(self.nodes_range(dimension)) - val)))
        ].id

    def discretize(self, n: int = 10) -> None:
        """Discretize the elements. Takes an already defined :class:`.SystemElements` object and increases the number
        of elements.

        Args:
            n (int, optional): Divide the elements into n sub-elements.. Defaults to 10.
        """
        ss = SystemElements(
            EA=self.EA, EI=self.EI, load_factor=self.load_factor, mesh=self.plotter.mesh
        )

        for element in self.element_map.values():
            g = self.element_map[element.id].dead_load
            mp = (
                self.non_linear_elements[element.id]
                if element.id in self.non_linear_elements
                else {}
            )

            el_v_range = vertex_range(element.vertex_1, element.vertex_2, n)
            for last_v, v in zip(el_v_range, el_v_range[1:]):
                loc = [last_v, v]
                ss.add_element(
                    loc,
                    EA=element.EA,
                    EI=element.EI,
                    g=g,
                    mp=mp,
                    spring=element.springs,
                )
                last_v = v

        # supports
        for node, direction, rotate in zip(
            self.supports_roll, self.supports_roll_direction, self.supports_roll_rotate
        ):
            ss.add_support_roll((node.id - 1) * n + 1, direction, None, rotate)
        for node in self.supports_fixed:
            ss.add_support_fixed((node.id - 1) * n + 1)
        for node in self.supports_hinged:
            ss.add_support_hinged((node.id - 1) * n + 1)
        for node in self.supports_rotational:
            ss.add_support_rotational((node.id - 1) * n + 1)
        for args in self.supports_spring_args:
            ss.add_support_spring((args[0] - 1) * n + 1, *args[1:])

        # loads
        for node_id, forces in self.loads_point.items():
            ss.point_load(
                (node_id - 1) * n + 1,
                Fx=forces[0] / self.load_factor,
                Fy=forces[1] / self.orientation_cs / self.load_factor,
            )

        for node_id, forces_moment in self.loads_moment.items():
            ss.moment_load((node_id - 1) * n + 1, forces_moment / self.load_factor)
        for element_id, forces_q in self.loads_q.items():
            ss.q_load(
                q=[i[1] / self.orientation_cs / self.load_factor for i in forces_q],
                element_id=element_id,
                direction="y",
                q_perp=[i[0] / self.load_factor for i in forces_q],
            )

        self.__dict__ = ss.__dict__.copy()

    def remove_loads(self, dead_load: bool = False) -> None:
        """Remove all the applied loads from the structure.

        Args:
            dead_load (bool, optional): Also remove the self-weights? Defaults to False.
        """
        self.loads_point = {}
        self.loads_q = {}
        self.loads_moment = {}

        for k in self.element_map:  # pylint: disable=consider-using-dict-items
            self.element_map[k].q_load = (0.0, 0.0)
            if dead_load:
                self.element_map[k].dead_load = 0
        if dead_load:
            self.loads_dead_load = set()

    def apply_load_case(self, loadcase: LoadCase) -> None:
        """Apply a load case to the structure.

        Args:
            loadcase (LoadCase): Load case to apply.
        """
        for method, kwargs in loadcase.spec.items():
            method = method.split("-")[0]
            kwargs = re.sub(r"[{}]", "", str(kwargs))
            # pass the groups that match back to the replace
            kwargs = re.sub(r".??(\w+).?:", r"\1=", kwargs)

            exec(f"self.{method}({kwargs})")  # pylint: disable=exec-used

    def get_stiffness_matrix(self, element_id: int) -> Optional[np.ndarray]:
        """
        Return the stiffness matrix for a specific element by its ID.
        Args:
            element_id (int): ID of the element.

        Returns:
        Optional[Union[list, None]]: The stiffness matrix of the element if it exists, otherwise None.
        """
        try:
            element = self.element_map[element_id]
        except KeyError:
            print(f"Element with ID {element_id} does not exist.")
            return None

        if isinstance(element, Element):
            if hasattr(element, "stiffness_matrix"):
                print(f"Stiffness Matrix for Element ID {element_id}:")
                return element.stiffness_matrix

            print(f"Element ID {element_id} does not have a stiffness matrix.")
        else:
            print(f"Invalid element type for element ID {element_id}.")
        return None

    def __deepcopy__(self, _: str) -> "SystemElements":
        """Deepcopy the SystemElements object.

        Args:
            _ (str): Unnecessary argument.

        Returns:
            SystemElements: Copied SystemElements object.
        """
        system = copy.copy(self)
        mesh = self.plotter.mesh
        system.plotter = None
        system.post_processor = None
        system.plot_values = None

        system.__dict__ = copy.deepcopy(system.__dict__)
        system.plotter = plotter.Plotter(system, mesh)
        system.post_processor = post_sl(system)
        system.plot_values = plotter.PlottingValues(system, mesh)

        return system


def _negative_index_to_id(idx: int, collection: Collection[int]) -> int:
    """Convert a negative index to a positive index. (That is, allowing the Pythonic negative indexing)

    Args:
        idx (int): Index to convert
        collection (Collection[int]): Collection of indices to check against

    Raises:
        TypeError: If the index is not an integer

    Returns:
        int: Positive index
    """
    if not isinstance(idx, int):
        if int(idx) == idx:  # if it can be non-destructively cast to an integer
            idx = int(idx)
        else:
            raise TypeError("Node or element id must be an integer")
    if idx >= 0:
        return idx
    return max(collection) + (idx + 1)
