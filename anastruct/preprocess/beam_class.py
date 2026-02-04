from abc import ABC, abstractmethod
from typing import Iterable, Literal, Optional, Sequence, Union

import numpy as np

from anastruct.fem.system import SystemElements
from anastruct.fem.system_components.util import add_node
from anastruct.types import LoadDirection, SectionProps
from anastruct.vertex import Vertex

DEFAULT_BEAM_SECTION: SectionProps = {
    "EI": 1e6,
    "EA": 1e8,
    "g": 0.0,
}


class Beam(ABC):
    """Abstract base class for 2D beam structures.

    Provides a framework for creating parametric beam geometries with automated
    node generation, connectivity, and support definitions. Subclasses implement
    specific beam types (simple, cantilever, etc.).

    The beam generation follows a two-phase process:
    1. define_nodes() - Generate node coordinates and span connectivity
    2. define_supports() - Define support locations and types

    Attributes:
        length (float): Total length of the beam (length units)
        angle (float): Angle of the beam (degrees; 0 = horizontal, positive = CCW); defaults to 0.0
        section (SectionProps): Section properties for all beam elements; defaults to DEFAULT_BEAM_SECTION
        supports_type (Literal["simple", "pinned", "fixed"]): Type of supports to apply; defaults to "simple"
        system (SystemElements): The FEM system containing all nodes, elements, and supports; initialized after adding elements
    """

    # Common geometry
    length: float
    span_lengths: list[float]
    angle: float

    # Material properties
    section: SectionProps

    # Defined by subclass (initialized in define_* methods)
    nodes: list[Vertex]
    node_ids: dict[int, list[int]]
    support_definitions: dict[int, Literal["fixed", "pinned", "roller"]]

    # Defined by main class (initialized in add_elements)
    element_ids: dict[int, list[int]]

    # System
    system: SystemElements

    def __init__(
        self,
        length: Optional[float] = None,
        span_lengths: Optional[list[float]] = None,
        angle: float = 0.0,
        section: Optional[SectionProps] = None,
    ):
        """Initialize a beam structure.

        Args:
            length (float): Total length of the beam (length units). Must be positive.
                Either length or span_lengths must be provided.
            span_lengths (list[float]): List of span lengths for each span. Must be
                positive. Either length or span_lengths must be provided.
            angle (float): Angle of the beam (degrees; 0 = horizontal, positive = CCW);
                defaults to 0.0
            section (SectionProps): Section properties for all beam elements; defaults
                to DEFAULT_BEAM_SECTION

        Raises:
            ValueError: If length or span_lengths are not positive, or if neither
                (or both) are provided.
        """
        if length is None and span_lengths is None:
            raise ValueError("Either length or span_lengths must be provided.")
        if length is not None and span_lengths is not None:
            raise ValueError("Only one of length or span_lengths may be provided.")
        if span_lengths is not None:
            if any(l <= 0 for l in span_lengths):
                raise ValueError(
                    f"All span lengths must be positive, got {span_lengths}"
                )
            self.span_lengths = span_lengths
            self.length = sum(span_lengths)
        if length is not None:
            if length <= 0:
                raise ValueError(f"length must be positive, got {length}")
            self.length = length
            self.span_lengths = [length]

        if angle != 0.0 and -2 * np.pi <= angle <= 2 * np.pi:
            import warnings

            warnings.warn(
                f"A very small angle was provided ({angle}). "
                f"Please ensure input units are degrees, not radians.",
                stacklevel=2,
            )
        if angle < 0 or angle >= 360:
            angle = angle % 360

        self.angle = angle
        self.section = section or DEFAULT_BEAM_SECTION

        self.dx = np.cos(self.angle * np.pi / 180)
        self.dy = np.sin(self.angle * np.pi / 180)

        # Initialize mutable attributes (prevents sharing between instances)
        self.nodes = []
        self.node_ids = {}
        self.support_definitions = {}
        self.element_ids = {}

        self.define_nodes()
        self.define_supports()

        self.system = SystemElements()
        self.add_nodes()
        self.add_elements()
        self.add_supports()

    @property
    @abstractmethod
    def type(self) -> str:
        """Return the human-readable name of the beam type."""

    @abstractmethod
    def define_nodes(self) -> None:
        """Generate node coordinates and populate self.nodes list.

        Must be implemented by subclasses. Should create Vertex objects
        representing all node locations in the beam. Should also populate
        self.node_ids dictionary mapping spanwise node indices to global node IDs.
        """

    @abstractmethod
    def define_supports(self) -> None:
        """Define support locations and types by populating self.support_definitions.

        Must be implemented by subclasses.
        """

    def add_nodes(self) -> None:
        """Add all nodes from self.nodes to the SystemElements."""
        for i, vertex in enumerate(self.nodes):
            add_node(self.system, point=vertex, node_id=i + 1)

    def add_elements(self) -> None:
        """Create elements from connectivity definitions and add to SystemElements.

        Populates element ID list self.element_ids.
        """

        def add_span_elements(
            node_pairs: Iterable[tuple[int, int]],
            section: SectionProps,
        ) -> list[int]:
            """Helper to add a sequence of connected elements.

            Args:
                node_pairs (Iterable[tuple[int, int]]): Pairs of node IDs to connect
                section (SectionProps): Section properties for the elements

            Returns:
                list[int]: Element IDs of created elements
            """
            element_ids = []
            for i, j in node_pairs:
                element_ids.append(
                    self.system.add_element(
                        location=(self.nodes[i - 1], self.nodes[j - 1]),
                        EA=section["EA"],
                        EI=section["EI"],
                        g=section["g"],
                        spring=None,
                    )
                )
            return element_ids

        # Element creation per span
        self.element_ids = {}
        for span, span_node_ids in self.node_ids.items():
            self.element_ids[span] = add_span_elements(
                node_pairs=zip(span_node_ids[:-1], span_node_ids[1:]),
                section=self.section,
            )

    def add_supports(self) -> None:
        """Add supports from self.support_definitions to the SystemElements."""
        for node_id, support_type in self.support_definitions.items():
            if support_type == "fixed":
                self.system.add_support_fixed(node_id=node_id)
            elif support_type == "pinned":
                self.system.add_support_hinged(node_id=node_id)
            elif support_type == "roller":
                self.system.add_support_roll(node_id=node_id)

    def get_element_ids_of_spans(
        self, spans: Optional[Union[int, Sequence[int]]]
    ) -> list[int]:
        """Get element IDs for a span.

        Args:
            span_ids (int, sequence, None): The ID of the span to query. If None, returns
                element IDs for all spans. If a sequence, returns IDs for all specified spans.

        Returns:
            list[int]: Element IDs of the requested span

        Raises:
            KeyError: If span_id does not exist
        """
        # Normalize spans to a list
        if spans is None:
            # Assume all spans by default
            spans = list(self.element_ids.keys())

        elif isinstance(spans, int):
            spans = [spans]

        element_ids: list[int] = []
        for span in spans:
            if span not in self.element_ids:
                available = list(self.element_ids.keys())
                raise KeyError(
                    f"span number '{span}' not found. " f"Available spans: {available}"
                )
            element_ids.extend(self.element_ids[span])
        return element_ids

    def apply_q_load_to_spans(
        self,
        q: Union[float, Sequence[float]],
        direction: Union[LoadDirection, Sequence[LoadDirection]] = "element",
        rotation: Optional[Union[float, Sequence[float]]] = None,
        q_perp: Optional[Union[float, Sequence[float]]] = None,
        spans: Optional[Union[int, Sequence[int]]] = None,
    ) -> None:
        """Apply distributed load to all elements within one or more spans.

        Args:
            q (Union[float, Sequence[float]]): Load magnitude (force/length units)
            direction (Union[LoadDirection, Sequence[LoadDirection]]): Load direction.
                Options: "element", "x", "y", "parallel", "perpendicular", "angle"
            rotation (Optional[Union[float, Sequence[float]]]): Rotation angle in degrees
                (used with direction="angle")
            q_perp (Optional[Union[float, Sequence[float]]]): Perpendicular load component
        """
        element_ids = self.get_element_ids_of_spans(spans=spans)
        for el_id in element_ids:
            self.system.q_load(
                element_id=el_id,
                q=q,
                direction=direction,
                rotation=rotation,
                q_perp=q_perp,
            )

    def apply_point_load_to_spans(
        self,
        Fx: Union[float, Sequence[float]] = 0.0,
        Fy: Union[float, Sequence[float]] = 0.0,
        rotation: Union[float, Sequence[float]] = 0.0,
        absolute_location: Optional[float] = None,
        relative_location: Optional[float] = None,
        spans: Optional[Union[int, Sequence[int]]] = None,
        tolerance: Optional[float] = None,
    ) -> None:
        """Apply point load to elements within one or more spans.

        Args:
            Fx (Union[float, Sequence[float]]): Horizontal load component (force units)
            Fy (Union[float, Sequence[float]]): Vertical load component (force units)
            rotation (Union[float, Sequence[float]]): Rotation angle in degrees
            absolute_location (Optional[float]): Absolute location along the beam length (length units).
                Either absolute_location or relative_location must be provided.
            relative_location (Optional[float]): Relative location along the beam length
                (0.0 = start of span, 1.0 = end of span). Either absolute_location or
                relative_location must be provided.
            spans (Optional[Union[int, Sequence[int]]]): Span(s) to apply the load to. If None,
                applies to all spans.
            tolerance (float): Tolerance for matching existing node locations (length units). Defaults to beam length * 1e-4.
        """
        if spans is None:
            spans = list(self.element_ids.keys())
        elif isinstance(spans, int):
            spans = [spans]

        if absolute_location is None and relative_location is None:
            raise ValueError(
                "Either absolute_location or relative_location must be provided."
            )
        if absolute_location is not None and relative_location is not None:
            raise ValueError(
                "Only one of absolute_location or relative_location may be provided."
            )

        if tolerance is None:
            tolerance = self.length * 1e-4

        for span in spans:
            span_node_ids = self.node_ids[span]
            span_start = self.nodes[span_node_ids[0] - 1]
            span_end = self.nodes[span_node_ids[-1] - 1]
            span_length = np.sqrt(
                (span_end.x - span_start.x) ** 2 + (span_end.y - span_start.y) ** 2
            )

            if relative_location is not None:
                # Check if location is within this span
                if relative_location < 0 or relative_location > 1.0:
                    continue
                # Compute absolute location within the span
                span_abs_location = relative_location * span_length
            else:
                assert absolute_location is not None
                span_abs_location = absolute_location

            # Compute load location
            load_x = span_start.x + self.dx * span_abs_location
            load_y = span_start.y + self.dy * span_abs_location

            # Determine if a node already exists at (or very near to) the load location
            node_id = self.system.find_node_id(
                vertex=Vertex(load_x, load_y), tolerance=tolerance
            )

            # If no existing node, insert a new node into the appropriate element
            if node_id is None:
                # Identify the element to insert the node into
                elem_start = 0.0
                for i, elem_id in enumerate(self.element_ids[span]):
                    elem_start_v = self.system.element_map[elem_id].vertex_1
                    elem_end_v = self.system.element_map[elem_id].vertex_2
                    elem_length = np.sqrt(
                        (elem_end_v.x - elem_start_v.x) ** 2
                        + (elem_end_v.y - elem_start_v.y) ** 2
                    )
                    elem_end = elem_start + elem_length

                    if elem_start <= span_abs_location <= elem_end:
                        # Insert node into this element
                        result = self.system.insert_node(
                            element_id=elem_id, location=Vertex(load_x, load_y)
                        )

                        # Update our internal node and element lists
                        self.node_ids[span].insert(i + 1, result["new_node_id"])
                        self.element_ids[span].remove(elem_id)
                        self.element_ids[span].insert(i, result["new_element_id1"])
                        self.element_ids[span].insert(i + 1, result["new_element_id2"])
                        node_id = result["new_node_id"]
                        break

            # Apply point load at the identified or newly created node
            assert node_id is not None
            self.system.point_load(node_id=node_id, Fx=Fx, Fy=Fy, rotation=rotation)

    def validate(self) -> bool:
        """Validate beam geometry and connectivity.

        Checks for common beam definition issues:
        - All node IDs in span lists reference valid nodes
        - No duplicate nodes at the same location
        - All elements have non-zero length

        Returns:
            bool: True if validation passes

        Raises:
            ValueError: If validation fails with description of the issue
        """
        # Check that all node IDs in connectivity are valid
        max_node_id = len(self.nodes)

        # Validate node ID list
        for span, span_node_ids in self.node_ids.items():
            for node_id in span_node_ids:
                if node_id < 1 or node_id > max_node_id:
                    raise ValueError(
                        f"Span number '{span}' references invalid node ID {node_id}. "
                        f"Valid range: 1-{max_node_id}"
                    )

        # Check for duplicate node locations (within tolerance)
        tolerance = 1e-6
        for i, node_i in enumerate(self.nodes):
            for j in range(i + 1, len(self.nodes)):
                node_j = self.nodes[j]
                dx = abs(node_i.x - node_j.x)
                dy = abs(node_i.y - node_j.y)
                if dx < tolerance and dy < tolerance:
                    raise ValueError(
                        f"Duplicate nodes at position ({node_i.x:.6f}, {node_i.y:.6f}): "
                        f"node {i} and node {j}"
                    )

        # Check for zero-length elements
        def check_element_length(
            node_a_id: int, node_b_id: int, element_type: str
        ) -> None:
            node_a = self.nodes[node_a_id - 1]
            node_b = self.nodes[node_b_id - 1]
            dx = node_b.x - node_a.x
            dy = node_b.y - node_a.y
            length = np.sqrt(dx**2 + dy**2)
            if length < tolerance:
                raise ValueError(
                    f"Zero-length element in {element_type}: nodes {node_a_id} and {node_b_id} "
                    f"at position ({node_a.x:.6f}, {node_a.y:.6f})"
                )

        # Check span elements
        for span, span_node_ids in self.node_ids.items():
            for i in range(len(span_node_ids) - 1):
                node_a = span_node_ids[i]
                node_b = span_node_ids[i + 1]
                check_element_length(node_a, node_b, f"span {span}")
        return True

    def show_structure(self) -> None:
        """Display the beam structure using matplotlib."""
        self.system.show_structure()
