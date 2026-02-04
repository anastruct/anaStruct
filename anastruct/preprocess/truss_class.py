from abc import ABC, abstractmethod
from typing import Iterable, Literal, Optional, Sequence, Union, overload

import numpy as np

from anastruct.fem.system import SystemElements
from anastruct.fem.system_components.util import add_node
from anastruct.types import LoadDirection, SectionProps
from anastruct.vertex import Vertex

DEFAULT_TRUSS_SECTION: SectionProps = {
    "EI": 1e6,
    "EA": 1e8,
    "g": 0.0,
}


class Truss(ABC):
    """Abstract base class for 2D truss structures.

    Provides a framework for creating parametric truss geometries with automated
    node generation, connectivity, and support definitions. Subclasses implement
    specific truss types (Howe, Pratt, Warren, etc.).

    The truss generation follows a three-phase process:
    1. define_nodes() - Generate node coordinates
    2. define_connectivity() - Define which nodes connect to form elements
    3. define_supports() - Define support locations and types

    Attributes:
        width (float): Total span of the truss (length units)
        height (float): Height of the truss (length units)
        top_chord_section (SectionProps): Section properties for top chord elements
        bottom_chord_section (SectionProps): Section properties for bottom chord elements
        web_section (SectionProps): Section properties for diagonal web elements
        web_verticals_section (SectionProps): Section properties for vertical web elements
        top_chord_continuous (bool): If True, top chord is continuous; if False, pinned at joints
        bottom_chord_continuous (bool): If True, bottom chord is continuous; if False, pinned at joints
        supports_type (Literal["simple", "pinned", "fixed"]): Type of supports to apply
        system (SystemElements): The FEM system containing all nodes, elements, and supports
    """

    # Common geometry
    width: float
    height: float

    # Material properties
    top_chord_section: SectionProps
    bottom_chord_section: SectionProps
    web_section: SectionProps
    web_verticals_section: SectionProps

    # Configuration
    top_chord_continuous: bool
    bottom_chord_continuous: bool
    supports_type: Literal["simple", "pinned", "fixed"]

    # Defined by subclass (initialized in define_* methods)
    nodes: list[Vertex]
    top_chord_node_ids: Union[list[int], dict[str, list[int]]]
    bottom_chord_node_ids: Union[list[int], dict[str, list[int]]]
    web_node_pairs: list[tuple[int, int]]
    web_verticals_node_pairs: list[tuple[int, int]]
    support_definitions: dict[int, Literal["fixed", "pinned", "roller"]]
    top_chord_length: float
    bottom_chord_length: float

    # Defined by main class (initialized in add_elements)
    top_chord_element_ids: Union[list[int], dict[str, list[int]]]
    bottom_chord_element_ids: Union[list[int], dict[str, list[int]]]
    web_element_ids: list[int]
    web_verticals_element_ids: list[int]

    # System
    system: SystemElements

    def __init__(
        self,
        width: float,
        height: float,
        top_chord_section: Optional[SectionProps] = None,
        bottom_chord_section: Optional[SectionProps] = None,
        web_section: Optional[SectionProps] = None,
        web_verticals_section: Optional[SectionProps] = None,
        top_chord_continuous: bool = True,
        bottom_chord_continuous: bool = True,
        supports_type: Literal["simple", "pinned", "fixed"] = "simple",
    ):
        """Initialize a truss structure.

        Args:
            width (float): Total span of the truss. Must be positive.
            height (float): Height of the truss. Must be positive.
            top_chord_section (Optional[SectionProps]): Section properties for top chord.
                Defaults to DEFAULT_TRUSS_SECTION if not provided.
            bottom_chord_section (Optional[SectionProps]): Section properties for bottom chord.
                Defaults to DEFAULT_TRUSS_SECTION if not provided.
            web_section (Optional[SectionProps]): Section properties for diagonal web members.
                Defaults to DEFAULT_TRUSS_SECTION if not provided.
            web_verticals_section (Optional[SectionProps]): Section properties for vertical web members.
                Defaults to web_section if not provided.
            top_chord_continuous (bool): If True, top chord is continuous at joints (moment connection).
                If False, top chord is pinned at joints. Defaults to True.
            bottom_chord_continuous (bool): If True, bottom chord is continuous at joints.
                If False, bottom chord is pinned at joints. Defaults to True.
            supports_type (Literal["simple", "pinned", "fixed"]): Type of supports.
                "simple" creates pinned+roller, "pinned" creates pinned+pinned, "fixed" creates fixed+fixed.
                Defaults to "simple".

        Raises:
            ValueError: If width or height is not positive.
        """
        if width <= 0:
            raise ValueError(f"width must be positive, got {width}")
        if height <= 0:
            raise ValueError(f"height must be positive, got {height}")

        self.width = width
        self.height = height
        self.top_chord_section = top_chord_section or DEFAULT_TRUSS_SECTION
        self.bottom_chord_section = bottom_chord_section or DEFAULT_TRUSS_SECTION
        self.web_section = web_section or DEFAULT_TRUSS_SECTION
        self.web_verticals_section = web_verticals_section or self.web_section
        self.top_chord_continuous = top_chord_continuous
        self.bottom_chord_continuous = bottom_chord_continuous
        self.supports_type = supports_type

        # Initialize mutable attributes (prevents sharing between instances)
        self.nodes = []
        self.web_node_pairs = []
        self.web_verticals_node_pairs = []
        self.support_definitions = {}
        self.top_chord_length = 0.0
        self.bottom_chord_length = 0.0

        self.define_nodes()
        self.define_connectivity()
        self.define_supports()

        self.system = SystemElements()
        self.add_nodes()
        self.add_elements()
        self.add_supports()

    @property
    @abstractmethod
    def type(self) -> str:
        """Return the human-readable name of the truss type."""

    @abstractmethod
    def define_nodes(self) -> None:
        """Generate node coordinates and populate self.nodes list.

        Must be implemented by subclasses. Should create Vertex objects
        representing all node locations in the truss.
        """

    @abstractmethod
    def define_connectivity(self) -> None:
        """Define element connectivity by populating node ID lists.

        Must be implemented by subclasses. Should populate:
        - self.top_chord_node_ids
        - self.bottom_chord_node_ids
        - self.web_node_pairs
        - self.web_verticals_node_pairs
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

        Populates element ID lists:
        - self.top_chord_element_ids
        - self.bottom_chord_element_ids
        - self.web_element_ids
        - self.web_verticals_element_ids
        """

        def add_segment_elements(
            node_pairs: Iterable[tuple[int, int]],
            section: SectionProps,
            continuous: bool,
        ) -> list[int]:
            """Helper to add a sequence of connected elements.

            Args:
                node_pairs (Iterable[tuple[int, int]]): Pairs of node IDs to connect
                section (SectionProps): Section properties for the elements
                continuous (bool): If True, create moment connections; if False, pin connections

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
                        spring=None if continuous else {1: 0.0, 2: 0.0},
                    )
                )
            return element_ids

        # Bottom chord elements
        if isinstance(self.bottom_chord_node_ids, dict):
            self.bottom_chord_element_ids = {}
            for key, segment_node_ids in self.bottom_chord_node_ids.items():
                self.bottom_chord_element_ids[key] = add_segment_elements(
                    node_pairs=zip(segment_node_ids[:-1], segment_node_ids[1:]),
                    section=self.bottom_chord_section,
                    continuous=self.bottom_chord_continuous,
                )
        else:
            self.bottom_chord_element_ids = add_segment_elements(
                node_pairs=zip(
                    self.bottom_chord_node_ids[:-1], self.bottom_chord_node_ids[1:]
                ),
                section=self.bottom_chord_section,
                continuous=self.bottom_chord_continuous,
            )

        # Top chord elements
        if isinstance(self.top_chord_node_ids, dict):
            self.top_chord_element_ids = {}
            for key, segment_node_ids in self.top_chord_node_ids.items():
                self.top_chord_element_ids[key] = add_segment_elements(
                    node_pairs=zip(segment_node_ids[:-1], segment_node_ids[1:]),
                    section=self.top_chord_section,
                    continuous=self.top_chord_continuous,
                )
        else:
            self.top_chord_element_ids = add_segment_elements(
                node_pairs=zip(
                    self.top_chord_node_ids[:-1], self.top_chord_node_ids[1:]
                ),
                section=self.top_chord_section,
                continuous=self.top_chord_continuous,
            )

        # Web diagonal elements
        self.web_element_ids = add_segment_elements(
            node_pairs=self.web_node_pairs,
            section=self.web_section,
            continuous=False,
        )

        # Web vertical elements
        self.web_verticals_element_ids = add_segment_elements(
            node_pairs=self.web_verticals_node_pairs,
            section=self.web_verticals_section,
            continuous=False,
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

    def _resolve_support_type(
        self, is_primary: bool = True
    ) -> Literal["fixed", "pinned", "roller"]:
        """Helper to resolve support type from "simple" to specific type.

        Args:
            is_primary (bool): If True, this is the primary (left) support.
                If False, this is the secondary (right) support.

        Returns:
            Literal["fixed", "pinned", "roller"]: The resolved support type.
                For "simple", returns "pinned" if primary, "roller" if secondary.
        """
        if self.supports_type != "simple":
            return self.supports_type
        return "pinned" if is_primary else "roller"

    @overload
    def get_element_ids_of_chord(
        self, chord: Literal["top", "bottom"], chord_segment: None = None
    ) -> list[int]: ...

    @overload
    def get_element_ids_of_chord(
        self, chord: Literal["top", "bottom"], chord_segment: str
    ) -> list[int]: ...

    def get_element_ids_of_chord(
        self, chord: Literal["top", "bottom"], chord_segment: Optional[str] = None
    ) -> list[int]:
        """Get element IDs for a chord (top or bottom).

        Args:
            chord (Literal["top", "bottom"]): Which chord to query
            chord_segment (Optional[str]): If the chord is segmented (dict of segments),
                specify which segment to get. If None and chord is segmented, returns
                all element IDs from all segments concatenated.

        Returns:
            list[int]: Element IDs of the requested chord (segment)

        Raises:
            ValueError: If chord is not "top" or "bottom"
            KeyError: If chord_segment is specified but doesn't exist in the chord
        """
        if chord == "top":
            if isinstance(self.top_chord_element_ids, dict):
                if chord_segment is None:
                    all_ids = []
                    for ids in self.top_chord_element_ids.values():
                        all_ids.extend(ids)
                    return all_ids
                if chord_segment not in self.top_chord_element_ids:
                    available = list(self.top_chord_element_ids.keys())
                    raise KeyError(
                        f"chord_segment '{chord_segment}' not found. "
                        f"Available segments: {available}"
                    )
                return self.top_chord_element_ids[chord_segment]
            return self.top_chord_element_ids

        if chord == "bottom":
            if isinstance(self.bottom_chord_element_ids, dict):
                if chord_segment is None:
                    all_ids = []
                    for ids in self.bottom_chord_element_ids.values():
                        all_ids.extend(ids)
                    return all_ids
                if chord_segment not in self.bottom_chord_element_ids:
                    available = list(self.bottom_chord_element_ids.keys())
                    raise KeyError(
                        f"chord_segment '{chord_segment}' not found. "
                        f"Available segments: {available}"
                    )
                return self.bottom_chord_element_ids[chord_segment]
            return self.bottom_chord_element_ids

        raise ValueError("chord must be either 'top' or 'bottom'.")

    def apply_q_load_to_top_chord(
        self,
        q: Union[float, Sequence[float]],
        direction: Union[LoadDirection, Sequence[LoadDirection]] = "element",
        rotation: Optional[Union[float, Sequence[float]]] = None,
        q_perp: Optional[Union[float, Sequence[float]]] = None,
        chord_segment: Optional[str] = None,
    ) -> None:
        """Apply distributed load to all elements in the top chord.

        Args:
            q (Union[float, Sequence[float]]): Load magnitude (force/length units)
            direction (Union[LoadDirection, Sequence[LoadDirection]]): Load direction.
                Options: "element", "x", "y", "parallel", "perpendicular", "angle"
            rotation (Optional[Union[float, Sequence[float]]]): Rotation angle in degrees
                (used with direction="angle")
            q_perp (Optional[Union[float, Sequence[float]]]): Perpendicular load component
            chord_segment (Optional[str]): If specified, apply load only to this segment
                (for trusses with segmented chords like roof trusses)
        """
        element_ids = self.get_element_ids_of_chord(
            chord="top", chord_segment=chord_segment
        )
        for el_id in element_ids:
            self.system.q_load(
                element_id=el_id,
                q=q,
                direction=direction,
                rotation=rotation,
                q_perp=q_perp,
            )

    def apply_q_load_to_bottom_chord(
        self,
        q: Union[float, Sequence[float]],
        direction: Union[LoadDirection, Sequence[LoadDirection]] = "element",
        rotation: Optional[Union[float, Sequence[float]]] = None,
        q_perp: Optional[Union[float, Sequence[float]]] = None,
        chord_segment: Optional[str] = None,
    ) -> None:
        """Apply distributed load to all elements in the bottom chord.

        Args:
            q (Union[float, Sequence[float]]): Load magnitude (force/length units)
            direction (Union[LoadDirection, Sequence[LoadDirection]]): Load direction.
                Options: "element", "x", "y", "parallel", "perpendicular", "angle"
            rotation (Optional[Union[float, Sequence[float]]]): Rotation angle in degrees
                (used with direction="angle")
            q_perp (Optional[Union[float, Sequence[float]]]): Perpendicular load component
            chord_segment (Optional[str]): If specified, apply load only to this segment
                (for trusses with segmented chords like roof trusses)
        """
        element_ids = self.get_element_ids_of_chord(
            chord="bottom", chord_segment=chord_segment
        )
        for el_id in element_ids:
            self.system.q_load(
                element_id=el_id,
                q=q,
                direction=direction,
                rotation=rotation,
                q_perp=q_perp,
            )

    def validate(self) -> bool:
        """Validate truss geometry and connectivity.

        Checks for common truss definition issues:
        - All node IDs in connectivity lists reference valid nodes
        - No duplicate nodes at the same location
        - All elements have non-zero length

        Returns:
            bool: True if validation passes

        Raises:
            ValueError: If validation fails with description of the issue
        """
        # Check that all node IDs in connectivity are valid (1-based)
        max_node_id = len(self.nodes)

        # Helper to validate node ID list
        def validate_node_ids(
            node_ids: Union[list[int], dict[str, list[int]]], name: str
        ) -> None:
            if isinstance(node_ids, dict):
                for segment_name, ids in node_ids.items():
                    for node_id in ids:
                        if node_id < 1 or node_id > max_node_id:
                            raise ValueError(
                                f"{name} segment '{segment_name}' references invalid node ID {node_id}. "
                                f"Valid range: 1-{max_node_id}"
                            )
            else:
                for node_id in node_ids:
                    if node_id < 1 or node_id > max_node_id:
                        raise ValueError(
                            f"{name} references invalid node ID {node_id}. "
                            f"Valid range: 1-{max_node_id}"
                        )

        validate_node_ids(self.top_chord_node_ids, "top_chord_node_ids")
        validate_node_ids(self.bottom_chord_node_ids, "bottom_chord_node_ids")

        for i, (node_a, node_b) in enumerate(self.web_node_pairs):
            if node_a < 1 or node_a > max_node_id:
                raise ValueError(
                    f"web_node_pairs[{i}] references invalid node ID {node_a}. "
                    f"Valid range: 1-{max_node_id}"
                )
            if node_b < 1 or node_b > max_node_id:
                raise ValueError(
                    f"web_node_pairs[{i}] references invalid node ID {node_b}. "
                    f"Valid range: 1-{max_node_id}"
                )

        for i, (node_a, node_b) in enumerate(self.web_verticals_node_pairs):
            if node_a < 1 or node_a > max_node_id:
                raise ValueError(
                    f"web_verticals_node_pairs[{i}] references invalid node ID {node_a}. "
                    f"Valid range: 1-{max_node_id}"
                )
            if node_b < 1 or node_b > max_node_id:
                raise ValueError(
                    f"web_verticals_node_pairs[{i}] references invalid node ID {node_b}. "
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

        # Check chord elements
        def check_chord_elements(
            node_ids: Union[list[int], dict[str, list[int]]], chord_name: str
        ) -> None:
            if isinstance(node_ids, dict):
                for segment_name, ids in node_ids.items():
                    for i in range(len(ids) - 1):
                        check_element_length(
                            ids[i], ids[i + 1], f"{chord_name} segment '{segment_name}'"
                        )
            else:
                for i in range(len(node_ids) - 1):
                    check_element_length(node_ids[i], node_ids[i + 1], chord_name)

        check_chord_elements(self.top_chord_node_ids, "top chord")
        check_chord_elements(self.bottom_chord_node_ids, "bottom chord")

        for i, (node_a, node_b) in enumerate(self.web_node_pairs):
            check_element_length(node_a, node_b, f"web diagonal {i}")

        for i, (node_a, node_b) in enumerate(self.web_verticals_node_pairs):
            check_element_length(node_a, node_b, f"web vertical {i}")

        return True

    def show_structure(self) -> None:
        """Display the truss structure using matplotlib."""
        self.system.show_structure()


class FlatTruss(Truss):
    """Abstract base class for flat (parallel chord) truss structures.

    Flat trusses have parallel top and bottom chords and are divided into
    repeating panel units. Specific truss patterns (Howe, Pratt, Warren)
    are implemented by subclasses.

    Attributes:
        unit_width (float): Width of each panel/bay
        end_type (EndType): Configuration of truss ends - "flat", "triangle_down", or "triangle_up"
        supports_loc (SupportLoc): Where supports are placed - "bottom_chord", "top_chord", or "both"
        min_end_fraction (float): Minimum width of end panels as fraction of unit_width
        enforce_even_units (bool): If True, ensure even number of panels for symmetry
        n_units (int): Computed number of panel units
        end_width (float): Computed width of end panels
    """

    # Data types specific to this truss type
    EndType = Literal["flat", "triangle_down", "triangle_up"]
    SupportLoc = Literal["bottom_chord", "top_chord", "both"]

    # Additional geometry for this truss type
    unit_width: float
    end_type: EndType
    supports_loc: SupportLoc

    # Additional configuration
    min_end_fraction: float
    enforce_even_units: bool

    # Computed properties
    n_units: int
    end_width: float

    @property
    @abstractmethod
    def type(self) -> str:
        return "[Generic] Flat Truss"

    def __init__(
        self,
        width: float,
        height: float,
        unit_width: float,
        end_type: EndType = "triangle_down",
        supports_loc: SupportLoc = "bottom_chord",
        min_end_fraction: float = 0.5,
        enforce_even_units: bool = True,
        top_chord_section: Optional[SectionProps] = None,
        bottom_chord_section: Optional[SectionProps] = None,
        web_section: Optional[SectionProps] = None,
        web_verticals_section: Optional[SectionProps] = None,
    ):
        """Initialize a flat truss.

        Args:
            width (float): Total span of the truss. Must be positive.
            height (float): Height of the truss. Must be positive.
            unit_width (float): Width of each panel. Must be positive and less than
                width - 2*min_end_fraction*unit_width.
            end_type (EndType): End panel configuration. "triangle_down" has diagonals
                pointing down at ends, "triangle_up" has diagonals pointing up,
                "flat" has vertical end panels.
            supports_loc (SupportLoc): Location of supports - "bottom_chord" (typical),
                "top_chord" (hanging truss), or "both" (supported at both chords).
            min_end_fraction (float): Minimum end panel width as fraction of unit_width.
                Must be between 0 and 1. Defaults to 0.5.
            enforce_even_units (bool): If True, ensure even number of units for symmetry.
                Defaults to True.
            top_chord_section (Optional[SectionProps]): Section properties for top chord
            bottom_chord_section (Optional[SectionProps]): Section properties for bottom chord
            web_section (Optional[SectionProps]): Section properties for diagonal webs
            web_verticals_section (Optional[SectionProps]): Section properties for vertical webs

        Raises:
            ValueError: If dimensions are invalid or result in negative/zero units
        """
        if unit_width <= 0:
            raise ValueError(f"unit_width must be positive, got {unit_width}")
        if not 0 < min_end_fraction <= 1:
            raise ValueError(
                f"min_end_fraction must be in (0, 1], got {min_end_fraction}"
            )

        self.unit_width = unit_width
        self.end_type = end_type
        self.supports_loc = supports_loc
        self.min_end_fraction = min_end_fraction
        self.enforce_even_units = enforce_even_units

        # Compute number of units
        n_units_float = (width - unit_width * 2 * min_end_fraction) / unit_width
        if n_units_float < 1:
            raise ValueError(
                f"Width {width} is too small for unit_width {unit_width} and "
                f"min_end_fraction {min_end_fraction}. Would result in {n_units_float:.2f} units."
            )

        self.n_units = int(np.floor(n_units_float))
        if self.enforce_even_units and self.n_units % 2 != 0:
            self.n_units -= 1

        if self.n_units < 2:
            raise ValueError(
                f"Truss must have at least 2 units. Computed {self.n_units} units. "
                f"Reduce unit_width or increase width."
            )

        self.end_width = (width - self.n_units * unit_width) / 2
        super().__init__(
            width,
            height,
            top_chord_section,
            bottom_chord_section,
            web_section,
            web_verticals_section,
        )

    @abstractmethod
    def define_nodes(self) -> None:
        pass

    @abstractmethod
    def define_connectivity(self) -> None:
        pass

    def define_supports(self) -> None:
        """Define support locations for flat trusses.

        Default implementation places supports at the ends of the truss.
        Assumes single-segment (non-dict) chord node ID lists.
        """
        assert isinstance(self.bottom_chord_node_ids, list)
        assert isinstance(self.top_chord_node_ids, list)
        bottom_left = 1
        bottom_right = max(self.bottom_chord_node_ids)
        top_left = min(self.top_chord_node_ids)
        top_right = max(self.top_chord_node_ids)
        if self.supports_loc in ["bottom_chord", "both"]:
            self.support_definitions[bottom_left] = self._resolve_support_type(
                is_primary=True
            )
            self.support_definitions[bottom_right] = self._resolve_support_type(
                is_primary=False
            )
        if self.supports_loc in ["top_chord", "both"]:
            self.support_definitions[top_left] = self._resolve_support_type(
                is_primary=True
            )
            self.support_definitions[top_right] = self._resolve_support_type(
                is_primary=False
            )


class RoofTruss(Truss):
    """Abstract base class for peaked roof truss structures.

    Roof trusses have sloped top chords meeting at a peak, forming a triangular
    profile. Height is computed from span and roof pitch. Specific truss patterns
    (King Post, Queen Post, Fink, etc.) are implemented by subclasses.

    Attributes:
        overhang_length (float): Length of roof overhang beyond supports
        roof_pitch_deg (float): Roof pitch angle in degrees
        roof_pitch (float): Roof pitch angle in radians (computed)
    """

    # Additional geometry for this truss type
    overhang_length: float
    roof_pitch_deg: float

    # Computed properties
    roof_pitch: float

    @property
    @abstractmethod
    def type(self) -> str:
        return "[Generic] Roof Truss"

    def __init__(
        self,
        width: float,
        roof_pitch_deg: float,
        overhang_length: float = 0.0,
        top_chord_section: Optional[SectionProps] = None,
        bottom_chord_section: Optional[SectionProps] = None,
        web_section: Optional[SectionProps] = None,
        web_verticals_section: Optional[SectionProps] = None,
    ):
        """Initialize a roof truss.

        Args:
            width (float): Total span of the truss (building width). Must be positive.
            roof_pitch_deg (float): Roof pitch angle in degrees. Must be positive and
                less than 90 degrees. Common values: 18-45 degrees.
            overhang_length (float): Length of roof overhang beyond the supports.
                Must be non-negative. Defaults to 0.0.
            top_chord_section (Optional[SectionProps]): Section properties for top chord
            bottom_chord_section (Optional[SectionProps]): Section properties for bottom chord
            web_section (Optional[SectionProps]): Section properties for diagonal webs
            web_verticals_section (Optional[SectionProps]): Section properties for vertical webs

        Raises:
            ValueError: If dimensions or angles are invalid
        """
        if roof_pitch_deg <= 0 or roof_pitch_deg >= 90:
            raise ValueError(
                f"roof_pitch_deg must be between 0 and 90, got {roof_pitch_deg}"
            )
        if overhang_length < 0:
            raise ValueError(
                f"overhang_length must be non-negative, got {overhang_length}"
            )

        self.roof_pitch_deg = roof_pitch_deg
        self.roof_pitch = np.radians(roof_pitch_deg)
        height = (width / 2) * np.tan(self.roof_pitch)
        self.overhang_length = overhang_length
        super().__init__(
            width,
            height,
            top_chord_section,
            bottom_chord_section,
            web_section,
            web_verticals_section,
        )

    @abstractmethod
    def define_nodes(self) -> None:
        pass

    @abstractmethod
    def define_connectivity(self) -> None:
        pass

    def define_supports(self) -> None:
        """Define support locations for roof trusses.

        Default implementation places supports at the ends of the bottom chord.
        Assumes single-segment (non-dict) bottom chord node ID list.
        """
        assert isinstance(self.bottom_chord_node_ids, list)

        bottom_left = 1
        bottom_right = max(self.bottom_chord_node_ids)
        self.support_definitions[bottom_left] = self._resolve_support_type(
            is_primary=True
        )
        self.support_definitions[bottom_right] = self._resolve_support_type(
            is_primary=False
        )
