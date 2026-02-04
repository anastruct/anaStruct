from typing import Any, Literal, Optional

import numpy as np

from anastruct.preprocess.truss_class import FlatTruss, RoofTruss, Truss
from anastruct.types import SectionProps
from anastruct.vertex import Vertex


class HoweFlatTruss(FlatTruss):
    """Howe flat truss with vertical web members and diagonal members in compression.

    The Howe truss features vertical web members and diagonal members sloping toward
    the center. Under gravity loads, diagonals are typically in compression and
    verticals in tension, making it efficient for steel trusses.
    """

    @property
    def type(self) -> str:
        return "Howe Flat Truss"

    def define_nodes(self) -> None:
        # Bottom chord nodes
        if self.end_type != "triangle_up":
            self.nodes.append(Vertex(0.0, 0.0))
        for i in range(int(self.n_units) + 1):
            x = self.end_width + i * self.unit_width
            self.nodes.append(Vertex(x, 0.0))
        if self.end_type != "triangle_up":
            self.nodes.append(Vertex(self.width, 0.0))

        # Top chord nodes
        if self.end_type != "triangle_down":
            self.nodes.append(Vertex(0, self.height))
        for i in range(int(self.n_units) + 1):
            x = self.end_width + i * self.unit_width
            self.nodes.append(Vertex(x, self.height))
        if self.end_type != "triangle_down":
            self.nodes.append(Vertex(self.width, self.height))

    def define_connectivity(self) -> None:
        n_bottom_nodes = (
            int(self.n_units) + 1 + (2 if self.end_type != "triangle_up" else 0)
        )
        n_top_nodes = (
            int(self.n_units) + 1 + (2 if self.end_type != "triangle_down" else 0)
        )

        # Bottom chord connectivity
        self.bottom_chord_node_ids = list(range(1, n_bottom_nodes + 1))

        # Top chord connectivity
        self.top_chord_node_ids = list(
            range(n_bottom_nodes + 1, n_bottom_nodes + n_top_nodes + 1)
        )

        # Web diagonals connectivity
        start_bot = 0
        start_top = 0
        end_bot = None  # a None index means go to the end
        end_top = None
        if self.end_type == "triangle_up":
            # special case: end diagonal slopes in the opposite direction
            self.web_node_pairs.append((1, n_bottom_nodes + 1))
            self.web_node_pairs.append(
                (n_bottom_nodes, n_bottom_nodes + n_top_nodes)
            )
            start_top = 2
            end_top = -3
        elif self.end_type == "flat":
            start_top = 1
            end_top = -2
        mid_bot = len(self.bottom_chord_node_ids) // 2
        mid_top = len(self.top_chord_node_ids) // 2
        for b, t in zip(
            self.bottom_chord_node_ids[start_bot : mid_bot + 1],
            self.top_chord_node_ids[start_top : mid_top + 1],
        ):
            self.web_node_pairs.append((b, t))
        for b, t in zip(
            self.bottom_chord_node_ids[end_bot : mid_bot - 1 : -1],
            self.top_chord_node_ids[end_top : mid_top - 1 : -1],
        ):
            self.web_node_pairs.append((b, t))

        # Web verticals connectivity
        start_bot = 0
        start_top = 0
        end_bot = None
        end_top = None
        if self.end_type == "triangle_up":
            start_top = 1
            end_top = -1
        elif self.end_type == "triangle_down":
            start_bot = 1
            end_bot = -1
        for b, t in zip(
            self.bottom_chord_node_ids[start_bot:end_bot],
            self.top_chord_node_ids[start_top:end_top],
        ):
            self.web_verticals_node_pairs.append((b, t))


class PrattFlatTruss(FlatTruss):
    """Pratt flat truss with vertical web members and diagonal members in tension.

    The Pratt truss features vertical web members and diagonal members sloping away
    from the center. Under gravity loads, diagonals are typically in tension and
    verticals in compression, making it efficient for a wide range of applications.
    """

    @property
    def type(self) -> str:
        return "Pratt Flat Truss"

    def define_nodes(self) -> None:
        # Bottom chord nodes
        if self.end_type != "triangle_up":
            self.nodes.append(Vertex(0.0, 0.0))
        for i in range(int(self.n_units) + 1):
            x = self.end_width + i * self.unit_width
            self.nodes.append(Vertex(x, 0.0))
        if self.end_type != "triangle_up":
            self.nodes.append(Vertex(self.width, 0.0))

        # Top chord nodes
        if self.end_type != "triangle_down":
            self.nodes.append(Vertex(0, self.height))
        for i in range(int(self.n_units) + 1):
            x = self.end_width + i * self.unit_width
            self.nodes.append(Vertex(x, self.height))
        if self.end_type != "triangle_down":
            self.nodes.append(Vertex(self.width, self.height))

    def define_connectivity(self) -> None:
        n_bottom_nodes = (
            int(self.n_units) + 1 + (2 if self.end_type != "triangle_up" else 0)
        )
        n_top_nodes = (
            int(self.n_units) + 1 + (2 if self.end_type != "triangle_down" else 0)
        )

        # Bottom chord connectivity
        self.bottom_chord_node_ids = list(range(1, n_bottom_nodes + 1))

        # Top chord connectivity
        self.top_chord_node_ids = list(
            range(n_bottom_nodes + 1, n_bottom_nodes + n_top_nodes + 1)
        )

        # Web diagonals connectivity
        start_bot = 0
        start_top = 0
        end_bot = None  # a None index means go to the end
        end_top = None
        if self.end_type == "triangle_down":
            # special case: end diagonal slopes in the opposite direction
            self.web_node_pairs.append((n_bottom_nodes + 1, 1))
            self.web_node_pairs.append(
                (n_bottom_nodes + n_top_nodes, n_bottom_nodes)
            )
            start_bot = 2
            end_bot = -3
        elif self.end_type == "flat":
            start_bot = 1
            end_bot = -2
        mid_bot = len(self.bottom_chord_node_ids) // 2
        mid_top = len(self.top_chord_node_ids) // 2
        for b, t in zip(
            self.bottom_chord_node_ids[start_bot : mid_bot + 1],
            self.top_chord_node_ids[start_top : mid_top + 1],
        ):
            self.web_node_pairs.append((b, t))
        for b, t in zip(
            self.bottom_chord_node_ids[end_bot : mid_bot - 1 : -1],
            self.top_chord_node_ids[end_top : mid_top - 1 : -1],
        ):
            self.web_node_pairs.append((b, t))

        # Web verticals connectivity
        start_bot = 0
        start_top = 0
        end_bot = None
        end_top = None
        if self.end_type == "triangle_up":
            start_top = 1
            end_top = -1
        elif self.end_type == "triangle_down":
            start_bot = 1
            end_bot = -1
        for b, t in zip(
            self.bottom_chord_node_ids[start_bot:end_bot],
            self.top_chord_node_ids[start_top:end_top],
        ):
            self.web_verticals_node_pairs.append((b, t))


class WarrenFlatTruss(FlatTruss):
    """Warren flat truss with diagonal-only web members forming a zigzag pattern.

    The Warren truss has no vertical web members (except optionally at midspan).
    Diagonal members alternate direction, creating a series of equilateral or
    isosceles triangles. This configuration is simple and efficient.

    Note: Warren trusses don't support the "flat" end_type - only "triangle_down"
    or "triangle_up".
    """

    # Data types specific to this truss type
    EndType = Literal["triangle_down", "triangle_up"]
    SupportLoc = Literal["bottom_chord", "top_chord", "both"]

    # Additional geometry for this truss type
    unit_width: float
    end_type: EndType
    supports_loc: SupportLoc

    # Computed properties
    n_units: int
    end_width: float

    @property
    def type(self) -> str:
        return "Warren Flat Truss"

    def __init__(
        self,
        width: float,
        height: float,
        unit_width: float,
        end_type: EndType = "triangle_down",
        supports_loc: SupportLoc = "bottom_chord",
        top_chord_section: Optional[SectionProps] = None,
        bottom_chord_section: Optional[SectionProps] = None,
        web_section: Optional[SectionProps] = None,
        web_verticals_section: Optional[SectionProps] = None,
    ):
        # Note that the maths for a Warren truss is simpler than for Howe/Pratt, because there
        # cannot be any option for non-even number of units, and there are no special cases for
        # web verticals.
        min_end_fraction = 0.5  # Not used for Warren truss
        enforce_even_units = True  # Handled internally for Warren truss
        super().__init__(
            width,
            height,
            unit_width,
            end_type,
            supports_loc,
            min_end_fraction,
            enforce_even_units,
            top_chord_section,
            bottom_chord_section,
            web_section,
            web_verticals_section,
        )
        self.end_width = (width - self.n_units * unit_width) / 2 + (unit_width / 2)

    def define_nodes(self) -> None:
        # Bottom chord nodes
        if self.end_type == "triangle_down":
            self.nodes.append(Vertex(0.0, 0.0))
        else:
            self.nodes.append(Vertex(self.end_width - self.unit_width / 2, 0.0))
        for i in range(int(self.n_units) + 1):
            x = self.end_width + i * self.unit_width
            self.nodes.append(Vertex(x, 0.0))
        if self.end_type == "triangle_down":
            self.nodes.append(Vertex(self.width, 0.0))
        else:
            self.nodes.append(
                Vertex(self.width - (self.end_width - self.unit_width / 2), 0.0)
            )

        # Top chord nodes
        if self.end_type == "triangle_up":
            self.nodes.append(Vertex(0, self.height))
        else:
            self.nodes.append(Vertex(self.end_width - self.unit_width / 2, self.height))
        for i in range(int(self.n_units) + 1):
            x = self.end_width + i * self.unit_width
            self.nodes.append(Vertex(x, self.height))
        if self.end_type == "triangle_up":
            self.nodes.append(Vertex(self.width, self.height))
        else:
            self.nodes.append(
                Vertex(self.width - (self.end_width - self.unit_width / 2), self.height)
            )

    def define_connectivity(self) -> None:
        n_bottom_nodes = int(self.n_units) + (
            1 if self.end_type == "triangle_down" else 0
        )
        n_top_nodes = int(self.n_units) + (1 if self.end_type == "triangle_up" else 0)

        # Bottom chord connectivity
        self.bottom_chord_node_ids = list(range(1, n_bottom_nodes + 1))

        # Top chord connectivity
        self.top_chord_node_ids = list(
            range(n_bottom_nodes + 1, n_bottom_nodes + n_top_nodes + 1)
        )

        # Web diagonals connectivity
        # sloping up from bottom left to top right
        top_start = 0 if self.end_type == "triangle_down" else 1
        for b, t in zip(
            self.bottom_chord_node_ids,
            self.top_chord_node_ids[top_start:],
        ):
            self.web_node_pairs.append((b, t))
        # sloping down from top left to bottom right
        bot_start = 0 if self.end_type == "triangle_up" else 1
        for b, t in zip(
            self.top_chord_node_ids,
            self.bottom_chord_node_ids[bot_start:],
        ):
            self.web_node_pairs.append((b, t))


class KingPostRoofTruss(RoofTruss):
    """King Post roof truss - simplest pitched roof truss with single center vertical.

    Features a single vertical member (king post) at the center supporting the peak.
    Suitable for short spans (up to ~8m). No diagonal web members.
    """

    @property
    def type(self) -> str:
        return "King Post Roof Truss"

    def define_nodes(self) -> None:
        # Bottom chord nodes
        self.nodes.append(Vertex(0.0, 0.0))
        self.nodes.append(Vertex(self.width / 2, 0.0))
        self.nodes.append(Vertex(self.width, 0.0))

        # Top chord nodes
        # self.nodes.append(Vertex(0.0, 0.0))
        self.nodes.append(Vertex(self.width / 2, self.height))
        # self.nodes.append(Vertex(self.width, 0.0))
        if self.overhang_length > 0:
            self.nodes.append(
                Vertex(
                    -self.overhang_length * np.cos(self.roof_pitch),
                    -self.overhang_length * np.sin(self.roof_pitch),
                )
            )
            self.nodes.append(
                Vertex(
                    self.width + self.overhang_length * np.cos(self.roof_pitch),
                    -self.overhang_length * np.sin(self.roof_pitch),
                )
            )

    def define_connectivity(self) -> None:
        # Bottom chord connectivity
        self.bottom_chord_node_ids = [1, 2, 3]
        left_v = 1
        right_v = 3

        # Top chord connectivity (left and right slopes stored separately)
        self.top_chord_node_ids = {"left": [left_v, 4], "right": [4, right_v]}
        if self.overhang_length > 0:
            self.top_chord_node_ids["left"].insert(0, 5)  # left overhang
            self.top_chord_node_ids["right"].append(6)  # right overhang

        # Web verticals connectivity
        self.web_verticals_node_pairs.append((2, 4))  # center vertical


class QueenPostRoofTruss(RoofTruss):
    """Queen Post roof truss with two vertical members and diagonal bracing.

    Features two vertical members (queen posts) at quarter points with diagonal
    members from center to quarter points. Suitable for medium spans (8-15m).
    More efficient than King Post for longer spans.
    """

    @property
    def type(self) -> str:
        return "Queen Post Roof Truss"

    def define_nodes(self) -> None:
        # Bottom chord nodes: [0=left, 1=center, 2=right]
        self.nodes.append(Vertex(0.0, 0.0))
        self.nodes.append(Vertex(self.width / 2, 0.0))
        self.nodes.append(Vertex(self.width, 0.0))

        # Top chord nodes: [3=left quarter, 4=peak, 5=right quarter]
        self.nodes.append(Vertex(self.width / 4, self.height / 2))
        self.nodes.append(Vertex(self.width / 2, self.height))
        self.nodes.append(Vertex(3 * self.width / 4, self.height / 2))

        # Optional overhang nodes
        if self.overhang_length > 0:
            self.nodes.append(
                Vertex(
                    -self.overhang_length * np.cos(self.roof_pitch),
                    -self.overhang_length * np.sin(self.roof_pitch),
                )
            )
            self.nodes.append(
                Vertex(
                    self.width + self.overhang_length * np.cos(self.roof_pitch),
                    -self.overhang_length * np.sin(self.roof_pitch),
                )
            )

    def define_connectivity(self) -> None:
        # Bottom chord connectivity
        self.bottom_chord_node_ids = [1, 2, 3]
        left_v = 1
        right_v = 3

        # Top chord connectivity (left and right slopes stored separately)
        self.top_chord_node_ids = {"left": [left_v, 4, 5], "right": [5, 6, right_v]}
        if self.overhang_length > 0:
            self.top_chord_node_ids["left"].insert(0, 7)  # left overhang
            self.top_chord_node_ids["right"].append(8)  # right overhang

        # Web diagonals connectivity
        self.web_node_pairs.append(
            (2, 4)
        )  # left diagonal from center bottom to left quarter top
        self.web_node_pairs.append(
            (2, 6)
        )  # right diagonal from center bottom to right quarter top

        # Web verticals connectivity - Fixed: should connect to peak (node 5), not node 4
        self.web_verticals_node_pairs.append(
            (2, 5)
        )  # center vertical from center bottom to peak


class FinkRoofTruss(RoofTruss):
    """Fink roof truss with W-shaped web configuration.

    Features diagonal members forming a W pattern between peak and supports.
    Efficient for medium to long spans (10-20m). The symmetrical W pattern
    distributes loads effectively with minimal material usage.
    """

    @property
    def type(self) -> str:
        return "Fink Roof Truss"

    def define_nodes(self) -> None:
        # Bottom chord nodes
        self.nodes.append(Vertex(0.0, 0.0))
        self.nodes.append(Vertex(1 * self.width / 3, 0.0))
        self.nodes.append(Vertex(2 * self.width / 3, 0.0))
        self.nodes.append(Vertex(self.width, 0.0))

        # Top chord nodes
        # self.nodes.append(Vertex(0.0, 0.0))
        self.nodes.append(Vertex(1 * self.width / 4, self.height / 2))
        self.nodes.append(Vertex(self.width / 2, self.height))
        self.nodes.append(Vertex(3 * self.width / 4, self.height / 2))
        # self.nodes.append(Vertex(self.width, 0.0))
        if self.overhang_length > 0:
            self.nodes.append(
                Vertex(
                    -self.overhang_length * np.cos(self.roof_pitch),
                    -self.overhang_length * np.sin(self.roof_pitch),
                )
            )
            self.nodes.append(
                Vertex(
                    self.width + self.overhang_length * np.cos(self.roof_pitch),
                    -self.overhang_length * np.sin(self.roof_pitch),
                )
            )

    def define_connectivity(self) -> None:
        # Bottom chord connectivity
        self.bottom_chord_node_ids = [1, 2, 3, 4]
        left_v = 1
        right_v = 4

        # Top chord connectivity (left and right slopes stored separately)
        self.top_chord_node_ids = {"left": [left_v, 5, 6], "right": [6, 7, right_v]}
        if self.overhang_length > 0:
            self.top_chord_node_ids["left"].insert(0, 8)  # left overhang
            self.top_chord_node_ids["right"].append(9)  # right overhang

        # Web diagonals connectivity
        self.web_node_pairs.append((2, 5))
        self.web_node_pairs.append((2, 6))
        self.web_node_pairs.append((3, 6))
        self.web_node_pairs.append((3, 7))


class HoweRoofTruss(RoofTruss):
    """Howe roof truss with vertical posts and diagonal compression members.

    Features vertical posts with diagonals sloping toward the peak. Under gravity
    loads, diagonals are in compression and verticals in tension. Suitable for
    medium to long spans with good load distribution.
    """

    @property
    def type(self) -> str:
        return "Howe Roof Truss"

    def define_nodes(self) -> None:
        # Bottom chord nodes
        self.nodes.append(Vertex(0.0, 0.0))
        self.nodes.append(Vertex(1 * self.width / 4, 0.0))
        self.nodes.append(Vertex(self.width / 2, 0.0))
        self.nodes.append(Vertex(3 * self.width / 4, 0.0))
        self.nodes.append(Vertex(self.width, 0.0))

        # Top chord nodes
        # self.nodes.append(Vertex(0.0, 0.0))
        self.nodes.append(Vertex(1 * self.width / 4, self.height / 2))
        self.nodes.append(Vertex(self.width / 2, self.height))
        self.nodes.append(Vertex(3 * self.width / 4, self.height / 2))
        # self.nodes.append(Vertex(self.width, 0.0))
        if self.overhang_length > 0:
            self.nodes.append(
                Vertex(
                    -self.overhang_length * np.cos(self.roof_pitch),
                    -self.overhang_length * np.sin(self.roof_pitch),
                )
            )
            self.nodes.append(
                Vertex(
                    self.width + self.overhang_length * np.cos(self.roof_pitch),
                    -self.overhang_length * np.sin(self.roof_pitch),
                )
            )

    def define_connectivity(self) -> None:
        # Bottom chord connectivity
        self.bottom_chord_node_ids = [1, 2, 3, 4, 5]
        left_v = 1
        right_v = 5

        # Top chord connectivity (left and right slopes stored separately)
        self.top_chord_node_ids = {"left": [left_v, 6, 7], "right": [7, 8, right_v]}
        if self.overhang_length > 0:
            self.top_chord_node_ids["left"].insert(0, 9)  # left overhang
            self.top_chord_node_ids["right"].append(10)  # right overhang

        # Web diagonals connectivity
        self.web_node_pairs.append((3, 6))  # left diagonal
        self.web_node_pairs.append((3, 8))  # right diagonal

        # Web verticals connectivity
        self.web_verticals_node_pairs.append((2, 6))  # left vertical
        self.web_verticals_node_pairs.append((3, 7))  # centre vertical
        self.web_verticals_node_pairs.append((4, 8))  # right vertical


class PrattRoofTruss(RoofTruss):
    """Pratt roof truss with vertical posts and diagonal tension members.

    Features vertical posts with diagonals sloping away from the peak. Under gravity
    loads, diagonals are in tension and verticals in compression. Widely used for
    its efficiency and simple construction.
    """

    @property
    def type(self) -> str:
        return "Pratt Roof Truss"

    def define_nodes(self) -> None:
        # Bottom chord nodes
        self.nodes.append(Vertex(0.0, 0.0))
        self.nodes.append(Vertex(1 * self.width / 4, 0.0))
        self.nodes.append(Vertex(self.width / 2, 0.0))
        self.nodes.append(Vertex(3 * self.width / 4, 0.0))
        self.nodes.append(Vertex(self.width, 0.0))

        # Top chord nodes
        # self.nodes.append(Vertex(0.0, 0.0))
        self.nodes.append(Vertex(1 * self.width / 4, self.height / 2))
        self.nodes.append(Vertex(self.width / 2, self.height))
        self.nodes.append(Vertex(3 * self.width / 4, self.height / 2))
        # self.nodes.append(Vertex(self.width, 0.0))
        if self.overhang_length > 0:
            self.nodes.append(
                Vertex(
                    -self.overhang_length * np.cos(self.roof_pitch),
                    -self.overhang_length * np.sin(self.roof_pitch),
                )
            )
            self.nodes.append(
                Vertex(
                    self.width + self.overhang_length * np.cos(self.roof_pitch),
                    -self.overhang_length * np.sin(self.roof_pitch),
                )
            )

    def define_connectivity(self) -> None:
        # Bottom chord connectivity
        self.bottom_chord_node_ids = [1, 2, 3, 4, 5]
        left_v = 1
        right_v = 5

        # Top chord connectivity (left and right slopes stored separately)
        self.top_chord_node_ids = {"left": [left_v, 6, 7], "right": [7, 8, right_v]}
        if self.overhang_length > 0:
            self.top_chord_node_ids["left"].insert(0, 9)  # left overhang
            self.top_chord_node_ids["right"].append(10)  # right overhang

        # Web diagonals connectivity
        self.web_node_pairs.append((2, 7))  # left diagonal
        self.web_node_pairs.append((4, 7))  # right diagonal

        # Web verticals connectivity
        self.web_verticals_node_pairs.append((2, 6))  # left vertical
        self.web_verticals_node_pairs.append((3, 7))  # centre vertical
        self.web_verticals_node_pairs.append((4, 8))  # right vertical


class FanRoofTruss(RoofTruss):
    """Fan roof truss with radiating diagonal members forming a fan pattern.

    Features diagonal members radiating from lower chord panel points up to the
    top chord, creating a fan-like appearance. Provides excellent load distribution
    for longer spans (15-25m).
    """

    @property
    def type(self) -> str:
        return "Fan Roof Truss"

    def define_nodes(self) -> None:
        # Bottom chord nodes
        self.nodes.append(Vertex(0.0, 0.0))
        self.nodes.append(Vertex(1 * self.width / 3, 0.0))
        self.nodes.append(Vertex(2 * self.width / 3, 0.0))
        self.nodes.append(Vertex(self.width, 0.0))

        # Top chord nodes
        # self.nodes.append(Vertex(0.0, 0.0))
        self.nodes.append(Vertex(1 * self.width / 6, self.height / 3))
        self.nodes.append(Vertex(2 * self.width / 6, 2 * self.height / 3))
        self.nodes.append(Vertex(self.width / 2, self.height))
        self.nodes.append(Vertex(4 * self.width / 6, 2 * self.height / 3))
        self.nodes.append(Vertex(5 * self.width / 6, self.height / 3))
        # self.nodes.append(Vertex(self.width, 0.0))
        if self.overhang_length > 0:
            self.nodes.append(
                Vertex(
                    -self.overhang_length * np.cos(self.roof_pitch),
                    -self.overhang_length * np.sin(self.roof_pitch),
                )
            )
            self.nodes.append(
                Vertex(
                    self.width + self.overhang_length * np.cos(self.roof_pitch),
                    -self.overhang_length * np.sin(self.roof_pitch),
                )
            )

    def define_connectivity(self) -> None:
        # Bottom chord connectivity
        self.bottom_chord_node_ids = [1, 2, 3, 4]
        left_v = 1
        right_v = 4

        # Top chord connectivity (left and right slopes stored separately)
        self.top_chord_node_ids = {
            "left": [left_v, 5, 6, 7],
            "right": [7, 8, 9, right_v],
        }
        if self.overhang_length > 0:
            self.top_chord_node_ids["left"].insert(0, 10)  # left overhang
            self.top_chord_node_ids["right"].append(11)  # right overhang

        # Web diagonals connectivity
        self.web_node_pairs.append((2, 5))
        self.web_node_pairs.append((2, 7))
        self.web_node_pairs.append((3, 7))
        self.web_node_pairs.append((3, 9))

        # Web verticals connectivity
        self.web_verticals_node_pairs.append((2, 6))
        self.web_verticals_node_pairs.append((3, 8))


class ModifiedQueenPostRoofTruss(RoofTruss):
    """Modified Queen Post roof truss with enhanced web configuration.

    An enhanced version of the Queen Post truss with additional web members
    for better load distribution and reduced member forces. Suitable for
    medium to long spans (12-20m).
    """

    @property
    def type(self) -> str:
        return "Modified Queen Post Roof Truss"

    def define_nodes(self) -> None:
        # Bottom chord nodes
        self.nodes.append(Vertex(0.0, 0.0))
        self.nodes.append(Vertex(1 * self.width / 4, 0.0))
        self.nodes.append(Vertex(self.width / 2, 0.0))
        self.nodes.append(Vertex(3 * self.width / 4, 0.0))
        self.nodes.append(Vertex(self.width, 0.0))

        # Top chord nodes
        # self.nodes.append(Vertex(0.0, 0.0))
        self.nodes.append(Vertex(1 * self.width / 6, self.height / 3))
        self.nodes.append(Vertex(2 * self.width / 6, 2 * self.height / 3))
        self.nodes.append(Vertex(self.width / 2, self.height))
        self.nodes.append(Vertex(4 * self.width / 6, 2 * self.height / 3))
        self.nodes.append(Vertex(5 * self.width / 6, self.height / 3))
        # self.nodes.append(Vertex(self.width, 0.0))
        if self.overhang_length > 0:
            self.nodes.append(
                Vertex(
                    -self.overhang_length * np.cos(self.roof_pitch),
                    -self.overhang_length * np.sin(self.roof_pitch),
                )
            )
            self.nodes.append(
                Vertex(
                    self.width + self.overhang_length * np.cos(self.roof_pitch),
                    -self.overhang_length * np.sin(self.roof_pitch),
                )
            )

    def define_connectivity(self) -> None:
        # Bottom chord connectivity
        self.bottom_chord_node_ids = [1, 2, 3, 4, 5]
        left_v = 1
        right_v = 5

        # Top chord connectivity (left and right slopes stored separately)
        self.top_chord_node_ids = {
            "left": [left_v, 6, 7, 8],
            "right": [8, 9, 10, right_v],
        }
        if self.overhang_length > 0:
            self.top_chord_node_ids["left"].insert(0, 11)  # left overhang
            self.top_chord_node_ids["right"].append(12)  # right overhang

        # Web diagonals connectivity
        self.web_node_pairs.append((2, 6))
        self.web_node_pairs.append((2, 7))
        self.web_node_pairs.append((3, 7))
        self.web_node_pairs.append((3, 9))
        self.web_node_pairs.append((4, 9))
        self.web_node_pairs.append((4, 10))

        # Web verticals connectivity
        self.web_verticals_node_pairs.append((3, 8))  # center vertical


class DoubleFinkRoofTruss(RoofTruss):
    """Double Fink roof truss with two W-shaped web patterns.

    An extension of the Fink truss with additional web members creating two
    W patterns. Suitable for longer spans (20-30m) where a standard Fink would
    have excessive member lengths.
    """

    @property
    def type(self) -> str:
        return "Double Fink Roof Truss"

    def define_nodes(self) -> None:
        # Bottom chord nodes
        self.nodes.append(Vertex(0.0, 0.0))
        self.nodes.append(Vertex(1 * self.width / 5, 0.0))
        self.nodes.append(Vertex(2 * self.width / 5, 0.0))
        self.nodes.append(Vertex(3 * self.width / 5, 0.0))
        self.nodes.append(Vertex(4 * self.width / 5, 0.0))
        self.nodes.append(Vertex(self.width, 0.0))

        # Top chord nodes
        # self.nodes.append(Vertex(0.0, 0.0))
        self.nodes.append(Vertex(1 * self.width / 6, self.height / 3))
        self.nodes.append(Vertex(2 * self.width / 6, 2 * self.height / 3))
        self.nodes.append(Vertex(self.width / 2, self.height))
        self.nodes.append(Vertex(4 * self.width / 6, 2 * self.height / 3))
        self.nodes.append(Vertex(5 * self.width / 6, self.height / 3))
        # self.nodes.append(Vertex(self.width, 0.0))
        if self.overhang_length > 0:
            self.nodes.append(
                Vertex(
                    -self.overhang_length * np.cos(self.roof_pitch),
                    -self.overhang_length * np.sin(self.roof_pitch),
                )
            )
            self.nodes.append(
                Vertex(
                    self.width + self.overhang_length * np.cos(self.roof_pitch),
                    -self.overhang_length * np.sin(self.roof_pitch),
                )
            )

    def define_connectivity(self) -> None:
        # Bottom chord connectivity
        self.bottom_chord_node_ids = [1, 2, 3, 4, 5, 6]
        left_v = 1
        right_v = 6

        # Top chord connectivity (left and right slopes stored separately)
        self.top_chord_node_ids = {
            "left": [left_v, 7, 8, 9],
            "right": [9, 10, 11, right_v],
        }
        if self.overhang_length > 0:
            self.top_chord_node_ids["left"].insert(0, 12)  # left overhang
            self.top_chord_node_ids["right"].append(13)  # right overhang

        # Web diagonals connectivity
        self.web_node_pairs.append((2, 7))
        self.web_node_pairs.append((2, 8))
        self.web_node_pairs.append((3, 8))
        self.web_node_pairs.append((3, 9))
        self.web_node_pairs.append((4, 9))
        self.web_node_pairs.append((4, 10))
        self.web_node_pairs.append((5, 10))
        self.web_node_pairs.append((5, 11))


class DoubleHoweRoofTruss(RoofTruss):
    """Double Howe roof truss with enhanced vertical and diagonal web pattern.

    An extension of the Howe truss with additional verticals and diagonals for
    increased load capacity and reduced member lengths. Suitable for long spans
    (20-30m) or heavy loading conditions.
    """

    @property
    def type(self) -> str:
        return "Double Howe Roof Truss"

    def define_nodes(self) -> None:
        # Bottom chord nodes
        self.nodes.append(Vertex(0.0, 0.0))
        self.nodes.append(Vertex(1 * self.width / 6, 0.0))
        self.nodes.append(Vertex(2 * self.width / 6, 0.0))
        self.nodes.append(Vertex(self.width / 2, 0.0))
        self.nodes.append(Vertex(4 * self.width / 6, 0.0))
        self.nodes.append(Vertex(5 * self.width / 6, 0.0))
        self.nodes.append(Vertex(self.width, 0.0))

        # Top chord nodes
        # self.nodes.append(Vertex(0.0, 0.0))
        self.nodes.append(Vertex(1 * self.width / 6, self.height / 3))
        self.nodes.append(Vertex(2 * self.width / 6, 2 * self.height / 3))
        self.nodes.append(Vertex(self.width / 2, self.height))
        self.nodes.append(Vertex(4 * self.width / 6, 2 * self.height / 3))
        self.nodes.append(Vertex(5 * self.width / 6, self.height / 3))
        # self.nodes.append(Vertex(self.width, 0.0))
        if self.overhang_length > 0:
            self.nodes.append(
                Vertex(
                    -self.overhang_length * np.cos(self.roof_pitch),
                    -self.overhang_length * np.sin(self.roof_pitch),
                )
            )
            self.nodes.append(
                Vertex(
                    self.width + self.overhang_length * np.cos(self.roof_pitch),
                    -self.overhang_length * np.sin(self.roof_pitch),
                )
            )

    def define_connectivity(self) -> None:
        # Bottom chord connectivity
        self.bottom_chord_node_ids = [1, 2, 3, 4, 5, 6, 7]
        left_v = 1
        right_v = 7

        # Top chord connectivity (left and right slopes stored separately)
        self.top_chord_node_ids = {
            "left": [left_v, 8, 9, 10],
            "right": [10, 11, 12, right_v],
        }
        if self.overhang_length > 0:
            self.top_chord_node_ids["left"].insert(0, 13)  # left overhang
            self.top_chord_node_ids["right"].append(14)  # right overhang

        # Web diagonals connectivity
        self.web_node_pairs.append((3, 8))
        self.web_node_pairs.append((4, 9))
        self.web_node_pairs.append((4, 11))
        self.web_node_pairs.append((5, 12))

        # Web verticals connectivity
        self.web_verticals_node_pairs.append((2, 8))
        self.web_verticals_node_pairs.append((3, 9))
        self.web_verticals_node_pairs.append((4, 10))  # center vertical
        self.web_verticals_node_pairs.append((5, 11))
        self.web_verticals_node_pairs.append((6, 12))


class ModifiedFanRoofTruss(RoofTruss):
    """Modified Fan roof truss with enhanced radiating web pattern.

    An enhanced version of the Fan truss with additional web members for
    improved structural performance. Suitable for long spans (20-30m) with
    excellent load distribution characteristics.
    """

    @property
    def type(self) -> str:
        return "Modified Fan Roof Truss"

    def define_nodes(self) -> None:
        # Bottom chord nodes
        self.nodes.append(Vertex(0.0, 0.0))
        self.nodes.append(Vertex(1 * self.width / 4, 0.0))
        self.nodes.append(Vertex(self.width / 2, 0.0))
        self.nodes.append(Vertex(3 * self.width / 4, 0.0))
        self.nodes.append(Vertex(self.width, 0.0))

        # Top chord nodes
        # self.nodes.append(Vertex(0.0, 0.0))
        self.nodes.append(Vertex(1 * self.width / 8, 1 * self.height / 4))
        self.nodes.append(Vertex(2 * self.width / 8, 2 * self.height / 4))
        self.nodes.append(Vertex(3 * self.width / 8, 3 * self.height / 4))
        self.nodes.append(Vertex(self.width / 2, self.height))
        self.nodes.append(Vertex(5 * self.width / 8, 3 * self.height / 4))
        self.nodes.append(Vertex(6 * self.width / 8, 2 * self.height / 4))
        self.nodes.append(Vertex(7 * self.width / 8, 1 * self.height / 4))
        # self.nodes.append(Vertex(self.width, 0.0))
        if self.overhang_length > 0:
            self.nodes.append(
                Vertex(
                    -self.overhang_length * np.cos(self.roof_pitch),
                    -self.overhang_length * np.sin(self.roof_pitch),
                )
            )
            self.nodes.append(
                Vertex(
                    self.width + self.overhang_length * np.cos(self.roof_pitch),
                    -self.overhang_length * np.sin(self.roof_pitch),
                )
            )

    def define_connectivity(self) -> None:
        # Bottom chord connectivity
        self.bottom_chord_node_ids = [1, 2, 3, 4, 5]
        left_v = 1
        right_v = 5

        # Top chord connectivity (left and right slopes stored separately)
        self.top_chord_node_ids = {
            "left": [left_v, 6, 7, 8, 9],
            "right": [9, 10, 11, 12, right_v],
        }
        if self.overhang_length > 0:
            self.top_chord_node_ids["left"].insert(0, 13)  # left overhang
            self.top_chord_node_ids["right"].append(14)  # right overhang

        # Web diagonals connectivity
        self.web_node_pairs.append((2, 6))
        self.web_node_pairs.append((2, 8))
        self.web_node_pairs.append((3, 8))
        self.web_node_pairs.append((3, 10))
        self.web_node_pairs.append((4, 10))
        self.web_node_pairs.append((4, 12))

        # Web verticals connectivity
        self.web_verticals_node_pairs.append((2, 7))
        self.web_verticals_node_pairs.append((3, 9))  # center vertical
        self.web_verticals_node_pairs.append((4, 11))


class AtticRoofTruss(RoofTruss):
    """Attic (or Room-in-Roof) truss with habitable space under the roof.

    Creates a truss with vertical walls and a flat ceiling to provide usable attic
    space. The geometry includes:
    - Vertical attic walls at the edges of the attic space
    - Horizontal ceiling beam
    - Sloped top chords from walls to peak
    - Diagonal and vertical web members for support

    The attic space is defined by attic_width (floor width) and attic_height
    (ceiling height). If attic_height is not specified, it defaults to the height
    where the vertical walls meet the sloped roof.

    Attributes:
        attic_width (float): Width of the attic floor (interior dimension)
        attic_height (float): Height of the attic ceiling
        wall_x (float): Horizontal position where attic walls are located
        wall_y (float): Height at top of attic walls where they meet the roof slope
        ceiling_y (float): Vertical position of the ceiling beam (equals attic_height)
        ceiling_x (float): Horizontal position where ceiling meets the sloped top chord
        wall_ceiling_intersect (bool): True if wall top and ceiling intersection coincide
    """

    # Additional properties for this truss type
    attic_width: float
    attic_height: float

    # Computed properties for this truss type
    wall_x: float
    wall_y: float
    ceiling_y: float
    ceiling_x: float
    wall_ceiling_intersect: bool = False

    @property
    def type(self) -> str:
        return "Attic Roof Truss"

    def __init__(
        self,
        width: float,
        roof_pitch_deg: float,
        attic_width: float,
        attic_height: Optional[float] = None,
        overhang_length: float = 0.0,
        top_chord_section: Optional[SectionProps] = None,
        bottom_chord_section: Optional[SectionProps] = None,
        web_section: Optional[SectionProps] = None,
        web_verticals_section: Optional[SectionProps] = None,
    ):
        """Initialize an attic roof truss.

        Args:
            width (float): Total span of the truss
            roof_pitch_deg (float): Roof pitch angle in degrees
            attic_width (float): Interior width of the attic space. Must be less than width.
            attic_height (Optional[float]): Height of the attic ceiling. If None, defaults
                to the height where vertical walls meet the roof slope. Must be at least
                as high as the wall intersection point.
            overhang_length (float): Length of roof overhang. Defaults to 0.0.
            top_chord_section (Optional[SectionProps]): Section properties for top chord
            bottom_chord_section (Optional[SectionProps]): Section properties for bottom chord
            web_section (Optional[SectionProps]): Section properties for diagonal webs
            web_verticals_section (Optional[SectionProps]): Section properties for vertical webs

        Raises:
            ValueError: If attic dimensions are invalid or create impossible geometry
        """
        # NOTE: Must compute attic geometry BEFORE calling super().__init__() because
        # define_nodes() needs these values, and it's called within super().__init__()

        if attic_width <= 0:
            raise ValueError(f"attic_width must be positive, got {attic_width}")
        if attic_width >= width:
            raise ValueError(
                f"attic_width ({attic_width}) must be less than truss width ({width})"
            )

        self.attic_width = attic_width

        # Compute roof pitch first (needed for geometry calculations)
        roof_pitch = np.radians(roof_pitch_deg)

        # Calculate horizontal position of attic walls (from centerline)
        wall_x = width / 2 - attic_width / 2

        # Calculate height where vertical wall meets the sloped roof
        # Using: wall_y = wall_x * tan(roof_pitch)
        wall_y = wall_x * np.tan(roof_pitch)

        # Set ceiling height
        if attic_height is None:
            # Default: ceiling at the wall-roof intersection
            ceiling_y = wall_y
        else:
            ceiling_y = attic_height

        # Calculate peak height for this width and pitch
        peak_height = (width / 2) * np.tan(roof_pitch)

        # Calculate horizontal position where ceiling meets the sloped top chord
        # From peak: horizontal_distance = (peak_height - ceiling_height) / tan(roof_pitch)
        # From centerline: ceiling_x = centerline - horizontal_distance
        ceiling_x = width / 2 - (peak_height - ceiling_y) / np.tan(roof_pitch)

        # Validate geometry: ceiling must be at or above the wall intersection
        # Use tolerance for floating point comparison
        tolerance = 1e-6
        if ceiling_y < wall_y - tolerance or ceiling_x < wall_x - tolerance:
            raise ValueError(
                f"Attic height ({ceiling_y:.2f}) is too low. "
                f"Minimum attic height for this configuration is {wall_y:.2f}. "
                f"Please increase attic_height or decrease attic_width."
            )

        # Store computed geometry
        self.attic_height = (
            ceiling_y  # Use the computed ceiling_y which is always a float
        )
        self.wall_x = wall_x
        self.wall_y = wall_y
        self.ceiling_y = ceiling_y
        self.ceiling_x = ceiling_x

        # Check if wall top and ceiling intersection are at the same point
        self.wall_ceiling_intersect = self.ceiling_y == self.wall_y

        # Now call super().__init__() which will call define_nodes/connectivity/supports
        super().__init__(
            width=width,
            roof_pitch_deg=roof_pitch_deg,
            overhang_length=overhang_length,
            top_chord_section=top_chord_section,
            bottom_chord_section=bottom_chord_section,
            web_section=web_section,
            web_verticals_section=web_verticals_section,
        )

    def define_nodes(self) -> None:
        # Bottom chord nodes
        self.nodes.append(Vertex(0.0, 0.0))
        self.nodes.append(Vertex(self.wall_x, 0.0))
        self.nodes.append(Vertex(self.width - self.wall_x, 0.0))
        self.nodes.append(Vertex(self.width, 0.0))

        # Top chord nodes
        # self.nodes.append(Vertex(0.0, 0.0))
        self.nodes.append(Vertex(self.wall_x / 2, self.wall_y / 2))
        self.nodes.append(Vertex(self.wall_x, self.wall_y))
        if not self.wall_ceiling_intersect:
            self.nodes.append(Vertex(self.ceiling_x, self.ceiling_y))
        self.nodes.append(Vertex(self.width / 2, self.height))
        if not self.wall_ceiling_intersect:
            self.nodes.append(Vertex(self.width - self.ceiling_x, self.ceiling_y))
        self.nodes.append(Vertex(self.width - self.wall_x, self.wall_y))
        self.nodes.append(Vertex(self.width - self.wall_x / 2, self.wall_y / 2))
        self.nodes.append(
            Vertex(self.width / 2, self.ceiling_y)
        )  # special node in the middle of the ceiling beam
        # self.nodes.append(Vertex(self.width, 0.0))
        if self.overhang_length > 0:
            self.nodes.append(
                Vertex(
                    -self.overhang_length * np.cos(self.roof_pitch),
                    -self.overhang_length * np.sin(self.roof_pitch),
                )
            )
            self.nodes.append(
                Vertex(
                    self.width + self.overhang_length * np.cos(self.roof_pitch),
                    -self.overhang_length * np.sin(self.roof_pitch),
                )
            )

    def define_connectivity(self) -> None:
        # Bottom chord connectivity
        self.bottom_chord_node_ids = [1, 2, 3, 4]
        left_v = 1
        right_v = 4

        if self.wall_ceiling_intersect:
            # Top chord connectivity (left and right slopes stored separately)
            self.top_chord_node_ids = {
                "left": [left_v, 5, 6, 7],
                "right": [7, 8, 9, right_v],
                "ceiling": [6, 10, 8],  # attic ceiling
            }
            if self.overhang_length > 0:
                self.top_chord_node_ids["left"].insert(0, 11)  # left overhang
                self.top_chord_node_ids["right"].append(12)  # right overhang

            # Web diagonals connectivity
            self.web_node_pairs.append((2, 5))
            self.web_node_pairs.append(
                (10, 7)
            )  # special case: this is actually the center vertical post
            self.web_node_pairs.append((3, 9))

            # Web verticals connectivity
            self.web_verticals_node_pairs.append((2, 6))
            self.web_verticals_node_pairs.append((3, 8))

        else:
            # Top chord connectivity (left and right slopes stored separately)
            self.top_chord_node_ids = {
                "left": [left_v, 5, 6, 7, 8],
                "right": [8, 9, 10, 11, right_v],
                "ceiling": [7, 12, 9],  # attic ceiling
            }
            if self.overhang_length > 0:
                self.top_chord_node_ids["left"].insert(0, 13)  # left overhang
                self.top_chord_node_ids["right"].append(14)  # right overhang

            # Web diagonals connectivity
            self.web_node_pairs.append((2, 5))
            self.web_node_pairs.append(
                (12, 8)
            )  # special case: this is actually the center vertical post
            self.web_node_pairs.append((3, 11))

            # Web verticals connectivity
            self.web_verticals_node_pairs.append((2, 6))
            self.web_verticals_node_pairs.append((3, 10))


def create_truss(truss_type: str, **kwargs: Any) -> "Truss":
    """Factory function to create truss instances by type name.

    Provides a convenient way to create trusses without importing specific classes.
    Type names are case-insensitive and can use underscores or hyphens as separators.

    Args:
        truss_type (str): Name of the truss type. Supported types:
            Flat trusses: "howe", "pratt", "warren"
            Roof trusses: "king_post", "queen_post", "fink", "howe_roof", "pratt_roof",
                "fan", "modified_queen_post", "double_fink", "double_howe",
                "modified_fan", "attic"
        **kwargs: Arguments to pass to the truss constructor

    Returns:
        Truss: An instance of the requested truss type

    Raises:
        ValueError: If truss_type is not recognized

    Examples:
        >>> truss = create_truss("howe", width=20, height=2.5, unit_width=2.0)
        >>> truss = create_truss("king-post", width=10, roof_pitch_deg=30)
    """
    # Normalize the truss type name
    normalized = truss_type.lower().replace("-", "_").replace(" ", "_")

    # Map of normalized names to classes
    truss_map = {
        # Flat trusses
        "howe": HoweFlatTruss,
        "howe_flat": HoweFlatTruss,
        "pratt": PrattFlatTruss,
        "pratt_flat": PrattFlatTruss,
        "warren": WarrenFlatTruss,
        "warren_flat": WarrenFlatTruss,
        # Roof trusses
        "king_post": KingPostRoofTruss,
        "kingpost": KingPostRoofTruss,
        "queen_post": QueenPostRoofTruss,
        "queenpost": QueenPostRoofTruss,
        "fink": FinkRoofTruss,
        "howe_roof": HoweRoofTruss,
        "pratt_roof": PrattRoofTruss,
        "fan": FanRoofTruss,
        "modified_queen_post": ModifiedQueenPostRoofTruss,
        "modified_queenpost": ModifiedQueenPostRoofTruss,
        "double_fink": DoubleFinkRoofTruss,
        "doublefink": DoubleFinkRoofTruss,
        "double_howe": DoubleHoweRoofTruss,
        "doublehowe": DoubleHoweRoofTruss,
        "modified_fan": ModifiedFanRoofTruss,
        "modifiedfan": ModifiedFanRoofTruss,
        "attic": AtticRoofTruss,
        "attic_roof": AtticRoofTruss,
    }

    if normalized not in truss_map:
        available = sorted(set(truss_map.keys()))
        raise ValueError(
            f"Unknown truss type '{truss_type}'. Available types: {', '.join(available)}"
        )

    truss_class = truss_map[normalized]
    assert issubclass(truss_class, Truss)
    return truss_class(**kwargs)


__all__ = [
    "HoweFlatTruss",
    "PrattFlatTruss",
    "WarrenFlatTruss",
    "KingPostRoofTruss",
    "QueenPostRoofTruss",
    "FinkRoofTruss",
    "HoweRoofTruss",
    "PrattRoofTruss",
    "FanRoofTruss",
    "ModifiedQueenPostRoofTruss",
    "DoubleFinkRoofTruss",
    "DoubleHoweRoofTruss",
    "ModifiedFanRoofTruss",
    "AtticRoofTruss",
    "create_truss",
]
