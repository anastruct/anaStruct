from typing import Any, Literal, Optional

import numpy as np

from anastruct.preprocess.beam_class import Beam
from anastruct.types import SectionProps
from anastruct.vertex import Vertex


class SimpleBeam(Beam):
    """Simple beam with a pin support at one end, and a roller support at the other."""

    def __init__(
        self,
        length: float,
        angle: float = 0.0,
        section: Optional[SectionProps] = None,
    ) -> None:
        super().__init__(
            length=length,
            angle=angle,
            section=section,
        )

    @property
    def type(self) -> str:
        return "Simple Beam"

    def define_nodes(self) -> None:
        self.nodes.append(Vertex(0.0, 0.0))
        self.nodes.append(Vertex(self.dx * self.length, self.dy * self.length))

    def define_supports(self) -> None:
        self.support_definitions[0] = "pinned"
        self.support_definitions[1] = "roller"


class CantileverBeam(Beam):
    """Cantilever beam with a fixed support at one end, and free at the other."""

    def __init__(
        self,
        length: float,
        cantilever_side: Literal["left", "right"] = "right",
        angle: float = 0.0,
        section: Optional[SectionProps] = None,
    ) -> None:
        super().__init__(
            length=length,
            angle=angle,
            section=section,
        )
        self.cantilever_side = cantilever_side.lower()
        if self.cantilever_side not in ["left", "right"]:
            raise ValueError(
                "cantilever_side must be either 'left' or 'right', "
                f"got '{cantilever_side}'"
            )

    @property
    def type(self) -> str:
        return "Cantilever Beam"

    def define_nodes(self) -> None:
        self.nodes.append(Vertex(0.0, 0.0))
        self.nodes.append(Vertex(self.dx * self.length, self.dy * self.length))

    def define_supports(self) -> None:
        self.support_definitions[1 if self.cantilever_side == "left" else 0] = "fixed"


class RightCantileverBeam(CantileverBeam):
    """Cantilever beam with a fixed support at the left end, and free at the right."""

    def __init__(
        self,
        length: float,
        angle: float = 0.0,
        section: Optional[SectionProps] = None,
    ) -> None:
        super().__init__(
            length=length,
            cantilever_side="right",
            angle=angle,
            section=section,
        )

    @property
    def type(self) -> str:
        return "Right Cantilever Beam"


class LeftCantileverBeam(CantileverBeam):
    """Cantilever beam with a free support at the left end, and fixed at the right."""

    def __init__(
        self,
        length: float,
        angle: float = 0.0,
        section: Optional[SectionProps] = None,
    ) -> None:
        super().__init__(
            length=length,
            cantilever_side="left",
            angle=angle,
            section=section,
        )

    @property
    def type(self) -> str:
        return "Left Cantilever Beam"


class MultiSpanBeam(Beam):
    """Simply supported multi-span beam. Assumes equal spans unless span_lengths provided."""

    def __init__(
        self,
        length: Optional[float] = None,
        num_spans: Optional[int] = None,
        span_lengths: Optional[list[float]] = None,
        cantilevers: Optional[Literal["left", "right", "both"]] = None,
        angle: float = 0.0,
        section: Optional[SectionProps] = None,
    ) -> None:
        if span_lengths is None and num_spans is None:
            raise ValueError("Either num_spans or span_lengths must be provided.")
        if span_lengths is not None and num_spans is not None:
            raise ValueError("Only one of num_spans or span_lengths may be provided.")
        if num_spans is not None and length is None:
            raise ValueError("If num_spans is provided, length must also be provided.")
        if num_spans is not None and length is not None:
            span_lengths = [length / num_spans] * num_spans

        super().__init__(
            length=sum(span_lengths) if span_lengths else num_spans,
            span_lengths=span_lengths,
            angle=angle,
            section=section,
        )
        self.num_spans = num_spans
        if cantilevers not in [None, "left", "right", "both"]:
            raise ValueError(
                "cantilevers must be either None, 'left', 'right', or 'both', "
                f"got '{cantilevers}'"
            )
        self.cantilevers = cantilevers

    @property
    def type(self) -> str:
        return "Multi-Span Beam"

    def define_nodes(self) -> None:
        current_length = 0.0
        self.nodes.append(Vertex(0.0, 0.0))
        for span in self.span_lengths:
            current_length += span
            self.nodes.append(
                Vertex(
                    self.dx * current_length,
                    self.dy * current_length,
                )
            )

    def define_supports(self) -> None:
        first_support = 0 if self.cantilevers in [None, "right"] else 1
        last_support = (
            len(self.span_lengths)
            if self.cantilevers in [None, "left"]
            else len(self.span_lengths) - 1
        )
        self.support_definitions[first_support] = "pinned"
        for i in range(first_support + 1, last_support):
            self.support_definitions[i] = "roller"


class TwoSpanBeam(MultiSpanBeam):
    """Simply supported two-span beam with equal spans."""

    def __init__(
        self,
        length: float,
        angle: float = 0.0,
        section: Optional[SectionProps] = None,
    ) -> None:
        super().__init__(
            length=length,
            num_spans=2,
            angle=angle,
            section=section,
        )

    @property
    def type(self) -> str:
        return "Two-Span Beam"


class ThreeSpanBeam(MultiSpanBeam):
    """Simply supported three-span beam with equal spans."""

    def __init__(
        self,
        length: float,
        angle: float = 0.0,
        section: Optional[SectionProps] = None,
    ) -> None:
        super().__init__(
            length=length,
            num_spans=3,
            angle=angle,
            section=section,
        )

    @property
    def type(self) -> str:
        return "Three-Span Beam"


class FourSpanBeam(MultiSpanBeam):
    """Simply supported four-span beam with equal spans."""

    def __init__(
        self,
        length: float,
        angle: float = 0.0,
        section: Optional[SectionProps] = None,
    ) -> None:
        super().__init__(
            length=length,
            num_spans=4,
            angle=angle,
            section=section,
        )

    @property
    def type(self) -> str:
        return "Four-Span Beam"


def create_beam(beam_type: str, **kwargs: Any) -> Beam:
    """Factory function to create beam instances by type name.

    Provides a convenient way to create beams without importing specific classes.
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
