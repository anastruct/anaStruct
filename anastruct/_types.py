from typing import TYPE_CHECKING, Dict, Literal, Sequence, TypedDict, Union

import numpy as np

if TYPE_CHECKING:
    from anastruct.vertex import Vertex

AxisNumber = Literal[1, 2, 3]
Dimension = Literal["x", "y", "y_neg", "both"]
ElementType = Literal["general", "truss"]
LoadDirection = Literal["element", "x", "y", "parallel", "perpendicular", "angle"]
MpType = Dict[Literal[1, 2], float]
NumberLike = Union[float, int, np.number]
OrientAxis = Literal["y", "z"]
Spring = Dict[Literal[1, 2], float]
SupportDirection = Literal["x", "y", "1", "2", 1, 2]
VertexLike = Union[Sequence[Union[float, int]], np.ndarray, "Vertex"]

SectionProps = TypedDict(
    "SectionProps",
    {
        "EI": float,
        "EA": float,
        "g": float,
    },
)
