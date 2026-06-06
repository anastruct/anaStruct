from __future__ import annotations

import math
from typing import TYPE_CHECKING, Sequence, Union

import numpy as np

if TYPE_CHECKING:
    from anastruct._types import NumberLike, VertexLike


class Vertex:
    """
    Utility point in 2D.
    """

    def __init__(
        self,
        x: Union["VertexLike", "NumberLike"],
        y: Union["NumberLike", None] = None,
    ):
        """Create a Vertex object

        Args:
            x (Union[VertexLike, NumberLike]): X coordinate or a Vertex object, or an object that
                can be converted to a Vertex
            y (Union[NumberLike, None], optional): Y coordinate. Defaults to None.
        """
        if isinstance(x, Vertex):
            self.coordinates: np.ndarray = np.array(x.coordinates, dtype=np.float32)
        elif (
            isinstance(x, (Sequence, np.ndarray))
            and len(x) == 2
            and isinstance(x[0], (float, int, np.number))
            and isinstance(x[1], (float, int, np.number))
        ):
            self.coordinates = np.array([x[0], x[1]], dtype=np.float32)
        elif isinstance(x, (float, int, np.number)) and isinstance(
            y, (float, int, np.number)
        ):
            self.coordinates = np.array([x, y], dtype=np.float32)
        else:
            raise TypeError(
                "Points must be convertable to a Vertex object: (x, y) or [x, y] or np.array([x, y]) or Vertex(x, y)"
            )

    @property
    def x(self) -> float:
        """X coordinate

        Returns:
            float: X coordinate
        """
        return float(self.coordinates[0])

    @property
    def y(self) -> float:
        """Y coordinate

        Returns:
            float: Y coordinate
        """
        return float(self.coordinates[1])

    @property
    def y_neg(self) -> float:
        """Y negative coordinate (equals negative of Y coordinate)

        Returns:
            float: Y_neg coordinate
        """
        return float(self.coordinates[1] * -1)

    def modulus(self) -> float:
        """Magnitude of the vector from the origin to the Vertex

        Returns:
            float: Magnitude of the vector from the origin to the Vertex
        """
        return float(np.sqrt(np.sum(self.coordinates**2)))

    def unit(self) -> Vertex:
        """Unit vector from the origin to the Vertex

        Returns:
            Vertex: Unit vector from the origin to the Vertex
        """
        return (1 / self.modulus()) * self

    def displace_polar(
        self, alpha: float, radius: float, inverse_y_axis: bool = False
    ) -> None:
        """Displace the Vertex by a polar coordinate

        Args:
            alpha (float): Angle in radians
            radius (float): Radius
            inverse_y_axis (bool, optional): Return a negative Y coordinate. Defaults to False.
        """
        if inverse_y_axis:
            self.coordinates[0] += math.cos(alpha) * radius
            self.coordinates[1] -= math.sin(alpha) * radius
        else:
            self.coordinates[0] += math.cos(alpha) * radius
            self.coordinates[1] += math.sin(alpha) * radius

    def __add__(self, other: Union["VertexLike", "NumberLike"]) -> Vertex:
        """Add two Vertex objects

        Args:
            other (Union[VertexLike, NumberLike]): Vertex to add

        Returns:
            Vertex: Sum of the two Vertex objects
        """
        other = det_coordinates(other)
        return Vertex(self.coordinates + other)

    def __radd__(self, other: Union["VertexLike", "NumberLike"]) -> Vertex:
        """Add two Vertex objects

        Args:
            other (Union[VertexLike, NumberLike]): Vertex to add

        Returns:
            Vertex: Sum of the two Vertex objects
        """
        return self.__add__(other)

    def __sub__(self, other: Union["VertexLike", "NumberLike"]) -> Vertex:
        """Subtract two Vertex objects

        Args:
            other (Union[VertexLike, NumberLike]): Vertex to subtract

        Returns:
            Vertex: Difference of the two Vertex objects
        """
        other = det_coordinates(other)
        return Vertex(self.coordinates - other)

    def __rsub__(self, other: Union["VertexLike", "NumberLike"]) -> Vertex:
        """Subtract two Vertex objects

        Args:
            other (Union[VertexLike, NumberLike]): Vertex to subtract

        Returns:
            Vertex: Difference of the two Vertex objects
        """
        return self.__sub__(other)

    def __mul__(self, other: Union["VertexLike", "NumberLike"]) -> Vertex:
        """Multiply two Vertex objects

        Args:
            other (Union[VertexLike, NumberLike]): Vertex to multiply

        Returns:
            Vertex: Product of the two Vertex objects
        """
        other = det_coordinates(other)
        return Vertex(self.coordinates * other)

    def __rmul__(self, other: Union["VertexLike", "NumberLike"]) -> Vertex:
        """Multiply two Vertex objects

        Args:
            other (Union[VertexLike, NumberLike]): Vertex to multiply

        Returns:
            Vertex: Product of the two Vertex objects
        """
        return self.__mul__(other)

    def __truediv__(self, other: Union["VertexLike", "NumberLike"]) -> Vertex:
        """Divide two Vertex objects

        Args:
            other (Union[VertexLike, NumberLike]): Vertex to divide

        Returns:
            Vertex: Quotient of the two Vertex objects
        """
        other = det_coordinates(other)
        return Vertex(self.coordinates / other)

    def __eq__(self, other: object) -> bool:
        """Check if two Vertex objects are equal

        Args:
            other (object): Object to compare

        Raises:
            NotImplementedError: If the object is not a Vertex object or a tuple or list of length 2

        Returns:
            bool: True if the two Vertex objects are equal
        """
        if isinstance(other, Vertex):
            return self.x == other.x and self.y == other.y
        if (
            isinstance(other, (np.ndarray, Sequence))
            and len(other) == 2
            and isinstance(other[0], (int, float))
            and isinstance(other[1], (int, float))
        ):
            return self.x == other[0] and self.y == other[1]
        return NotImplemented

    def __str__(self) -> str:
        """String representation of the Vertex object

        Returns:
            str: String representation of the Vertex object
        """
        return f"Vertex({self.x}, {self.y})"

    def __hash__(self) -> int:
        """Hash of the Vertex object

        Returns:
            int: Hash of the Vertex object
        """
        return hash(str(self))


def vertex_range(v1: Vertex, v2: Vertex, n: int) -> list:
    """Create a list of n Vertex objects between two Vertex objects

    Args:
        v1 (Vertex): Starting Vertex
        v2 (Vertex): Ending Vertex
        n (int): Number of Vertex objects to create

    Returns:
        list: List of n Vertex objects between v1 and v2
    """
    dv = v2 - v1
    return [v1 + dv * i / n for i in range(n + 1)]


def det_coordinates(point: Union["VertexLike", "NumberLike"]) -> np.ndarray:
    """Convert a point to coordinates

    Args:
        point (Union[VertexLike, NumberLike]): Point to convert

    Raises:
        TypeError: If the point is not convertable to a Vertex object

    Returns:
        np.ndarray: Coordinates of the point
    """
    if isinstance(point, Vertex):
        return point.coordinates
    if (
        isinstance(point, (np.ndarray, Sequence))
        and len(point) == 2
        and isinstance(point[0], (float, int, np.number))
        and isinstance(point[1], (float, int, np.number))
    ):
        return np.asarray(point)
    if isinstance(point, (float, int, np.number)):
        return np.array([point, point])
    raise TypeError(
        "Points must be convertable to a Vertex object: (x, y) or [x, y] or np.array([x, y]) or Vertex(x, y)"
    )
