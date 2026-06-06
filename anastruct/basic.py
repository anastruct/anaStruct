from typing import Any, Sequence, Tuple

import numpy as np

try:
    from anastruct.cython.cbasic import (  # type: ignore # pylint: disable=unused-import
        angle_x_axis,
        converge,
    )
except ImportError:
    from anastruct.cython.basic import angle_x_axis, converge


def find_nearest(array: np.ndarray, value: float) -> Tuple[float, int]:
    """Find the nearest value in an array

    Args:
        array (np.ndarray): array to search within
        value (float): value to search for

    Returns:
        Tuple[float, int]: Nearest value, index
    """

    # Subtract the value of the value's in the array. Make the values absolute.
    # The lowest value is the nearest.
    index: int = int((np.abs(array - value)).argmin())
    return array[index], index


def integrate_array(y: np.ndarray, dx: float) -> np.ndarray:
    """Integrate an array y*dx using the trapezoidal rule

    Args:
        y (np.ndarray): Array to integrate
        dx (float): Step size

    Returns:
        np.ndarray: Integrated array
    """
    return np.cumsum(y) * dx


class FEMException(Exception):
    def __init__(self, type_: str, message: str):
        """Exception for FEM

        Args:
            type_ (str): Type of exception
            message (str): Message for exception
        """
        self.type = type_
        self.message = message


def arg_to_list(arg: Any, n: int) -> list:
    """Convert an argument to a list of length n

    Args:
        arg (Any): Argument to convert
        n (int): Length of list to create

    Returns:
        list: List of length n
    """
    if isinstance(arg, Sequence) and not isinstance(arg, str) and len(arg) == n:
        return list(arg)
    if isinstance(arg, Sequence) and not isinstance(arg, str) and len(arg) == 1:
        return [arg[0] for _ in range(n)]
    return [arg for _ in range(n)]


def rotation_matrix(angle: float) -> np.ndarray:
    """Create a 2x2 rotation matrix

    Args:
        angle (float): Angle in radians

    Returns:
        np.ndarray: 2x2 Euclidean rotation matrix
    """
    s = np.sin(angle)
    c = np.cos(angle)
    return np.array([[c, -s], [s, c]])


def rotate_xy(a: np.ndarray, angle: float) -> np.ndarray:
    """Rotate a 2D matrix around the origin

    Args:
        a (np.ndarray): Matrix to rotate
        angle (float): Angle in radians

    Returns:
        np.ndarray: Rotated matrix
    """
    b: np.ndarray = np.array(a)
    b[:, 0] -= a[0, 0]
    b[:, 1] -= a[0, 1]
    b = np.dot(b, rotation_matrix(angle))
    b[:, 0] += a[0, 0]
    b[:, 1] += a[0, 1]
    return b
