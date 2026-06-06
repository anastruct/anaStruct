import math
from typing import Any, Dict, Tuple

from anastruct.sectionbase.sectionbase import section_base


def steel_section_properties(
    **kwargs: Dict[str, Any],
) -> Tuple[str, float, float, float]:
    """Get steel section properties

    Raises:
        ValueError: Raised if orient is not defined

    Returns:
        Tuple[str, float, float, float]: Section name, EA, EI, g
    """
    steel_section = kwargs.get("steelsection", "IPE 300")
    orient = kwargs.get("orient", "y")
    E = kwargs.get("E", 210e9)
    sw = kwargs.get("sw", False)

    assert isinstance(steel_section, str)
    param = section_base.get_section_parameters(steel_section)
    EA: float = E * param["Ax"]
    if orient == "y":
        EI: float = E * param["Iy"]
    elif orient == "z":
        EI = E * param["Iz"]
    else:
        raise ValueError("Orient should be defined.")
    if sw:
        g: float = param["swdl"]
    else:
        g = 0
    section_name = f"{steel_section}({orient})"
    return section_name, EA, EI, g


def rectangle_properties(**kwargs: Dict[str, Any]) -> Tuple[str, float, float, float]:
    """Get rectangle section properties

    Returns:
        Tuple[str, float, float, float]: Section name, EA, EI, g
    """
    b = kwargs.get("b", 0.1)
    h = kwargs.get("h", 0.5)
    E = kwargs.get("E", 210e9)
    gamma = kwargs.get("gamma", 10000)
    sw = kwargs.get("sw", False)
    assert isinstance(b, float)
    assert isinstance(h, float)
    assert isinstance(E, float)
    assert isinstance(gamma, int)

    A = b * h
    I = b * h**3 / 12
    EA = E * A
    EI = E * I
    if sw:
        g = A * gamma
    else:
        g = 0
    section_name = f"rect {b}x{h}"
    return section_name, EA, EI, g


def circle_properties(**kwargs: Dict[str, Any]) -> Tuple[str, float, float, float]:
    """Get circle section properties

    Returns:
        Tuple[str, float, float, float]: Section name, EA, EI, g
    """
    d = kwargs.get("d", 0.4)
    E = kwargs.get("E", 210e9)
    gamma = kwargs.get("gamma", 10000)
    sw = kwargs.get("sw", False)
    assert isinstance(d, float)
    assert isinstance(E, float)
    assert isinstance(gamma, int)

    A = math.pi * d**2 / 4
    I = math.pi * d**4 / 64
    EA = E * A
    EI = E * I
    if sw:
        g = A * gamma
    else:
        g = 0
    section_name = "fi {d}"
    return section_name, EA, EI, g
