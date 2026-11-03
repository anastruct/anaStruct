"""``solve(geometrical_non_linear=True)`` against closed-form beam-column
results: a simply supported member, uniform transverse load q, axial
force N, meshed into 20 elements.

Exact mid-span moments (P = |N|, k = sqrt(P / EI), u = kL / 2):

- compression: M = q EI / P · (sec u − 1)
- tension:     M = q EI / P · (1 − sech u)

Buckling factor: N_cr / P in compression (N_cr = π² EI / L²); no buckling
(infinite factor) when the member is in tension.
"""

import math

import numpy as np
import pytest

from anastruct import SystemElements

EI = 200e6 * 9.05e-6  # kNm²
EA = 200e6 * 2300e-6  # kN
L = 6.0
Q = 5.0  # kN/m
N_CR = math.pi**2 * EI / L**2
N_ELEMENTS = 20


def _member(axial: float, *, vertical: bool = False) -> SystemElements:
    """``axial`` > 0 is tension. Pin at the start, roller at the end, and
    the load perpendicular to the member."""
    ss = SystemElements(EA=EA, EI=EI)
    end = [0, L] if vertical else [L, 0]
    ids = ss.add_multiple_elements([[0, 0], end], N_ELEMENTS)
    ss.add_support_hinged(1)
    ss.add_support_roll(ss.id_last_node, direction=1 if vertical else 2)
    ss.q_load(q=Q if vertical else -Q, element_id=ids)
    if vertical:
        ss.point_load(ss.id_last_node, Fy=axial)
    else:
        ss.point_load(ss.id_last_node, Fx=axial)
    return ss


def _max_moment(ss: SystemElements) -> float:
    return max(np.max(np.abs(r["M"])) for r in ss.get_element_results(verbose=True))


def _exact(ratio: float, compression: bool) -> float:
    p = ratio * N_CR
    u = L / 2 * math.sqrt(p / EI)
    if compression:
        return Q * EI / p * (1 / math.cos(u) - 1)
    return Q * EI / p * (1 - 1 / math.cosh(u))


@pytest.mark.parametrize("vertical", [False, True], ids=["horizontal", "vertical"])
@pytest.mark.parametrize("ratio", [0.3, 0.6])
def test_compression_amplifies_moment(ratio, vertical):
    ss = _member(-ratio * N_CR, vertical=vertical)
    ss.solve(geometrical_non_linear=True)
    assert _max_moment(ss) == pytest.approx(_exact(ratio, True), rel=1e-3)
    assert ss.buckling_factor == pytest.approx(1 / ratio, rel=1e-3)


@pytest.mark.parametrize("vertical", [False, True], ids=["horizontal", "vertical"])
@pytest.mark.parametrize("ratio", [0.3, 0.6])
def test_tension_reduces_moment_and_cannot_buckle(ratio, vertical):
    ss = _member(ratio * N_CR, vertical=vertical)
    ss.solve(geometrical_non_linear=True)
    assert _max_moment(ss) == pytest.approx(_exact(ratio, False), rel=1e-3)
    assert ss.buckling_factor == math.inf


def test_no_axial_force_matches_first_order():
    first = _member(0.0)
    first.solve()
    second = _member(0.0)
    second.solve(geometrical_non_linear=True)
    assert _max_moment(second) == pytest.approx(_max_moment(first), rel=1e-12)
    assert second.buckling_factor == math.inf
