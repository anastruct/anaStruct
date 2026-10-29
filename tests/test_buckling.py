import copy

import numpy as np
from pytest import approx

from anastruct import SystemElements
from anastruct.fem.elements import geometric_stiffness_matrix
from anastruct.fem.system_components.solver import smallest_buckling_factor

"""
NOTE: These tests validate the buckling factor of `solve(geometrical_non_linear=True)`
against the Euler buckling loads. General definitions:
- L = column length
- EI = bending stiffness
- P = 1 is the load on the column, so that the buckling factor equals the buckling load
- Pcr = pi**2 * EI / (mu * L)**2, with the effective length factor mu: 1 (pinned-pinned),
  2 (cantilever), 0.699 (fixed-pinned) and 0.5 (fixed-fixed)

The geometric stiffness matrix of an element is exact for a prismatic beam with a constant
normal force only for a cubic displacement field, so a column needs to be discretized to
find the Euler load (the factor is found from above, and converges with the number of
elements).

The tests are plain test functions (not nested describe/context blocks), so that every
assertion is guaranteed to be executed.
"""

EI = 5.0e3
EA = 1.0e6
L = 10.0
EULER = np.pi**2 * EI / L**2  # pinned-pinned
FIXED_PINNED = 20.190728556 * EI / L**2  # 2.0457 * pi**2 * EI / L**2


def column(kind, load=-1.0, **element_kwargs):
    system = SystemElements(EI=EI, EA=EA)
    system.add_element([[0, 0], [0, L]], **element_kwargs)
    if kind == "pinned":
        system.add_support_hinged(1)
        system.add_support_roll(2, "y")
    elif kind == "cantilever":
        system.add_support_fixed(1)
    elif kind == "fixed-pinned":
        system.add_support_fixed(1)
        system.add_support_roll(2, "y")
    elif kind == "fixed-fixed":
        system.add_support_fixed(1)
        system.add_support_roll(2, "y", rotate=False)
    system.point_load(2, Fy=load)
    return system


def buckling_factor(system, n=20):
    system.solve(geometrical_non_linear=True, discretize_kwargs={"n": n})
    return system.buckling_factor


# ---------------------------------------------------------------------------
# Euler columns
# ---------------------------------------------------------------------------


def test_pinned_pinned_column():
    assert buckling_factor(column("pinned")) == approx(EULER, rel=1e-3)


def test_cantilever_column():
    assert buckling_factor(column("cantilever")) == approx(EULER / 4, rel=1e-3)


def test_fixed_pinned_column():
    assert buckling_factor(column("fixed-pinned")) == approx(FIXED_PINNED, rel=1e-3)


def test_fixed_fixed_column():
    assert buckling_factor(column("fixed-fixed")) == approx(4 * EULER, rel=1e-3)


def test_buckling_factor_converges_from_above_with_the_number_of_elements():
    coarse = buckling_factor(column("pinned"), n=4)
    fine = buckling_factor(column("pinned"), n=16)
    assert coarse > fine > EULER * (1 - 1e-6)
    assert fine == approx(EULER, rel=2e-3)


def test_single_element_column_gives_the_buckling_load_of_one_element():
    # one cubic element: 12 * EI / L**2 instead of pi**2 * EI / L**2
    system = column("pinned")
    system.solve(geometrical_non_linear=True)
    assert system.buckling_factor == approx(12 * EI / L**2)


def test_buckling_factor_is_inversely_proportional_to_the_load():
    assert buckling_factor(column("pinned", load=-2.0)) == approx(EULER / 2, rel=1e-3)


def test_column_in_tension_has_no_buckling_factor():
    assert buckling_factor(column("cantilever", load=1.0)) == float("inf")


# ---------------------------------------------------------------------------
# frames
# ---------------------------------------------------------------------------


def test_sway_buckling_of_a_portal_frame_with_a_rigid_beam():
    # the rigid beam has no normal force, so the geometric stiffness matrix is
    # singular; with hinged bases each column buckles as a cantilever
    system = SystemElements(EI=EI, EA=EA)
    system.add_element([[0, 0], [0, L]])
    system.add_element([[0, L], [6, L]], EI=1e9)
    system.add_element([[6, L], [6, 0]])
    system.add_support_hinged([1, 4])
    system.point_load([2, 3], Fy=[-1, -1])

    assert buckling_factor(system, n=10) == approx(EULER / 4, rel=2e-3)


# ---------------------------------------------------------------------------
# releases: hinges and springs
# ---------------------------------------------------------------------------


def test_hinge_at_the_end_of_an_element_does_not_change_the_buckling_factor():
    # the rotation of the roller is free, so a hinge at the end changes nothing; the
    # geometric stiffness must be condensed like the elastic stiffness (965 without)
    plain = buckling_factor(column("fixed-pinned"), n=8)
    hinged = buckling_factor(column("fixed-pinned", spring={2: 0}), n=8)
    assert hinged == approx(plain, rel=1e-4)


def test_spring_at_the_end_of_an_element_is_a_rotational_spring_support():
    for k in (500.0, 5.0e3, 5.0e4):
        with_spring = SystemElements(EI=EI, EA=EA)
        with_spring.add_element([[0, 0], [0, L]], spring={1: k})
        with_spring.add_support_fixed(1)
        with_spring.point_load(2, Fy=-1)

        with_support = SystemElements(EI=EI, EA=EA)
        with_support.add_element([[0, 0], [0, L]])
        with_support.add_support_hinged(1)
        with_support.add_support_spring(1, translation=3, k=k)
        with_support.point_load(2, Fy=-1)

        assert buckling_factor(with_spring) == approx(
            buckling_factor(with_support), rel=1e-6
        )


def test_cantilever_column_with_a_rotational_spring_at_the_base():
    # Beam-column theory: the moment at the base P * delta equals k * y'(0), which gives
    # mu * L * tan(mu * L) = k * L / EI with P = mu**2 * EI. For k -> infinity this is
    # the cantilever (mu * L = pi / 2).
    for k in (500.0, 5.0e3, 5.0e4):
        target = k * L / EI
        low, high = 0.0, np.pi / 2 - 1e-12
        for _ in range(200):  # bisection on the increasing function x * tan(x)
            middle = (low + high) / 2
            if middle * np.tan(middle) < target:
                low = middle
            else:
                high = middle
        expected = low**2 * EI / L**2

        system = SystemElements(EI=EI, EA=EA)
        system.add_element([[0, 0], [0, L]], spring={1: k})
        system.add_support_fixed(1)
        system.point_load(2, Fy=-1)

        assert buckling_factor(system) == approx(expected, rel=2e-3)
        assert expected < EULER / 4


# ---------------------------------------------------------------------------
# repeated calculations
# ---------------------------------------------------------------------------


def loaded_column():
    system = SystemElements(EI=EI, EA=EA)
    system.add_element([[0, 0], [0, 4]])
    system.add_support_fixed(1)
    system.point_load(2, Fy=-100, Fx=0.01)
    return system


def test_repeated_second_order_calculations_give_the_same_buckling_factor():
    # the elements must not keep the geometric stiffness of the previous calculation
    system = loaded_column()
    factors = []
    for _ in range(3):
        system.solve(geometrical_non_linear=True)
        factors.append(system.buckling_factor)
    assert factors == approx([factors[0]] * 3)
    assert factors[0] == approx(np.pi**2 * EI / (4 * 4.0**2) / 100, rel=1e-2)

    copied = copy.deepcopy(system)
    copied.solve(geometrical_non_linear=True)
    assert copied.buckling_factor == approx(factors[0])


def test_linear_calculation_after_a_second_order_calculation_is_linear():
    reference = loaded_column()
    reference.solve()

    system = loaded_column()
    system.solve(geometrical_non_linear=True)
    second_order = system.get_node_displacements(2)["ux"]
    system.solve()

    assert second_order > reference.get_node_displacements(2)["ux"]
    assert system.get_node_displacements(2)["ux"] == approx(
        reference.get_node_displacements(2)["ux"]
    )


def test_second_order_displacement_is_amplified_by_the_compression():
    # cantilever column with a horizontal load H and a compressive load P = Pcr / 2: the
    # tip displacement is amplified by about 1 / (1 - P / Pcr) = 2 (one step of the
    # second order analysis with the first order normal forces)
    pcr = np.pi**2 * EI / (4 * L**2)
    amplification = []
    for load in (-0.5 * pcr, 0.5 * pcr):  # compression, tension
        first = SystemElements(EI=EI, EA=EA)
        first.add_element([[0, 0], [0, L]])
        first.add_support_fixed(1)
        first.point_load(2, Fx=0.01, Fy=load)
        first.solve()

        second = SystemElements(EI=EI, EA=EA)
        second.add_element([[0, 0], [0, L]])
        second.add_support_fixed(1)
        second.point_load(2, Fx=0.01, Fy=load)
        second.solve(geometrical_non_linear=True, discretize_kwargs={"n": 20})
        amplification.append(
            second.get_node_displacements(2)["ux"]
            / first.get_node_displacements(2)["ux"]
        )

    assert amplification[0] == approx(2.0, rel=0.05)
    assert amplification[1] < 1.0  # tension stiffens the column


def test_truss_geometric_stiffness_has_no_rotation_terms():
    # a truss element has no bending stiffness: the geometric stiffness is that of a string,
    # N / l times the transverse translation, so the buckling factor is finite
    matrix = geometric_stiffness_matrix(4.0, 2.0, 0.0, 0.0, 0.0, truss=True)
    assert matrix[[2, 5], :] == approx(0.0)
    assert matrix[:, [2, 5]] == approx(0.0)
    # horizontal member: only the transverse (y) translations, with +-N / l
    assert abs(matrix[1][1]) == approx(2.0 / 4.0)
    assert matrix[1][4] == approx(-matrix[1][1])

    system = SystemElements(EA=1.0e5)
    system.add_truss_element([[0, 0], [3, 0]])
    system.add_truss_element([[3, 0], [1.5, 2]])
    system.add_truss_element([[0, 0], [1.5, 2]])
    system.add_support_hinged(1)
    system.add_support_roll(2)
    system.point_load(3, Fy=-1)
    system.solve(geometrical_non_linear=True)

    assert np.isfinite(system.buckling_factor)
    assert system.buckling_factor > 1.0e3


# ---------------------------------------------------------------------------
# the eigenvalue problem
# ---------------------------------------------------------------------------


def test_smallest_buckling_factor_of_a_diagonal_problem():
    k0 = np.diag([2.0, 3.0])
    assert smallest_buckling_factor(k0, -np.eye(2)) == approx(2.0)
    # tension (positive geometric stiffness) is no buckling mode
    assert smallest_buckling_factor(k0, np.diag([-1.0, 1.0])) == approx(2.0)
    assert smallest_buckling_factor(k0, np.eye(2)) == float("inf")


def test_smallest_buckling_factor_with_a_singular_geometric_stiffness():
    k0 = np.diag([2.0, 3.0])
    assert smallest_buckling_factor(k0, np.diag([-1.0, 0.0])) == approx(2.0)


def test_smallest_buckling_factor_without_a_positive_definite_stiffness():
    k0 = np.diag([2.0, -3.0])
    assert smallest_buckling_factor(k0, np.diag([-1.0, 0.0])) == approx(2.0)


# ---------------------------------------------------------------------------
# geometric stiffness matrix and repeated solves
# ---------------------------------------------------------------------------


def test_geometric_stiffness_matrix_is_symmetric_for_different_end_angles():
    # an inclined roller support changes the angle at one end of the element
    matrix = geometric_stiffness_matrix(3.0, 2.0, 0.3, 0.7)
    assert matrix == approx(matrix.T)


def test_solving_twice_gives_the_same_result_for_an_axially_loaded_column():
    # the transverse displacements are exactly 0 after the first solve; that must not
    # make them restrained degrees of freedom in the second solve
    system = column("cantilever")
    first = np.array(system.solve())
    second = np.array(system.solve())
    assert second == approx(first)
    # Fy = -1 pushes the top of the column towards its base: it shortens by P * L / EA
    assert system.get_node_displacements(2)["uy"] == approx(-1.0 * L / EA)


def test_solving_twice_with_an_imposed_displacement():
    system = SystemElements(EA=EA, EI=1.0e4)
    system.add_element([[0, 0], [10, 0]])
    system.add_support_fixed(1)
    system.add_support_roll(2)
    system.imposed_displacement(2, uy=0.02)
    system.solve()
    first = system.reaction_forces[2].Fy
    system.solve()

    assert first == approx(-3 * 1.0e4 * 0.02 / 10**3)
    assert system.reaction_forces[2].Fy == approx(first)
