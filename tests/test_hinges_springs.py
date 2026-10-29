import numpy as np
from pytest import approx

from anastruct import SystemElements
from anastruct.fem.elements import constitutive_matrix

"""
NOTE: These tests validate the element end releases (hinges and rotational springs,
`spring={node: k}` and internal hinges) against analytical results. A hinge is
modelled by static condensation of the end rotation (an element with one hinge has the
rotational stiffness 3EI/l instead of 4EI/l); a rotational spring k acts in series with
the element. General definitions:
- L = member length
- EA = axial stiffness
- EI = bending stiffness
- k = stiffness of the rotational spring at the element end

The tests are plain test functions (not nested describe/context blocks), so that every
assertion is guaranteed to be executed.
"""

EA = 1.0e9
EI = 1.0e4
L = 4.0


# ---------------------------------------------------------------------------
# constitutive matrix
# ---------------------------------------------------------------------------


def test_constitutive_matrix_without_releases():
    matrix = constitutive_matrix(EA, EI, L, None, False, False)
    expected = np.array(
        [
            [EA / L, 0, 0],
            [0, 4 * EI / L, -2 * EI / L],
            [0, -2 * EI / L, 4 * EI / L],
        ]
    )
    assert matrix == approx(expected)


def test_constitutive_matrix_with_one_hinge_is_condensed():
    # fixed-hinge: 3EI/l, no coupling with the released rotation
    for hinge_1, hinge_2, spring in (
        (False, True, None),
        (False, False, {2: 0}),
        (False, True, {2: 5.0}),  # the hinge prevails over a spring
    ):
        matrix = constitutive_matrix(EA, EI, L, spring, hinge_1, hinge_2)
        assert matrix[1][1] == approx(3 * EI / L)
        assert matrix[1][2] == 0 and matrix[2][1] == 0 and matrix[2][2] == 0

    matrix = constitutive_matrix(EA, EI, L, None, True, False)
    assert matrix[2][2] == approx(3 * EI / L)
    assert matrix[1][1] == 0 and matrix[1][2] == 0 and matrix[2][1] == 0


def test_constitutive_matrix_with_two_hinges_has_no_bending_stiffness():
    matrix = constitutive_matrix(EA, EI, L, {1: 0, 2: 0}, False, False)
    assert matrix[1:, 1:] == approx(np.zeros((2, 2)))
    assert matrix[0][0] == approx(EA / L)


def test_constitutive_matrix_with_springs_is_finite_and_symmetric():
    # the series of two flexibilities used to give NaN/inf for a finite spring
    matrix = constitutive_matrix(EA, EI, L, {1: 500.0, 2: 800.0}, False, False)
    assert np.all(np.isfinite(matrix))
    assert matrix == approx(matrix.T)
    flexibility = np.linalg.inv(matrix[1:, 1:])
    expected = np.linalg.inv(EI / L * np.array([[4.0, -2.0], [-2.0, 4.0]]))
    assert flexibility == approx(expected + np.diag([1 / 500.0, 1 / 800.0]))


def test_constitutive_matrix_for_a_rigid_spring_is_the_rigid_element():
    matrix = constitutive_matrix(EA, EI, L, {1: 1e15, 2: 1e15}, False, False)
    assert matrix[1][1] == approx(4 * EI / L)
    assert matrix[1][2] == approx(-2 * EI / L)


# ---------------------------------------------------------------------------
# mechanical loads
# ---------------------------------------------------------------------------


def test_fixed_hinge_fixed_beam_with_point_load_at_the_hinge():
    # two propped cantilevers sharing the loaded hinge: u = P * l**3 / (6 * EI)
    p = 100.0
    system = SystemElements(EA=EA, EI=EI)
    system.add_element([[0, 0], [L, 0]])
    system.add_element([[L, 0], [2 * L, 0]])
    system.add_support_fixed([1, 3])
    system.add_internal_hinge(2)
    system.point_load(2, Fy=-p)
    system.solve()

    assert abs(system.get_node_displacements(2)["uy"]) == approx(p * L**3 / (6 * EI))
    assert system.reaction_forces[1].Fy == approx(system.reaction_forces[3].Fy)


def test_element_end_hinge_is_equivalent_to_a_hinged_support():
    # fixed-pinned beam under a q-load: M_fixed = q * L**2 / 8
    q = -10.0
    length = 6.0

    def solve(spring):
        system = SystemElements(EA=EA, EI=EI)
        system.add_element([[0, 0], [length, 0]], spring=spring)
        system.add_support_fixed(1)
        system.add_support_hinged(2)
        system.q_load(q, 1)
        system.solve()
        return system

    plain = solve(None)
    released = solve({2: 0})
    assert abs(plain.reaction_forces[1].Tz) == approx(abs(q) * length**2 / 8, rel=1e-4)
    assert released.reaction_forces[1].Tz == approx(plain.reaction_forces[1].Tz)
    assert released.reaction_forces[2].Fy == approx(plain.reaction_forces[2].Fy)


def test_cantilever_with_a_rotational_spring_at_the_fixed_end():
    # tip deflection: P * L**3 / (3 * EI) + P * L**2 / k  (the spring rotates by M / k)
    p = 10.0
    for k in (50.0, 1e3, 1e9):
        system = SystemElements(EA=EA, EI=EI)
        system.add_element([[0, 0], [L, 0]], spring={1: k})
        system.add_support_fixed(1)
        system.point_load(2, Fy=-p)
        system.solve()

        assert abs(system.get_node_displacements(2)["uy"]) == approx(
            p * L**3 / (3 * EI) + p * L**2 / k
        )


def test_simply_supported_beam_with_a_rotational_spring_at_the_end():
    # end rotation under an end moment M: M * (L / (3 * EI) + 1 / k)
    m = 10.0
    for k in (500.0, 1e3, 1e8):
        system = SystemElements(EA=EA, EI=EI)
        system.add_element([[0, 0], [L, 0]], spring={2: k})
        system.add_support_hinged(1)
        system.add_support_roll(2)
        system.moment_load(2, Tz=m)
        system.solve()

        assert abs(system.get_node_displacements(2)["phi_z"]) == approx(
            m * (L / (3 * EI) + 1 / k)
        )


def test_springs_at_both_ends_with_imposed_rotation():
    # fixed-fixed element with rotational springs at both ends: the end moment follows
    # from the series of the element and the springs
    k1, k2 = 300.0, 700.0
    phi = 0.01
    system = SystemElements(EA=EA, EI=EI)
    system.add_element([[0, 0], [L, 0]], spring={1: k1, 2: k2})
    system.add_support_fixed([1, 2])
    system.imposed_displacement(2, phi_z=phi)
    system.solve()

    element = EI / L * np.array([[4.0, -2.0], [-2.0, 4.0]])
    series = np.linalg.inv(np.linalg.inv(element) + np.diag([1 / k1, 1 / k2]))
    assert system.reaction_forces[2].Tz == approx(series[1][1] * phi)
    # the far-end moment is reported with the opposite sign of the coupling term
    # (without springs: M_far = +2 * EI * phi / L)
    assert system.reaction_forces[1].Tz == approx(-series[0][1] * phi)


def test_q_load_on_a_fixed_fixed_element_with_rotational_springs():
    # the fixed-end moment is reduced by the springs: the beam end rotation
    # q * L**3 / (24 * EI) - M * L / (2 * EI) equals the spring rotation M / k, so
    # M = q * L**3 / (24 * EI) / (L / (2 * EI) + 1 / k)
    q = 10.0
    length = 6.0
    for k in (500.0, 50.0, 1e9):
        system = SystemElements(EA=EA, EI=EI)
        system.add_element([[0, 0], [length, 0]], spring={1: k, 2: k})
        system.add_support_fixed([1, 2])
        system.q_load(-q, 1)
        system.solve()

        expected = q * length**3 / (24 * EI) / (length / (2 * EI) + 1 / k)
        assert abs(system.reaction_forces[1].Tz) == approx(expected)
        assert abs(system.reaction_forces[2].Tz) == approx(expected)
        assert system.get_element_results(1)["Mmax"] == approx(expected)


def test_q_load_on_an_element_with_a_start_hinge():
    # hinge at the start, fixed end: M_fixed = q * L**2 / 8, R_hinge = 3 * q * L / 8
    q = 10.0
    length = 6.0
    system = SystemElements(EA=EA, EI=EI)
    system.add_element([[0, 0], [length, 0]], spring={1: 0})
    system.add_support_hinged(1)
    system.add_support_fixed(2)
    system.q_load(-q, 1)
    system.solve()

    assert abs(system.reaction_forces[2].Tz) == approx(q * length**2 / 8, rel=1e-4)
    assert abs(system.reaction_forces[1].Fy) == approx(3 * q * length / 8, rel=1e-4)


def test_q_load_on_a_beam_with_an_internal_hinge():
    # fixed - span - hinge - span - roller (statically determinate):
    # R_roller = q * L / 2, M_fixed = q * L**2 (the shear at the hinge q * L / 2 acts
    # on the cantilever)
    q = 10.0
    length = 6.0
    system = SystemElements(EA=EA, EI=EI)
    system.add_element([[0, 0], [length, 0]])
    system.add_element([[length, 0], [2 * length, 0]])
    system.add_support_fixed(1)
    system.add_support_roll(3)
    system.add_internal_hinge(2)
    system.q_load(-q, [1, 2])
    system.solve()

    assert abs(system.reaction_forces[3].Fy) == approx(q * length / 2)
    assert abs(system.reaction_forces[1].Tz) == approx(q * length**2)


def test_q_load_on_an_element_with_a_rotational_spring_and_a_plastic_hinge():
    # a spring and a plastic moment at the same end: the iteration converges and the
    # end moment is limited to mp; the other end follows from equilibrium (75)
    mp = 30.0
    system = SystemElements()
    system.add_element([[0, 0], [6, 0]], EA=5e4, EI=2e3, spring={1: 1500.0}, mp={1: mp})
    system.add_support_fixed([1, 2])
    system.q_load(q=-20, element_id=1)
    system.solve()

    assert system.element_map[1].bending_moment is not None
    assert abs(system.reaction_forces[1].Tz) == approx(mp, rel=5e-3)
    assert abs(system.reaction_forces[2].Tz) == approx(75.0, rel=5e-3)


def test_trapezoidal_q_load_on_a_truss_element():
    # a truss element carries no moment: simply supported reactions
    # R1 = (2 * q1 + q2) * L / 6, R2 = (q1 + 2 * q2) * L / 6
    q1, q2, length = 6.0, 18.0, 5.0
    system = SystemElements()
    system.add_truss_element([[0, 0], [length, 0]], EA=1e5)
    system.add_support_hinged(1)
    system.add_support_roll(2)
    system.q_load(q=[-q1, -q2], element_id=1)
    system.solve()

    assert abs(system.reaction_forces[1].Fy) == approx((2 * q1 + q2) * length / 6)
    assert abs(system.reaction_forces[2].Fy) == approx((q1 + 2 * q2) * length / 6)


# ---------------------------------------------------------------------------
# thermal loads
# ---------------------------------------------------------------------------


def test_thermal_gradient_on_an_element_with_an_end_hinge():
    # fixed-pinned beam with a temperature gradient (warmer upper face):
    # M_fixed = 3/2 * EI * kappa (sagging, so negative in anaStruct), also when the hinge
    # is an element end release
    kappa = 1.2e-5 * 15.0 / 0.5
    length = 10.0

    def solve(spring):
        system = SystemElements(EA=EA, EI=EI)
        system.add_element([[0, 0], [length, 0]], spring=spring)
        system.add_support_fixed(1)
        system.add_support_hinged(2)
        system.thermal_load(1, delta_t=0, alpha=1.2e-5, delta_t_gradient=15, h=0.5)
        system.solve()
        return system

    for spring in (None, {2: 0}):
        system = solve(spring)
        assert system.reaction_forces[1].Tz == approx(-1.5 * EI * kappa)
        assert system.reaction_forces[2].Fy == approx(-1.5 * EI * kappa / length)


def test_thermal_gradient_on_an_element_with_a_rotational_spring():
    # a spring on the fixed end of a cantilever does not carry a moment (the cantilever
    # curves freely), so the spring does not change the free thermal response
    kappa = 1.2e-5 * 15.0 / 0.5
    system = SystemElements(EA=EA, EI=EI)
    system.add_element([[0, 0], [10, 0]], spring={1: 500.0})
    system.add_support_fixed(1)
    system.thermal_load(1, delta_t=0, alpha=1.2e-5, delta_t_gradient=15, h=0.5)
    system.solve()

    # the tip moves down: the warmer face is the upper face
    assert system.get_node_displacements(2)["uy"] == approx(-kappa * 10.0**2 / 2)
    assert system.reaction_forces[1].Tz == approx(0.0, abs=1e-9)
