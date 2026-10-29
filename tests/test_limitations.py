import numpy as np
from pytest import approx, mark, raises

from anastruct import LoadCase, LoadCombination, SystemElements
from anastruct.basic import FEMException

"""
NOTE: These tests cover former limitations of anaStruct: loads on nodes with an inclined
roller support, member buckling of bars, plastic hinges in a second order analysis, load
combinations of non linear calculations and changes of the model after a solve. General
definitions:
- L = member length
- EA = axial stiffness
- EI = bending stiffness
- Pcr = Euler buckling load of a column
- reaction_forces[i].Fx is the force of the support on the structure in the +x direction
  and reaction_forces[i].Fy the same force counted positive downwards

The tests are plain test functions (not nested describe/context blocks), so that every
assertion is guaranteed to be executed.
"""

EA = 1.0e6
EI = 1.0e4
L = 4.0


# ---------------------------------------------------------------------------
# point loads on a node with an inclined roller support
# ---------------------------------------------------------------------------


@mark.parametrize("angle", [30, 60, 100, 135, 180, 225, -40])
@mark.parametrize("load", [(0.0, -10.0), (6.0, -10.0), (6.0, 0.0)])
def test_point_load_on_an_inclined_roller_satisfies_equilibrium(angle, load):
    # beam hinged at x = 0, roller at x = L whose free direction has the angle `angle`:
    # the reaction R of the roller is normal to the free direction, and the moment about
    # the hinge gives R * n_y * L + F_y * L = 0
    system = SystemElements(EA=EA, EI=EI)
    system.add_element([[0, 0], [L, 0]])
    system.add_support_hinged(1)
    system.add_support_roll(2, angle=angle)
    system.point_load(2, Fx=load[0], Fy=load[1])
    system.solve()

    radians = np.radians(angle)
    normal = np.array([-np.sin(radians), np.cos(radians)])
    force = np.array(load)
    roller = -force[1] / normal[1] * normal
    hinge = -(force + roller)

    reaction_roller = system.reaction_forces[2]
    reaction_hinge = system.reaction_forces[1]
    assert [reaction_roller.Fx, -reaction_roller.Fy] == approx(list(roller), abs=1e-6)
    assert [reaction_hinge.Fx, -reaction_hinge.Fy] == approx(list(hinge), abs=1e-6)


@mark.parametrize(
    "beam, plain, angle, load",
    [
        ([[0, 0], [L, 0]], "x", 0, (3.0, -10.0)),
        ([[0, 0], [L, 0]], "x", 180, (7.0, -10.0)),
        ([[0, 0], [0, L]], "y", 90, (5.0, 3.0)),
    ],
)
def test_point_load_on_an_inclined_roller_equals_a_plain_roller(
    beam, plain, angle, load
):
    # inclined rollers of 0, 90 and 180 degrees are the plain rollers
    results = []
    for kind in (plain, angle):
        system = SystemElements(EA=EA, EI=EI)
        system.add_element(beam)
        system.add_support_hinged(1)
        if isinstance(kind, str):
            system.add_support_roll(2, kind)
        else:
            system.add_support_roll(2, angle=kind)
        system.point_load(2, Fx=load[0], Fy=load[1])
        system.solve()
        results.append(
            [
                system.reaction_forces[1].Fx,
                system.reaction_forces[1].Fy,
                system.reaction_forces[2].Fx,
                system.reaction_forces[2].Fy,
                *system.get_node_displacements(2).values(),
            ]
        )
    assert results[0] == approx(results[1], abs=1e-9)


# ---------------------------------------------------------------------------
# member buckling of bars
# ---------------------------------------------------------------------------


def test_bar_with_hinges_at_both_ends_buckles_like_an_euler_column():
    # a bar is an element with hinges at both ends: discretize puts the hinges at the
    # ends of the bar only, so the bar can buckle between the new nodes
    ei, length = 5.0e3, 10.0
    system = SystemElements(EI=ei, EA=EA)
    system.add_element([[0, 0], [0, length]], spring={1: 0, 2: 0})
    system.add_support_hinged(1)
    system.add_support_roll(2, "y")
    system.point_load(2, Fy=-1)
    system.solve(geometrical_non_linear=True, discretize_kwargs={"n": 20})

    assert system.buckling_factor == approx(np.pi**2 * ei / length**2, rel=1e-3)


def test_member_buckling_of_a_truss_bar():
    # symmetric two-bar truss with an apex load of 1: N = 1 / (2 sin 45 deg) in each bar,
    # the bars (l = 2 sqrt 2) buckle at Pcr / N = pi**2 EI / l**2 / N
    ei = 5.0e3
    length = np.hypot(2, 2)
    expected = np.pi**2 * ei / length**2 / (1 / (2 * np.sin(np.pi / 4)))

    system = SystemElements(EA=EA, EI=ei)
    system.add_element([[0, 0], [2, 2]], spring={1: 0, 2: 0})
    system.add_element([[2, 2], [4, 0]], spring={1: 0, 2: 0})
    system.add_support_hinged(1)
    system.add_support_hinged(3)
    system.point_load(2, Fy=-1)
    system.solve(geometrical_non_linear=True, discretize_kwargs={"n": 20})

    assert system.buckling_factor == approx(expected, rel=1e-3)


def test_truss_elements_have_no_member_buckling():
    # truss elements have no bending stiffness: only the geometry of the whole truss
    system = SystemElements(EA=EA)
    system.add_truss_element([[0, 0], [2, 2]])
    system.add_truss_element([[2, 2], [4, 0]])
    system.add_support_hinged(1)
    system.add_support_hinged(3)
    system.point_load(2, Fy=-1)
    system.solve(geometrical_non_linear=True)

    assert system.buckling_factor > 1.0e5


# ---------------------------------------------------------------------------
# plastic hinges in a second order analysis
# ---------------------------------------------------------------------------


def plastic_column(horizontal, axial, mp, **solve_kwargs):
    # fixed base (plastic moment mp), lateral support at the top, a horizontal load at
    # mid-height and an axial compression at the top
    system = SystemElements(EI=5.0e3, EA=EA)
    system.add_element([[0, 0], [0, 2]], mp={1: mp})
    system.add_element([[0, 2], [0, 4]])
    system.add_support_fixed(1)
    system.add_support_roll(3, "y")
    system.point_load(2, Fx=horizontal)
    system.point_load(3, Fy=-axial)
    system.solve(**solve_kwargs)
    return system


def test_second_order_analysis_without_axial_force_is_first_order():
    first = plastic_column(60.0, 0.0, 30.0)
    second = plastic_column(60.0, 0.0, 30.0, geometrical_non_linear=True)
    assert second.get_node_displacements(2)["ux"] == approx(
        first.get_node_displacements(2)["ux"]
    )


def test_plastic_hinge_in_a_second_order_analysis():
    # the base moment is limited by mp, the compression amplifies the displacement
    first = plastic_column(60.0, 150.0, 30.0)
    second = plastic_column(60.0, 150.0, 30.0, geometrical_non_linear=True)
    elastic_second = plastic_column(60.0, 150.0, 1.0e9, geometrical_non_linear=True)

    assert abs(first.reaction_forces[1].Tz) == approx(30.0, rel=5e-3)
    assert abs(second.reaction_forces[1].Tz) == approx(30.0, rel=5e-3)
    assert abs(elastic_second.reaction_forces[1].Tz) > 45.0  # not limited
    assert abs(second.get_node_displacements(2)["ux"]) > abs(
        first.get_node_displacements(2)["ux"]
    )


def test_buckling_factor_with_plastic_hinges_is_the_elastic_one():
    # the elastic cantilever column, one element: the load of 0.3 Pcr
    ei, length = 5.0e3, 4.0
    pcr = np.pi**2 * ei / (4 * length**2)
    system = SystemElements(EI=ei, EA=EA)
    system.add_element([[0, 0], [0, length]], mp={1: 30.0})
    system.add_support_fixed(1)
    system.point_load(2, Fx=4.0, Fy=-0.3 * pcr)
    system.solve(geometrical_non_linear=True, discretize_kwargs={"n": 20})

    assert system.buckling_factor == approx(1 / 0.3, rel=1e-3)
    # (the moment of 4 * 4 + the second order part stays below mp: elastic result)
    elastic = SystemElements(EI=ei, EA=EA)
    elastic.add_element([[0, 0], [0, length]])
    elastic.add_support_fixed(1)
    elastic.point_load(2, Fx=4.0, Fy=-0.3 * pcr)
    elastic.solve(geometrical_non_linear=True)
    assert system.get_node_displacements(2)["ux"] == approx(
        elastic.get_node_displacements(2)["ux"]
    )


# ---------------------------------------------------------------------------
# load combinations of non linear calculations
# ---------------------------------------------------------------------------


def two_element_column():
    system = SystemElements(EI=5.0e3, EA=EA)
    system.add_element([[0, 0], [0, 2]])
    system.add_element([[0, 2], [0, 4]])
    system.add_support_fixed(1)
    return system


def test_second_order_combination_is_one_calculation_with_all_the_loads():
    pcr = np.pi**2 * 5.0e3 / (4 * 4.0**2)
    horizontal, axial = 5.0, 0.3 * pcr
    wind = LoadCase("wind")
    wind.point_load(2, Fx=1.0)
    wind.q_load(-2.0, 2)
    gravity = LoadCase("gravity")
    gravity.point_load(3, Fy=-1.0)
    combination = LoadCombination("ULS")
    combination.add_load_case(wind, horizontal)
    combination.add_load_case(gravity, axial)
    results = combination.solve(two_element_column(), geometrical_non_linear=True)

    direct = two_element_column()
    direct.point_load(2, Fx=horizontal)
    direct.point_load(3, Fy=-axial)
    direct.q_load(-2.0 * horizontal, 2)
    direct.solve(geometrical_non_linear=True)

    combined = results["combination"]
    for node_id in (1,):
        assert combined.reaction_forces[node_id].Fx == approx(
            direct.reaction_forces[node_id].Fx
        )
        assert combined.reaction_forces[node_id].Fy == approx(
            direct.reaction_forces[node_id].Fy
        )
        assert combined.reaction_forces[node_id].Tz == approx(
            direct.reaction_forces[node_id].Tz
        )
    assert combined.get_node_displacements(3)["ux"] == approx(
        direct.get_node_displacements(3)["ux"]
    )
    assert combined.buckling_factor == approx(direct.buckling_factor)

    # superposing the second order results of the load cases would not be second order
    superposed = sum(
        results[name].get_node_displacements(3)["ux"] for name in ("wind", "gravity")
    )
    assert abs(combined.get_node_displacements(3)["ux"]) > abs(superposed) * 1.05


def test_second_order_combination_with_thermal_loads_and_factors():
    def build():
        system = SystemElements(EI=5.0e3, EA=EA)
        system.add_element([[0, 0], [0, 2]])
        system.add_element([[0, 2], [0, 4]])
        system.add_support_fixed(1)
        return system

    q_case = LoadCase("q")
    q_case.q_load(-3.0, 1)
    thermal_case = LoadCase("thermal")
    thermal_case.thermal_load(2, delta_t=15, alpha=1.2e-5, delta_t_gradient=10, h=0.4)
    point_case = LoadCase("point")
    point_case.point_load(3, Fx=2.0, Fy=-50.0)
    combination = LoadCombination("c")
    combination.add_load_case(q_case, 1.5)
    combination.add_load_case(thermal_case, 1.2)
    combination.add_load_case(point_case, 1.1)
    combined = combination.solve(build(), geometrical_non_linear=True)["combination"]

    direct = build()
    direct.q_load(-3.0 * 1.5, 1)
    direct.thermal_load(
        2, delta_t=15 * 1.2, alpha=1.2e-5, delta_t_gradient=10 * 1.2, h=0.4
    )
    direct.point_load(3, Fx=2.0 * 1.1, Fy=-50.0 * 1.1)
    direct.solve(geometrical_non_linear=True)

    assert combined.reaction_forces[1].Tz == approx(direct.reaction_forces[1].Tz)
    assert combined.get_node_displacements(3)["ux"] == approx(
        direct.get_node_displacements(3)["ux"]
    )


def test_linear_combination_is_still_superposed():
    wind = LoadCase("wind")
    wind.point_load(3, Fx=1.0)
    gravity = LoadCase("gravity")
    gravity.point_load(3, Fy=-1.0)
    combination = LoadCombination("c")
    combination.add_load_case(wind, 5.0)
    combination.add_load_case(gravity, 40.0)
    results = combination.solve(two_element_column())

    expected = sum(
        results[name].get_node_displacements(3)["ux"] for name in ("wind", "gravity")
    )
    assert results["combination"].get_node_displacements(3)["ux"] == approx(expected)
    assert results["combination"].buckling_factor is None


def test_combination_with_plastic_hinges_is_one_calculation():
    def build():
        system = SystemElements(EI=5.0e3, EA=EA)
        system.add_element([[0, 0], [0, 2]], mp={1: 30.0})
        system.add_element([[0, 2], [0, 4]])
        system.add_support_fixed(1)
        system.add_support_roll(3, "y")
        return system

    first = LoadCase("first")
    first.point_load(2, Fx=1.0)
    combination = LoadCombination("c")
    combination.add_load_case(first, 60.0)
    combined = combination.solve(build())["combination"]

    direct = build()
    direct.point_load(2, Fx=60.0)
    direct.solve()

    assert abs(combined.reaction_forces[1].Tz) == approx(30.0, rel=5e-3)
    assert combined.reaction_forces[1].Tz == approx(direct.reaction_forces[1].Tz)
    assert combined.get_node_displacements(2)["ux"] == approx(
        direct.get_node_displacements(2)["ux"]
    )


# ---------------------------------------------------------------------------
# changes of the model after a solve
# ---------------------------------------------------------------------------


def chain():
    system = SystemElements(EA=EA, EI=EI)
    system.add_element([[0, 0], [4, 0]])
    system.add_element([[4, 0], [8, 0]])
    system.add_element([[8, 0], [12, 0]])
    system.add_support_fixed(1)
    system.add_support_roll(4)
    system.q_load(-10, [1, 2, 3])
    return system


def test_support_added_after_a_solve_is_used():
    expected = chain()
    expected.add_support_hinged(3)
    expected.solve()

    system = chain()
    system.solve()
    system.add_support_hinged(3)
    system.solve()

    assert system.reaction_forces[3].Fy == approx(expected.reaction_forces[3].Fy)
    assert system.reaction_forces[1].Tz == approx(expected.reaction_forces[1].Tz)


def test_insert_node_after_a_solve():
    def build(solve_first):
        system = chain()
        if solve_first:
            system.solve()
        ids = system.insert_node(2, factor=0.5)
        system.q_load(-10, [ids["new_element_id1"], ids["new_element_id2"]])
        system.solve()
        return system

    after, before = build(True), build(False)
    assert after.reaction_forces[1].Fy == approx(before.reaction_forces[1].Fy)
    assert after.reaction_forces[1].Tz == approx(before.reaction_forces[1].Tz)


def test_element_added_after_a_solve():
    def build(solve_first):
        system = chain()
        if solve_first:
            system.solve()
        system.add_element([[12, 0], [16, 0]])
        system.add_support_roll(5)
        system.q_load(-10, 4)
        system.solve()
        return system

    after, before = build(True), build(False)
    assert after.reaction_forces[1].Fy == approx(before.reaction_forces[1].Fy)


def test_imposed_displacement_after_a_change_of_the_model():
    system = chain()
    system.solve()
    system.add_support_hinged(3)  # the solution is reset, so a displacement can follow
    system.imposed_displacement(3, uy=-0.01)
    system.solve()

    expected = chain()
    expected.add_support_hinged(3)
    expected.imposed_displacement(3, uy=-0.01)
    expected.solve()
    assert system.reaction_forces[3].Fy == approx(expected.reaction_forces[3].Fy)


def test_spring_dictionaries_are_not_shared_between_elements():
    spring = {1: 0}
    multiple = SystemElements(EA=EA, EI=EI)
    multiple.add_multiple_elements([[0, 0], [8, 0]], n=2, spring=spring)
    grid = SystemElements(EA=EA, EI=EI)
    grid.add_element_grid([0, 3, 6], [0, 0, 0], spring=spring)

    elements = list(multiple.element_map.values()) + list(grid.element_map.values())
    assert len({id(element.springs) for element in elements}) == len(elements)
    assert spring == {1: 0}


# ---------------------------------------------------------------------------
# review findings: hinges added after a solve, repeated and forced linear calculations
# ---------------------------------------------------------------------------


def test_internal_hinge_added_after_a_solve():
    # fixed - 4 m - hinge - 4 m - fixed with a load P = 10 at the hinge: two cantilevers
    # with M = P / 2 * l = 20 and u = P / 2 * l**3 / (3 EI)
    def build():
        system = SystemElements()
        system.add_element([[0, 0], [4, 0]])
        system.add_element([[4, 0], [8, 0]])
        system.add_support_fixed([1, 3])
        system.point_load(2, Fy=-10)
        return system

    after = build()
    after.solve()
    after.add_internal_hinge(2)
    after.solve()

    assert abs(after.reaction_forces[1].Tz) == approx(20.0)
    assert abs(after.get_node_displacements(2)["uy"]) == approx(
        5.0 * 4.0**3 / (3 * 5.0e3)
    )


def plastic_beam_column(mp, **solve_kwargs):
    # fixed base (plastic moment mp), lateral support at the top, a horizontal load at
    # mid-height and an axial compression at the top
    system = SystemElements(EI=2.0e3, EA=5.0e4)
    system.add_element([[0, 0], [0, 2]], mp={1: mp})
    system.add_element([[0, 2], [0, 4]])
    system.add_support_fixed(1)
    system.add_support_roll(3, "y")
    system.point_load(2, Fx=10.0)
    system.point_load(3, Fy=-80.0)
    system.solve(**solve_kwargs)
    return system


def test_forced_linear_second_order_analysis_ignores_the_plastic_moments():
    plastic = plastic_beam_column(4.0, geometrical_non_linear=True, force_linear=True)
    elastic = plastic_beam_column(1.0e9, geometrical_non_linear=True, force_linear=True)

    assert plastic.reaction_forces[1].Tz == approx(elastic.reaction_forces[1].Tz)
    assert plastic.get_node_displacements(2)["ux"] == approx(
        elastic.get_node_displacements(2)["ux"]
    )
    assert abs(plastic.reaction_forces[1].Tz) > 4.0  # the elastic moment, not mp


def test_repeated_second_order_calculations_with_plastic_hinges_are_identical():
    system = plastic_beam_column(4.0, geometrical_non_linear=True)
    elastic = plastic_beam_column(1.0e9, geometrical_non_linear=True)
    first = (
        system.buckling_factor,
        system.reaction_forces[1].Tz,
        system.get_node_displacements(2)["ux"],
    )
    for _ in range(2):
        system.solve(geometrical_non_linear=True)
        again = (
            system.buckling_factor,
            system.reaction_forces[1].Tz,
            system.get_node_displacements(2)["ux"],
        )
        assert again == approx(first, rel=1e-9)

    # the buckling factor is that of the elastic structure
    assert first[0] == approx(elastic.buckling_factor)
    assert abs(first[1]) == approx(4.0, rel=5e-3)  # limited by mp


def test_repeated_plastic_calculations_start_from_the_elastic_stiffness():
    system = SystemElements(EI=5.0e3, EA=1.0e6)
    system.add_element([[0, 0], [0, 2]], mp={1: 30.0})
    system.add_element([[0, 2], [0, 4]])
    system.add_support_fixed(1)
    system.add_support_roll(3, "y")
    system.point_load(2, Fx=60.0)
    results = []
    for _ in range(3):
        system.solve()
        results.append(
            (system.reaction_forces[1].Tz, system.get_node_displacements(2)["ux"])
        )
    assert results[1] == approx(results[0], rel=1e-9)
    assert results[2] == approx(results[0], rel=1e-9)


def test_plastic_iteration_that_does_not_converge_is_an_error():
    # a cantilever with a plastic hinge at the base that carries less than the load:
    # a mechanism, there is no solution
    system = SystemElements(EI=5.0e3, EA=1.0e6)
    system.add_element([[0, 0], [0, 4]], mp={1: 30.0})
    system.add_support_fixed(1)
    system.point_load(2, Fx=10.0)
    with raises(FEMException):
        system.solve(max_iter=30)


def test_combination_counts_the_loads_of_the_system_once():
    # the self-weight and a load of the system itself are part of every load case result,
    # but only once of the combination (linear as well as non linear)
    def build():
        system = SystemElements(EI=EI, EA=EA)
        system.add_element([[0, 0], [4, 0]], g=1.5)
        system.add_element([[4, 0], [8, 0]], g=1.5)
        system.add_support_fixed(1)
        system.add_support_roll(3)
        system.point_load(2, Fy=-3.0)
        return system

    first = LoadCase("first")
    first.q_load(-2.0, 1)
    second = LoadCase("second")
    second.point_load(3, Fy=-4.0)
    combination = LoadCombination("c")
    combination.add_load_case(first, 1.5)
    combination.add_load_case(second, 1.2)
    combined = combination.solve(build())["combination"]

    direct = build()
    direct.q_load(-2.0 * 1.5, 1)
    direct.point_load(3, Fy=-4.0 * 1.2)
    direct.solve()

    assert combined.reaction_forces[1].Fy == approx(direct.reaction_forces[1].Fy)
    assert combined.reaction_forces[1].Tz == approx(direct.reaction_forces[1].Tz)
    assert combined.get_node_displacements(2)["uy"] == approx(
        direct.get_node_displacements(2)["uy"]
    )


# ---------------------------------------------------------------------------
# imposed displacements in a direction that no support restrains
# ---------------------------------------------------------------------------


def fixed_beam():
    system = SystemElements(EA=EA, EI=EI)
    system.add_element([[0, 0], [10, 0]])
    system.add_support_fixed(1)
    return system


def message_of(action):
    with raises(FEMException) as error:
        action()
    return error.value.message


def test_imposed_displacement_in_the_free_direction_of_a_roller_explains_why():
    def action():
        system = fixed_beam()
        system.add_support_roll(2, "x")
        system.imposed_displacement(2, ux=0.01)
        system.solve()

    message = message_of(action)
    assert "imposed ux at node 2" in message
    assert "roller support that is free to move in the x direction" in message
    assert "restrains the y translation" in message


def test_imposed_rotation_at_a_hinged_support_explains_why():
    def action():
        system = fixed_beam()
        system.add_support_hinged(2)
        system.imposed_displacement(2, phi_z=0.01)
        system.solve()

    message = message_of(action)
    assert "imposed phi_z at node 2" in message
    assert "hinged support" in message and "not the rotation" in message


def test_imposed_displacement_on_an_inclined_roller_explains_the_restrained_direction():
    def action():
        system = fixed_beam()
        system.add_support_roll(2, angle=30)
        system.imposed_displacement(2, uy=0.01)

    message = message_of(action)
    assert "imposed uy at node 2" in message
    assert "inclined roller support" in message
    assert "direction of 30 degrees" in message  # the free direction
    assert (
        "direction of 120 degrees" in message
    )  # the restrained direction, normal to it


def test_inclined_roller_added_after_the_imposed_displacement_is_reported_when_solving():
    def action():
        system = fixed_beam()
        system.imposed_displacement(2, uy=0.01)
        system.add_support_roll(2, angle=45)
        system.solve()

    message = message_of(action)
    assert "imposed uy at node 2" in message
    assert "inclined roller support" in message


def test_imposed_displacement_without_a_support_and_on_a_spring_support():
    def without_support():
        system = fixed_beam()
        system.imposed_displacement(2, uy=0.01)
        system.solve()

    def with_spring():
        system = fixed_beam()
        system.add_support_spring(2, translation=2, k=500.0)
        system.imposed_displacement(2, uy=0.01)
        system.solve()

    assert "no support restrains" in message_of(without_support)
    assert "spring support, which yields" in message_of(with_spring)


# ---------------------------------------------------------------------------
# findings of the second review
# ---------------------------------------------------------------------------


def test_second_order_solves_of_a_horizontal_column_are_repeatable():
    # the transverse displacements of a horizontal column are exactly 0: they must not
    # become restrained degrees of freedom in the second solve
    def build():
        system = SystemElements(EI=5.0e3, EA=1.0e5)
        system.add_multiple_elements([[0, 0], [10, 0]], 10)
        system.add_support_hinged(1)
        system.add_support_roll(11, "x")
        system.point_load(11, Fx=-1)
        return system

    system = build()
    factors = []
    for _ in range(3):
        system.solve(geometrical_non_linear=True)
        factors.append(system.buckling_factor)
    assert np.isfinite(factors[0])
    assert factors == approx([factors[0]] * 3)
    assert factors[0] == approx(np.pi**2 * 5.0e3 / 10.0**2, rel=2e-3)


def test_linear_calculation_after_a_plastic_one_is_elastic():
    def build():
        system = SystemElements(EA=EA, EI=EI)
        system.add_element([[0, 0], [3, 0]], mp={1: 20.0})
        system.add_element([[3, 0], [6, 0]], mp={2: 20.0})
        system.add_support_fixed([1, 3])
        system.q_load(-10, [1, 2])
        return system

    plastic = build()
    plastic.solve()
    assert abs(plastic.reaction_forces[1].Tz) == approx(20.0, rel=5e-3)
    plastic.solve(force_linear=True)

    elastic = build()
    elastic.solve(force_linear=True)
    assert abs(elastic.reaction_forces[1].Tz) == approx(10 * 6**2 / 12)
    assert plastic.reaction_forces[1].Tz == approx(elastic.reaction_forces[1].Tz)


def test_combination_with_an_imposed_displacement_when_the_system_was_solved():
    def build():
        system = chain()
        system.q_load(-10, [1, 2, 3])
        return system

    solved = build()
    solved.solve()
    settlement = LoadCase("settlement")
    settlement.imposed_displacement(4, uy=-0.01)
    combination = LoadCombination("c")
    combination.add_load_case(settlement, 1.0)
    combined = combination.solve(solved)["combination"]

    fresh = build()
    fresh.imposed_displacement(4, uy=-0.01)
    fresh.solve()
    assert combined.reaction_forces[1].Fy == approx(fresh.reaction_forces[1].Fy)
    assert combined.reaction_forces[1].Tz == approx(fresh.reaction_forces[1].Tz)


def test_buckling_factor_without_free_degrees_of_freedom_is_infinite():
    system = SystemElements(EA=EA, EI=EI)
    system.add_element([[0, 0], [L, 0]])
    system.add_support_fixed([1, 2])
    system.q_load(-1, 1)
    system.solve(geometrical_non_linear=True)
    assert system.buckling_factor == float("inf")


@mark.parametrize("k", [500.0, 5.0e3, 5.0e4])
def test_buckling_of_one_element_with_a_spring_is_close_to_the_rotational_support(k):
    # the geometric stiffness of a released end is condensed with the elastic stiffness: one
    # element per member is within 0.5 % (it was several times too high before)
    with_spring = SystemElements(EI=5.0e3, EA=EA)
    with_spring.add_element([[0, 0], [0, 10]], spring={1: k})
    with_spring.add_support_fixed(1)
    with_spring.point_load(2, Fy=-1)
    with_spring.solve(geometrical_non_linear=True)

    with_support = SystemElements(EI=5.0e3, EA=EA)
    with_support.add_element([[0, 0], [0, 10]])
    with_support.add_support_hinged(1)
    with_support.add_support_spring(1, translation=3, k=k)
    with_support.point_load(2, Fy=-1)
    with_support.solve(geometrical_non_linear=True)

    assert with_spring.buckling_factor == approx(with_support.buckling_factor, rel=5e-3)


def test_plastic_moment_of_zero_is_an_error():
    system = SystemElements(EA=EA, EI=EI)
    system.add_element([[0, 0], [L, 0]], mp={1: 0.0})
    system.add_support_fixed(1)
    system.add_support_roll(2)
    system.point_load(2, Fy=-1.0)
    with raises(AssertionError, match="mp = 0"):
        system.solve()


@mark.parametrize("g", [2.0, -2.0])
def test_second_order_self_weight_column_does_not_depend_on_the_drawing_direction(g):
    # the normal force varies along the element: the geometric stiffness uses its mean
    def build(drawn_up):
        system = SystemElements(EI=5.0e3, EA=1.0e6)
        if drawn_up:
            system.add_element([[0, 0], [0, 10]], g=g)
            system.add_support_fixed(1)
        else:
            system.add_element([[0, 10], [0, 0]], g=g)
            system.add_support_fixed(2)
        system.solve(geometrical_non_linear=True, discretize_kwargs={"n": 8})
        return system.buckling_factor

    up, down = build(True), build(False)
    assert up == approx(down, rel=1e-6)


@mark.parametrize("options", [{}, {"load_factor": 2.0}, {"invert_y_loads": False}])
def test_insert_node_copies_the_q_load_with_factor_and_orientation(options):
    def build(split):
        system = SystemElements(EA=EA, EI=EI, **options)
        system.add_element([[0, 0], [4, 0]])
        system.add_support_fixed(1)
        system.add_support_roll(2)
        system.q_load([-4, -8], 1)
        if split:
            system.insert_node(1, factor=0.3)
        system.solve()
        return system

    plain, split = build(False), build(True)
    for node_id in (1, 2):
        assert split.reaction_forces[node_id].Fy == approx(
            plain.reaction_forces[node_id].Fy, rel=1e-5
        )
    assert split.reaction_forces[1].Tz == approx(plain.reaction_forces[1].Tz, rel=1e-5)


def test_insert_node_keeps_a_truss_element_a_truss():
    system = SystemElements(EA=1.0e5)
    system.add_truss_element([[0, 0], [3, 0]])
    system.add_support_hinged(1)
    system.add_support_roll(2)
    system.insert_node(1, factor=0.5)
    assert {element.type for element in system.element_map.values()} == {"truss"}


def test_internal_hinges_stay_consistent_when_hinges_are_added_and_removed():
    system = SystemElements(EA=EA, EI=EI)
    for x in range(5):
        system.add_element([[x * 3, 0], [(x + 1) * 3, 0]])
    for node_id in (4, 2, 5, 3):
        system.add_internal_hinge(node_id)
    assert sorted(h.id for h in system.internal_hinges) == [2, 3, 4, 5]
    index = system._internal_hinge_index
    assert all(system.internal_hinges[i].id == node_id for node_id, i in index.items())

    system.remove_element(5)  # the node 6 is removed, the hinge at node 5 stays
    system.remove_element(4)  # the node 5 is removed with its hinge
    assert sorted(h.id for h in system.internal_hinges) == [2, 3, 4]
    index = system._internal_hinge_index
    assert sorted(index) == [2, 3, 4]
    assert all(system.internal_hinges[i].id == node_id for node_id, i in index.items())


def test_remove_loads_removes_the_perpendicular_q_load():
    def build():
        system = SystemElements(EA=EA, EI=EI)
        system.add_element([[0, 0], [L, 0]])
        system.add_support_fixed(1)
        return system

    system = build()
    system.q_load(q=-5, element_id=1, direction="x", q_perp=-3)
    system.remove_loads()
    system.point_load(2, Fy=-1.0)
    system.solve()

    fresh = build()
    fresh.point_load(2, Fy=-1.0)
    fresh.solve()
    assert system.get_node_displacements(2)["uy"] == approx(
        fresh.get_node_displacements(2)["uy"]
    )


def test_combination_adds_the_loads_of_the_cases_to_the_loads_of_the_system():
    def build():
        system = SystemElements(EA=EA, EI=EI)
        system.add_element([[0, 0], [4, 0]])
        system.add_element([[4, 0], [8, 0]])
        system.add_support_fixed(1)
        system.add_support_roll(3)
        system.point_load(2, Fy=-3.0)
        system.moment_load(2, Tz=2.0)
        system.imposed_displacement(3, uy=-0.01)
        return system

    first = LoadCase("first")
    first.point_load(2, Fy=-1.0)
    first.moment_load(2, Tz=1.0)
    first.imposed_displacement(3, uy=-0.005)
    second = LoadCase("second")
    second.point_load(2, Fy=-1.0)
    second.moment_load(2, Tz=1.0)
    second.imposed_displacement(3, uy=-0.005)
    combination = LoadCombination("c")
    combination.add_load_case(first, 1.0)
    combination.add_load_case(second, 2.0)
    combined = combination.solve(build())["combination"]

    # the system itself counts once: -3 - 1 - 2, 2 + 1 + 2, -0.01 - 0.005 - 0.01
    direct = SystemElements(EA=EA, EI=EI)
    direct.add_element([[0, 0], [4, 0]])
    direct.add_element([[4, 0], [8, 0]])
    direct.add_support_fixed(1)
    direct.add_support_roll(3)
    direct.point_load(2, Fy=-6.0)
    direct.moment_load(2, Tz=5.0)
    direct.imposed_displacement(3, uy=-0.025)
    direct.solve()

    assert combined.loads_point[2] == approx(direct.loads_point[2])
    assert combined.reaction_forces[1].Fy == approx(direct.reaction_forces[1].Fy)
    assert combined.reaction_forces[1].Tz == approx(direct.reaction_forces[1].Tz)
    assert combined.get_node_displacements(2)["uy"] == approx(
        direct.get_node_displacements(2)["uy"]
    )


def test_combination_does_not_show_rounding_noise_in_the_q_load():
    system = SystemElements(EA=EA, EI=EI)
    system.add_element([[0, 0], [0, 4]])
    system.add_support_fixed(1)
    case = LoadCase("q")
    case.q_load(-1.5, 1)
    combination = LoadCombination("c")
    combination.add_load_case(case, 1.0)
    results = combination.solve(system)

    values = [
        *results["combination"].get_element_results(1)["q"],
        *results["combination"].element_map[1].q_load,
        *results["combination"].element_map[1].q_perp_load,
    ]
    assert all(value == 0 or abs(value) > 1e-9 for value in values)
    assert results["combination"].get_element_results(1)["q"] == approx(
        results["q"].get_element_results(1)["q"]
    )


@mark.parametrize("support", ["fixed", "spring"])
def test_supports_added_after_a_solve_are_used(support):
    def add(system):
        if support == "fixed":
            system.add_support_fixed(3)
        else:
            system.add_support_spring(3, translation=2, k=500.0)

    expected = chain()
    add(expected)
    expected.solve()

    system = chain()
    system.solve()
    add(system)
    system.solve()
    assert system.reaction_forces[1].Tz == approx(expected.reaction_forces[1].Tz)
    assert system.reaction_forces[1].Fy == approx(expected.reaction_forces[1].Fy)


def test_element_with_a_hinge_added_after_a_solve():
    def build(solve_first):
        system = SystemElements(EA=EA, EI=EI)
        system.add_element([[0, 0], [4, 0]], spring={2: 0})
        system.add_support_fixed(1)
        if solve_first:
            system.point_load(2, Fy=-1.0)
            system.solve()
        system.add_element([[4, 0], [8, 0]], spring={1: 0})
        system.add_support_hinged(3)
        system.remove_loads()
        system.q_load(-10, [1, 2])
        system.solve()
        return system

    after, before = build(True), build(False)
    assert after.reaction_forces[1].Fy == approx(before.reaction_forces[1].Fy)
    assert after.reaction_forces[1].Tz == approx(before.reaction_forces[1].Tz)
    assert [hinge.id for hinge in after.internal_hinges] == [2]


def test_discretize_keeps_the_direction_of_a_q_load_on_an_inclined_element():
    def build(n=None):
        system = SystemElements(EA=EA, EI=EI)
        system.add_element([[0, 0], [5, 3]])
        system.add_support_fixed(1)
        system.add_support_hinged(2)
        system.q_load(-4, 1, direction="x")
        if n:
            system.discretize(n)
        system.solve()
        return system

    plain, fine = build(), build(4)
    for point in ([0, 0], [5, 3]):
        reaction_plain = plain.reaction_forces[plain.find_node_id(point)]
        reaction_fine = fine.reaction_forces[fine.find_node_id(point)]
        assert reaction_fine.Fx == approx(reaction_plain.Fx, abs=1e-3)
        assert reaction_fine.Fy == approx(reaction_plain.Fy, abs=1e-3)
