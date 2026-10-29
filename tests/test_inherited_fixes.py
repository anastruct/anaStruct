import numpy as np
from pytest import approx, mark, raises

from anastruct import LoadCase, LoadCombination, SystemElements
from anastruct.basic import FEMException
from anastruct.vertex import Vertex

"""
NOTE: These tests cover problems that anaStruct had before: the model of the plastic hinge,
the node ids after remove_element, LoadCase.dead_load, validate() before the first solve, the
precision of the coordinates, zero length elements and infinite springs, and the load
directions with invert_y_loads=False.

The tests are plain test functions (not nested describe/context blocks), so that every
assertion is guaranteed to be executed.
"""

EA = 1.0e6
EI = 5.0e3


# ---------------------------------------------------------------------------
# plastic hinges: a spring in series with the element, secant stiffness
# ---------------------------------------------------------------------------


def two_span_beam(q1, q2, mp):
    # fixed at x = 0 with the plastic moment mp, rollers at x = 4 and x = 8
    system = SystemElements(EA=EA, EI=EI)
    system.add_element([[0, 0], [4, 0]], mp={1: mp})
    system.add_element([[4, 0], [8, 0]])
    system.add_support_fixed(1)
    system.add_support_roll(2)
    system.add_support_roll(3)
    system.q_load(-q1, 1)
    system.q_load(-q2, 2)
    system.solve()
    return system


@mark.parametrize("q1, q2", [(1.0, 1.0), (3.0, 1.0)])
def test_plastic_hinge_at_a_fixed_end_redistributes_the_moments_exactly(q1, q2):
    # a hinge with the moment mp at the fixed end: the three-moment equation with the end
    # moment M_A = -mp gives M_B = (-(q1 L1**3 + q2 L2**3) / 4 + mp L1) / (2 (L1 + L2))
    mp = 1.0
    system = two_span_beam(q1, q2, mp)

    expected = (-(q1 * 4**3 + q2 * 4**3) / 4 + mp * 4) / 16
    assert abs(system.reaction_forces[1].Tz) == approx(mp, rel=3e-3)
    assert system.element_map[1].bending_moment[-1] == approx(-expected, rel=2e-3)


def test_elastic_result_when_the_plastic_moment_is_not_reached():
    system = two_span_beam(2.0, 5.0, 100.0)
    elastic = SystemElements(EA=EA, EI=EI)
    elastic.add_element([[0, 0], [4, 0]])
    elastic.add_element([[4, 0], [8, 0]])
    elastic.add_support_fixed(1)
    elastic.add_support_roll(2)
    elastic.add_support_roll(3)
    elastic.q_load(-2.0, 1)
    elastic.q_load(-5.0, 2)
    elastic.solve()
    assert system.reaction_forces[1].Tz == approx(elastic.reaction_forces[1].Tz)
    assert system.reaction_forces[2].Fy == approx(elastic.reaction_forces[2].Fy)


def test_rotation_of_a_propped_cantilever_with_a_plastic_hinge():
    # the hinge at the fixed end leaves a simply supported beam with the end moment mp:
    # the rotation at the roller is q L**3 / (24 EI) - mp L / (6 EI)
    length, q, mp = 4.0, 3.0, 1.0
    system = SystemElements(EA=EA, EI=EI)
    system.add_element([[0, 0], [length, 0]], mp={1: mp})
    system.add_support_fixed(1)
    system.add_support_roll(2)
    system.q_load(-q, 1)
    system.solve()

    expected = q * length**3 / (24 * EI) - mp * length / (6 * EI)
    assert abs(system.get_node_displacements(2)["phi_z"]) == approx(expected, rel=2e-3)


def test_fixed_fixed_beam_with_plastic_hinges_at_both_ends():
    # M = mp at both ends, so the moment at mid-span is q L**2 / 8 - mp (statics)
    length, q, mp = 6.0, 10.0, 20.0
    system = SystemElements(EA=EA, EI=EI)
    system.add_element([[0, 0], [length / 2, 0]], mp={1: mp})
    system.add_element([[length / 2, 0], [length, 0]], mp={2: mp})
    system.add_support_fixed([1, 3])
    system.q_load(-q, [1, 2])
    system.solve()

    assert abs(system.reaction_forces[1].Tz) == approx(mp, rel=5e-3)
    assert abs(system.reaction_forces[3].Tz) == approx(mp, rel=5e-3)
    assert abs(system.element_map[1].bending_moment[-1]) == approx(
        q * length**2 / 8 - mp, rel=5e-3
    )


def test_plastic_hinge_with_a_spring_of_the_user_acts_in_series():
    # a plastic hinge and a rotational spring at the same end: the end carries mp
    system = SystemElements(EA=EA, EI=EI)
    system.add_element([[0, 0], [4, 0]], mp={1: 1.0}, spring={1: 800.0})
    system.add_support_fixed(1)
    system.add_support_roll(2)
    system.q_load(-3.0, 1)
    system.solve()
    assert abs(system.reaction_forces[1].Tz) == approx(1.0, rel=5e-3)


# ---------------------------------------------------------------------------
# remove_element: gaps in the node ids
# ---------------------------------------------------------------------------


def three_elements():
    system = SystemElements(EA=EA, EI=1.0e4)
    system.add_element([[0, 0], [4, 0]])
    system.add_element([[4, 0], [8, 0]])
    system.add_element([[8, 0], [12, 0]])
    return system


def test_solve_after_removing_the_first_element():
    # the node 1 disappears: the ids 2, 3, 4 remain, and the degrees of freedom follow each other
    system = three_elements()
    system.remove_element(1)
    assert sorted(system.node_map) == [2, 3, 4]
    system.add_support_fixed(2)
    system.add_support_roll(4)
    system.q_load(-10, [2, 3])
    system.solve()

    reference = SystemElements(EA=EA, EI=1.0e4)
    reference.add_element([[4, 0], [8, 0]])
    reference.add_element([[8, 0], [12, 0]])
    reference.add_support_fixed(1)
    reference.add_support_roll(3)
    reference.q_load(-10, [1, 2])
    reference.solve()

    assert system.reaction_forces[2].Fy == approx(reference.reaction_forces[1].Fy)
    assert system.reaction_forces[2].Tz == approx(reference.reaction_forces[1].Tz)
    assert system.get_node_displacements(3)["uy"] == approx(
        reference.get_node_displacements(2)["uy"]
    )


def test_supports_springs_and_loads_with_gaps_in_the_node_ids():
    system = three_elements()
    system.remove_element(1)
    system.add_support_fixed(2)
    system.add_support_spring(4, translation=2, k=2.0e3)
    system.point_load(3, Fy=-5.0)
    system.solve()

    reference = SystemElements(EA=EA, EI=1.0e4)
    reference.add_element([[4, 0], [8, 0]])
    reference.add_element([[8, 0], [12, 0]])
    reference.add_support_fixed(1)
    reference.add_support_spring(3, translation=2, k=2.0e3)
    reference.point_load(2, Fy=-5.0)
    reference.solve()

    assert system.reaction_forces[2].Tz == approx(reference.reaction_forces[1].Tz)
    assert system.get_node_displacements(4)["uy"] == approx(
        reference.get_node_displacements(3)["uy"]
    )


def test_solve_after_removing_a_middle_element_and_adding_one():
    system = three_elements()
    system.remove_element(2)
    system.add_element([[4, 0], [8, 0]])  # a new element between the same nodes
    system.add_support_fixed(1)
    system.add_support_roll(4)
    system.q_load(-10, [1, 3, 4])
    system.solve()

    reference = three_elements()
    reference.add_support_fixed(1)
    reference.add_support_roll(4)
    reference.q_load(-10, [1, 2, 3])
    reference.solve()
    assert system.reaction_forces[1].Tz == approx(reference.reaction_forces[1].Tz)


# ---------------------------------------------------------------------------
# LoadCase.dead_load
# ---------------------------------------------------------------------------


def beam_with_supports(**element):
    system = SystemElements(EA=EA, EI=1.0e4)
    system.add_element([[0, 0], [6, 0]], **element)
    system.add_support_fixed(1)
    system.add_support_roll(2)
    return system


def test_dead_load_of_a_load_case():
    reference = beam_with_supports(g=3.0)
    reference.solve()

    system = beam_with_supports()
    case = LoadCase("self-weight")
    case.dead_load(element_id=1, g=3.0)
    system.apply_load_case(case)
    system.solve()

    assert system.reaction_forces[2].Fy == approx(reference.reaction_forces[2].Fy)
    assert system.reaction_forces[1].Tz == approx(reference.reaction_forces[1].Tz)


def test_dead_load_method_uses_the_load_factor():
    system = beam_with_supports()
    system.load_factor = 1.5
    system.dead_load(1, 2.0)
    system.solve()

    reference = beam_with_supports(g=3.0)
    reference.solve()
    assert system.reaction_forces[2].Fy == approx(reference.reaction_forces[2].Fy)


def test_dead_load_in_a_load_combination():
    wind = LoadCase("wind")
    wind.point_load(2, Fy=-1.0)
    weight = LoadCase("weight")
    weight.dead_load(1, 1.0)
    combination = LoadCombination("c")
    combination.add_load_case(wind, 2.0)
    combination.add_load_case(weight, 3.0)
    combined = combination.solve(beam_with_supports(g=0.5))["combination"]

    direct = beam_with_supports(g=0.5 + 3.0 * 1.0)
    direct.point_load(2, Fy=-2.0)
    direct.solve()
    assert combined.reaction_forces[1].Tz == approx(direct.reaction_forces[1].Tz)
    assert combined.reaction_forces[2].Fy == approx(direct.reaction_forces[2].Fy)


def test_q_load_after_removing_the_dead_load():
    # the q-loads of an element are processed also after remove_loads(dead_load=True)
    system = beam_with_supports(g=2.0)
    system.remove_loads(dead_load=True)
    system.q_load(-3.0, 1)
    system.solve()
    assert abs(system.reaction_forces[1].Tz) == approx(3.0 * 6**2 / 8)


# ---------------------------------------------------------------------------
# validate() before the first solve, and errors for unusual input
# ---------------------------------------------------------------------------


def test_validate_before_the_first_solve():
    supported = beam_with_supports()
    supported.q_load(-1.0, 1)
    assert supported.validate() is True
    supported.solve()  # and the solve still works afterwards
    assert abs(supported.reaction_forces[1].Tz) == approx(1.0 * 6**2 / 8)

    unsupported = SystemElements(EA=EA, EI=EI)
    unsupported.add_element([[0, 0], [4, 0]])
    unsupported.point_load(2, Fy=-1.0)
    assert unsupported.validate() is False
    with raises(FEMException) as error:
        unsupported.solve()
    assert error.value.type == "StabilityError"


def test_zero_length_element_is_an_error():
    system = SystemElements()
    with raises(FEMException) as error:
        system.add_element([[1, 1], [1, 1]])
    assert "two different locations" in error.value.message
    assert system.element_map == {}  # nothing was added


@mark.parametrize("k", [float("inf"), -1.0, float("nan")])
def test_spring_support_needs_a_finite_stiffness(k):
    system = SystemElements()
    system.add_element([[0, 0], [1, 0]])
    with raises(FEMException) as error:
        system.add_support_spring(1, translation=2, k=k)
    assert "finite" in error.value.message


# ---------------------------------------------------------------------------
# coordinates in double precision
# ---------------------------------------------------------------------------


def test_coordinates_are_stored_in_double_precision():
    # (rounded to 12 significant digits, so that computed coordinates match)
    vertex = Vertex(1000.123456789, 0.1)
    assert vertex.x == approx(1000.123456789, rel=1e-11)
    assert vertex.y == 0.1  # float32 would give 0.10000000149011612

    system = SystemElements()
    system.add_element([[0, 0], [1000.123456789, 0]])
    assert system.element_map[1].l == approx(1000.123456789, rel=1e-11)


def test_computed_and_given_coordinates_are_the_same_node():
    # 0.1 + 0.2 is 0.30000000000000004: still the node at 0.3
    system = SystemElements()
    system.add_element([[0, 0], [0.1 + 0.2, 0]])
    system.add_element([[0.3, 0], [0.6, 0]])
    assert len(system.node_map) == 3
    assert Vertex(0.1 + 0.2, -0.0) == Vertex(0.3, 0)
    assert str(Vertex(-0.0, 1)) == "Vertex(0.0, 1.0)"


# ---------------------------------------------------------------------------
# invert_y_loads=False: the x-direction does not change
# ---------------------------------------------------------------------------


def cantilever_displacement(invert_y_loads, apply_load):
    system = SystemElements(EA=EA, EI=1.0e4, invert_y_loads=invert_y_loads)
    system.add_element([[0, 0], [4, 0]])
    system.add_support_fixed(1)
    apply_load(system)
    system.solve()
    displacement = system.get_node_displacements(2)
    return displacement["ux"], displacement["uy"]


@mark.parametrize("direction", ["x", "parallel"])
def test_q_load_in_the_x_direction_is_the_same_in_both_modes(direction):
    default = cantilever_displacement(
        True, lambda s: s.q_load(1.0, 1, direction=direction)
    )
    inverted = cantilever_displacement(
        False, lambda s: s.q_load(1.0, 1, direction=direction)
    )
    assert default[0] > 0
    assert inverted == approx(default)


@mark.parametrize("direction", ["y", "element", "perpendicular"])
def test_q_load_in_the_y_direction_is_reversed_with_invert_y_loads_false(direction):
    default = cantilever_displacement(
        True, lambda s: s.q_load(1.0, 1, direction=direction)
    )
    inverted = cantilever_displacement(
        False, lambda s: s.q_load(1.0, 1, direction=direction)
    )
    assert default[1] > 0
    assert inverted[1] == approx(-default[1])


def test_q_perp_of_an_x_load_is_a_y_load():
    default = cantilever_displacement(
        True, lambda s: s.q_load(1.0, 1, direction="x", q_perp=1.0)
    )
    inverted = cantilever_displacement(
        False, lambda s: s.q_load(1.0, 1, direction="x", q_perp=1.0)
    )
    assert inverted[0] == approx(default[0])
    assert inverted[1] == approx(-default[1])


def test_point_load_rotation_in_both_modes():
    # clockwise rotation: with the default (y up) Fy = 1 (up) turns to +x and Fx = 1 to -y; with
    # invert_y_loads=False the y-axis points down: Fy = 1 (down) turns to -x
    rotated_fy = {
        mode: cantilever_displacement(
            mode, lambda s: s.point_load(2, Fy=1.0, rotation=90.0)
        )
        for mode in (True, False)
    }
    assert rotated_fy[True][0] > 0
    assert rotated_fy[False][0] == approx(-rotated_fy[True][0])
    for mode in (True, False):
        rotated_fx = cantilever_displacement(
            mode, lambda s: s.point_load(2, Fx=1.0, rotation=90.0)
        )
        assert rotated_fx[1] < 0  # down
        assert np.isclose(rotated_fx[0], 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# findings of the review: ids after a gap, stale supports, order of load cases
# ---------------------------------------------------------------------------


def test_new_element_after_a_gap_gets_new_node_ids():
    system = three_elements()
    system.remove_element(1)  # the ids 2, 3, 4 remain
    system.add_element([[12, 0], [16, 0]])
    assert sorted(system.node_map) == [2, 3, 4, 5]
    assert system.element_map[4].node_id1 == 4
    assert system.element_map[4].node_id2 == 5

    system.add_support_fixed(2)
    system.add_support_roll(5)
    system.q_load(-10, [2, 3, 4])
    system.solve()

    reference = SystemElements(EA=EA, EI=1.0e4)
    for i in range(3):
        reference.add_element([[4 + 4 * i, 0], [8 + 4 * i, 0]])
    reference.add_support_fixed(1)
    reference.add_support_roll(4)
    reference.q_load(-10, [1, 2, 3])
    reference.solve()
    assert system.reaction_forces[2].Tz == approx(reference.reaction_forces[1].Tz)


def test_solve_remove_element_solve_again():
    # the supports of the removed node are removed also after a solve (new node objects)
    system = three_elements()
    system.add_support_fixed(1)
    system.add_support_roll(4)
    system.q_load(-10, [1, 2, 3])
    system.solve()
    system.remove_element(1)
    system.add_support_fixed(2)
    system.q_load(-10, [2, 3])
    system.solve()

    reference = SystemElements(EA=EA, EI=1.0e4)
    reference.add_element([[4, 0], [8, 0]])
    reference.add_element([[8, 0], [12, 0]])
    reference.add_support_fixed(1)
    reference.add_support_roll(3)
    reference.q_load(-10, [1, 2])
    reference.solve()
    assert system.reaction_forces[2].Tz == approx(reference.reaction_forces[1].Tz)


def test_spring_support_and_imposed_displacement_of_a_removed_node_are_removed():
    system = three_elements()
    system.add_support_spring(1, translation=2, k=100.0)
    system.add_support_fixed(1)
    system.imposed_displacement(1, uy=0.01)
    system.remove_element(1)
    system.add_support_fixed(2)
    system.add_support_roll(4)
    system.q_load(-10, [2, 3])
    system.solve()
    assert sorted(system.reaction_forces) == [2, 4]


def test_dead_load_case_after_a_q_load_case_in_a_combination():
    q_case = LoadCase("q")
    q_case.q_load(-2.0, 1)
    dead_case = LoadCase("g")
    dead_case.dead_load(1, 1.5)
    combination = LoadCombination("c")
    combination.add_load_case(q_case, 1.0)
    combination.add_load_case(dead_case, 2.0)
    combined = combination.solve(beam_with_supports())["combination"]

    direct = beam_with_supports(g=3.0)
    direct.q_load(-2.0, 1)
    direct.solve()
    assert combined.reaction_forces[1].Tz == approx(direct.reaction_forces[1].Tz)
    assert combined.reaction_forces[2].Fy == approx(direct.reaction_forces[2].Fy)


@mark.parametrize("mp", [5.99, 5.95])
def test_plastic_moment_just_below_the_elastic_moment_converges(mp):
    # the elastic fixed-end moment is q L**2 / 8 = 6
    system = SystemElements(EA=EA, EI=EI)
    system.add_element([[0, 0], [4, 0]], mp={1: mp})
    system.add_support_fixed(1)
    system.add_support_roll(2)
    system.q_load(-3.0, 1)
    system.solve()
    assert abs(system.reaction_forces[1].Tz) == approx(mp, rel=2e-3)


def test_failed_zero_length_element_keeps_the_previous_point():
    system = SystemElements()
    system.add_element([[0, 0], [1, 0]])
    with raises(FEMException):
        system.add_element([1, 0])
    system.add_element([2, 0])  # continues from the previous point (1, 0)
    assert system.element_map[2].vertex_1 == Vertex(1, 0)


@mark.parametrize("rotation, direction", [(0, "x"), (90, "y")])
@mark.parametrize("perpendicular", [False, True])
def test_q_load_rotation_is_the_same_as_the_direction_with_invert_y_loads_false(
    rotation, direction, perpendicular
):
    def displacement(**load):
        system = SystemElements(EA=EA, EI=1.0e4, invert_y_loads=False)
        system.add_element([[0, 0], [4, 3]])
        system.add_support_fixed(1)
        system.q_load(
            q=0.0 if perpendicular else 1.0,
            element_id=1,
            **load,
            **({"q_perp": 1.0} if perpendicular else {}),
        )
        system.solve()
        return system.get_node_displacements(2)

    by_rotation = displacement(rotation=rotation)
    by_direction = displacement(direction=direction)
    assert by_rotation["ux"] == approx(by_direction["ux"])
    assert by_rotation["uy"] == approx(by_direction["uy"])


def test_plastic_frame_beyond_the_collapse_load_is_a_mechanism():
    mps = [[14.67, 17.98], [10.95, 16.87], [12.48, 13.77], [24.51, 14.17]]
    points = [[0, 0], [0, 4], [3, 4], [6, 4], [6, 0]]
    system = SystemElements(EA=1.0e8, EI=EI)
    for i, (start, end) in enumerate(zip(points[:-1], points[1:])):
        system.add_element([start, end], mp={1: mps[i][0], 2: mps[i][1]})
    system.add_support_fixed([1, 5])
    system.point_load(2, Fx=1.526)
    system.point_load(3, Fy=-9.892)
    system.q_load(-4.731, [2, 3])
    with raises(FEMException, match="mechanism"):
        system.solve()
