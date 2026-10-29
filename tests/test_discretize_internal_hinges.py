import itertools

import numpy as np

from pytest import approx, mark, raises

from anastruct import SystemElements
from anastruct.basic import FEMException

"""
NOTE: These tests cover the detection of internal hinges (`check_internal_hinges`) and
`SystemElements.discretize()`. The hinge status of a node follows from the springs given
by the user and does not depend on the order in which the elements were added; a
discretized system gives the same results as the original system.

The tests are plain test functions (not nested describe/context blocks), so that every
assertion is guaranteed to be executed.
"""

EA = 1.0e9
EI = 1.0e4
L = 3.0


def reaction(system, point):
    """Reaction forces at the node at the given location (the ids change in discretize)"""
    return system.reaction_forces[system.find_node_id(point)]


# ---------------------------------------------------------------------------
# internal hinges
# ---------------------------------------------------------------------------


def tee(order, moment=10.0):
    """Beam with a column and a beam whose start is a hinge, meeting in one node.

    The left beam and the column are rigidly connected, so they share the rotation of
    the node and carry the moment; the right beam is released.
    """
    system = SystemElements(EA=EA, EI=EI)
    elements = {
        "left": lambda: system.add_element([[0, 0], [L, 0]]),
        "right": lambda: system.add_element([[L, 0], [2 * L, 0]], spring={1: 0}),
        "column": lambda: system.add_element([[L, 0], [L, -L]]),
    }
    for name in order:
        elements[name]()
    junction = system.find_node_id([L, 0])
    system.add_support_fixed(
        [system.find_node_id([0, 0]), system.find_node_id([L, -L])]
    )
    system.add_support_hinged(system.find_node_id([2 * L, 0]))
    system.moment_load(junction, Tz=moment)
    system.solve()
    return system, junction


@mark.parametrize("order", list(itertools.permutations(["left", "right", "column"])))
def test_junction_with_two_rigid_elements_is_not_a_hinge_in_any_order(order):
    # the rotation of the junction is shared by the left beam and the column, each
    # fixed at the far end: phi = M / (4 * EI / L + 4 * EI / L)
    moment = 10.0
    system, junction = tee(order, moment)

    assert system.internal_hinges == []
    assert not system.node_map[junction].hinge
    assert abs(system.get_node_displacements(junction)["phi_z"]) == approx(
        moment / (8 * EI / L), rel=1e-4  # EA is finite
    )


def test_internal_hinge_is_detected_when_all_elements_are_released():
    system = SystemElements(EA=EA, EI=EI)
    system.add_element([[0, 0], [L, 0]], spring={2: 0})
    system.add_element([[L, 0], [2 * L, 0]], spring={1: 0})
    junction = system.find_node_id([L, 0])
    assert [hinge.id for hinge in system.internal_hinges] == [junction]
    assert system.node_map[junction].hinge

    # a rigid element takes the node out of the hinges, removing it again brings it back
    column = system.add_element([[L, 0], [L, -L]])
    assert system.internal_hinges == []
    assert not system.node_map[junction].hinge
    system.remove_element(column)
    assert [hinge.id for hinge in system.internal_hinges] == [junction]
    assert system.node_map[junction].hinge


def test_a_node_with_one_rigid_element_is_solved_with_its_own_rotation():
    # rigid cantilever + a released beam in one node: the node rotation is a normal
    # degree of freedom, so a moment load is carried by the rigid element
    # (it used to be dropped silently)
    moment = 10.0
    system = SystemElements(EA=EA, EI=EI)
    system.add_element([[0, 0], [L, 0]])
    system.add_element([[L, 0], [2 * L, 0]], spring={1: 0})
    system.add_support_fixed(1)
    system.add_support_hinged(3)
    system.moment_load(2, Tz=moment)
    system.solve()

    assert abs(reaction(system, [0, 0]).Tz) == approx(moment)
    assert reaction(system, [2 * L, 0]).Fy == approx(0.0, abs=1e-9)


def test_a_fixed_support_before_a_released_element_stays_fixed():
    # cantilever (support at its right end) and a beam that is hinged to the support:
    # q on the cantilever gives M = q * L**2 / 2 in the support
    q = 10.0
    system = SystemElements(EA=EA, EI=EI)
    system.add_element([[0, 0], [L, 0]])
    system.add_support_fixed(2)
    system.add_element([[L, 0], [2 * L, 0]], spring={1: 0})
    system.add_support_hinged(3)
    system.q_load(-q, 1)
    system.solve()

    assert abs(reaction(system, [L, 0]).Tz) == approx(q * L**2 / 2)


def test_hinges_stay_known_after_solving_more_than_once():
    system = SystemElements(EA=EA, EI=EI)
    system.add_element([[0, 0], [L, 0]], spring={2: 0})
    system.add_element([[L, 0], [2 * L, 0]], spring={1: 0})
    system.add_support_fixed(1)
    system.add_support_roll(3)
    junction = system.find_node_id([L, 0])
    for _ in range(2):
        system.remove_loads()
        system.q_load(-10, [1, 2])
        system.solve()
        assert [hinge.id for hinge in system.internal_hinges] == [junction]
        assert system.node_map[junction].hinge
        assert len(system.node_map[junction].elements) == 2

    # elements can be removed after solving; the remaining element is still released
    # at the node, so it stays a hinge
    system.remove_element(2)
    assert [hinge.id for hinge in system.internal_hinges] == [junction]
    system.remove_element(1)
    assert system.internal_hinges == []


def test_a_translational_spring_support_is_allowed_at_an_internal_hinge():
    system = SystemElements(EA=EA, EI=EI)
    system.add_element([[0, 0], [L, 0]])
    system.add_element([[L, 0], [2 * L, 0]])
    system.add_internal_hinge(2)
    system.add_support_spring(2, translation=2, k=500.0)  # does not restrain rotation
    with raises(FEMException):
        system.add_support_spring(2, translation=3, k=500.0)


def test_springs_given_by_the_user_are_not_modified():
    spring = {1: 0}
    system = SystemElements(EA=EA, EI=EI)
    system.add_element([[0, 0], [L, 0]])
    system.add_element([[L, 0], [2 * L, 0]], spring=spring)
    system.add_element([[2 * L, 0], [3 * L, 0]], spring=spring)

    # the rigid element at the hinge node is released through the node, not by a spring
    assert system.element_map[1].springs == {}
    assert system.element_map[2].springs == {1: 0}
    assert system.element_map[3].springs == {1: 0}
    assert spring == {1: 0}
    assert system.element_map[2].springs is not system.element_map[3].springs


def test_a_spring_dictionary_of_the_caller_is_not_changed_when_the_element_is_reversed():
    spring = {1: 500.0}
    system = SystemElements(EA=EA, EI=EI)
    system.add_element([[L, 0], [0, 0]], spring=spring)  # drawn right to left

    assert spring == {1: 500.0}
    assert system.element_map[1].springs == {2: 500.0}


def test_explicit_internal_hinge_is_a_hinge_immediately():
    system = SystemElements(EA=EA, EI=EI)
    system.add_element([[0, 0], [L, 0]])
    system.add_element([[L, 0], [2 * L, 0]])
    system.add_internal_hinge(2)

    assert system.node_map[2].hinge
    assert system.internal_hinges == [system.node_map[2]]
    with raises(FEMException):
        system.add_support_fixed(2)


def test_explicit_internal_hinge_with_a_third_rigid_element_stays_a_hinge():
    # a declared hinge releases all the elements, also when three are connected (a low EA, so
    # that the load is not carried by the normal force of the column alone)
    system = SystemElements(EA=1.0e3, EI=EI)
    system.add_element([[0, 0], [L, 0]])
    system.add_element([[L, 0], [2 * L, 0]])
    system.add_element([[L, 0], [L, -L]])
    system.add_internal_hinge(2)
    system.add_support_fixed([1, 3, 4])
    system.point_load(2, Fy=-10.0)
    assert system.node_map[2].hinge
    system.solve()

    # (the hinge flag is reset by the post-processing and set again by every solve)
    assert [hinge.id for hinge in system.internal_hinges] == [2]
    # the moment is zero at the hinge end of each of the three released elements; the two
    # beams carry the load by shear into their supports (the column only by a normal force)
    for element in system.element_map.values():
        moments = element.bending_moment
        at_hinge = moments[0] if element.node_id1 == 2 else moments[-1]
        assert at_hinge == approx(0.0, abs=1e-6)
    beams = [system.element_map[1], system.element_map[2]]
    assert all(np.max(np.abs(beam.bending_moment)) > 0.05 for beam in beams)
    assert np.max(np.abs(system.element_map[3].bending_moment)) < 1e-3


def test_moment_load_at_an_internal_hinge_is_an_error():
    # no element transfers a moment in a hinge: the load would vanish silently
    system = SystemElements(EA=EA, EI=EI)
    system.add_element([[0, 0], [L, 0]])
    system.add_element([[L, 0], [2 * L, 0]])
    system.add_support_fixed([1, 3])
    system.add_internal_hinge(2)
    system.moment_load(2, Tz=10.0)
    with raises(FEMException) as error:
        system.solve()
    assert "internal hinge" in error.value.message


def test_moment_load_at_a_node_with_only_truss_elements_is_an_error():
    system = SystemElements(EA=EA)
    system.add_truss_element([[0, 0], [L, 0]])
    system.add_truss_element([[L, 0], [2 * L, 0]])
    system.add_support_hinged([1, 3])
    system.moment_load(2, Tz=10.0)
    with raises(FEMException) as error:
        system.solve()
    assert "truss" in error.value.message


def test_insert_node_keeps_the_internal_hinge_of_a_released_node():
    def build(split):
        system = SystemElements(EA=EA, EI=EI)
        system.add_element([[0, 0], [2 * L, 0]])
        system.add_element([[2 * L, 0], [4 * L, 0]], spring={1: 0})
        system.add_support_fixed(1)
        system.add_support_roll(3)
        if split:
            system.insert_node(1, factor=0.5)
        system.q_load(-10, [1, 2] if not split else [2, 3, 4])
        system.solve()
        return system

    plain, split = build(False), build(True)
    assert reaction(split, [0, 0]).Tz == approx(reaction(plain, [0, 0]).Tz, rel=1e-6)
    assert reaction(split, [4 * L, 0]).Fy == approx(
        reaction(plain, [4 * L, 0]).Fy, rel=1e-6
    )


# ---------------------------------------------------------------------------
# discretize
# ---------------------------------------------------------------------------


def test_discretize_a_portal_frame_gives_the_same_results():
    # a model that is not a simple chain of elements (node ids do not follow the
    # element ids)
    def build(n=None):
        system = SystemElements(EA=EA, EI=EI)
        system.add_element([[0, 0], [0, 4]])
        system.add_element([[0, 4], [6, 4]])
        system.add_element([[6, 4], [6, 0]])
        system.add_support_fixed(1)
        system.add_support_hinged(4)
        system.q_load(-10, 2)
        system.point_load(2, Fx=5)
        if n:
            system.discretize(n)
        system.solve()
        return system

    plain, fine = build(), build(4)
    for point in ([0, 0], [6, 0]):
        for component in ("Fx", "Fy", "Tz"):
            assert getattr(reaction(fine, point), component) == approx(
                getattr(reaction(plain, point), component), abs=1e-4
            )
    assert len(fine.element_map) == 12


def test_discretize_with_a_trapezoidal_q_load_dead_load_and_a_spring():
    def build(n=None):
        system = SystemElements(EA=EA, EI=EI)
        system.add_element([[0, 0], [0, 4]])
        system.add_element([[0, 4], [6, 4]], spring={2: 800.0}, g=2.0)
        system.add_element([[6, 4], [6, 0]])
        system.add_support_fixed([1, 4])
        system.q_load([-4, -12], 2)
        if n:
            system.discretize(n)
        system.solve()
        return system

    plain, fine = build(), build(5)
    for point in ([0, 0], [6, 0]):
        for component in ("Fx", "Fy", "Tz"):
            assert getattr(reaction(fine, point), component) == approx(
                getattr(reaction(plain, point), component), abs=1e-3
            )


def test_discretize_gives_the_releases_to_the_outer_sub_elements_only():
    system = SystemElements(EA=EA, EI=EI)
    system.add_element([[0, 0], [L, 0]], spring={1: 0, 2: 500.0}, mp={1: 30.0, 2: 40.0})
    system.discretize(4)

    springs = [system.element_map[i].springs for i in sorted(system.element_map)]
    assert springs == [{1: 0}, {}, {}, {2: 500.0}]
    assert len({id(spring) for spring in springs}) == 4  # no shared dictionaries
    assert [system.non_linear_elements.get(i) for i in sorted(system.element_map)] == [
        {1: 30.0},
        None,
        None,
        {2: 40.0},
    ]


def test_discretize_keeps_the_hinge_of_the_original_element_only():
    def build(n=None):
        system = SystemElements(EA=EA, EI=EI)
        system.add_element([[0, 0], [0, 4]])
        system.add_element([[0, 4], [6, 4]], spring={1: 0})
        system.add_element([[6, 4], [6, 0]])
        system.add_support_fixed([1, 4])
        system.q_load(-10, 2)
        if n:
            system.discretize(n)
        system.solve()
        return system

    plain, fine = build(), build(4)
    assert [
        element.springs for element in fine.element_map.values() if element.springs
    ] == [{1: 0}]
    assert reaction(fine, [0, 0]).Tz == approx(reaction(plain, [0, 0]).Tz, abs=1e-3)
    assert reaction(fine, [6, 0]).Tz == approx(reaction(plain, [6, 0]).Tz, abs=1e-3)


def test_discretize_does_not_divide_truss_elements():
    # divided truss elements would leave mechanisms at the new nodes: the apex of
    # this two-bar truss moves N * l / EA / sin(45 deg) = 0.02828
    def build(n=None):
        system = SystemElements(EA=1000.0)
        system.add_truss_element([[0, 0], [2, 2]])
        system.add_truss_element([[2, 2], [4, 0]])
        system.add_support_hinged(1)
        system.add_support_hinged(3)
        system.point_load(2, Fy=-10)
        if n:
            system.discretize(n)
        system.solve()
        return system

    fine = build(3)
    assert len(fine.element_map) == 2
    apex = fine.find_node_id([2, 2])
    assert abs(fine.get_node_displacements(apex)["uy"]) == approx(0.028284, rel=1e-4)
    assert abs(build().get_node_displacements(2)["uy"]) == approx(0.028284, rel=1e-4)


def test_discretize_with_a_spring_support_given_by_a_negative_node_id():
    # the negative id refers to the last node at the moment the support is added
    def build(n=None):
        system = SystemElements(EA=EA, EI=EI)
        system.add_element([[0, 0], [4, 0]])
        system.add_support_spring(-1, 2, 800.0)
        system.add_element([[4, 0], [8, 0]])
        system.add_support_fixed(1)
        system.q_load(-10, [1, 2])
        if n:
            system.discretize(n)
        system.solve()
        return system

    plain, fine = build(), build(2)
    assert reaction(fine, [0, 0]).Tz == approx(reaction(plain, [0, 0]).Tz, abs=1e-3)
    assert reaction(fine, [0, 0]).Fy == approx(reaction(plain, [0, 0]).Fy, abs=1e-3)
    assert fine.supports_spring_y[0][0].vertex.x == approx(4.0)


def test_geometrical_non_linear_solve_keeps_the_plotter_of_the_user_system():
    system = SystemElements(EI=5e3, EA=1e5)
    system.add_element([0, 10])
    system.add_support_hinged(1)
    system.add_support_roll(2, 1)
    system.point_load(2, Fy=-1)
    system.solve(geometrical_non_linear=True, discretize_kwargs={"n": 5})

    assert len(system.element_map) == 1
    assert system.plotter.system is system
    assert system.plot_values.system is system


def test_discretize_keeps_the_truss_elements():
    def build(n=None):
        system = SystemElements(EA=1e5, EI=EI)
        system.add_truss_element([[0, 0], [3, 0]])
        system.add_truss_element([[3, 0], [3, 4]])
        system.add_truss_element([[0, 0], [3, 4]])
        system.add_support_hinged(1)
        system.add_support_roll(2)
        system.point_load(3, Fx=10, Fy=-20)
        if n:
            system.discretize(n)
        system.solve()
        return system

    plain, fine = build(), build(3)
    assert {element.type for element in fine.element_map.values()} == {"truss"}
    for point in ([0, 0], [3, 0]):
        for component in ("Fx", "Fy"):
            # the bars have no bending stiffness (EI = 1e-14), so the transverse
            # degrees of freedom of the inner nodes are almost singular
            assert getattr(reaction(fine, point), component) == approx(
                getattr(reaction(plain, point), component), abs=1e-4
            )


def test_discretize_keeps_an_inclined_roller_support():
    def build(n=None):
        system = SystemElements(EA=EA, EI=EI)
        system.add_element([[0, 0], [L, 0]])
        system.add_element([[L, 0], [2 * L, 0]])
        system.add_support_hinged(1)
        system.add_support_roll(3, angle=30)
        system.q_load(-10, [1, 2])
        if n:
            system.discretize(n)
        system.solve()
        return system

    plain, fine = build(), build(3)
    assert reaction(fine, [2 * L, 0]).Fx == approx(
        reaction(plain, [2 * L, 0]).Fx, abs=1e-4
    )
    assert reaction(fine, [2 * L, 0]).Fy == approx(
        reaction(plain, [2 * L, 0]).Fy, abs=1e-4
    )


def test_discretize_keeps_an_explicit_internal_hinge_and_a_spring_support():
    def build(n=None):
        system = SystemElements(EA=EA, EI=EI)
        system.add_element([[0, 0], [L, 0]])
        system.add_element([[L, 0], [2 * L, 0]])
        system.add_support_fixed(1)
        system.add_support_spring(3, translation=2, k=500.0)
        system.add_internal_hinge(2)
        system.point_load(2, Fy=-100)
        if n:
            system.discretize(n)
        system.solve()
        return system

    plain, fine = build(), build(4)
    assert len(fine.internal_hinges) == 1
    assert reaction(fine, [0, 0]).Fy == approx(reaction(plain, [0, 0]).Fy, abs=1e-4)
    assert reaction(fine, [0, 0]).Tz == approx(reaction(plain, [0, 0]).Tz, abs=1e-4)
    assert fine.get_node_displacements(fine.find_node_id([2 * L, 0]))["uy"] == approx(
        plain.get_node_displacements(3)["uy"], abs=1e-9
    )


def test_discretize_keeps_the_plotter_and_post_processor_bound_to_the_system():
    system = SystemElements(EA=EA, EI=EI)
    system.add_element([[0, 0], [L, 0]])
    system.add_support_fixed(1)
    system.q_load(-10, 1)
    system.point_load(2, Fy=-5)
    max_q = system.plotter.max_q
    system.discretize(3)

    assert system.plotter.system is system
    assert system.plot_values.system is system
    assert system.post_processor.system is system
    assert system.plotter.max_q == max_q
    system.solve()
    assert system.get_element_results(1) is not None
