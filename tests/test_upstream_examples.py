import numpy as np
from pytest import approx

from anastruct import SystemElements

"""
NOTE: The upstream tests of these examples (tests/test_e2e.py) are nested describe/context/it
blocks, which pytest-describe does not run in this environment. These plain tests execute
them. The expectations for the hinge in example 4 and the rotational spring in example 7
follow from the condensation of hinges and springs (the older values, 3.125e-3 and
0.2356/0.0967/0.0643, used 4EI/l for an element with a hinge, and a wrong series of the
spring), and are verified independently here.
"""


def test_example_4_column_pinned_at_both_ends():
    # a moment M at the top of a column that is pinned at both ends (the beam, with a hinge
    # at the column, only gives a horizontal support): the rotation is M * l / (3 EI)
    system = SystemElements()
    system.add_element(location=[[0, 0], [5, 0]], EA=5e9, EI=8000, spring={2: 0})
    system.add_element(location=[[5, 0], [5, 5]], EA=5e9, EI=4000)
    system.moment_load(Tz=10, node_id=3)
    system.add_support_hinged(node_id=1)
    system.add_support_hinged(node_id=3)
    solution = system.solve()

    assert solution[8] == approx(10 * 5 / (3 * 4000), rel=1e-6)
    assert solution[5] == approx(
        -10 * 5 / (6 * 4000), rel=1e-6
    )  # the column end at the hinge


def test_example_7_rotational_spring():
    # beam elements with the degrees of freedom (v up, theta counter-clockwise): the free
    # degrees of freedom are v2, theta2, the rotation r of the end of the beam behind the
    # spring, and theta3; P = 100 down at node 2, spring k = 100 between theta2 and r
    ei, length, load, k = 5.0e3, 3.5, 100.0, 100.0

    def beam():
        return (
            ei
            / length**3
            * np.array(
                [
                    [12, 6 * length, -12, 6 * length],
                    [6 * length, 4 * length**2, -6 * length, 2 * length**2],
                    [-12, -6 * length, 12, -6 * length],
                    [6 * length, 2 * length**2, -6 * length, 4 * length**2],
                ]
            )
        )

    stiffness = np.zeros((4, 4))
    stiffness[0:2, 0:2] += beam()[2:4, 2:4]  # the end of the left element is node 2
    right = beam()  # (v2, r, v3, theta3), v3 is restrained
    index = [0, 2, None, 3]
    for a in range(4):
        for b in range(4):
            if index[a] is not None and index[b] is not None:
                stiffness[index[a], index[b]] += right[a, b]
    stiffness[1, 1] += k
    stiffness[2, 2] += k
    stiffness[1, 2] -= k
    stiffness[2, 1] -= k
    v2, theta2, _, theta3 = np.linalg.solve(stiffness, np.array([-load, 0, 0, 0]))

    system = SystemElements()
    system.add_element([[0.0, 0.0], [3.5, 0.0]])
    system.add_element([7, 0], spring={1: 100})
    system.add_support_fixed([1])
    system.add_support_hinged(3)
    system.point_load(2, Fy=-100)
    solution = system.solve()

    assert abs(solution[4]) == approx(abs(v2), rel=1e-6)
    assert abs(solution[5]) == approx(abs(theta2), rel=1e-6)
    assert abs(solution[8]) == approx(abs(theta3), rel=1e-6)


def test_inclined_roller_of_0_degrees_is_the_plain_roller():
    results = []
    for roller in ({"direction": "x"}, {"angle": 0}):
        system = SystemElements()
        system.add_element_grid([0, 1, 2], [0, 1, 0])
        system.add_support_hinged(1)
        system.add_support_roll(3, **roller)
        system.point_load(2, Fy=-100)
        system.solve()
        results.append(system.get_node_results_system(3))
    assert results[0]["Fx"] == approx(results[1]["Fx"])
    assert results[0]["Fy"] == approx(results[1]["Fy"])
    assert results[0]["Tz"] == approx(results[1]["Tz"])


def test_inclined_roller_of_45_degrees_reactions():
    # the roller reaction is normal to its free direction (45 degrees): equal components
    system = SystemElements()
    system.add_element_grid([0, 1, 2], [0, 0, 0])
    system.add_support_hinged(1)
    system.add_support_roll(3, angle=45)
    system.point_load(2, Fy=-100)
    system.solve()

    assert system.get_node_results_system(3)["Fx"] == approx(50)
    assert system.get_node_results_system(3)["Fy"] == approx(-50)
    assert system.reaction_forces[3].Fx == approx(-50)
    assert system.reaction_forces[3].Fy == approx(-50)


def test_inclined_roller_with_a_q_load():
    system = SystemElements(EA=356000, EI=1330)
    system.add_element(location=[[0, 0], [10, 0]])
    system.add_support_hinged(node_id=1)
    system.add_support_roll(node_id=-1, angle=45)
    system.q_load(q=-1, element_id=1, direction="element")
    system.solve()

    assert system.get_node_results_system(1)["Fx"] == approx(-5)
    assert system.get_node_results_system(1)["Fy"] == approx(-5)
    assert system.get_element_results(1)["Nmax"] == approx(-5)
    assert system.get_element_results(1)["Nmin"] == approx(-5)


def test_buckling_factor_of_example_18_with_one_element():
    # one cubic element: 12 EI / L**2 for a column that is pinned at both ends
    system = SystemElements(EI=5e3, EA=1e5)
    system.add_element([0, 10])
    system.add_support_hinged(1)
    system.add_support_roll(2, 1)
    system.point_load(2, Fy=-1)
    system.solve(geometrical_non_linear=True)
    assert system.buckling_factor == approx(600)
