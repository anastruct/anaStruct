import numpy as np
from pytest import approx, raises

from anastruct import LoadCase, LoadCombination, SystemElements

from .fixtures.e2e_fixtures import *
from .utils import pspec_context

"""
NOTE: Several tests in this file validate that the correct numerical engineering results
occur in specific tested scenarios. These results are thereby validated against analytical
results for the same scenarios (which comes from engineering theory). Rather than comment
every single occurrence of every engineering variable, the following general definitions
apply to any tests in which these occur:
- L = total member length (typically in metres)
- l = length of individual member span
- w = peak distributed load magnitude (typically in kN/m)
- W = total distributed load (often w * l, typically in kN)
- P = concentrated / point load magnitude (typically in kN)
- a = location of concentrated / point load from left end
- b = location of concentrated / point load from right end of
- EI = bending stiffness (typically in kN*m^2)
- EA = axial stiffness (typically in kN*m/m)
"""


def describe_analytical_validation_tests():
    # Analytical validation tests, ensuring correct results for full FEA runs

    def context_simply_supported_beam_validation():
        # Simply-supported beam with UDL validation

        w = 1
        l = 2
        EI = 10000
        EA = 1000

        system = SystemElements(EI=EI, EA=EA, mesh=10000)
        system.add_element([[0, 0], [l, 0]])
        system.add_support_hinged([1, 2])
        system.q_load(w, element_id=1)
        system.solve()

        def it_results_in_correct_moment_shear():
            assert system.get_element_results(1)["Mmax"] == approx(w * l**2 / 8)
            assert system.get_element_results(1)["Qmax"] == approx(w * l / 2)

        def it_results_in_correct_reactions():
            assert system.get_node_results_system(1)["Fy"] == approx(w * l / 2)
            assert system.get_node_results_system(2)["Fy"] == approx(w * l / 2)

        def it_results_in_correct_deflections():
            assert system.get_element_results(1)["wtotmax"] == approx(
                -5 * w * l**4 / (384 * EI)
            )

    def context_simply_supported_two_point_loads():
        # Validation tests of results, based upon analytical equations

        p = 80
        l = 5
        a = l / 3
        EI = 14000

        system = SystemElements(EI=EI, mesh=10000)
        system.add_element([[0, 0], [a, 0]])
        system.add_element([[a, 0], [2 * a, 0]])
        system.add_element([[2 * a, 0], [l, 0]])
        system.add_support_hinged([1, 4])
        system.point_load(node_id=2, Fy=p)
        system.point_load(node_id=3, Fy=p)
        system.solve()

        def it_results_in_correct_moment_shear():
            assert system.get_element_results(2)["Mmax"] == approx(p * a)
            assert system.get_element_results(1)["Qmax"] == approx(p)

        def it_results_in_correct_reactions():
            assert system.get_node_results_system(1)["Fy"] == approx(p)
            assert system.get_node_results_system(4)["Fy"] == approx(p)

        def it_results_in_correct_deflections():
            assert system.get_element_results(2)["wtotmax"] == approx(
                -((p * a) / (24 * EI)) * (3 * (l**2) - 4 * (a**2))
            )

    def context_simply_supported_beam_point_load_validation():
        # Simply-supported beam with point load at center validation

        w = 1
        l = 2
        EI = 10000
        EA = 1000
        p = 1

        system = SystemElements(EI=EI, EA=EA, mesh=10000)
        system.add_element([[0, 0], [l / 2, 0]])
        system.add_element([[l / 2, 0], [l, 0]])
        system.add_support_hinged([1, 3])
        system.point_load(2, Fx=0, Fy=p, rotation=0)
        system.solve()

        def it_results_in_correct_moment_shear():
            assert system.get_element_results(1)["Mmax"] == approx(p * l / 4)
            assert system.get_element_results(1)["Qmax"] == approx(p / 2)

        def it_results_in_correct_reactions():
            assert system.get_node_results_system(1)["Fy"] == approx(p / 2)
            assert system.get_node_results_system(3)["Fy"] == approx(p / 2)

        def it_results_in_correct_deflections():
            assert system.get_node_results_system(2)["uy"] == approx(
                -p * l**3 / (48 * EI)
            )

    def context_fixed_ends_beam_point_load_validation():
        # Beam fixed at both ends with concentrated load at center

        p = 200
        l = 8
        x = 3
        EI = 23000

        system = SystemElements(EI=EI, mesh=9000)
        system.add_element([[0, 0], [x, 0]])
        system.add_element([[x, 0], [l / 2, 0]])
        system.add_element([[l / 2, 0], [l, 0]])
        system.add_support_fixed([1, 4])
        system.point_load(3, Fx=0, Fy=200, rotation=0)
        system.solve()

        def it_results_in_correct_moment_shear():
            assert system.get_element_results(2)["Mmin"] == approx(p / 8 * (4 * x - l))
            assert system.get_element_results(2)["Qmax"] == approx(p / 2)

        def it_results_in_correct_reactions():
            assert system.get_node_results_system(1)["Fy"] == approx(p / 2)
            assert system.get_node_results_system(4)["Fy"] == approx(p / 2)

        def it_results_in_correct_deflections():
            assert system.get_node_results_system(2)["uy"] == approx(
                -p * x**2 / (48 * EI) * (3 * l - 4 * x)
            )

    def context_3_span_continuous_beam_2_UDL_loads_validation():
        # Continuous Beam spanning over three supports with two UDL in the outer spans

        w = 70
        l = 5
        EI = 15000

        system = SystemElements(EI=EI, mesh=20000)
        system.add_element([[0, 0], [l, 0]])
        system.add_element([[l, 0], [2 * l, 0]])
        system.add_element([[2 * l, 0], [3 * l, 0]])
        system.add_support_hinged([1, 2, 3, 4])
        system.q_load(w, element_id=1)
        system.q_load(w, element_id=3)
        system.solve()

        def it_results_in_correct_moment_shear():
            assert system.get_element_results(1)["Mmax"] == approx(0.10125 * w * l**2)
            assert system.get_element_results(2)["Mmax"] == approx(-0.050 * w * l**2)
            assert system.get_element_results(1)["Qmin"] == approx(-0.55 * w * l)
            assert system.get_element_results(2)["Qmax"] == approx(0)

        def it_results_in_correct_reactions():
            assert system.get_node_results_system(1)["Fy"] == approx(0.45 * w * l)
            assert system.get_node_results_system(3)["Fy"] == approx(0.55 * w * l)

        def it_results_in_correct_deflections():
            assert system.get_element_results(1)["wtotmax"] == approx(
                -0.009917469 * w * l**4 / EI
            )

            def it_results_in_correct_deflections():
                assert system.get_node_results_system(2)["uy"] == approx(
                    -p * l**3 / (48 * EI)
                )

    def context_fixed1end_simplySupported_UDL_validation():
        # beam fixed at one end and simply supported at the other with UDL

        EA = 3420000  # KN.m/m
        EI = 83100  # KN.m^2
        w = 1  # KN/m
        l = 2  # m

        system = SystemElements(EA=EA, EI=EI)
        system.add_element([[0, 0], [l, 0]])
        system.add_support_hinged(1)
        system.add_support_fixed(2)
        system.q_load(w, element_id=1)
        system.solve()

        def it_results_in_correct_reactions():
            assert system.get_node_results_system(1)["Fy"] == approx(3 * w * l / 8)
            assert system.get_node_results_system(2)["Fy"] == approx(5 * w * l / 8)
            assert system.get_node_results_system(2)["Tz"] == approx(-w * (l**2) / 8)

        def it_results_in_correct_deflections():
            assert system.get_element_results(1)["wtotmax"] == approx(
                -w * (l**4) / (185 * EI), rel=1e-3
            )

    def context_fixed1end_simplySupported_pointLoad_validation():
        # beam fixed at one end and simply supported on other with point load at the middle

        EA = 3420000  # KN.m/m
        EI = 83100  # KN.m^2
        p = 1  # KN
        l = 2  # m

        system = SystemElements(EA=EA, EI=EI, mesh=10000)
        system.add_element([[0, 0], [l * (1 / 5) ** 0.5, 0]])
        system.add_element([[l * (1 / 5) ** 0.5, 0], [l / 2, 0]])
        system.add_element([[l / 2, 0], [l, 0]])
        system.add_support_hinged(1)
        system.add_support_fixed(4)
        system.point_load(3, Fx=0, Fy=p, rotation=0)
        system.solve()

        def it_results_in_correct_reactions():
            assert system.get_node_results_system(1)["Fy"] == approx(5 * p / 16)
            assert system.get_node_results_system(4)["Fy"] == approx(11 * p / 16)
            assert system.get_node_results_system(4)["Tz"] == approx(-3 * p * l / 16)

        def it_results_in_correct_deflections():
            assert system.get_node_results_system(2)["uy"] == approx(
                -p * (l**3) / (48 * EI * 5**0.5)
            )

    def context_beam_fixed_on_both_ends_UDL():
        # beam with fixed supports at both ends and UDL

        EA = 3420000  # KN.m/m
        EI = 83100  # KN.m^2
        w = 1  # KN/m
        l = 2  # m

        system = SystemElements(EA=EA, EI=EI, mesh=2000)
        system.add_element([[0, 0], [l, 0]])
        system.add_support_fixed([1, 2])
        system.q_load(w, 1)
        system.solve()

        def it_results_in_correct_reactions():
            assert system.get_node_results_system(1)["Fy"] == approx(w * l / 2)
            assert system.get_node_results_system(2)["Fy"] == approx(w * l / 2)
            assert system.get_node_results_system(1)["Tz"] == approx(w * l**2 / 12)
            assert system.get_node_results_system(2)["Tz"] == approx(-w * l**2 / 12)

        def it_results_in_correct_deflections():
            assert system.get_element_results(1)["wtotmax"] == approx(
                -w * l**4 / (384 * EI)
            )

    def context_cantilever_UDL_validation():
        # Cantilever beam with UDL validation

        w = 10
        l = 3
        EI = 10000
        EA = 1000

        system = SystemElements(EI=EI, EA=EA, mesh=100)
        system.add_element([[0, 0], [l, 0]])
        system.add_support_fixed([2])
        system.q_load(w, element_id=1)
        system.solve()

        def it_results_in_correct_moment_shear():
            assert system.get_element_results(1)["Mmin"] == approx(-w * l**2 / 2)
            assert system.get_element_results(1)["Qmin"] == approx(-w * l)

        def it_results_in_correct_reactions():
            assert system.get_node_results_system(2)["Fy"] == approx(w * l)

        def it_results_in_correct_deflections():
            assert system.get_node_results_system(1)["uy"] == approx(
                -w * l**4 / (8 * EI)
            )

    def context_remove_element():
        # Removing an element from a propped beam

        system = SystemElements()
        system.add_element([[0, 0], [10, 0]])
        system.add_element([[10, 0], [15, 0]])
        system.add_support_hinged(1)
        system.add_support_hinged(2)
        system.q_load(q=-1, element_id=1)
        system.q_load(q=-1, element_id=2)
        system.point_load(node_id=3, Fy=-10)
        system.remove_element(2)
        system.solve()

        def it_removes_element():
            assert len(system.element_map) == 1
            assert not (2 in system.loads_q)

        def it_removes_orphaned_node():
            assert len(system.node_map) == 2
            assert not (3 in system.loads_point)

        def it_results_in_correct_reactions():
            assert system.get_node_results_system(1)["Fy"] == approx(-5)
            assert system.get_node_results_system(2)["Fy"] == approx(-5)
