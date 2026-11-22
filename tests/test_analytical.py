import numpy as np
from pytest import approx, mark, raises

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

    def context_semi_rigid_beam_UDL_validation():
        # Beams with rotational spring (semi-rigid) end connections and UDL.
        # Validated against N. Kotlyar, "Formulas for Beams with Semi-rigid
        # Connections", AISC Engineering Journal, Vol. 33, No. 4 (1996), pp. 142-146.
        # Kotlyar uses i = EI / l and mu = i / C, where C is the rotational stiffness
        # of the connection. Both element ends are restrained (a released end sits on
        # a hinged support), so the support reactions are the fixed-end forces.

        w = 10  # kN/m
        l = 4  # m
        EI = 5000  # kN.m^2
        EA = 5e6  # kN

        def _solve(spring, EI=EI, l=l, w=w):
            system = SystemElements(EI=EI, EA=EA)
            system.add_element([[0, 0], [l, 0]], spring=spring)
            system.add_support_fixed(1)
            if spring.get(2) == 0:
                system.add_support_hinged(2)
            else:
                system.add_support_fixed(2)
            system.q_load(w, element_id=1)
            system.solve()
            return system

        @mark.parametrize("mu", [0.01, 0.1, 0.5, 1.0, 5.0])
        def it_results_in_correct_reactions_with_springs_at_both_ends(mu):
            # Kotlyar Table 2, row 1: M_A = M_B = w * l^2 / 12 * 1 / (2 * mu + 1)
            C = EI / l / mu
            system = _solve({1: C, 2: C})
            M = w * l**2 / 12 / (2 * mu + 1)
            assert system.get_node_results_system(1)["Tz"] == approx(M)
            assert system.get_node_results_system(2)["Tz"] == approx(-M)
            assert system.get_node_results_system(1)["Fy"] == approx(w * l / 2)
            assert system.get_node_results_system(2)["Fy"] == approx(w * l / 2)

        @mark.parametrize("mu", [0.01, 0.1, 0.5, 1.0, 5.0])
        def it_results_in_correct_reactions_with_spring_and_pin(mu):
            # Kotlyar Table 2, row 3: M_A = w * l^2 / 8 * 1 / (3 * mu + 1), B pinned
            C = EI / l / mu
            system = _solve({1: C, 2: 0})
            M = w * l**2 / 8 / (3 * mu + 1)
            assert system.get_node_results_system(1)["Tz"] == approx(M)
            assert system.get_node_results_system(2)["Tz"] == approx(0, abs=1e-9)
            assert system.get_node_results_system(1)["Fy"] == approx(w * l / 2 + M / l)
            assert system.get_node_results_system(2)["Fy"] == approx(w * l / 2 - M / l)

        def it_results_in_fixed_end_moments_for_rigid_ends():
            # mu = 0: w * l^2 / 12 with both ends rigid, w * l^2 / 8 with B pinned
            system = _solve({})
            assert system.get_node_results_system(1)["Tz"] == approx(w * l**2 / 12)
            assert system.get_node_results_system(2)["Tz"] == approx(-w * l**2 / 12)
            system = _solve({2: 0})
            assert system.get_node_results_system(1)["Tz"] == approx(w * l**2 / 8)

        def it_matches_the_worked_example():
            # Kotlyar's example: W18x50 beam, I = 800 in^4, l = 240 in, E = 29,000 ksi,
            # bolted flange-plate connection C = 3,347,325 kip-in/rad, so mu = 0.0288
            # and M_A = M_B = 0.945 * w * l^2 / 12.
            EI_ex, l_ex, w_ex, C = 29000 * 800, 240, 0.1, 3347325
            system = _solve({1: C, 2: C}, EI=EI_ex, l=l_ex, w=w_ex)
            M_fixed = w_ex * l_ex**2 / 12
            assert system.get_node_results_system(1)["Tz"] / M_fixed == approx(
                0.945, abs=5e-4
            )
            assert system.get_node_results_system(2)["Tz"] / M_fixed == approx(
                -0.945, abs=5e-4
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
