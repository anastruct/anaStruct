from pytest import approx, raises

from anastruct import LoadCase, LoadCombination, SystemElements
from anastruct.basic import FEMException

"""
NOTE: These tests validate imposed displacements/rotations of supports (support
settlements) and thermal loads on elements against analytical results from
engineering theory, and cover their interplay with load combinations, model edits
(insert_node, discretize, remove_loads) and the sign conventions. General
definitions:
- L = member length
- EA = axial stiffness
- EI = bending stiffness
- alpha = coefficient of thermal expansion
- dT = uniform temperature change
- dT_grad = temperature difference over the section depth h
- kappa = alpha * dT_grad / h = free thermal curvature
- delta = imposed support settlement
- phi = imposed support rotation

The tests are plain test functions (not nested describe/context blocks), so that every
assertion is guaranteed to be executed.
"""

EA = 1.0e6
EI = 1.0e4
L = 10.0
ALPHA = 1.2e-5
DT = 20.0
H = 0.5
DT_GRAD = 15.0
KAPPA = ALPHA * DT_GRAD / H


def propped_cantilever(**kwargs):
    system = SystemElements(EA=EA, EI=EI, **kwargs)
    system.add_element([[0, 0], [L, 0]])
    system.add_support_fixed(1)
    system.add_support_roll(2)
    return system


def fixed_fixed():
    system = SystemElements(EA=EA, EI=EI)
    system.add_element([[0, 0], [L, 0]])
    system.add_support_fixed([1, 2])
    return system


def cantilever():
    system = SystemElements(EA=EA, EI=EI)
    system.add_element([[0, 0], [L, 0]])
    system.add_support_fixed(1)
    return system


# ---------------------------------------------------------------------------
# imposed displacements
# ---------------------------------------------------------------------------


def test_propped_cantilever_settlement():
    # Propped cantilever with an imposed displacement delta of the prop:
    # |R_prop| = 3 * EI * delta / L**3, |M_fixed| = 3 * EI * delta / L**2.
    # uy is positive in the +y direction (upwards, like a load Fy): a settlement is a
    # negative uy and gives the reactions of the opposite sign.
    delta = 0.02
    for sign in (1, -1):  # +1: the prop moves up, -1: the prop settles
        system = propped_cantilever()
        system.imposed_displacement(2, uy=sign * delta)
        system.solve()

        assert system.reaction_forces[2].Fy == approx(-sign * 3 * EI * delta / L**3)
        assert system.reaction_forces[1].Tz == approx(-sign * 3 * EI * delta / L**2)
        # vertical equilibrium
        assert system.reaction_forces[1].Fy + system.reaction_forces[2].Fy == approx(0)
        # the imposed displacement is reported in the direction of +y
        assert system.get_node_displacements(2)["uy"] == approx(sign * delta)
        assert system.reaction_forces[2].uy == approx(sign * delta)


def test_positive_load_and_imposed_displacement_act_in_the_same_direction():
    # a positive Fy and a positive uy both act in the +y direction (upwards); with
    # invert_y_loads=False they act in the -y direction
    for invert, direction in ((True, 1), (False, -1)):
        loaded = SystemElements(EA=EA, EI=EI, invert_y_loads=invert)
        loaded.add_element([[0, 0], [L, 0]])
        loaded.add_support_fixed(1)
        loaded.point_load(2, Fy=1.0)
        loaded.solve()
        assert direction * loaded.get_node_displacements(2)["uy"] > 0

        moved = propped_cantilever(invert_y_loads=invert)
        moved.imposed_displacement(2, uy=0.01)
        moved.solve()
        assert direction * moved.get_node_displacements(2)["uy"] > 0


def test_positive_fy_acts_in_the_y_direction_of_the_structure():
    # a force along +y on the top node of a column (y = +L) pulls it: tension
    system = SystemElements(EA=EA, EI=EI)
    system.add_element([[0, 0], [0, L]])
    system.add_support_fixed(1)
    system.point_load(2, Fy=1.0)
    system.solve()

    assert system.element_map[1].N_1 == approx(1.0)


def test_fixed_fixed_imposed_rotation():
    # Fixed-fixed beam with an imposed rotation phi at node 2:
    # M_near = 4 * EI * phi / L, M_far = 2 * EI * phi / L, R = 6 * EI * phi / L**2
    phi = 0.01
    system = fixed_fixed()
    system.imposed_displacement(2, phi_z=phi)
    system.solve()

    assert system.reaction_forces[2].Tz == approx(4 * EI * phi / L)
    assert system.reaction_forces[1].Tz == approx(2 * EI * phi / L)
    assert system.reaction_forces[1].Fy == approx(-6 * EI * phi / L**2)
    assert system.reaction_forces[2].Fy == approx(6 * EI * phi / L**2)
    # reported rotations have the opposite sign of the system displacement
    # vector, consistent with the moment_load sign convention
    assert system.get_node_displacements(2)["phi_z"] == approx(-phi)


def test_fixed_fixed_imposed_axial_displacement():
    # Fixed-fixed bar with an imposed axial displacement dx at node 2:
    # N = EA * dx / L (tension)
    dx = 0.01
    system = fixed_fixed()
    system.imposed_displacement(2, ux=dx)
    system.solve()

    assert system.reaction_forces[1].Fx == approx(-EA * dx / L)
    assert system.reaction_forces[2].Fx == approx(EA * dx / L)
    assert system.get_node_displacements(2)["ux"] == approx(dx)


def test_two_span_beam_middle_support_settlement():
    # Two equal spans (continuous beam), settlement delta of the middle support:
    # the middle reaction is reduced by 6 * EI * delta / L**3 (L = span length)
    delta = 0.01
    system = SystemElements(EA=EA, EI=EI)
    system.add_element([[0, 0], [L, 0]])
    system.add_element([[L, 0], [2 * L, 0]])
    system.add_support_hinged(1)
    system.add_support_roll(2)
    system.add_support_roll(3)
    system.imposed_displacement(2, uy=delta)
    system.solve()

    assert system.reaction_forces[2].Fy == approx(-6 * EI * delta / L**3)
    assert system.reaction_forces[1].Fy == approx(3 * EI * delta / L**3)
    assert system.reaction_forces[3].Fy == approx(3 * EI * delta / L**3)


def test_settlement_superposition_with_q_load():
    # Propped cantilever with a q-load and a settlement of the prop:
    # R_prop = 3 * q * L / 8 - 3 * EI * delta / L**3
    q = 10.0
    delta = 0.02
    system = propped_cantilever()
    system.q_load(q, 1)
    system.imposed_displacement(2, uy=delta)
    system.solve()

    assert system.reaction_forces[2].Fy == approx(3 * q * L / 8 - 3 * EI * delta / L**3)


def test_imposed_displacement_on_unrestrained_dof_raises_on_solve():
    system = cantilever()
    system.imposed_displacement(2, uy=0.01)
    with raises(FEMException):
        system.solve()


def test_imposed_displacement_requires_a_value():
    system = SystemElements(EA=EA, EI=EI)
    system.add_element([[0, 0], [L, 0]])
    with raises(FEMException):
        system.imposed_displacement(1)


def test_imposed_displacement_after_solve_raises():
    system = propped_cantilever()
    system.point_load(2, Fy=1.0)
    system.solve()
    with raises(FEMException):
        system.imposed_displacement(2, uy=0.01)


def test_imposed_displacement_on_inclined_roller_raises():
    system = SystemElements(EA=EA, EI=EI)
    system.add_element([[0, 0], [L, 0]])
    system.add_support_hinged(1)
    system.add_support_roll(2, angle=45)
    with raises(FEMException):
        system.imposed_displacement(2, uy=0.01)


def test_imposed_rotation_on_internal_hinge_raises():
    system = SystemElements(EA=EA, EI=EI)
    system.add_element([[0, 0], [L, 0]])
    system.add_element([[L, 0], [2 * L, 0]])
    system.add_support_fixed(1)
    system.add_support_roll(3)
    system.add_internal_hinge(2)
    with raises(FEMException):
        system.imposed_displacement(2, phi_z=0.01)


def test_imposed_displacement_is_equivalent_for_invert_y_loads_false():
    delta = 0.02
    system_inverted = propped_cantilever()
    system_inverted.imposed_displacement(2, uy=delta)
    system_inverted.solve()
    system_plain = propped_cantilever(invert_y_loads=False)
    system_plain.imposed_displacement(2, uy=-delta)
    system_plain.solve()

    assert system_plain.reaction_forces[2].Fy == approx(
        system_inverted.reaction_forces[2].Fy
    )


# ---------------------------------------------------------------------------
# thermal loads
# ---------------------------------------------------------------------------


def test_fixed_fixed_uniform_temperature():
    # Fixed-fixed beam with a uniform temperature increase:
    # N = -EA * alpha * dT (compression), no bending
    system = fixed_fixed()
    system.thermal_load(1, delta_t=DT, alpha=ALPHA)
    system.solve()

    assert system.reaction_forces[1].Fx == approx(EA * ALPHA * DT)
    assert system.reaction_forces[2].Fx == approx(-EA * ALPHA * DT)
    assert system.reaction_forces[1].Tz == approx(0)
    assert system.reaction_forces[1].Fy == approx(0)
    el = system.get_element_results(1)
    assert el["Nmin"] == approx(-EA * ALPHA * DT)
    assert el["Nmax"] == approx(-EA * ALPHA * DT)
    assert el["Mmin"] == approx(0)
    assert el["Mmax"] == approx(0)


def test_cantilever_uniform_temperature():
    # Cantilever with a uniform temperature increase expands freely:
    # tip displacement = alpha * dT * L, no internal forces
    system = cantilever()
    system.thermal_load(1, delta_t=DT, alpha=ALPHA)
    system.solve()

    assert system.get_node_displacements(2)["ux"] == approx(ALPHA * DT * L)
    assert system.reaction_forces[1].Fx == approx(0)
    el = system.get_element_results(1)
    assert el["Nmin"] == approx(0)
    assert el["Nmax"] == approx(0)


def test_fixed_fixed_temperature_gradient():
    # Fixed-fixed beam with a temperature gradient over the section depth (warmer upper
    # face): the upper fibres are restrained in compression and the lower fibres in
    # tension, a sagging moment of the constant value EI * kappa, no shear. Sagging
    # moments are negative in anaStruct.
    system = fixed_fixed()
    system.thermal_load(1, delta_t=0, alpha=ALPHA, delta_t_gradient=DT_GRAD, h=H)
    system.solve()

    m = EI * KAPPA
    assert system.reaction_forces[1].Tz == approx(-m)
    assert system.reaction_forces[2].Tz == approx(m)
    assert system.reaction_forces[1].Fy == approx(0)
    assert system.reaction_forces[2].Fy == approx(0)
    el = system.get_element_results(1)
    assert el["Mmin"] == approx(-m)
    assert el["Mmax"] == approx(-m)
    assert el["Qmin"] == approx(0)
    assert el["Qmax"] == approx(0)


def test_cantilever_temperature_gradient():
    # Cantilever with a temperature gradient curves freely with a constant curvature
    # kappa: tip deflection = kappa * L**2 / 2, no internal forces. A positive gradient is
    # a warmer upper face: the upper fibres are longer, so the tip moves down towards the
    # colder lower face. get_node_displacements is positive in the direction of +y.
    system = cantilever()
    system.thermal_load(1, delta_t=0, alpha=ALPHA, delta_t_gradient=DT_GRAD, h=H)
    system.solve()

    assert system.get_node_displacements(2)["uy"] == approx(-KAPPA * L**2 / 2)
    assert system.reaction_forces[1].Tz == approx(0)
    assert system.reaction_forces[1].Fy == approx(0)


def test_propped_cantilever_temperature_gradient():
    # Fixed-pinned beam with a temperature gradient (warmer upper face):
    # M_fixed = 3/2 * EI * kappa (sagging, so negative in anaStruct)
    system = SystemElements(EA=EA, EI=EI)
    system.add_element([[0, 0], [L, 0]])
    system.add_support_fixed(1)
    system.add_support_hinged(2)
    system.thermal_load(1, delta_t=0, alpha=ALPHA, delta_t_gradient=DT_GRAD, h=H)
    system.solve()

    assert system.reaction_forces[1].Tz == approx(-1.5 * EI * KAPPA)
    assert system.reaction_forces[1].Fy == approx(1.5 * EI * KAPPA / L)
    assert system.reaction_forces[2].Fy == approx(-1.5 * EI * KAPPA / L)


def test_simply_supported_beam_with_a_warmer_upper_face_cambers_upwards():
    # an arch-shaped beam: the middle rises by kappa * L**2 / 8 relative to the supports
    # (a sun-heated bridge girder) and there are no internal forces
    system = SystemElements(EA=EA, EI=EI)
    system.add_element([[0, 0], [L, 0]])
    system.add_support_hinged(1)
    system.add_support_roll(2)
    system.thermal_load(1, delta_t=0, alpha=ALPHA, delta_t_gradient=DT_GRAD, h=H)
    middle = system.insert_node(1, factor=0.5)["new_node_id"]
    system.solve()

    assert system.get_node_displacements(middle)["uy"] == approx(KAPPA * L**2 / 8)
    assert system.reaction_forces[1].Fy == approx(0.0, abs=1e-9)


def test_beam_with_a_warmer_lower_face_sags_downwards():
    # the same beam with a negative gradient (warmer lower face) hangs through
    system = SystemElements(EA=EA, EI=EI)
    system.add_element([[0, 0], [L, 0]])
    system.add_support_hinged(1)
    system.add_support_roll(2)
    system.thermal_load(1, delta_t=0, alpha=ALPHA, delta_t_gradient=-DT_GRAD, h=H)
    middle = system.insert_node(1, factor=0.5)["new_node_id"]
    system.solve()

    assert system.get_node_displacements(middle)["uy"] == approx(-KAPPA * L**2 / 8)


def test_truss_uniform_temperature():
    # Fully restrained truss bar with a uniform temperature increase:
    # N = -EA * alpha * dT (compression)
    system = SystemElements()
    system.add_truss_element([[0, 0], [L, 0]], EA=EA)
    system.add_support_hinged(1)
    system.add_support_hinged(2)
    system.thermal_load(1, delta_t=DT, alpha=ALPHA)
    system.solve()

    assert system.reaction_forces[1].Fx == approx(EA * ALPHA * DT)
    assert system.reaction_forces[2].Fx == approx(-EA * ALPHA * DT)
    el = system.get_element_results(1)
    assert el["Nmin"] == approx(-EA * ALPHA * DT)
    assert el["Nmax"] == approx(-EA * ALPHA * DT)


def test_gradient_requires_section_depth():
    system = SystemElements(EA=EA, EI=EI)
    system.add_element([[0, 0], [L, 0]])
    with raises(FEMException):
        system.thermal_load(1, delta_t=0, alpha=ALPHA, delta_t_gradient=DT_GRAD)


def test_thermal_load_with_element_lists():
    # two elements with different temperature changes, free expansion
    system = SystemElements(EA=EA, EI=EI)
    system.add_element([[0, 0], [L, 0]])
    system.add_element([[L, 0], [2 * L, 0]])
    system.add_support_fixed(1)
    system.thermal_load([1, 2], delta_t=[DT, 2 * DT], alpha=ALPHA)
    system.solve()

    assert system.get_node_displacements(3)["ux"] == approx(3 * ALPHA * DT * L)


def test_thermal_load_with_negative_ids():
    system = fixed_fixed()
    system.thermal_load(-1, delta_t=DT, alpha=ALPHA)
    system.solve()

    assert system.reaction_forces[1].Fx == approx(EA * ALPHA * DT)


def test_thermal_load_and_settlement_superposition():
    # propped cantilever, gradient and settlement of the prop: the prop reaction is
    # the sum of the separate results
    delta = 0.02
    thermal = propped_cantilever()
    thermal.thermal_load(1, delta_t=0, alpha=ALPHA, delta_t_gradient=DT_GRAD, h=H)
    thermal.solve()
    settlement = propped_cantilever()
    settlement.imposed_displacement(2, uy=delta)
    settlement.solve()
    both = propped_cantilever()
    both.thermal_load(1, delta_t=0, alpha=ALPHA, delta_t_gradient=DT_GRAD, h=H)
    both.imposed_displacement(2, uy=delta)
    both.solve()

    assert both.reaction_forces[2].Fy == approx(
        thermal.reaction_forces[2].Fy + settlement.reaction_forces[2].Fy
    )
    assert both.reaction_forces[2].Fy != approx(0)


def test_gradient_warmer_face_on_the_left_of_a_vertical_element():
    # column fixed at the base, the face on the left of the direction from its first to
    # its second node is warmer: the free end moves to the colder face. Drawn upwards the
    # left face is the face towards -x, so the free end moves to +x; drawn downwards the
    # left face is the face towards +x, so the free end moves to -x.
    up = SystemElements(EA=EA, EI=EI)
    up.add_element([[0, 0], [0, L]])
    up.add_support_fixed(1)
    up.thermal_load(1, delta_t=0, alpha=ALPHA, delta_t_gradient=DT_GRAD, h=H)
    up.solve()
    assert up.get_node_displacements(2)["ux"] == approx(KAPPA * L**2 / 2)

    down = SystemElements(EA=EA, EI=EI)
    down.add_element([[0, L], [0, 0]])
    down.add_support_fixed(2)
    down.thermal_load(1, delta_t=0, alpha=ALPHA, delta_t_gradient=DT_GRAD, h=H)
    down.solve()
    assert down.get_node_displacements(1)["ux"] == approx(-KAPPA * L**2 / 2)


def test_gradient_warmer_face_up_for_a_right_to_left_element():
    system = SystemElements(EA=EA, EI=EI)
    system.add_element([[L, 0], [0, 0]])
    system.add_support_fixed(2)
    system.thermal_load(1, delta_t=0, alpha=ALPHA, delta_t_gradient=DT_GRAD, h=H)
    system.solve()

    # the warmer face is the upper face, whatever the order of the end points
    assert system.get_node_displacements(1)["uy"] == approx(-KAPPA * L**2 / 2)


# ---------------------------------------------------------------------------
# load cases and load combinations
# ---------------------------------------------------------------------------


def test_thermal_load_in_load_case():
    lc = LoadCase("thermal")
    lc.thermal_load(1, delta_t=DT, alpha=ALPHA)
    system = fixed_fixed()
    system.apply_load_case(lc)
    system.solve()

    assert system.reaction_forces[1].Fx == approx(EA * ALPHA * DT)


def test_thermal_load_in_load_combination_applies_the_factor():
    lc = LoadCase("thermal")
    lc.thermal_load(1, delta_t=DT, alpha=ALPHA)
    comb = LoadCombination("ULS")
    comb.add_load_case(lc, factor=1.5)
    results = comb.solve(fixed_fixed())

    assert results["combination"].reaction_forces[1].Fx == approx(1.5 * EA * ALPHA * DT)


def test_imposed_displacement_in_load_combination():
    # a settlement belongs to one load case: it must be counted once and be
    # multiplied by the factor of its load case
    delta = 0.02
    q = -10.0
    lc_settlement = LoadCase("settlement")
    lc_settlement.imposed_displacement(2, uy=1.0)  # unit settlement
    lc_q = LoadCase("q")
    lc_q.q_load(q, 1)
    comb = LoadCombination("ULS")
    comb.add_load_case(lc_settlement, factor=delta)
    comb.add_load_case(lc_q, factor=1.5)
    result = comb.solve(propped_cantilever())["combination"]

    # superposition of independent analyses
    settlement_only = propped_cantilever()
    settlement_only.imposed_displacement(2, uy=delta)
    settlement_only.solve()
    q_only = propped_cantilever(load_factor=1.5)
    q_only.q_load(q, 1)
    q_only.solve()
    expected = settlement_only.reaction_forces[2].Fy + q_only.reaction_forces[2].Fy
    assert result.reaction_forces[2].Fy == approx(expected)
    assert result.get_node_displacements(2)["uy"] == approx(delta)


# ---------------------------------------------------------------------------
# model edits
# ---------------------------------------------------------------------------


def test_insert_node_carries_the_thermal_load_to_both_new_elements():
    system = cantilever()
    system.thermal_load(1, delta_t=0, alpha=ALPHA, delta_t_gradient=DT_GRAD, h=H)
    result = system.insert_node(1, factor=0.5)
    system.solve()

    assert 1 not in system.loads_temperature
    assert set(system.loads_temperature) == {
        result["new_element_id1"],
        result["new_element_id2"],
    }
    # free end of the cantilever: -kappa * L**2 / 2 (down), new middle node:
    # -kappa * (L/2)**2 / 2
    assert system.get_node_displacements(2)["uy"] == approx(-KAPPA * L**2 / 2)
    middle = result["new_node_id"]
    assert system.get_node_displacements(middle)["uy"] == approx(
        -KAPPA * (L / 2) ** 2 / 2
    )


def test_discretize_keeps_the_thermal_load():
    system = fixed_fixed()
    system.thermal_load(1, delta_t=DT, alpha=ALPHA)
    system.discretize(4)
    system.solve()

    assert system.reaction_forces[1].Fx == approx(EA * ALPHA * DT)


def test_discretize_keeps_the_imposed_displacement():
    delta = 0.02
    reference = propped_cantilever()
    reference.imposed_displacement(2, uy=delta)
    reference.solve()

    system = propped_cantilever()
    system.imposed_displacement(2, uy=delta)
    system.discretize(4)
    system.solve()

    node_id = max(system.reaction_forces)
    assert system.reaction_forces[node_id].Fy == approx(reference.reaction_forces[2].Fy)


def test_remove_loads_removes_thermal_loads_and_imposed_displacements():
    system = propped_cantilever()
    system.thermal_load(1, delta_t=DT, alpha=ALPHA)
    system.imposed_displacement(2, uy=0.02)
    system.remove_loads()

    assert system.loads_temperature == {}
    assert system.loads_displacement == {}
    assert system.element_map[1].temperature_load == (0.0, 0.0)

    system.point_load(2, Fy=1.0)
    system.solve()
    assert system.reaction_forces[1].Fx == approx(0.0, abs=1e-9)
    assert system.get_node_displacements(2)["uy"] == approx(0.0, abs=1e-12)
