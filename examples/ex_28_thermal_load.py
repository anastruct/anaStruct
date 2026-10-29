import math

from anastruct.fem.system import SystemElements

EA = 1e6
EI = 1e4
L = 10.0
ALPHA = 1.2e-5

# Fixed-fixed beam with a uniform temperature increase of 20 degrees.
# The restrained free expansion causes a constant compression force
# N = -EA * alpha * dT.
ss = SystemElements(EA=EA, EI=EI)
ss.add_element(location=[(0, 0), (L, 0)])
ss.add_support_fixed(node_id=[1, 2])
ss.thermal_load(element_id=1, delta_t=20, alpha=ALPHA)

# Fixed-fixed beam with a linear temperature gradient of 15 degrees over a
# section depth of 0.5 m (upper face warmer). The restrained free curvature
# kappa = alpha * dT_gradient / h causes a constant bending moment
# M = EI * alpha * dT_gradient / h.
ss2 = SystemElements(EA=EA, EI=EI)
ss2.add_element(location=[(0, 0), (L, 0)])
ss2.add_support_fixed(node_id=[1, 2])
ss2.thermal_load(element_id=1, delta_t=0, alpha=ALPHA, delta_t_gradient=15, h=0.5)

# Cantilever with the same gradient curves freely: the tip deflects by
# kappa * L**2 / 2 and no internal forces develop.
ss3 = SystemElements(EA=EA, EI=EI)
ss3.add_element(location=[(0, 0), (L, 0)])
ss3.add_support_fixed(node_id=1)
ss3.thermal_load(element_id=1, delta_t=0, alpha=ALPHA, delta_t_gradient=15, h=0.5)


def check(name, value, expected):
    """Print a result next to its analytical value and fail if they differ."""
    print(f"{name}: {value:.6f} (analytical: {expected:.6f})")
    assert math.isclose(value, expected, rel_tol=1e-6, abs_tol=1e-9), name


if __name__ == "__main__":
    delta_t = 20
    kappa = ALPHA * 15 / 0.5  # free curvature of the gradient

    ss.solve()
    print("Uniform dT on fixed-fixed beam:")
    # the supports push inwards with the compression force N = EA alpha dT
    check("Axial reaction at node 1", ss.reaction_forces[1].Fx, EA * ALPHA * delta_t)
    check("Axial reaction at node 2", ss.reaction_forces[2].Fx, -EA * ALPHA * delta_t)
    ss.show_axial_force()

    ss2.solve()
    print("\nTemperature gradient on fixed-fixed beam:")
    # constant moment EI kappa (sagging is negative in anaStruct), no shear force
    check("End moment at node 1", ss2.reaction_forces[1].Tz, -EI * kappa)
    check("End moment at node 2", ss2.reaction_forces[2].Tz, EI * kappa)
    ss2.show_bending_moment()

    ss3.solve()
    print("\nTemperature gradient on cantilever:")
    tip = ss3.get_node_displacements(2)
    # the upper fibres are longer: the tip goes down by kappa L**2 / 2 and rotates by kappa L
    check("Tip displacement uy", tip["uy"], -kappa * L**2 / 2)
    check("Tip rotation phi_z", tip["phi_z"], kappa * L)
    check("Reaction Fy (no internal forces)", ss3.reaction_forces[1].Fy, 0.0)
    ss3.show_displacement()
