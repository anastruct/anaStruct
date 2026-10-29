import math

from anastruct.fem.system import SystemElements

EI = 1e4
L = 10.0

# Propped cantilever with a support settlement of 20 mm at the roller. Like a load Fy, uy
# is positive in the +y direction (upwards), so a settlement is a negative uy.
ss = SystemElements(EA=1e6, EI=EI)
ss.add_element(location=[(0, 0), (L, 0)])
ss.add_support_fixed(node_id=1)
ss.add_support_roll(node_id=2)
ss.imposed_displacement(node_id=2, uy=-0.02)

# Fixed-fixed beam with an imposed rotation of 0.01 rad at the right end.
ss2 = SystemElements(EA=1e6, EI=EI)
ss2.add_element(location=[(0, 0), (L, 0)])
ss2.add_support_fixed(node_id=[1, 2])
ss2.imposed_displacement(node_id=2, phi_z=0.01)


def check(name, value, expected):
    """Print a result next to its analytical value and fail if they differ."""
    print(f"{name}: {value:.4f} (analytical: {expected:.4f})")
    assert math.isclose(value, expected, rel_tol=1e-6, abs_tol=1e-9), name


if __name__ == "__main__":
    delta = 0.02  # settlement
    phi = 0.01  # imposed rotation

    ss.solve()
    print("Settlement of 20 mm at node 2:")
    # propped cantilever: R = 3 EI delta / L**3, M = 3 EI delta / L**2
    check("Roller reaction Fy", ss.reaction_forces[2].Fy, 3 * EI * delta / L**3)
    check("Fixed end moment Tz", ss.reaction_forces[1].Tz, 3 * EI * delta / L**2)
    ss.show_bending_moment()

    ss2.solve()
    print("\nImposed rotation of 0.01 rad at node 2:")
    # fixed-fixed beam: the far end gets 2 EI phi / L, the end that rotates 4 EI phi / L
    check("Moment at the far end (node 1)", ss2.reaction_forces[1].Tz, 2 * EI * phi / L)
    check(
        "Moment at the rotated end (node 2)",
        ss2.reaction_forces[2].Tz,
        4 * EI * phi / L,
    )
    ss2.show_bending_moment()
