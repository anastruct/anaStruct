import math

import matplotlib.pyplot as plt

from anastruct import LoadCase, LoadCombination
from anastruct.fem.system import SystemElements

EI = 1e4
L = 6.0  # length of each span
ALPHA = 1.2e-5

# Two-span continuous beam exposed to a uniform temperature change and a
# temperature gradient, combined with a q-load in a load combination.
ss = SystemElements(EA=1e6, EI=EI)
ss.add_element(location=[(0, 0), (L, 0)])
ss.add_element(location=[(L, 0), (2 * L, 0)])
ss.add_support_hinged(node_id=1)
ss.add_support_roll(node_id=2)
ss.add_support_roll(node_id=3)

lc_thermal = LoadCase("thermal")
lc_thermal.thermal_load(element_id=[1, 2], delta_t=15, alpha=ALPHA)

lc_gradient = LoadCase("gradient")
lc_gradient.thermal_load(
    element_id=[1, 2], delta_t=0, alpha=ALPHA, delta_t_gradient=10, h=0.5
)

lc_traffic = LoadCase("traffic")
lc_traffic.q_load(q=-10, element_id=[1, 2])  # negative q acts downwards by default

combination = LoadCombination("SLS")
combination.add_load_case(lc_thermal, factor=1.0)
combination.add_load_case(lc_gradient, factor=1.0)
combination.add_load_case(lc_traffic, factor=1.0)


def check(name, value, expected):
    """Print a result next to its analytical value and fail if they differ."""
    print(f"{name}: {value:.4f} (analytical: {expected:.4f})")
    assert math.isclose(value, expected, rel_tol=1e-6, abs_tol=1e-6), name


if __name__ == "__main__":
    results = combination.solve(ss)

    # Analytical reactions of the middle support of a two-span beam with equal spans L:
    kappa = ALPHA * 10 / 0.5  # free curvature of the gradient
    expected = {
        # a uniform temperature change expands the beam freely on the rollers
        "thermal": 0.0,
        # constant curvature kappa: R = 3 EI kappa / L (upwards)
        "gradient": 3 * EI * kappa / L,
        # uniform load q on both spans: R = 5 q L / 4 (q = -10 acts downwards)
        "traffic": 5 * -10 * L / 4,
    }
    # the combination is the sum of the load cases (all factors are 1)
    expected["combination"] = sum(expected.values())

    for name, system in results.items():
        check(
            f"{name} -> middle support reaction Fy",
            system.reaction_forces[2].Fy,
            expected[name],
        )
        system.show_bending_moment(show=False)
        plt.title(name)
        plt.show()  # type: ignore
