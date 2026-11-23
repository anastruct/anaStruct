import math

import numpy as np
from pytest import approx, raises

from anastruct import SystemElements
from anastruct.basic import FEMException
from anastruct.fem.system_components.solver import det_linear_buckling

EI = 1810.0
EA = 460000.0
L = 6.0


def _beam(left: str, right: str, axial: float = 0.0, n: int = 12) -> SystemElements:
    ss = SystemElements()
    xs = np.linspace(0, L, n + 1)
    for a, b in zip(xs[:-1], xs[1:]):
        ss.add_element([[a, 0], [b, 0]], EA=EA, EI=EI)
    for node, support in ((1, left), (n + 1, right)):
        if support == "fixed":
            ss.add_support_fixed(node)
        else:
            ss.add_support_roll(node, direction=2)
    ss.q_load(q=-5.0, element_id=list(range(1, n + 1)), direction="element")
    if axial:
        ss.point_load(n + 1, Fx=axial)
    return ss


def describe_validate():
    # validate() checks the eigenvalues of the symmetric stiffness matrix against
    # min_eigen with a Cholesky factorisation

    def it_rejects_a_mechanism():
        # two rollers: nothing stops the beam sliding along its axis
        with raises(FEMException, match="StabilityError"):
            _beam("roller", "roller", n=1).solve()

    def it_compares_the_smallest_eigenvalue_with_min_eigen():
        # Cantilever with EI = l = 1: the bending stiffness at the free end is
        # [[12, 6], [6, 4]], whose smallest eigenvalue is 8 - sqrt(52) = 0.789.
        # Loaded in x and y, so every free degree of freedom is displaced.
        ss = SystemElements(EA=1000.0, EI=1.0)
        ss.add_element([[0, 0], [1, 0]])
        ss.add_support_fixed(1)
        ss.point_load(2, Fx=1.0, Fy=-1.0)
        ss.solve()
        assert ss.validate(min_eigen=0.75)
        assert not ss.validate(min_eigen=0.8)


def describe_buckling_factor():
    # The buckling factor is found from the symmetric form of the eigenproblem,
    # using the Cholesky factor of k0

    def it_matches_the_euler_load_of_a_fixed_pinned_member():
        # N_cr = π²EI / (0.6992 L)²
        n_cr = math.pi**2 * EI / (0.6992 * L) ** 2
        ss = _beam("fixed", "roller", axial=-100.0, n=24)
        assert det_linear_buckling(ss) == approx(n_cr / 100.0, rel=2e-3)
