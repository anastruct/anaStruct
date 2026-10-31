import unittest

import numpy as np

from anastruct.fem.elements import constitutive_matrix


class TestConstitutiveMatrixSprings(unittest.TestCase):
    def test_symmetric_with_springs_on_both_nodes(self):
        # A linear-elastic element's constitutive matrix must be symmetric,
        # spring release or not.
        matrix = constitutive_matrix(
            EA=1e9,
            EI=15000.0,
            l=4.0,
            spring={1: 1000.0, 2: 2000.0},
            node_1_hinge=False,
            node_2_hinge=False,
        )
        np.testing.assert_allclose(matrix, matrix.T)

    def test_spring_reduces_only_its_own_node_diagonal_term(self):
        # A rotational spring in series with the beam's own rotational
        # stiffness should only soften that node's diagonal term (per the
        # flexibility matrix in the function's own comment); the
        # off-diagonal (moment/rotation coupling) term is unaffected.
        EI, l = 15000.0, 4.0
        k1, k2 = 1000.0, 2000.0

        matrix = constitutive_matrix(
            EA=1e9,
            EI=EI,
            l=l,
            spring={1: k1, 2: k2},
            node_1_hinge=False,
            node_2_hinge=False,
        )

        c11_expected = 1 / (1 / (4 * EI / l) + 1 / k1)
        c22_expected = 1 / (1 / (4 * EI / l) + 1 / k2)

        np.testing.assert_allclose(matrix[1][1], c11_expected)
        np.testing.assert_allclose(matrix[2][2], c22_expected)
        np.testing.assert_allclose(matrix[1][2], -2 * EI / l)
        np.testing.assert_allclose(matrix[2][1], -2 * EI / l)

    def test_spring_at_node_2_only(self):
        # Regression test: a spring at node 2 alone used to leave
        # matrix[2][2] untouched and corrupt the off-diagonal terms instead.
        EI, l = 15000.0, 4.0
        k2 = 2000.0

        matrix = constitutive_matrix(
            EA=1e9,
            EI=EI,
            l=l,
            spring={2: k2},
            node_1_hinge=False,
            node_2_hinge=False,
        )

        c22_expected = 1 / (1 / (4 * EI / l) + 1 / k2)

        np.testing.assert_allclose(matrix[1][1], 4 * EI / l)
        np.testing.assert_allclose(matrix[2][2], c22_expected)
        np.testing.assert_allclose(matrix[1][2], -2 * EI / l)
        np.testing.assert_allclose(matrix[2][1], -2 * EI / l)


if __name__ == "__main__":
    unittest.main()
