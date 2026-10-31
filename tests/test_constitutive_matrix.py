import unittest

import numpy as np

from anastruct import SystemElements
from anastruct.fem.elements import constitutive_matrix

EI, L = 15000.0, 4.0


def series_spring_block(k1=None, k2=None):
    """Rotational block with springs in series with the beam flexibility."""
    stiffness = np.array([[4 * EI / L, -2 * EI / L], [-2 * EI / L, 4 * EI / L]])
    flexibility = np.linalg.inv(stiffness)
    for index, k in enumerate((k1, k2)):
        if k is not None:
            flexibility[index, index] += 1 / k
    return np.linalg.inv(flexibility)


class TestConstitutiveMatrixSprings(unittest.TestCase):
    def test_springs_add_in_series_with_bending_flexibility(self):
        for spring, (k1, k2) in (
            ({1: 1000.0}, (1000.0, None)),
            ({2: 2000.0}, (None, 2000.0)),
            ({1: 1000.0, 2: 2000.0}, (1000.0, 2000.0)),
        ):
            matrix = constitutive_matrix(1e9, EI, L, spring, False, False)
            np.testing.assert_allclose(matrix[1:, 1:], series_spring_block(k1, k2))
            self.assertEqual(matrix[0][0], 1e9 / L)

    def test_spring_matrix_is_symmetric_positive_definite(self):
        # Soft springs used to leave an indefinite matrix, which made
        # SystemElements.solve() raise a stability error.
        for spring in ({1: 100.0}, {2: 100.0}, {1: 100.0, 2: 50.0}):
            block = constitutive_matrix(1e9, EI, L, spring, False, False)[1:, 1:]
            np.testing.assert_allclose(block, block.T)
            self.assertTrue(np.all(np.linalg.eigvalsh(block) > 0))

    def test_hinge_with_spring_on_other_node(self):
        matrix = constitutive_matrix(1e9, EI, L, {2: 2000.0}, True, False)
        self.assertEqual(matrix[1][1], 0)
        self.assertEqual(matrix[1][2], 0)
        np.testing.assert_allclose(matrix[2][2], 1 / (L / (4 * EI) + 1 / 2000.0))


class TestRotationalSpringStructures(unittest.TestCase):
    P = 10.0

    def test_cantilever_with_base_spring_matches_closed_form(self):
        for k in (2000.0, 20000.0, 2e5):
            ss = SystemElements(EI=EI, EA=1e9)
            ss.add_element([[0, 0], [L, 0]], spring={1: k})
            ss.add_support_fixed(1)
            ss.point_load(2, Fy=-self.P)
            ss.solve()
            expected = self.P * L**3 / (3 * EI) + self.P * L * L / k
            self.assertAlmostEqual(
                abs(ss.get_node_displacements(2)["uy"]), expected, places=10
            )

    def test_cantilever_with_spring_at_element_end_matches_closed_form(self):
        for k in (2000.0, 20000.0, 2e5):
            ss = SystemElements(EI=EI, EA=1e9)
            ss.add_element([[0, 0], [L, 0]], spring={2: k})
            ss.add_element([[L, 0], [2 * L, 0]])
            ss.add_support_fixed(1)
            ss.point_load(3, Fy=-self.P)
            ss.solve()
            # The moment P*L at the spring opens a rotation P*L/k, which the
            # second span carries as a rigid rotation over length L.
            expected = self.P * (2 * L) ** 3 / (3 * EI) + self.P * L / k * L
            self.assertAlmostEqual(
                abs(ss.get_node_displacements(3)["uy"]), expected, places=10
            )


if __name__ == "__main__":
    unittest.main()
