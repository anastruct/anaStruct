"""Geometric stiffness matrix Kg used by ``solve(geometrical_non_linear=True)``.

The reference matrix (H. P. Gavin, CEE 421 notes, linked in elements.py) has
y up; anastruct's y is opposite, so the y DOFs must be flipped in rows *and*
columns. Flipping only the columns made compression stiffen an element.
"""

import math

import numpy as np
import pytest

from anastruct.fem.elements import geometric_stiffness_matrix

ANGLES = [0.0, 0.3, math.pi / 4, math.pi / 2, 2.0, math.pi]


@pytest.mark.parametrize("angle", ANGLES)
def test_geometric_stiffness_matrix_is_symmetric(angle):
    kg = geometric_stiffness_matrix(2.5, -100.0, angle, angle)
    np.testing.assert_allclose(kg, kg.T, atol=1e-12)


@pytest.mark.parametrize(
    "a1, a2", [(0.0, 0.0), (0.2, 0.5), (1.0, 1.3), (math.pi / 2, 0.4)]
)
def test_geometric_stiffness_forces_are_in_equilibrium(a1, a2):
    """The nodal forces Kg produces balance: node 2's x / y rows are the
    negatives of node 1's (rows 3 and 4 equal minus rows 0 and 1)."""
    kg = geometric_stiffness_matrix(2.5, -100.0, a1, a2)
    np.testing.assert_allclose(kg[3], -kg[0], atol=1e-12)
    np.testing.assert_allclose(kg[4], -kg[1], atol=1e-12)


@pytest.mark.parametrize("angle", ANGLES)
def test_compression_softens_and_tension_stiffens(angle):
    """Axial compression (N < 0) reduces the stiffness of every element
    DOF set it acts on; tension increases it."""
    compression = geometric_stiffness_matrix(2.5, -100.0, angle, angle)
    tension = geometric_stiffness_matrix(2.5, 100.0, angle, angle)
    assert np.all(np.linalg.eigvalsh(compression) <= 1e-9)
    assert np.all(np.linalg.eigvalsh(tension) >= -1e-9)
    assert np.linalg.eigvalsh(compression).min() < 0
    assert np.linalg.eigvalsh(tension).max() > 0


def test_horizontal_element_matches_the_reference_matrix():
    """For a horizontal element Kg is the reference N/l matrix on the
    transverse and rotational DOFs, in anastruct's y convention."""
    l, n = 2.0, -50.0
    kg = geometric_stiffness_matrix(l, n, 0.0, 0.0)
    reference = (
        n
        / l
        * np.array(
            [
                [6 / 5, l / 10, -6 / 5, l / 10],
                [l / 10, 2 * l**2 / 15, -l / 10, -(l**2) / 30],
                [-6 / 5, -l / 10, 6 / 5, -l / 10],
                [l / 10, -(l**2) / 30, -l / 10, 2 * l**2 / 15],
            ]
        )
    )
    flip = np.diag([-1, 1, -1, 1])  # anastruct's y is opposite the reference's
    expected = flip @ reference @ flip
    dofs = [1, 2, 4, 5]
    np.testing.assert_allclose(kg[np.ix_(dofs, dofs)], expected, atol=1e-12)
    assert not np.any(kg[[0, 3], :]) and not np.any(kg[:, [0, 3]])
