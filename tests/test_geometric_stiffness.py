"""Geometric stiffness matrix Kg used by ``solve(geometrical_non_linear=True)``.

The reference matrix (H. P. Gavin, CEE 421 notes, linked in elements.py) has
y up; anastruct's y is opposite, so the y DOFs must be flipped in rows *and*
columns. Flipping only the columns made compression stiffen an element.

An inclined roller rotates the DOFs of one node, so the element's two end
angles a1 and a2 differ. Kg must then transform like the elastic stiffness
matrix: each node's DOFs rotated by its own angle.
"""

import math

import numpy as np
import pytest

from anastruct.fem.elements import (
    constitutive_matrix,
    geometric_stiffness_matrix,
    kinematic_matrix,
    stiffness_matrix,
)

ANGLES = [0.0, 0.3, math.pi / 4, math.pi / 2, 2.0, math.pi]
# (a1, a2) pairs: equal angles, then one end rotated as by an inclined roller
END_ANGLES = [(a, a) for a in ANGLES] + [
    (0.0, math.pi),
    (0.2, 0.9),
    (math.pi / 2, 0.0),
    (2.0, -1.0),
]


@pytest.mark.parametrize("a1, a2", END_ANGLES)
def test_geometric_stiffness_matrix_is_symmetric(a1, a2):
    kg = geometric_stiffness_matrix(2.5, -100.0, a1, a2)
    np.testing.assert_allclose(kg, kg.T, atol=1e-12)


def _node_rotation(d1: float, d2: float) -> np.ndarray:
    """Rotate the x / y DOFs of node 1 by d1 and node 2 by d2."""
    t = np.eye(6)
    for i, d in ((0, d1), (3, d2)):
        t[i : i + 2, i : i + 2] = [
            [math.cos(d), -math.sin(d)],
            [math.sin(d), math.cos(d)],
        ]
    return t


def _elastic_stiffness(a1: float, a2: float, l: float = 2.5) -> np.ndarray:
    return stiffness_matrix(
        constitutive_matrix(1000.0, 500.0, l, None, False, False),
        kinematic_matrix(a1, a2, l),
    )


@pytest.mark.parametrize("angle", ANGLES)
@pytest.mark.parametrize(
    "d1, d2", [(0.0, math.pi), (0.0, -math.pi / 2), (0.7, 0.0), (0.4, -1.1)]
)
def test_rotating_a_node_transforms_kg_like_the_elastic_stiffness(angle, d1, d2):
    """K(angle + d1, angle + d2) = Tᵀ K(angle, angle) T for the elastic
    stiffness, with T rotating each node's x / y DOFs. Kg must obey the same
    rule, or it is inconsistent with the elastic stiffness on inclined rollers."""
    t = _node_rotation(d1, d2)
    np.testing.assert_allclose(
        _elastic_stiffness(angle + d1, angle + d2),
        t.T @ _elastic_stiffness(angle, angle) @ t,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        geometric_stiffness_matrix(2.5, -100.0, angle + d1, angle + d2),
        t.T @ geometric_stiffness_matrix(2.5, -100.0, angle, angle) @ t,
        atol=1e-9,
    )


@pytest.mark.parametrize(
    "a1, a2", [(0.0, 0.0), (0.2, 0.5), (1.0, 1.3), (math.pi / 2, 0.4)]
)
def test_geometric_stiffness_forces_are_in_equilibrium(a1, a2):
    """The nodal forces Kg produces balance: once node 2's forces are rotated
    into node 1's frame, its x / y rows are the negatives of node 1's. When
    a1 != a2 (inclined roller) the raw rows are in different frames."""
    kg = _node_rotation(0.0, a2 - a1) @ geometric_stiffness_matrix(2.5, -100.0, a1, a2)
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
