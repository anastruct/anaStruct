import copy
import logging
import math
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np

from anastruct.basic import FEMException
from anastruct.fem.elements import PLASTIC_MECHANISM_STIFFNESS
from anastruct.fem.system_components.assembly import assemble_system_matrix

if TYPE_CHECKING:
    from anastruct.fem.system import SystemElements


def second_order_step(system: "SystemElements") -> None:
    """
    One linear step of a second order analysis with the current stiffness of the elements
    (which may be reduced by plastic hinges): a first order solution gives the normal
    forces N0, the geometric stiffness for N0 is added to the elements and the system
    is solved again, (K + Kg(N0)) u = F. The results stay on the system, without the
    post-processing.

    :param system: (SystemElements)
    """
    for el in system.element_map.values():
        el.compile_stiffness_matrix()  # the plastic reduction stays, the old Kg goes
    system.solve(True, naked=True)
    system.post_processor.node_results_elements()
    for el in system.element_map.values():
        el.N_1 = math.sin(el.angle) * el.node_1.Fy - math.cos(el.angle) * el.node_1.Fx
        el.N_2 = -math.sin(el.angle) * el.node_2.Fy + math.cos(el.angle) * el.node_2.Fx
        el.compile_geometric_non_linear_stiffness_matrix()
    system.solve(True, naked=True)


def stiffness_adaptation(
    system: "SystemElements",
    verbosity: int,
    max_iter: int,
    geometrical_non_linear: bool = False,
) -> np.ndarray:
    """
    Non linear solver for the nodes by adapting the stiffness of the elements (nodes).

    :param geometrical_non_linear: (bool) Every step of the iteration is a second order
                                   analysis, see :func:`second_order_step`.
    :return: Vector with displacements.
    """

    def solve_step() -> None:
        try:
            if geometrical_non_linear:
                second_order_step(system)
            else:
                system.solve(True, naked=True)
        except np.linalg.LinAlgError as error:
            raise FEMException(
                "ConvergenceError",
                "The structure is a mechanism: the plastic hinges can not carry the loads "
                "(check the plastic moments mp and the loads).",
            ) from error

    # start from the elastic stiffness: the plastic reduction of an earlier calculation
    # would otherwise be the starting point of this one
    system._reset_plastic_state()

    solve_step()
    if verbosity == 0:
        logging.info("Starting stiffness adaptation calculation.")

    # check validity
    assert all(
        mp > 0 for mpd in system.non_linear_elements.values() for mp in mpd.values()
    ), "Cannot solve for an mp = 0. If you want a hinge set the spring stiffness equal to 0."

    iteration = 0
    # (stiffness of the plastic hinge, |M|) of the previous iteration, per element end
    history: Dict[Tuple[int, int], Tuple[float, float]] = {}
    while iteration < max_iter:
        factors = []

        # update the elements stiffnesses
        for k, v in system.non_linear_elements.items():
            el = system.element_map[k]
            assert el.element_force_vector is not None

            for node_no, mp in v.items():
                if node_no == 1:
                    # Fast Tz
                    m_e = (
                        el.element_force_vector[2] + el.element_primary_force_vector[2]
                    )
                else:
                    # Fast Tz
                    m_e = (
                        el.element_force_vector[5] + el.element_primary_force_vector[5]
                    )

                # the end is plastic when the moment exceeds mp; a plastic hinge is a
                # spring in series, its stiffness follows from the secant: k <- k * mp / |M|
                # (a moment below mp makes it stiffer, until the end is elastic again)
                current = el.plastic_stiffness[node_no - 1]
                if current is None and abs(m_e) <= mp:
                    continue
                factor = mp / abs(m_e) if m_e != 0 else 1.0e3
                if current is not None:
                    # the moment is nearly linear in the flexibility 1/k of the spring: with
                    # two iterates, a secant step in the flexibility converges much faster
                    # than the plain secant when mp is close to the elastic moment
                    previous = history.get((k, node_no))
                    history[(k, node_no)] = (current, abs(m_e))
                    if previous is not None and previous[0] != current:
                        slope = (abs(m_e) - previous[1]) / (
                            1 / current - 1 / previous[0]
                        )
                        if slope < 0:
                            flexibility = 1 / current + (mp - abs(m_e)) / slope
                            if flexibility > 0:
                                factor = min(
                                    max(1 / (current * flexibility), 0.01), 100
                                )
                else:
                    history.pop((k, node_no), None)
                el.update_stiffness(factor, node_no)
                updated = el.plastic_stiffness[node_no - 1]
                if updated is None:
                    # the end became elastic again: the structure has to be solved again
                    factors.append(0.0)
                else:
                    if updated < PLASTIC_MECHANISM_STIFFNESS * 4 * el.EI / el.l:
                        raise FEMException(
                            "ConvergenceError",
                            "The structure is a mechanism: the plastic hinges can not carry "
                            "the loads (check the plastic moments mp and the loads).",
                        )
                    factors.append(factor)

        if not np.allclose(factors, 1, 1e-3):
            solve_step()
        else:
            system.post_processor.node_results_elements()
            system.post_processor.node_results_system()
            system.post_processor.reaction_forces()
            system.post_processor.element_results()
            break
        iteration += 1

    if geometrical_non_linear:
        # the results are second order; the elements go back to the first order stiffness
        for el in system.element_map.values():
            el.compile_stiffness_matrix()

    if iteration >= max_iter:
        # the iteration of the last step has no post-processed results, and the results of
        # an earlier calculation must not be mistaken for them
        raise FEMException(
            "ConvergenceError",
            f"The plastic iteration did not converge in max_iter={max_iter} iterations. "
            "The structure may be a mechanism (check the plastic moments mp and the loads), "
            "or the iteration needs more iterations: increase max_iter.",
        )
    if verbosity == 0:
        logging.info(f"Solved in {iteration} iterations")

    assert system.system_displacement_vector is not None
    return system.system_displacement_vector


def smallest_buckling_factor(k0: np.ndarray, kg: np.ndarray) -> float:
    """
    Solve the generalized eigenvalue problem (K0 + λ Kg) x = 0 and return the smallest
    positive λ.

    K0 is the (symmetric, positive definite) reduced stiffness matrix of the first order
    analysis. Kg is the reduced geometric stiffness matrix for the first order normal
    forces N0. Compression makes Kg negative, so a buckling mode, K0 x = -λ Kg x, has a
    positive λ for the loads that cause compression. A tensile normal force stiffens the
    structure and gives a negative λ, which is no buckling mode.

    Kg is singular when some members carry no normal force, so it is not inverted. The
    problem is solved for ν = -1 / λ, the eigenvalues of K0⁻¹ Kg.

    Args:
        k0 (np.ndarray): Reduced stiffness matrix of the first order analysis.
        kg (np.ndarray): Reduced geometric stiffness matrix for the first order normal forces.

    Returns:
        float: The smallest positive factor λ, or infinity when there is no buckling mode (no compression).
    """
    if k0.size == 0:
        # every degree of freedom is restrained: nothing can buckle
        logging.warning("There are no free degrees of freedom: no buckling mode.")
        return float("inf")

    try:
        # K0 = L Lᵀ: ν are the eigenvalues of the symmetric matrix L⁻¹ Kg L⁻ᵀ
        chol = np.linalg.cholesky(k0)
        half = np.linalg.solve(chol, kg)
        symmetric = np.linalg.solve(chol, half.T).T
        nu = np.linalg.eigvalsh((symmetric + symmetric.T) / 2)
    except np.linalg.LinAlgError:
        # the stiffness matrix is not positive definite: the general eigenvalue problem
        try:
            nu = np.linalg.eigvals(np.linalg.solve(k0, kg)).real
        except np.linalg.LinAlgError as error:
            raise FEMException(
                "StabilityError",
                "The stiffness matrix is singular, the buckling factor can not be "
                "determined. Check your support conditions.",
            ) from error

    # buckling modes have a negative ν (compression); ignore ν = 0 (no geometric effect)
    compressive = nu[nu < -1e-12 * max(float(np.max(np.abs(nu))), 1e-300)]
    if compressive.size == 0:
        logging.warning(
            "There is no buckling mode: no compressive normal force acts on the free "
            "degrees of freedom, so the buckling factor is infinite."
        )
        return float("inf")
    return float(np.min(-1.0 / compressive))


def det_linear_buckling(system: "SystemElements") -> float:
    """
    Determine linear buckling by solving the generalized eigenvalue problem (K0 + λKg0)x = 0.

    geometrical stiffness matrix at buckling point: Kg = f(N_max)
    1st order forces: N0
    Nmax = λN0
    Kg(Nmax) = λKg(N0) = λKg0

    2nd order analysis is solved by:
    (K0 + λKg0)U = F

    We are interested in the point that there is no additional load F and displacement U
    is possible.
    (K0 + λKg0)ΔU = ΔF = 0

    Is the generalized eigenvalue problem:
    (K0 + λKg0)x = 0, see :func:`smallest_buckling_factor`.

    :return: The factor the loads can be increased until the structure fails due to buckling.
    """
    # first order analysis: the stiffness matrix and the normal forces N0. The elements
    # may still hold the geometric stiffness of an earlier second order calculation.
    for el in system.element_map.values():
        el.compile_stiffness_matrix()
    system.solve(force_linear=True)  # also for non linear (plastic) elements
    free = list(system._remainder_indexes)
    k0 = np.array(system.reduced_system_matrix)  # copy

    # the same matrix with the geometric stiffness of the elements added
    for el in system.element_map.values():
        el.compile_geometric_non_linear_stiffness_matrix()
    assemble_system_matrix(system)
    assert system.system_matrix is not None
    kg = np.array(system.system_matrix)[np.ix_(free, free)] - k0

    # the elements are used again by the caller: back to the first order stiffness
    for el in system.element_map.values():
        el.compile_stiffness_matrix()

    return smallest_buckling_factor(k0, kg)


def determine_buckling_factor(
    system: "SystemElements", discretize_kwargs: Optional[dict] = None
) -> float:
    """
    Determine the buckling factor of the (elastic) structure for its loads.

    :param system: (SystemElements)
    :param discretize_kwargs: (dict) Containing the kwargs passed to the discretize function
    :return: The factor the loads can be increased until the structure fails due to buckling.
    """
    buckling_system = copy.copy(system)
    if discretize_kwargs is not None:
        buckling_system.discretize(**discretize_kwargs)

    # the elements may be shared with the system, and hold the plastic reduction of an
    # earlier calculation: use the elastic stiffness, and restore the elements afterwards
    states = [
        (el, el.constitutive_matrix, list(el.nodes_plastic), list(el.plastic_stiffness))
        for el in buckling_system.element_map.values()
    ]
    try:
        for el, _, _, _ in states:
            el.reset_plastic()
            el.refresh_nodes(buckling_system.node_map)
            el.compile_constitutive_matrix()
            el.compile_stiffness_matrix()
        return det_linear_buckling(buckling_system)
    finally:
        for el, constitutive, nodes_plastic, plastic_stiffness in states:
            el.constitutive_matrix = constitutive
            el.nodes_plastic = nodes_plastic
            el.plastic_stiffness = plastic_stiffness
            el.compile_stiffness_matrix()


def geometrically_non_linear(
    system: "SystemElements",
    verbosity: int = 0,
    return_buckling_factor: bool = True,
    discretize_kwargs: Optional[dict] = None,
) -> Optional[float]:
    """

    :param system: (SystemElements)
    :param verbosity: (int)
    :param return_buckling_factor: (bool)
    :param discretize_kwargs: (dict) Containing the kwargs passed to the discretize function
    :param discretize: (function) discretize function.
    :return: buckling_factor: (flt) The factor the loads can be increased until the structure
                              fails due to buckling.
    """
    # https://www.ethz.ch/content/dam/ethz/special-interest/baug/ibk/structural-mechanics-dam/education/femI/Lecture_2b.pdf
    if verbosity == 0:
        logging.info("Starting geometrical non linear calculation")

    buckling_factor: Optional[float] = None
    if return_buckling_factor:
        buckling_factor = determine_buckling_factor(system, discretize_kwargs)

    # first order analysis: the normal forces N0 (back to the first order stiffness,
    # the elements may hold the geometric stiffness of an earlier calculation)
    for el in system.element_map.values():
        el.compile_stiffness_matrix()
    system.solve(force_linear=True)

    # second order analysis
    for el in system.element_map.values():
        el.compile_geometric_non_linear_stiffness_matrix()

    system.solve(force_linear=True)

    # the results are second order; the elements go back to the first order stiffness, so
    # that a next (linear) calculation, or another second order calculation, starts from it
    for el in system.element_map.values():
        el.compile_stiffness_matrix()

    return buckling_factor
