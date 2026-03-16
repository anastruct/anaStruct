import copy
import logging
from typing import TYPE_CHECKING, Optional

import numpy as np
from anastruct.basic import converge

if TYPE_CHECKING:
    from anastruct.fem.system import SystemElements


def stiffness_adaptation(
    system: "SystemElements", verbosity: int, max_iter: int
) -> np.ndarray:
    """
    Non linear solver for the nodes by adapting the stiffness of the elements (nodes).

    :return: Vector with displacements.
    """
    system.solve(True, naked=True)
    if verbosity == 0:
        logging.info("Starting stiffness adaptation calculation.")

    # check validity
    assert all(
        mp > 0 for mpd in system.non_linear_elements.values() for mp in mpd
    ), "Cannot solve for an mp = 0. If you want a hinge set the spring stiffness equal to 0."

    iteration = 0
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

                if abs(m_e) > mp:
                    el.nodes_plastic[node_no - 1] = True
                if el.nodes_plastic[node_no - 1]:
                    factor = converge(m_e, mp)
                    factors.append(factor)
                    el.update_stiffness(factor, node_no)

        if not np.allclose(factors, 1, 1e-3):
            system.solve(force_linear=True, naked=True)
        else:
            system.post_processor.node_results_elements()
            system.post_processor.node_results_system()
            system.post_processor.reaction_forces()
            system.post_processor.element_results()
            break
        iteration += 1

    if iteration >= max_iter:
        logging.warning(
            f"Couldn't solve the in the amount of iterations given. max_iter={max_iter}"
        )
    elif verbosity == 0:
        logging.info(f"Solved in {iteration} iterations")

    assert system.system_displacement_vector is not None
    return system.system_displacement_vector


def det_linear_buckling(system: "SystemElements") -> float:
    """
    Determine linear buckling by solving the generalized eigenvalue problem (k -λkg)x = 0.

    geometrical stiffness matrix at buckling point: Kg = f(N_max)
    1st order forces: N0
    Nmax = λN0
    Kg(Nmax) = λ(Kg(N0) = λKg0

    2nd order analysis is solved by:
    (K + λKg0)U = F

    We are interested in the point that there is nog additional load F and displacement U
    is possible.
    (K + λKg0)ΔU = ΔF = 0
    (K + λKg0) = 0

    Is the generalized eigenvalue problem:
    (A - λB)x = 0

    :return: The factor the loads can be increased until the structure fails due to buckling.
    """
    system.solve()

    # buckling
    k0 = np.array(system.reduced_system_matrix)  # copy

    for el in system.element_map.values():
        el.compile_geometric_non_linear_stiffness_matrix()
        el.reset()

    system.solve()
    kg = system.reduced_system_matrix - k0
    # solve (k -λkg)x = 0

    # The following two lines are equivalent to the cleaner
    # scipy.linalg.eigvals(k0, kg), but doing this allows us
    # to drop the scipy dependency
    invkg_k0 = np.linalg.inv(kg) * k0
    eigenvalues = np.abs(np.linalg.eigvals(invkg_k0))
    return float(np.min(eigenvalues))


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
        buckling_system = copy.copy(system)
        if discretize_kwargs is not None:
            buckling_system.discretize(**discretize_kwargs)

        buckling_factor = det_linear_buckling(buckling_system)

    system.solve()

    for el in system.element_map.values():
        el.compile_geometric_non_linear_stiffness_matrix()

    system.solve()

    return buckling_factor
