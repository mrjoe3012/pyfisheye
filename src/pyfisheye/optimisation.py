from pyfisheye.utils.check_shapes import check_shapes
from pyfisheye.utils.common import unnormalize_coefficients
from scipy.optimize import minimize, LinearConstraint
from scipy.sparse import csc_matrix
from typing import Optional
import itertools
import numpy as np

class OptimRuntimeError(RuntimeError):
    def __init__(self, msg: Optional[str] = None) -> None:
        if msg is None:
            msg = 'Unexpected error.'
        super().__init__(msg)

@check_shapes({
    'pattern_observations' : 'n_obs,n_pat_pts,2',
    'pattern_world_coords' : 'n_pat_pts,3',
    'distortion_centre' : '2',
})
def partial_extrinsics(pattern_observations: np.ndarray,
                       pattern_world_coords: np.ndarray,
                       distortion_centre: np.ndarray) -> np.ndarray:
    """
    Computes four possible extrinsic configurations for each pattern observation by solving
        a system of linear homogenous equations.
    :param pattern_observations: Pixel coordinate pattern observations.
    :param pattern_world_coords: Pattern corner points in the pattern coordinate system (Z=0)
        for a single pattern.
    :param distortion_centre: x-y distortion centre.
    :returns: Extrinsic configuration as a N,4,3,3 array where the first 2 column are the first
        2 column vectors of the rotation matrix and the last column is the translation vector
        with the z-component set to NaN. The dimension with length 4 corresponds to the 4 possible
        solutions.
    """
    pattern_observations = pattern_observations - distortion_centre
    # generate the (empty) result N,3,3
    result = np.tile(
        np.full((3, 3), np.nan, dtype=pattern_observations.dtype),
        (pattern_observations.shape[0], 4, 1, 1) # four possible solutions per observation
    )
    # form the matrix containing the linear systems
    M = np.concatenate(
        [
            - pattern_observations[..., [1]] * pattern_world_coords[..., :2],
            pattern_observations[..., [0]] * pattern_world_coords[..., :2],
            -pattern_observations[..., [1]], pattern_observations[..., [0]]
        ],
        axis=-1
    )
    # batch compute the solutions
    Vh = np.linalg.svd(M)[2]
    H = Vh.swapaxes(1 ,2)[..., -1]
    B1 = np.sum(H[..., 0:4:2] ** 2, axis=-1) - np.sum(H[..., 1:4:2] ** 2, axis=-1)
    B2 = (H[..., None, 0:4:2] @ H[..., 1:4:2, None]).squeeze((-1, -2))
    B3 = B2 ** 2
    # compute R_31 and R_32 and scale solution by lambda
    # construct 4 possible solutions for each observation
    # - positive / negative R_13
    # - positive / negative lambda
    for i in range(pattern_observations.shape[0]):
        roots = np.roots(
            [
                1, B1[i], -B3[i]
            ]
        )
        real_roots = np.real(roots[np.isclose(np.imag(roots), 0)])
        positive_roots = real_roots[real_roots > 0]
        if len(positive_roots) != 1:
            raise OptimRuntimeError("Unexpected error.")
        for sol_idx, (lambda_sign, r_13_sign) in enumerate(itertools.product([1, -1], repeat=2)):
            result[i, sol_idx, :2, :2] = H[i][:4].reshape(2, 2)
            result[i, sol_idx, :2, -1] = H[i][-2:]
            result[i, sol_idx, 2, 0] = r_13_sign * positive_roots[0] ** 0.5
            result[i, sol_idx, 2, 1] = - B2[i] / result[i, sol_idx, 2, 0]
            result[i, sol_idx] /= lambda_sign * np.linalg.norm(result[i, sol_idx, :, 0])
    return result

@check_shapes({
    'pattern_observations' : 'n_obs,n_pat_pts,2',
    'pattern_world_coords' : 'n_pat_pts,3',
    'distortion_centre' : '2',
    'extrinsics' : 'n_obs,4,3,3'
})
def select_best_extrinsic_solution(pattern_observations: np.ndarray,
                                   pattern_world_coords: np.ndarray,
                                   distortion_centre: np.ndarray,
                                   extrinsics: np.ndarray,
                                   image_radius: float) -> np.ndarray:
    """
    Selects the extrinsic solution in the same quadrant as its corresponding observation
        and resulting in a polynomial fit which tends to positive infinity.
    :param pattern_observations: Pixel coordinate pattern observations.
    :param pattern_world_coords: Pattern corner points in the pattern coordinate system (Z=0)
        for a single pattern.
    :param distortion_centre: x-y distortion centre.
    :param extrinsics: 4 extrinsic solutions per observation.
    :param image_radius: The image radius in pixels.
    :returns: N,3,3 array containing the best solution for each observation.
    """
    pattern_observations_norm = pattern_observations - distortion_centre
    # first, select the solutions in the correct quadrant
    best_quadrant_index = np.argmin(
        np.linalg.norm(
            extrinsics[..., :2, -1] - pattern_observations_norm[:, None, 0],
            axis=-1
        ),
        axis=1
    )
    mask_having_right_quadrant = np.all(
        np.sign(extrinsics[..., :2, -1]) == np.sign(
            extrinsics[np.arange(len(extrinsics)), best_quadrant_index, :2, -1]
        )[:, None, :],
        axis=-1
    )
    if not np.all(np.any(mask_having_right_quadrant, axis=-1)):
        raise OptimRuntimeError()
    extr_correct_quadrant = extrinsics[np.where(mask_having_right_quadrant)]
    extr_correct_quadrant = extr_correct_quadrant.reshape(pattern_observations_norm.shape[0], 2, 3, 3)
    # now select the solution resulting in a polynomial tending to +ve infinity
    selection: list[int] = []
    for observation_idx in range(len(pattern_observations_norm)):
        best_solution = None
        for solution_idx in range(len(extrinsics[observation_idx])):
            intrinsics_norm, _ = intrinsics_and_z_translation(
                pattern_observations[[observation_idx]], pattern_world_coords, distortion_centre,
                extrinsics[[observation_idx], solution_idx], image_radius,
                monotonic=False,
            )
            if intrinsics_norm[-1] >= 0:
                best_solution = solution_idx
                break
        if best_solution is None:
            raise OptimRuntimeError()
        selection.append(best_solution)
    result = extr_correct_quadrant[np.arange(len(extr_correct_quadrant)), selection]
    return result

@check_shapes({
    'pattern_observations' : 'n_obs,n_pat_pts,2',
    'pattern_world_coords' : 'n_pat_pts,3',
    'distortion_centre' : '2',
    'extrinsics' : 'n_obs,3,3'
})
def intrinsics_and_z_translation(pattern_observations: np.ndarray,
                       pattern_world_coords: np.ndarray,
                       distortion_centre: np.ndarray,
                       extrinsics: np.ndarray,
                       image_radius: float,
                       monotonic: bool = False,
                       num_rho_samples: int = 250) -> tuple[np.ndarray, np.ndarray]:
    """
    :param pattern_observations: Pixel coordinate pattern observations.
    :param pattern_world_coords: Pattern corner points in the pattern coordinate system (Z=0)
        for a single pattern.
    :param distortion_centre: x-y distortion centre.
    :param extrinsics: 4 extrinsic solutions per observation.
    :param image_radius: The image radius in pixels.
    :param monotonic: If False, then the solution will be computed using least squares via
        pseudoinverse. Otherwise, a solver supporting inequality constraints will be used
        instead to enforce monotonicty, i.e. f'(rho) >= 0 for all rho.
    :param num_rho_samples: Matters only when monotonic=True. Determines the number of samples
        used to generate the constraint matrix. This can have a signficiant impact on the convergence
        time.
    :returns: The intrinsic parameters (5,) in ascending order of power followed by the
        z-translation for each observation.
    """
    pattern_observations = pattern_observations - distortion_centre
    if len(pattern_observations.shape) == 2:
        pattern_observations = pattern_observations.reshape(-1, *pattern_observations.shape)
    extrinsics = extrinsics.reshape(-1, 3, 3)
    # compute rho, the radial distance and normalize it
    rho = np.linalg.norm(pattern_observations, axis=-1)
    rho /= image_radius
    # setup A, B, C & D - Terms in the linear system of equations
    A = np.sum(
        pattern_world_coords[:, :2] * extrinsics[:, None, 1, :2],
        axis=-1
    ) + extrinsics[:, 1, [-1]]
    B = pattern_observations[..., 1] * np.sum(
        pattern_world_coords[:, :2] * extrinsics[:, None, 2, :2],
        axis=-1
    )
    C = np.sum(
        pattern_world_coords[:, :2] * extrinsics[:, None, 0, :2],
        axis=-1
    ) + extrinsics[:, 0, [-1]]
    D = pattern_observations[..., 0] * np.sum(
        pattern_world_coords[:, :2]  * extrinsics[:, None, 2, :2],
        axis=-1
    )
    # put the linear system in matrix form M * H = b
    poly_rho = np.stack(
        [
            np.ones_like(rho), rho ** 2,
            rho ** 3, rho ** 4
        ],
        axis=-1
    )
    M = np.concatenate(
        [
            np.stack(
                [
                    A[..., None] * poly_rho,
                    C[..., None] * poly_rho
                ],
                axis=-1
            ).swapaxes(-1, -2).reshape(-1, 4),
            np.stack(
                [
                    np.stack(
                        [np.diag(-a[:, 1]) for a in pattern_observations.swapaxes(0, 1)],
                        axis=0
                    ),
                    np.stack(
                        [np.diag(-a[:, 0]) for a in pattern_observations.swapaxes(0, 1)],
                        axis=0
                    ),
                ],
                axis=-1
            ).transpose(1, 0, 3, 2).reshape(-1, pattern_observations.shape[0]),
        ],
        axis=-1
    )
    b = np.stack(
        [
            B, D
        ],
        axis=-1
    ).flatten()
    if monotonic:
        # setup the montonicity constraint Nx >= 0
        rho_samples = np.linspace(
            np.finfo(pattern_observations.dtype).eps,
            image_radius,
            num_rho_samples
        )[..., None]
        rho_samples /= image_radius
        N = np.concatenate(
            [
                np.zeros(
                    (num_rho_samples, 1)
                ),
                2 * rho_samples,
                3 * rho_samples ** 2,
                4 * rho_samples ** 3,
                np.zeros(
                    (num_rho_samples, extrinsics.shape[0])
                )
            ],
            axis=-1
        )
        # precompute constant hessian matrix
        hess = 2 * M.T @ M
        # precompute constant part of jacobian
        jac_const = 2 * b.T @ M
        jac_scale = 2 * M.T @ M
        result = minimize( # type: ignore
            fun=lambda x: np.sum((M @ x - b) ** 2),
            x0=np.linalg.lstsq(M, b)[0].tolist(),
            method='trust-constr',
            constraints=[LinearConstraint(csc_matrix(N), lb=0)],
            options={
                'maxiter' : 10_000,
                'disp' : True,
            },
            jac=lambda x: jac_scale @ x - jac_const,
            hess=lambda x: hess
        )
        H = result.x
    else:
        H = np.linalg.lstsq(M, b)[0]
    intrinsics = unnormalize_coefficients(
        np.array([
            H[0], 0, *H[1:4]
        ]),
        image_radius
    )
    z_translation = H[4:]
    return intrinsics, z_translation

@check_shapes({
    'intrinsics' : '5',
    'image_radius' : '2',
})
def build_inv_lookup_table(intrinsics: np.ndarray, image_radius: np.ndarray,
                           n_samples: int = 10_000) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a lookup table for the inverse polynomial f(theta) = rho. This can be used to backproject
        points more quickly.
    :param intrinsics: The polynomial coefficients for the model in ascending order.
    :param image_radius: The image radius in pixels.
    :param n_samples: How many discrete samples should be included in the lookup table.
    :returns: A tuple containing the theta and rho values, sorted in ascending order by theta. Use
        np.interp to evaluate the lookup table.
    """
    samples = np.linspace(0, image_radius, n_samples)
    rho = np.sqrt(2) * samples
    f_rho = np.polyval(intrinsics[::-1], rho)
    theta = np.acos(f_rho / np.sqrt(f_rho ** 2 + 2 * samples ** 2))
    sorted_indices = np.argsort(theta)    
    return theta[sorted_indices], rho[sorted_indices]
