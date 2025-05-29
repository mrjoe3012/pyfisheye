from pyfisheye.internal.utils.check_shapes import check_shapes
from pyfisheye.internal.utils.common import get_3d_transformation, get_logger
from pyfisheye.internal.optimisation.optim_result import OptimResult
from pyfisheye.internal.projection import project
from typing import Optional
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
import numpy as np

__all__ = ['nonlinear_refinement']
__logger = get_logger()

def __unpack_arguments(N: int,
                       params: np.ndarray) -> tuple[np.ndarray, np.ndarray,
                                                    np.ndarray, np.ndarray]:
    """
    :returns: extrinsics, intrinsics, stretch_matrix, distortion_centre
    """ 
    num_quat_elems = 4 * N
    num_trans_elems = 3 * N
    num_intr_elems = 4
    num_scale_elems = 3
    num_dist_centre_elems = 2
    quats = params[:num_quat_elems].reshape(-1, 4)
    rot = Rotation.from_quat(quats).as_matrix()
    trans = params[num_quat_elems:num_quat_elems + num_trans_elems].reshape(-1, 3, 1)
    extr = np.concatenate([rot[..., :2], trans], axis=-1)
    intr_begin = num_quat_elems + num_trans_elems
    intr = params[intr_begin:intr_begin + num_intr_elems]
    intr = np.array([intr[0], 0, *intr[1:]])
    dist_centre_begin = intr_begin + num_intr_elems
    dist_centre = params[dist_centre_begin:dist_centre_begin + num_dist_centre_elems]
    scale_begin = dist_centre_begin + num_dist_centre_elems
    scale_mat = np.array([*params[scale_begin:scale_begin + num_scale_elems], 1]).reshape(2, 2)
    return extr, intr, scale_mat, dist_centre

def __pack_arguments(extr: np.ndarray, intr: np.ndarray, scale: np.ndarray,
            dist_centre: np.ndarray) -> np.ndarray:
    rot, trans = get_3d_transformation(extr)
    quats = Rotation.from_matrix(rot).as_quat()
    c, d, e = scale[0, 0], scale[0, 1], scale[1, 0]
    params = np.concatenate(
        [
            quats.flatten(),
            trans.flatten(),
            intr[[0, 2, 3, 4]],
            dist_centre,
            [c, d, e],
        ],
        axis=0
    ) 
    return params

@check_shapes({
    'pattern_observations' : 'N,M,2',
    'pattern_world_coords' : 'M,3',
    'distortion_centre' : '2',
    'extrinsics' : 'N,3,3',
    'intrinsics' : '5',
    'stretch_matrix' : '2,2'
})
def nonlinear_refinement(pattern_observations: np.ndarray,
                         pattern_world_coords: np.ndarray,
                         distortion_centre: np.ndarray,
                         extrinsics: np.ndarray,
                         intrinsics: np.ndarray,
                         stretch_matrix: np.ndarray,
                         wnls_threshold: float) -> OptimResult:
    """
    Use LM to perform a nonlinear refinement of all calibration parameters at once.
        Use Huber's function for robustness to outliers.

    :param pattern_observations:
    :param pattern_world_coords:
    :param distortion_centre:
    :param extrinsics:
    :param intrinsics:
    :param stretch_matrix:
    :returns: Optimisation result.
    """
    __logger.debug(f"Starting nonlinear refinement for {pattern_observations.shape[0]}"
                   " observations.")
    prev_residuals: Optional[np.ndarray] = None
    first_iteration = True
    def loss(params: np.ndarray) -> np.ndarray:
        nonlocal prev_residuals, first_iteration
        extr, intr, scale_mat, dist_centre = __unpack_arguments(
            pattern_observations.shape[0],
            params
        )
        transformed_world_coords = \
            (extr[:, None, :, :2] @ pattern_world_coords[None, :, :2, None]).squeeze(-1)
        transformed_world_coords += extr[:, None, :, 2]
        projected_pixels = project(
            transformed_world_coords,
            intr,
            dist_centre,
            scale_mat
        )
        residuals = (projected_pixels - pattern_observations).flatten()
        # Weighted least squares using Huber's function
        if prev_residuals is None:
            weights = np.array(1.0)
        else:
            abs_res = np.abs(prev_residuals)
            weights = wnls_threshold / abs_res
            weights = np.where(abs_res <= wnls_threshold, 1.0, weights)
        prev_residuals = residuals.copy()
        if first_iteration:
            residuals[np.isnan(residuals)] = 1000
            first_iteration = False
        return weights * residuals
    result = least_squares(  # type: ignore
        fun=loss,
        x0=__pack_arguments(extrinsics, intrinsics, stretch_matrix, distortion_centre),
        method='lm'
    )
    if not result['success']:
        raise RuntimeWarning("Nonlinear refinement failed to converge.")
    __logger.debug(f"Finished refinement in {result['nfev']} iterations. "
                   f"loss={result['cost']:.2f} message='{result['message']}'")
    extrinsics, intrinsics, stretch_matrix, distortion_centre = __unpack_arguments(
        pattern_observations.shape[0],
        result.x
    )
    return OptimResult(
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        dist_centre=distortion_centre,
        scaling_mat=stretch_matrix
    )