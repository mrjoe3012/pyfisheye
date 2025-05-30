import numpy as np
from typing import Optional
from pyfisheye.internal.utils.check_shapes import check_shapes
from pyfisheye.internal.utils.common import build_companion_matrix, get_logger

__all__ = ['project', 'backproject', 'project_fast']
__logger = get_logger()

@check_shapes({
    'points' : 'N*,3',
    'intrinsics' : '5',
    'scaling_matrix' : '2,2',
    'distortion_centre' : '2'
})
def project(points: np.ndarray, intrinsics: np.ndarray,
            distortion_centre: np.ndarray,
            scaling_matrix: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Project world points to pixel coordinates. Uses batched method for computing
        polynomial roots, but is still quite slow for large numbers of points.

    :param pixels: The pixels to project (in image coordinates.)
    :param distortion_centre: The camera's distortion centre in pixel coordinates.
    :param scaling_matrix: The scaling matrix, leave as None to use identity.
    :returns: Array with same shape as points except for last dimension having length two for
        the x-y pixel coordinates.
    """
    if scaling_matrix is None:
        scaling_matrix = np.eye(2, dtype=np.float64)
    if not np.issubdtype(points.dtype, np.floating):
        points = points.astype(np.float64)
    points_flat = points.reshape(-1, 3)
    norm = np.linalg.norm(points_flat[:, :2], axis=-1)
    coeffs = np.power(norm[:, None], [0, 1, 2, 3, 4]) * intrinsics
    coeffs[:, 1] -= points_flat[:, 2]
    companion = build_companion_matrix(coeffs)
    valid_companion_matrices = ~np.any(np.isnan(companion), axis=(1, 2))
    if not np.all(valid_companion_matrices):
        __logger.debug(f'{np.count_nonzero(~valid_companion_matrices)}/{companion.shape[0]}'
                       ' points have no projections! (NaNs in companion matrix)')
    roots = np.full(
        (companion.shape[0], companion.shape[1]),
        np.nan,
        dtype=np.complex64
    )
    if np.any(valid_companion_matrices):
        roots[valid_companion_matrices], _ = np.linalg.eig(companion[valid_companion_matrices])
    roots_real = np.real(roots)
    valid_roots_mask = np.isreal(roots) & (roots_real > 0)
    valid_roots_rows = np.where(np.any(valid_roots_mask, axis=-1))[0]
    valid_roots_column = np.argmax(valid_roots_mask[valid_roots_rows], axis=-1)
    lmbda = np.full((len(points_flat),), np.nan, points_flat.dtype)
    lmbda[valid_roots_rows] = roots_real[valid_roots_rows, valid_roots_column]
    if valid_roots_rows.shape[0] != points_flat.shape[0]:
        __logger.debug(f"{points_flat.shape[0] - valid_roots_rows.shape[0]} / "
                       f"{points_flat.shape[0]} points have no valid projections! "
                       "(no valid roots)")
    pixels_flat = lmbda[:, None] * points_flat[:, :2]
    pixels_flat = (pixels_flat + distortion_centre) @ np.linalg.inv(scaling_matrix).T
    pixels = pixels_flat.reshape(*points.shape[:-1], 2) 
    return pixels

@check_shapes({
    'points' : 'N*,3',
    'lookup_theta' : 'M',
    'lookup_rho' : 'M',
    'distortion_centre' : '2',
    'scaling_matrix' : '2,2'
})
def project_fast(points: np.ndarray,
                     lookup_theta: np.ndarray,
                     lookup_rho: np.ndarray,
                     distortion_centre: np.ndarray,
                     scaling_matrix: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Use a lookup table to perform a faster projection of 3D points onto the image space.

    :param points: The points to project, can be any shape as long as the last dimension
        has length 2.
    :param lookup_theta: The theta values in the lookup table in ascending order. Use
        build_inv_lookup_table() to compute.
    :param lookup_rho: The rho values in the lookup table. Use build_inv_lookup_table to
        compute.
    :param distortion_centre: Distortion centre in pixels.
    :param scaling_matrix: The scaling matrix. Leave as None to use the identity.
    :returns: The projected pixels.
    """
    if scaling_matrix is None:
        scaling_matrix = np.eye(2, dtype=np.float64)
    if not np.issubdtype(points.dtype, np.floating):
        points = points.astype(np.float64)
    points_flat = points.reshape(-1, 3)
    theta = np.acos(points_flat[:, 2] / np.linalg.norm(points_flat, axis=-1))
    rho = np.interp(theta, lookup_theta, lookup_rho, left=np.nan, right=np.nan)
    pixels_flat = (rho / np.linalg.norm(points_flat[:, :2], axis=-1))[:, None] * points_flat[:, :2]
    pixels_flat = (pixels_flat + distortion_centre) @ scaling_matrix.T
    pixels = pixels_flat.reshape(*points.shape[:-1], 2)
    return pixels

@check_shapes({
    'pixels' : 'N*,2',
    'distortion_centre' : '2',
    'stretch_matrix' : '2,2',
    'intrinsics' : '5'
})
def backproject(pixels: np.ndarray, intrinsics: np.ndarray,
            distortion_centre: np.ndarray, scaling_matrix: Optional[np.ndarray] = None,
            normalise_rays: bool = True) -> np.ndarray:
    """
    Backproject pixels into the world-space.

    :param pixels: The pixels to backproject. Any shape with last dimension having length 2.
    :param distortion_centre: The distortion centre x-y in pixels.
    :param intrinsics: The intrinsic parameters / polynomial coefficients in ascending order.
    :param scaling_matrix: The stretch matrix, leave as None to default to identity.
    :param normalise_rays: If set to True, all returned rays will have unit length.
    :returns: The projected rays with the same shape as pixels except for the last dimension
        having length 3.
    """
    if scaling_matrix is None:
        scaling_matrix = np.eye(2, dtype=np.float64)
    if not np.issubdtype(pixels.dtype, np.floating):
        pixels = pixels.astype(np.float64)
    pixels_flat = pixels.reshape(-1, 2)
    pixels_flat = (pixels_flat @ scaling_matrix.T) - distortion_centre
    rho = np.linalg.norm(pixels_flat, axis=-1)
    rays = np.concatenate(
        [
            pixels_flat,
            np.expand_dims(np.polyval(intrinsics[::-1], rho), -1)
        ],
        axis=-1
    )        
    if normalise_rays:
        rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    return rays.reshape(*pixels.shape[:-1], 3)
