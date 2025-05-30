import numpy as np
from typing import Optional
from pyfisheye.internal.utils.check_shapes import check_shapes
from rich.logging import RichHandler
from functools import cache
import logging

@check_shapes({
    'distortion_centre' : '2'
})
def compute_image_radius(image_width: int, image_height: int,
                         distortion_centre: np.ndarray) -> float:
    """
    Compute the maximum L2 norm between a pixel in the image and the distortion centre
        (for normalization).

    :param image_height:
    :param image_width:
    :param distortion_centre: x-y distortion centre.
    :returns: The image radius being the maximum L2 norm between any point on the image and the
        distortion centre.
    """
    return max(
        np.linalg.norm(distortion_centre).item(),
        np.linalg.norm([image_width, image_height] - distortion_centre).item()
    ) 

def generate_pattern_world_coords(num_rows: int, num_cols: int,
                                  pattern_tile_size_x: float,
                                  pattern_tile_size_y: Optional[float] = None) -> np.ndarray:
    """
    Generate a num_rows*num_cols,3 array containing the pattern coordinates in world space
        with Z = 0. They are returned in row-major ordering starting at (0, 0, 0).

    :param num_rows: The number of rows in the pattern - the number of corners in one column.
    :param num_cols: The number of columns in the pattern - the number of corners in one row.
    :param pattern_tile_size_x: The width of one tile in the pattern in metres.
    :param pattern_tile_size_y: The height of one tile in the pattern in metres. Leave as None
        to default to pattern_tile_size_x.
    :returns: num_rows*num_cols,3 array of pattern world coordinates with Z=0. (XYZ) ordering.
    """
    if pattern_tile_size_y is None:
        pattern_tile_size_y = pattern_tile_size_x
    pattern_world_coords = np.concatenate(
        [
            np.stack(
                np.meshgrid(
                    np.arange(num_cols) * pattern_tile_size_x,
                    np.arange(num_rows) * pattern_tile_size_y
                ),
                axis=-1
            ),
            np.zeros((num_rows, num_cols, 1))
        ],
        axis=-1
    ).astype(np.float64).reshape(-1, 3)
    return pattern_world_coords

@check_shapes({
    'coefficients' : '5'
})

def unnormalize_coefficients(coefficients: np.ndarray, image_radius: float) -> np.ndarray:
    """
    Convert the coefficients provided, which work on normalized values of rho.

    :param coefficients: The 5 coefficients for the omnidirectional camera model.
    :param image_radius: The image radius in pixels.
    :returns: The 5 coefficients, unnormalized.
    """
    return coefficients * [1 / image_radius ** i for i in range(5)]

@check_shapes({
    'extrinsics' : 'N*,3,3'
})
def get_3d_transformation(extr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the third column of the rotation matrix and return a 3x3 rotation matrix
        and 3, translation vector.

    :param extr: The extrinsic parameters as some ndarray with 3,3 in the last two dimensions.
    :returns: The rotation matrices and translation vectors for each of the transformations,
        shape=N*,3,3 and N*,3
    """
    rotation_matrices = np.concatenate(
        [
            extr[..., :2],
            np.expand_dims(np.cross(extr[..., 0], extr[..., 1]), axis=-1)
        ],
        axis=-1
    )
    translation_vectors = extr[..., -1]
    return rotation_matrices, translation_vectors

@check_shapes({
    'coeffs' : 'batch?,n_coeffs'
})
def build_companion_matrix(coeffs: np.ndarray) -> np.ndarray:
    """
    Construct a companion matrix for a batch of polynomials. The eigenvalues of each matrix
        are the roots of the corresponding polynomial.

    :param coeffs: The coefficients in ascending order, optionally batched for multiple
        polynomials (of the same degree).
    """
    if len(coeffs.shape) == 1:
        coeffs = coeffs.reshape(1, -1)
    n_coeffs = coeffs.shape[1]
    companion = np.concatenate(
        [
            np.tile(
                np.concatenate(
                    [
                        np.zeros((n_coeffs - 2,), dtype=coeffs.dtype).reshape(1, -1),
                        np.eye(n_coeffs - 2, dtype=coeffs.dtype),
                    ],
                    axis=0
                ),
                (len(coeffs), 1, 1)
            ),
            -(coeffs[:, :-1] / coeffs[:, [-1]])[:, :, None]
        ],
        axis=-1
    )
    if companion.shape[0] == 1:
        companion = companion.squeeze(0)
    return companion

@cache
def get_logger() -> logging.Logger:
    """
    :returns: The logger for the pyfisheye library.
    """
    logger = logging.getLogger('pyfisheye')
    handler = RichHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
