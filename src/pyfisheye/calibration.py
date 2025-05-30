from pyfisheye.calibration_options import CalibrationOptions
from pyfisheye.calibration_result import CalibrationResult
from pyfisheye.internal.utils.common import check_shapes
from typing import Optional, Any
from tqdm import tqdm
import pyfisheye.internal.optimisation as optim
import pyfisheye.internal.utils.common as common
import pyfisheye.internal.projection as projection
import numpy as np
import numpy.typing as npt

__all__ = ['calibrate']
_logger = common.get_logger()

def __calibrate(pattern_observations: np.ndarray,
                pattern_world_coords: np.ndarray,
                optimal_distortion_centre: np.ndarray,
                image_radius: np.ndarray,
                num_rho_samples: int,
                monotonic: bool) -> tuple[np.ndarray, np.ndarray]:
    extrinsics = optim.partial_extrinsics(
        pattern_observations,
        pattern_world_coords,
        optimal_distortion_centre
    )
    extrinsics = optim.select_best_extrinsic_solution(
        pattern_observations,
        pattern_world_coords,
        optimal_distortion_centre,
        extrinsics,
        image_radius
    )
    intrinsics_norm, z_translation = optim.intrinsics_and_z_translation(
        pattern_observations,
        pattern_world_coords,
        optimal_distortion_centre,
        extrinsics,
        image_radius,
        monotonic=monotonic,
        num_rho_samples=num_rho_samples
    )
    extrinsics[:, -1, -1] = z_translation
    extrinsics = optim.linear_refinement_extrinsics(
        pattern_observations,
        pattern_world_coords,
        optimal_distortion_centre,
        extrinsics,
        intrinsics_norm,
        image_radius
    )
    intrinsics_norm = optim.linear_refinement_intrinsics(
        pattern_observations,
        pattern_world_coords,
        optimal_distortion_centre,
        extrinsics,
        image_radius
    )
    return intrinsics_norm, extrinsics

@check_shapes({
    'pattern_observations' : 'N,M,2'
})
def calibrate(pattern_observations: np.ndarray,
              image_height: int,
              image_width: int,
              pattern_num_rows: int,
              pattern_num_cols: int,
              pattern_width_x: float,
              pattern_width_y: Optional[float] = None,
              calibration_options: CalibrationOptions = CalibrationOptions()) -> CalibrationResult:
    if pattern_width_y is None:
        pattern_width_y = pattern_width_x
    if pattern_num_cols <=  0:
        raise ValueError("Argument 'pattern_num_cols' must be greater than 0.")
    if pattern_num_rows <=  0:
        raise ValueError("Argument 'pattern_num_rows' must be greater than 0.")
    if pattern_width_x <=  0:
        raise ValueError("Argument 'pattern_width_x' must be greater than 0.")
    if pattern_width_y <=  0:
        raise ValueError("Argument 'pattern_width_y' must be greater than 0.")
    if image_height <= 0:
        raise ValueError("Argument 'image_height' must be greater than 0.")
    if image_width <= 0:
        raise ValueError("Argument 'image_width' must be greater than 0.")
    if len(pattern_observations.shape) != 3 or pattern_observations.shape[-1] != 2:
        raise ValueError("Expected argument 'pattern_observations' to have shape N,M,2 "
                         "where N is greater than 1 and M is equal to pattern_num_cols *"
                         " pattern_num_rows.")
    _logger.debug("Starting calibration procedure.")
    pattern_observations = pattern_observations.astype(np.float64)
    pattern_world_coords = common.generate_pattern_world_coords(
        pattern_num_rows, pattern_num_cols,
        pattern_width_x, pattern_width_y
    )
    img_centre_x, img_centre_y = image_width / 2, image_height / 2
    stretch_matrix = np.eye(2, dtype=np.float64)
    if calibration_options.optimise_distortion_centre:
        optimal_distortion_centre, centre_search_error = optimise_distortion_centre(
            pattern_observations,
            pattern_world_coords,
            stretch_matrix,
            image_width, image_height,
            calibration_options.distortion_centre_search_grid_size,
            calibration_options.distortion_centre_search_progress_bar
        )
        _logger.debug(f"{centre_search_error=}")
    else:
        optimal_distortion_centre = np.array([
            calibration_options.initial_distortion_centre_x or img_centre_x,
            calibration_options.initial_distortion_centre_y or img_centre_y
        ])
    image_radius = common.compute_image_radius(
        image_width, image_height,
        optimal_distortion_centre
    )
    intrinsics_norm, extrinsics = __calibrate(
        pattern_observations,
        pattern_world_coords,
        optimal_distortion_centre,
        image_radius,
        calibration_options.monotonicity_constraint_samples,
        True
    )
    intrinsics = common.unnormalize_coefficients(
        intrinsics_norm,
        image_radius
    )
    if calibration_options.nonlinear_refinement:
        result = optim.nonlinear_refinement(
            pattern_observations,
            pattern_world_coords,
            optimal_distortion_centre,
            extrinsics,
            intrinsics,
            stretch_matrix,
            calibration_options.robust_wnls_threshold
        )
        extrinsics, intrinsics = result.extrinsics, result.intrinsics
        optimal_distortion_centre = result.dist_centre
        stretch_matrix = result.scaling_mat
    return CalibrationResult(
        extrinsics, intrinsics,
        optimal_distortion_centre,
        stretch_matrix
    )

def __reproject(pattern_world_coords: np.ndarray,
                extrinsics: np.ndarray,
                intrinsics: np.ndarray,
                distortion_centre,
                scaling_matrix: np.ndarray) -> np.ndarray:
    transformed_world_coords = \
        (extrinsics[:, None, :, :2] @ pattern_world_coords[None, :, :2, None]).squeeze(-1)
    transformed_world_coords += extrinsics[:, None, :, -1]
    projected = projection.project(
        transformed_world_coords,
        intrinsics,
        distortion_centre,
        scaling_matrix
    )
    return projected

def reproject(pattern_num_rows: int,
              pattern_num_cols: int,
              calibration_result: CalibrationResult,
              pattern_width_x: float,
              pattern_width_y: Optional[float] = None) -> np.ndarray:
    pattern_world_coords = common.generate_pattern_world_coords(
        pattern_num_rows,
        pattern_num_cols,
        pattern_width_x,
        pattern_width_y
    )
    return __reproject(
        pattern_world_coords,
        calibration_result.extrinsics,
        calibration_result.intrinsics,
        calibration_result.optimal_distortion_centre,
        calibration_result.stretch_matrix
    )
 
@check_shapes({
    'pattern_observations' : 'N,M,2',
    'pattern_world_coords' : 'M,3',
    'stretch_matrix' : '2,2'
})
def optimise_distortion_centre(pattern_observations: np.ndarray,
                               pattern_world_coords: np.ndarray,
                               stretch_matrix: np.ndarray,
                               image_width: float, image_height: float,
                               search_grid_size: int,
                               progress_bar: bool) -> tuple[np.ndarray, float]:
    """
    Use a basic grid search to find an optimal distortion centre. This works by performing a linear
        calibration procedure (without monotonocity constraints) for each candidate distortion centre
        and takes the one with the minimum reprojection error.

    :param pattern_observations: The 2D pattern coordinates in the image.
    :param pattern_world_coords: The 3D pattern coordinates in the pattern space with Z equal to 0.
    :param stretch_matrix: The stretch matrix to use. The identity can be provided for no
        stretching effect.
    :param image_width: Width in pixels.
    :param image_height: Height in pixels.
    :param search_grid_size: The size of the grid to search. Total number of samples is the square
        of this value. Use a value around 10-20 for a quick calibration and around 100 for a more
        accurate calibration.
    :param progress_bar: If true, show a tqdm progress bar. This function can take several minutes
        to complete for grid sizes around 100.
    :returns: The optimal distortion centre followed by its mean reprojection error in pixels.
    """
    potential_centres: npt.NDArray[Any] = np.stack(
        np.meshgrid(
            np.linspace(0, image_width, search_grid_size),
            np.linspace(0, image_height, search_grid_size)
        ),
        axis=-1
    ).reshape(-1, 2)
    img_centre_x, img_centre_y = image_width / 2, image_height / 2
    # add image centre as well
    potential_centres = np.concatenate(
        [
            potential_centres,
            np.expand_dims([img_centre_x, img_centre_y], axis=0)
        ],
        axis=0
    )
    # sort candidates by distance from centre
    sorted_indices = np.argsort(
        np.linalg.norm(potential_centres - [img_centre_x, img_centre_y], axis=-1)
    )
    potential_centres = potential_centres[sorted_indices]
    iterator = potential_centres
    if progress_bar:
        iterator = tqdm(iterator, desc='Optimising image centre.')  # type: ignore
    optimal_distortion_centre = potential_centres[0]
    optimal_distortion_centre_mean_error = np.array(np.inf)
    _logger.debug(f"Searching through {len(potential_centres)} potential distortion centres.")
    # temporarily disable logging due to running calibration in a loop.
    prev_level = _logger.level
    _logger.setLevel('INFO')
    for distortion_centre in iterator:
        image_radius = common.compute_image_radius(
            image_width,
            image_height,
            distortion_centre
        )
        intr_norm, extr = __calibrate(
            pattern_observations,
            pattern_world_coords,
            distortion_centre,
            image_radius,
            monotonic=False,
            num_rho_samples=100  # ignored as monotonic=False
        )
        intr = common.unnormalize_coefficients(
            intr_norm,
            image_radius
        )
        reprojected = __reproject(
            pattern_world_coords,
            extr,
            intr,
            distortion_centre,
            stretch_matrix
        )
        mean_error = np.mean(
            np.linalg.norm(
                reprojected - pattern_observations,
                axis=-1
            )
        )
        if mean_error < optimal_distortion_centre_mean_error:
            optimal_distortion_centre_mean_error = mean_error
            optimal_distortion_centre = distortion_centre
    # restore logging level to previous value
    _logger.setLevel(prev_level)
    _logger.debug(f"Finished searching for distortion centre. {optimal_distortion_centre=}"
                      f" {optimal_distortion_centre_mean_error=}")
    return optimal_distortion_centre, float(optimal_distortion_centre_mean_error.item())