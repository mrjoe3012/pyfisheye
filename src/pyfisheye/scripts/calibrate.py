from argparse import ArgumentParser, Namespace, ArgumentTypeError
import os
from typing import Optional
from dataclasses import dataclass
from pyfisheye.internal.utils.common import get_logger, generate_pattern_world_coords
from pyfisheye.calibration import (calibrate as calibrate_camera, reproject,
                                   CalibrationOptions, CalibrationResult)
from pyfisheye.internal.utils.common import compute_image_radius
from pyfisheye.camera import Camera
from mpl_toolkits.mplot3d.axes3d import Axes3D  # type: ignore
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys

__all__ = ['Arguments', 'calibrate']
_logger = get_logger()
plt.rcParams['figure.max_open_warning'] = False

class Arguments(Namespace):
    """
    Typed program arguments for the calibration script.
    """
    images: list[str]
    pattern_num_rows: int
    pattern_num_cols: int
    pattern_tile_width: float
    pattern_tile_height: float
    show_corner_det: bool
    save_corner_det: Optional[str]
    optimise_distortion_centre: bool
    initial_distortion_centre_x: Optional[float]
    initial_distortion_centre_y: Optional[float]
    weighted_least_squares_threshold: float
    num_monotonicity_constraint_samples: int = 1000
    nonlinear_refinement: bool
    distortion_centre_search_grid_size: int = 20
    save_results: Optional[str]
    show_results: bool
    save_calibration_to: str
    log_level: str = 'INFO'

@dataclass
class Status:
    """
    Convenience for returning statuses from functions with
        errors & messages.
    """
    error: bool = False
    message: str = ''

    def handle(self) -> None:
        """
        Print the message and exit if there was an error.
        """
        if self.error:
            _logger.error(self.message)
            sys.exit(1)

def check_existing_file(x: str) -> str:
    """
    Argparse tpyechecking for a file that must exist.
    """
    if not os.path.isfile(x):
        raise ArgumentTypeError(f"'{x}' doesn't exist or is not a file.")
    return x

def check_bool(x: str) -> bool:
    """
    Argparse typechecking for boolean-like variables.
    """
    x = x.lower()
    if x in ['y', 'yes', 'true', 't', '1']:
        return True
    elif x in ['n', 'no', 'false', 'f', '0']:
        return False
    else:
        raise ArgumentTypeError(f"Bad value for boolean option: '{x}'")

def parse_args() -> Arguments:
    """
    :returns: Arguments parsed from CLA.
    """
    parser = ArgumentParser()
    parser.add_argument('images', nargs='+', type=check_existing_file)
    parser.add_argument('--pattern-num-rows', '-pr', type=int, required=True)
    parser.add_argument('--pattern-num-cols', '-pc', type=int, required=True)
    parser.add_argument('--pattern-tile-width', '-pw', type=float, required=True)
    parser.add_argument('--pattern-tile-height', '-ph', type=float, required=True)
    parser.add_argument('--show-corner-det', type=check_bool, default=True, help='If True,'
        ' corner detection will be shown with matplotlib.')
    parser.add_argument('--save-corner-det', type=str, help='Path in which to save images'
        'of detected corners. Leave as None to skip saving.')
    parser.add_argument('--optimise-distortion-centre', type=check_bool, default=True)
    parser.add_argument('--initial-distortion-centre-x', '-cx', type=float, default=None)
    parser.add_argument('--initial-distortion-centre-y', '-cy', type=float, default=None)
    parser.add_argument('--weighted-least-squares-threshold', type=float, default=1)
    parser.add_argument('--num-monotonicity-constraint-samples', '-m', type=int,
                        default=Arguments.num_monotonicity_constraint_samples)
    parser.add_argument('--nonlinear-refinement', type=check_bool, default=True)
    parser.add_argument('--distortion-centre-search-grid-size', type=int, default=20)
    parser.add_argument('--save-results', type=str, default=None)
    parser.add_argument('--show-results', type=check_bool, default=True)
    parser.add_argument('--save-calibration-to', type=str, default='calibration.json')
    parser.add_argument('--log-level', type=str, default=Arguments.log_level)
    args = parser.parse_args(namespace=Arguments())
    return args

def find_corners(images: list[str],
                 rows: int, cols: int) -> tuple[Status, np.ndarray, np.ndarray,
                                                tuple[int, int]]:
    """
    :param images: A list of image paths.
    :param rows: The number of rows in the calibration pattern.
    :param cols: The number of columns in the calibration pattern.
    :returns: The corners, mask for which images corners were found in and the image size
        (width, height).
    """    
    _logger.info(f"Finding pattern corners in {len(images)} images.")
    all_corners = []
    mask = []
    image_size: Optional[tuple[int, int]] = None
    for path in tqdm(images):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image_size is None:
            image_size = img.shape[:2][::-1]
        elif image_size != img.shape[:2][::-1]:
            _logger.warning(f"{path} has size {img.shape[:2][::-1]} which is inconsistent with"
                             f" previous images having size {image_size}.")
            mask.append(False)
            continue
        if img is None:
            _logger.warning(f"Could not load {path}.")
            mask.append(False)
            continue
        retval, corners = cv2.findChessboardCorners(
            img, (cols, rows), flags=cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_ADAPTIVE_THRESH
        )
        if retval is False:
            _logger.warning(f"Couldn't find any corners in {path}.")
            mask.append(False)
            continue
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(
            img, corners, (11, 11), (-1, -1), criteria
        ).squeeze()
        all_corners.append(corners)
        mask.append(True)
    if len(all_corners) == 0:
        status = Status(error=True, message='Pattern could not be detected in any of the '
                        'provided images. Please check the provided number of rows and'
                        ' columns')
    else:
        status = Status()
    _logger.info(f"Found corners in {np.count_nonzero(mask)}/{len(images)} images.")
    assert image_size is not None
    return status, np.array(all_corners), np.array(mask), image_size

def show_and_save_corners(images: list[str], corners: np.ndarray, mask: np.ndarray,
                          show: bool, save: Optional[str], num_cols: int) -> None:
    """
    Show the detected corners and save them. Each of these are only done if the corresponding
        option is set.

    :param images: A list of image paths.
    :param corners: Detected 2D corners.
    :param mask: A boolean mask indicating which of the image paths the corners were successfuly
        loaded from.
    :param show: Whether or not to show the corners.
    :param save: Whether or not to save the corners (visualisation).
    :param num_cols: The number of columns in the calibration pattern.
    """
    if not show and save is None:
        return
    _logger.info("Showing / saving corner detection.")
    figures = []
    j = 0
    figsize = (12, 8)
    for i in range(len(images)):
        img = cv2.imread(images[i], cv2.IMREAD_COLOR_RGB)
        if img is None:
            continue
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.imshow(img)
        if mask[i]:
            c = corners[j]
            ax.plot(
                *c[:num_cols].T,
                color='red'
            )
            ax.plot(
                *c[::num_cols].T,
                color='red'
            )
            ax.scatter(
                *c.T,
                marker='o'
            )
            j += 1
        if save is not None:
            os.makedirs(save, exist_ok=True)
            fig.savefig(os.path.join(save, f'corners_{i}.jpg'))
        figures.append(fig)
    if show:
        plt.show()
    for fig in figures:
        plt.close(fig)

def show_and_save_results(calib_result: CalibrationResult,
                          corners: np.ndarray,
                          mask: np.ndarray,
                          arguments: Arguments) -> None:
    """
    Show and/or save the calibration results. Whether or not they are shown or saved is controlled
        by the corresponding variables in 'options'. The following are shown/saved:
        - extrinsic configuration of detected calibration patterns
        - polynomial describing the camera's distortion
        - reprojection of the 3D pattern coordinates onto the image
        - undistorted (perspective projection) of each of the detected calibration pattern

    :param calib_result:
    :param corners:
    :param mask:
    :param arguments:
    """
    if not arguments.show_results and arguments.save_results is None:
        return
    _logger.info("Saving / showing results.")
    reprojected = reproject(
        arguments.pattern_num_rows,
        arguments.pattern_num_cols,
        calib_result,
        arguments.pattern_tile_width,
        arguments.pattern_tile_height
    )
    figures = []
    figsize=(12, 8)
    # plot extrinsics
    fig = plt.figure(figsize=figsize)
    figures.append(fig)
    ax: Axes3D = fig.add_subplot(1, 1, 1, projection='3d')
    ax.view_init(roll=50, elev=-50, azim=-130)
    ax.set_title('Extrinsic Parameters')
    world_coords = generate_pattern_world_coords(
        arguments.pattern_num_rows,
        arguments.pattern_num_cols,
        arguments.pattern_tile_width,
        arguments.pattern_tile_height
    )
    transformed_coords = \
        (calib_result.extrinsics[:, None, :, :2] @ world_coords[None, :, :2, None]).squeeze(-1)
    transformed_coords += calib_result.extrinsics[:, None, :, -1]
    for i in range(corners.shape[0]):
        ax.scatter(
            *transformed_coords[i].T
        )
    for vec, color in zip(np.eye(3), ('red', 'green', 'blue')):
        ax.quiver(
            0, 0, 0,
            *(vec * 0.1),
            color=color
        )
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect('equal')
    if arguments.save_results is not None:
        fig.savefig(os.path.join(arguments.save_results, 'extrinsics.jpg'))
    errors = []
    for corner_idx, img_idx in enumerate(np.where(mask)[0]):
        # plot reprojections
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        img = cv2.imread(arguments.images[img_idx], cv2.IMREAD_COLOR_RGB)
        h, w = img.shape[:2]
        ax.imshow(img)
        ax.scatter(*corners[corner_idx].T, marker='o', label='Detected Corners')
        ax.scatter(*reprojected[corner_idx].T, marker='x', label='Reprojected Corners')
        cx, cy = calib_result.optimal_distortion_centre
        ax.scatter(
            cx, cy,
            color='green',
            label='Distortion Centre'
        )
        ax_len = max(img.shape[0], img.shape[1]) * 0.1
        ax.plot(
            [cx, cx + ax_len],
            [cy, cy],
            color='green'
        )
        ax.plot(
            [cx, cx],
            [cy, cy + ax_len],
            color='green'
        )
        ax.legend()
        mean_pixel_error = np.mean(
            np.linalg.norm(
                reprojected[corner_idx] - corners[corner_idx],
                axis=-1
            )
        )
        errors.append(mean_pixel_error)
        ax.set_title(f'error={mean_pixel_error:.2f} pixels')
        figures.append(fig)
        if arguments.save_results is not None:
            fig.savefig(os.path.join(arguments.save_results, f'result_{img_idx}.jpg'))
        # plot undistorted pattern
        camera = Camera(calib_result.optimal_distortion_centre, calib_result.intrinsics,
                        calib_result.stretch_matrix, img.shape[:2][::-1])
        reprojected_img = camera.reproject_perspective(img, corners[corner_idx], img_width=800)
        fig, ax = plt.subplots(1, 1, figsize=(9, 9)) 
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.imshow(reprojected_img)
        figures.append(fig)
        if arguments.save_results is not None:
            fig.savefig(os.path.join(arguments.save_results, f'undistorted_{img_idx}.jpg'))
    # plot polynomial
    rho_samples = np.linspace(0, compute_image_radius(w, h, calib_result.optimal_distortion_centre), 300)
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    ax.set_title('Polynomial')
    ax.set_xlabel('Rho')
    ax.set_ylabel('f(Rho)')
    ax.plot(rho_samples, np.polyval(calib_result.intrinsics[::-1], rho_samples))
    figures.append(fig)
    if arguments.save_results is not None:
        fig.savefig(os.path.join(arguments.save_results, 'polynomial.jpg'))
    _logger.info(f'Pixel error - mean={np.mean(errors):.2f} std={np.std(errors):.2f}')
    if arguments.save_results is not None:
        np.savetxt(
            os.path.join(arguments.save_results, 'reprojection_error.txt'),
            errors
        )
    if arguments.show_results:
        plt.show()
    for fig in figures:
        plt.close(fig)

def calibrate(arguments: Optional[Arguments] = None) -> None:
    """
    Detect corners of the calibration pattern, alibrate the camera and show / save the results.

    :param arguments: Leave as None to parse arguments from CLA.
    """
    if arguments is None:
        return calibrate(parse_args())
    _logger.setLevel(arguments.log_level)
    status, pattern_observations, valid_img_mask, image_size = find_corners(
        arguments.images,
        arguments.pattern_num_rows,
        arguments.pattern_num_cols
    )
    status.handle()
    show_and_save_corners(
        arguments.images,
        pattern_observations,
        valid_img_mask,
        arguments.show_corner_det,
        arguments.save_corner_det,
        arguments.pattern_num_cols
    )
    options = CalibrationOptions(
        initial_distortion_centre_x=arguments.initial_distortion_centre_x,
        initial_distortion_centre_y=arguments.initial_distortion_centre_y,
        optimise_distortion_centre=arguments.optimise_distortion_centre,
        distortion_centre_search_grid_size=arguments.distortion_centre_search_grid_size,
        nonlinear_refinement=arguments.nonlinear_refinement,
        robust_wnls_threshold=arguments.weighted_least_squares_threshold,
        monotonicity_constraint_samples=arguments.num_monotonicity_constraint_samples
    )
    calib_result = calibrate_camera(
        pattern_observations,
        image_size[1],
        image_size[0],
        arguments.pattern_num_rows,
        arguments.pattern_num_cols,
        arguments.pattern_tile_width,
        arguments.pattern_tile_height,
        options
    )
    _logger.info("Finished calibration procedure.")
    show_and_save_results(
        calib_result,
        pattern_observations,
        valid_img_mask,
        arguments
    )
    camera = Camera(
        calib_result.optimal_distortion_centre,
        calib_result.intrinsics,
        calib_result.stretch_matrix,
        image_size,
    )
    camera.to_json(arguments.save_calibration_to)

if __name__ == '__main__':
    _logger.setLevel('INFO')
    calibrate()
