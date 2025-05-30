from dataclasses import dataclass
from typing import Optional

@dataclass
class CalibrationOptions:
    """
    Parameters for the camera calibration procedure.

    :param initial_distortion_centre_x: Specify an initial distortion centre to use for nonlinear
        optimisation. Leave as None to use half of the image dimensions. optimise_distortion_centre
        is True, this value will be added to the search grid and the normal search procedure is
        carried out.
    :param initial_distortion_centre_y: Similar to initial_distortion_centr_x but on the vertical
        dimension.
    :param optimise_distortion_centre: If True perform a grid search - trying out many different
        distortion centres - to find the best distortion centre to use in the camera model. This
        is the slowest part of the algorithm.
    :param distortion_centre_search_grid_size: Set the number of rows and columns in the 2D search
        grid used in the distortion centre search. Use 10-30 for a quick calibration
        and 100+ for a more accurate but slower (5+ minutes) calibration.
    :param distortion_centre_search_progress_bar: If True, use tqdm to display a progress bar in
        the terminal for the distortion centre search.
    :param nonlinear_refinement: If True, use the LM algorithm to perform a nonlinear refinement
        of the calibration parameters. This is recommended and takes a few seconds.
    :param robust_wnls_threshold: The threshold to use for weighted nonlinear least squares. This
        controls robustness to misdetected corners in the calibration pattern. See
        Improved Wide-Angle, Fisheye and Omnidirectional Camera Calibration et al.
    :param monotonicity_constraint_samples: The number of samples used to enforce a strictly
        decreasing camera model polynomial during the linear optimisation. Increasing this number
        can significantly impact the calibration time.
    """
    initial_distortion_centre_x: Optional[float] = None
    initial_distortion_centre_y: Optional[float] = None
    optimise_distortion_centre: bool = True
    distortion_centre_search_grid_size: int = 20
    distortion_centre_search_progress_bar: bool = True

    nonlinear_refinement: bool = True
    robust_wnls_threshold: float = 1.0

    monotonicity_constraint_samples: int = 500 

    def __post_init__(self) -> None:
        if self.initial_distortion_centre_x is not None and self.initial_distortion_centre_x <= 0:
            raise ValueError("Option 'initial_image_centre_x' must be greater than 0.")
        if self.initial_distortion_centre_y is not None and self.initial_distortion_centre_y <= 0:
            raise ValueError("Option 'initial_image_centre_y' must be greater than 0.")
        if self.optimise_distortion_centre and self.distortion_centre_search_grid_size <= 0:
            raise ValueError("Option 'distortion_centre_search_grid_size' must be greater than 0.")
        if self.nonlinear_refinement and self.robust_wnls_threshold <= 0:
            raise ValueError("Option 'robust_wnls_threshold' must be greater than 0.")
        if self.monotonicity_constraint_samples <= 0:
            raise ValueError("Option 'monotonicity_constraint_samples' must be greater than 0.")
