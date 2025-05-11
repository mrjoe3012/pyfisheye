from dataclasses import dataclass
from typing import Optional

@dataclass
class CalibrationOptions:
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
