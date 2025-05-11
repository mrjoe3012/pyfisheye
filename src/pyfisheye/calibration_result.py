from dataclasses import dataclass
import numpy as np

@dataclass
class CalibrationResult:
    extrinsics: np.ndarray
    intrinsics: np.ndarray
    optimal_distortion_centre: np.ndarray
    stretch_matrix: np.ndarray 
