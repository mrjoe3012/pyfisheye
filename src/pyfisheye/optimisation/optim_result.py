from typing import NamedTuple
import numpy as np

class OptimResult(NamedTuple):
    extrinsics: np.ndarray
    intrinsics: np.ndarray
    dist_centre: np.ndarray
    scaling_mat: np.ndarray
