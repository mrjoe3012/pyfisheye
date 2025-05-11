import numpy as np
from pyfisheye.utils.check_shapes import check_shapes
from typing import Optional
import pyfisheye.projection as projection
import pyfisheye.optimisation as optim
import pyfisheye.utils.common as common

@check_shapes({
    'distortion_centre' : '2',
    'intrinsics' : '5',
    'stretch_matrix' : '2,2',
    'image_size_wh' : '2'
})
class Camera:
    def __init__(self,
                 distortion_centre: np.ndarray,
                 intrinsics: np.ndarray,
                 stretch_matrix: np.ndarray = np.eye(2, dtype=np.float64),
                 image_size_wh: Optional[np.ndarray] = None,
                 precompute_lookup_table: bool = False) -> None:
        self._distortion_centre = distortion_centre
        self._intrinsics = intrinsics
        self._stretch_matrix = stretch_matrix
        self._image_size = image_size_wh 
        if precompute_lookup_table:
            self.__compute_lookup_table()
        else:
            self._lookup_table = None

    @check_shapes({
        'pixels' : 'N*,2'
    })
    def cam2world(self, pixels: np.ndarray, normalise: bool = True) -> np.ndarray:
        return projection.backproject(
            pixels, self._intrinsics,
            self._distortion_centre,
            self._stretch_matrix,
            normalise
        )

    @check_shapes({
        'points' : 'N*,3'
    })
    def world2cam(self, points: np.ndarray) -> np.ndarray:
        return projection.project(
            points,
            self._intrinsics,
            self._distortion_centre,
            self._stretch_matrix
        )

    @check_shapes({
        'points' : 'N*,3'
    })
    def world2cam_fast(self, points: np.ndarray) -> np.ndarray:
        if self._lookup_table is None:
            self.__compute_lookup_table()
        return projection.project_fast(
            points,
            *self._lookup_table,
            self._distortion_centre,
            self._stretch_matrix
        )

    def __compute_lookup_table(self) -> None:
        if self._image_size is None:
            raise RuntimeError("'image_size_wh' must be provided to Camera.__init__"
                                " in order to use world2cam_fast.")
        self._lookup_table = optim.linear.build_inv_lookup_table(
            self._intrinsics,
            common.compute_image_radius(*self._image_size, self._distortion_centre)
        )
