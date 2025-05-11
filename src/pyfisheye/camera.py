from __future__ import annotations
import numpy as np
from pyfisheye.internal.utils.check_shapes import check_shapes
from typing import Optional, TextIO
import pyfisheye.internal.projection as projection
import pyfisheye.internal.optimisation as optim
import pyfisheye.internal.utils.common as common
import json

class Camera:
    @check_shapes({
        'distortion_centre' : '2',
        'intrinsics' : '5',
        'stretch_matrix' : '2,2',
        'image_size_wh' : '2'
    })
    def __init__(self,
                 distortion_centre: np.ndarray,
                 intrinsics: np.ndarray,
                 stretch_matrix: np.ndarray = np.eye(2, dtype=np.float64),
                 image_size_wh: Optional[np.ndarray] = None,
                 precompute_lookup_table: bool = False) -> None:
        self._distortion_centre = np.array(distortion_centre)
        self._intrinsics = np.array(intrinsics)
        self._stretch_matrix = np.array(stretch_matrix)
        self._image_size = np.array(image_size_wh)
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

    def to_json(self, path: str) -> None:
        data = {
            'intrinsics' : self._intrinsics.tolist(),
            'distortion_centre' : self._distortion_centre.tolist(),
            'stretch_matrix' : self._stretch_matrix.tolist(),
        }
        if self._image_size is not None:
            data['image_size'] = self._image_size.tolist()
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def from_json(path: str) ->  Camera:
        with open(path, 'r') as f:
            data = json.load(f)
        kwargs = {
            k : np.array(v) for k, v in data.items()
        }
        return Camera(**kwargs)
