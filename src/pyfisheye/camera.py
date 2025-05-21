from __future__ import annotations
import numpy as np
from pyfisheye.internal.utils.check_shapes import check_shapes
from typing import Optional
from scipy.spatial.transform import Rotation
import pyfisheye.internal.projection as projection
import pyfisheye.internal.optimisation as optim
import pyfisheye.internal.utils.common as common
import json
import cv2

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
            data['image_size_wh'] = self._image_size.tolist()
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

    @check_shapes({
        'rotation' : '3,3'
    })
    def map_perspective(self, points_or_pixels: np.ndarray,
                        img_width: int, img_height: int,
                        rotation: np.ndarray = np.eye(3, dtype=np.float64)) -> np.ndarray:
        """
        :param points_or_pixels: 2D points in the image or 3D points/rays in the camera
            coordinate system. Any shape as long as the last dimension equals 2 or 3.
        :param img_width: Number of pixels in the x-axis for the perspective projection.
        :param img_height: Number of pixels in the y-axis for the perspective projection.
        :param rotation: The rotation matrix to apply to the imaginary perspective camera
            prior to reprojection. Defaults to the identity so that no rotation is applied.
        :returns: A max of size (height, width, 2) specifying the coordinates in the original
            for each pixel in the destination image. Uses the provided points_or_pixels to
            compute the correct perspective camera parameters such that all points lie in the
            final image.
        """
        # handle parameters and compute world rays by backprojection / normalising 3d points
        if img_width <= 0:
            raise ValueError("img_width must be greater than 0.")
        if img_height <= 0:
            raise ValueError("img_height must be greater than 0.")
        if len(points_or_pixels.shape) < 2:
            raise ValueError("At least two points must be provided to compute a perspective"
                " field of view.")
        if points_or_pixels.shape[-1] == 2:
            rays = self.cam2world(points_or_pixels, normalise=True)
        elif points_or_pixels.shape[-1] == 3:
            rays = points_or_pixels / np.linalg.norm(rays, axis=-1, keepdims=True)
        else:
            raise ValueError("points_or_pixels must have final dimension with length 2 (for pixels)"
            " or 3 (for points/rays)")
        # compute a rotation such that rays are roughly centred
        mean_ray = np.mean(rays.reshape(-1, 3), axis=0)
        phi = np.atan2(mean_ray[1], mean_ray[0])
        theta = np.atan2(
            *(Rotation.from_euler('zyx', [-phi, 0, 0]).apply(mean_ray)[[0, 2]])
        )
        # rotate rays so that they are centred
        rays_centred = Rotation.from_euler('zyx', [-phi, -theta, 0]).apply(rays)
        # compute the required camera FOV
        horizontal_angles = np.atan2(
            rays_centred[..., 0], rays_centred[..., 2]
        )
        vertical_angles = np.atan2(
            rays_centred[..., 1], rays_centred[..., 2]
        )
        horizontal_fov = horizontal_angles.max() - horizontal_angles.min()
        vertical_fov = vertical_angles.max() - vertical_angles.min()
        # compute the rays coming from the perspective camera
        f_x = img_width / (2 * np.tan(horizontal_fov / 2))
        f_y = img_height / (2 * np.tan(vertical_fov / 2))
        c_x = img_width / 2
        c_y = img_height / 2
        camera_matrix = np.array([
            [f_x, 0, c_x],
            [0, f_y, c_y],
            [0, 0, 1]
        ])
        perspective_pixels = np.stack(
            np.meshgrid(
                np.arange(img_width),
                np.arange(img_height)
            ),
            axis=-1
        ).astype(np.float64)
        sample_rays = (np.linalg.inv(camera_matrix) @ np.concatenate(
            [
                perspective_pixels,
                np.ones((*perspective_pixels.shape[:-1], 1))
            ],
            axis=-1
        )[..., None])
        # rotate perspective rays so that they point towards desired points/pixels with the
        # provided orientation
        perspective_transformation = \
            rotation @ Rotation.from_euler('zyx', [-phi, -theta, 0]).inv().as_matrix()
        sample_rays = (perspective_transformation @ sample_rays).squeeze(-1)
        # negate the xy-axis to align with internal left-handed coordinate system TODO: fix
        perspective_mapping = self.world2cam_fast(sample_rays)
        return perspective_mapping

    @check_shapes({
        'original_image' : 'width, height, dims'
    })
    def reproject_perspective(self, original_image: np.ndarray,
                              points_or_pixels: np.ndarray, img_width: int,
                              img_height: int) -> np.ndarray:
        """
        :param original_image: The original image to sample from. cv2.remap will be used
            to generate the perspective image with linear interpolation.
        :param points_or_pixels: 2D points in the image or 3D points/rays in the camera
            coordinate system. Any shape as long as the last dimension equals 2 or 3. The
            reprojected image will contain each point and surrounding pixels. It is sufficient
            to provide a bounding box or a small set of pixels of interest rather than a dense
            selection.
        :param img_width: Number of pixels in the x-axis for the perspective projection.
        :param img_height: Number of pixels in the y-axis for the perspective projection.
        :returns: A max of size (height, width, 2) specifying the coordinates in the original
            for each pixel in the destination image.
        """
        perspective_mapping = self.map_perspective(
            points_or_pixels,
            img_width,
            img_height
        )
        perspective_img = cv2.remap(
            original_image, perspective_mapping.astype(np.float32), None,
            interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        return perspective_img 
