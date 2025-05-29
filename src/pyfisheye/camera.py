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
    """
    Stores the calibration parameters for a camera and provides methods for projection,
        backprojection and more.
    """
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
        """
        :param distortion_centre: 2D point in pixel coordinates for the distortion centre.
        :param intrnisics: Array of 5 coefficients for the polynomial of degree 4, provided with
            lowest power first.
        :param stretch_matrix: The stretch matrix.
        :param image_size_wh: The image size. Required only when using the fast projection method.
        :param precompute_lookup_table: If True, the lookup table for fast projection is computed
            immediately. Otherwise, it is only computed the first time the method is called.
        """
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
        """
        Backproject pixel(s) in the image to a ray in 3D space.
 
        :param pixels: The array of pixels, can be any shape as long as the last dimension is of
            length 2.
        :param normalise: If True, all returned rays will lie on the unit sphere.
        :returns: 3D rays eminating from the camera in the camera's coordinate system.
        """
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
        """
        Project 3D points/rays in the camera coordinate system onto the image plane. This method
            requires a polynomial inversion (i.e. eigenvalue decomposition) for each provided point.

        :param points: An array with any number of dimensions as long as the last dimension
            has length 3.
        :returns: Pixel coordinates for each point. NaN is returned for failed projections.
        """
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
        """
        Project 3D points / rays in the camera coordinate system onto the image plane. This method
            uses a lookup table and linear interpolation to compute the projection. The lookup
            table will be computed if it was not precomputed when instantiating the camera.

        :param points: An array with any number of dimensions as long as the last dimension
            has length 3.
        :returns: Pixel coordinates for each point. NaN is returned for failed projections.
        """
        if self._lookup_table is None:
            self.__compute_lookup_table()
        return projection.project_fast(
            points,
            *self._lookup_table,
            self._distortion_centre,
            self._stretch_matrix
        )

    def __compute_lookup_table(self) -> None:
        """
        Compute the lookup table for the inverse mapping.
        """
        if self._image_size is None:
            raise RuntimeError("'image_size_wh' must be provided to Camera.__init__"
                                " in order to use world2cam_fast.")
        self._lookup_table = optim.linear.build_inv_lookup_table(
            self._intrinsics,
            common.compute_image_radius(*self._image_size, self._distortion_centre)
        )

    def to_json(self, path: str) -> None:
        """
        Export the calibration parameters to a json file.

        :param path: The path to write to.
        """
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
    def from_json(path: str, **kwargs) ->  Camera:
        """
        Instantiate a camera from a json file.

        :param \*\*kwargs: Optional overrides passed to __init__.
        :returns: The instantiated camera.
        """
        with open(path, 'r') as f:
            data = json.load(f)
        json_kwargs = {
            k : np.array(v) for k, v in data.items()
        }
        return Camera(**json_kwargs, **kwargs)

    @check_shapes({
        'rotation' : '3,3'
    })
    def map_perspective(self, points_or_pixels: np.ndarray,
                        img_width: Optional[int] = None,
                        img_height: Optional[int] = None,
                        rotation: np.ndarray = np.eye(3, dtype=np.float64)) -> np.ndarray:
        """
        Returns the map fed to cv2.remap in :func:`Camera.reproject_perspective`.

        :param points_or_pixels: 2D points in the image or 3D points/rays in the camera
            coordinate system. Any shape as long as the last dimension equals 2 or 3.
        :param img_width: Number of pixels in the x-axis for the perspective projection. Leave as
            None and provide a height to automatically set this and maintain the aspect ratio.
        :param img_height: Number of pixels in the y-axis for the perspective projection. Leave as
            None and provide a width to automatically set this and maintain the aspect ratio.
        :param rotation: The rotation matrix to apply to the imaginary perspective camera
            prior to reprojection. Defaults to the identity so that no rotation is applied. For
            example, the extrinsic orientation of the camera (camera -> world) can be provided
            to correct for skewed/rotated perspective projections.
        :returns: A max of size (height, width, 2) specifying the coordinates in the original
            for each pixel in the destination image. Uses the provided points_or_pixels to
            compute the correct perspective camera parameters such that all points lie in the
            final image.
        """
        if img_width is None and img_height is None:
            raise ValueError("At least one of img_width or img_height needs to be provided.")
        # handle parameters and compute world rays by backprojection / normalising 3d points
        if img_width is not None and img_width <= 0:
            raise ValueError("img_width must be greater than 0.")
        if img_height is not None and img_height <= 0:
            raise ValueError("img_height must be greater than 0.")
        if len(points_or_pixels.shape) < 2:
            raise ValueError("At least two points must be provided to compute a perspective"
                " field of view.")
        if points_or_pixels.shape[-1] == 2:
            rays = self.cam2world(points_or_pixels, normalise=True)
        elif points_or_pixels.shape[-1] == 3:
            rays = points_or_pixels / np.linalg.norm(points_or_pixels, axis=-1, keepdims=True)
        else:
            raise ValueError("points_or_pixels must have final dimension with length 2 (for pixels)"
            " or 3 (for points/rays)")
        # compute a rotation such that rays are roughly centred
        mean_ray = np.mean(rays.reshape(-1, 3), axis=0)
        phi = np.atan2(mean_ray[1], mean_ray[0])
        theta = np.atan2(
            *(Rotation.from_euler('zyx', [-phi, 0, 0]).apply(mean_ray)[[0, 2]])
        )
        # transformation of the perspective camera from its origin to the world origin in order
        # to capture the right light rays
        perspective_transformation = \
            rotation @ Rotation.from_euler('zyx', [-phi, -theta, 0]).inv().as_matrix()
        # transform rays into perspective camera's frame of reference
        rays_centred = Rotation.from_euler('zyx', [-phi, -theta, 0]).apply(rays)
        rays_centred = rays @ perspective_transformation # TODO: rename
        # compute the required camera FOV
        horizontal_angles = np.atan2(
            rays_centred[..., 0], rays_centred[..., 2]
        )
        vertical_angles = np.atan2(
            rays_centred[..., 1], rays_centred[..., 2]
        )
        horizontal_fov = horizontal_angles.max() - horizontal_angles.min()
        vertical_fov = vertical_angles.max() - vertical_angles.min()
        horizontal_fov = 2 * np.abs(horizontal_angles).max()
        vertical_fov = 2 * np.abs(vertical_angles).max()
        # use difference in FOV to set aspect ratio (if both dimensions not provided)
        if img_width is None:
            img_width = int(img_height * (horizontal_fov / vertical_fov))
        if img_height is None:
            img_height = int(img_width * (vertical_fov / horizontal_fov))
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
        sample_rays = (perspective_transformation @ sample_rays).squeeze(-1)
        perspective_mapping = self.world2cam_fast(sample_rays)
        return perspective_mapping

    @check_shapes({
        'original_image' : 'width, height, dims'
    })
    def reproject_perspective(self, original_image: np.ndarray,
                              points_or_pixels: np.ndarray,
                              rotation: np.ndarray = np.eye(3),
                              img_width: Optional[int] = None,
                              img_height: Optional[int] = None) -> np.ndarray:
        """
        Reproject a region of a fisheye image to a perspective one.

        :param original_image: The original image to sample from. cv2.remap will be used
            to generate the perspective image with linear interpolation.
        :param points_or_pixels: 2D points in the image or 3D points/rays in the camera
            coordinate system. Any shape as long as the last dimension equals 2 or 3. The
            reprojected image will contain each point and surrounding pixels. It is sufficient
            to provide a bounding box or a small set of pixels of interest rather than a dense
            selection.
        :param rotation: The rotation matrix to apply to the imaginary perspective camera
            prior to reprojection. Defaults to the identity so that no rotation is applied. For
            example, the extrinsic orientation of the camera (camera -> world) can be provided
            to correct for skewed/rotated perspective projections.
        :param img_width: Number of pixels in the x-axis for the perspective projection. Leave as
            None and provide a height to automatically set this and keep the correct aspect ratio.
        :param img_height: Number of pixels in the y-axis for the perspective projection. Leave as
            None and provide a width to automatically set this and keep the correct aspect ratio.
        :returns: A max of size (height, width, 2) specifying the coordinates in the original
            for each pixel in the destination image.
        """
        perspective_mapping = self.map_perspective(
            points_or_pixels,
            img_width,
            img_height,
            rotation
        )
        perspective_img = cv2.remap(  # type: ignore
            original_image, perspective_mapping.astype(np.float32), None,
            interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        return perspective_img 
