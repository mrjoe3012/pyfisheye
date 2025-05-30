Examples
========

The below examples demonstrate how you can use the `Camera` class to work with a calibrated camera. First,
we must instantiate a `Camera` object from the json file saved by the calibration procedure. See :ref:`Calibration<calibration>`.

.. code-block:: python

    from pyfisheye.camera import Camera
    cam = Camera.from_json('calibration.json')
    # optional to speed up first call of world2cam_fast
    cam = Camera.from_json('calibration.json', precompute_lookup_table=True)

cam2world
---------

This function converts pixels in the image into 3D vectors.

.. code-block:: python

    import numpy as np
    pixels = np.array([
        10.5, 333.5,
        1454.4, 323.4,
        1987.3, 115.9
    ]).reshape(-1, 2)
    # unit vectors are returned by default
    # +x : right, +y: down, +z: forward
    vectors = cam.cam2world(pixels)
    # can skip normalisation
    vectors = cam.cam2world(pixels, normalise=False)

world2cam
---------

This function converts 3D points (or rays) in the Camera's coordinate system into image pixel coordinates. It is recommended that you use `world2cam_fast`.

.. code-block:: python

    points = np.array([
        12.3, -5.0, 10.0,
        -3.0, -2.0, -5.0
    ]).reshape(-1, 3)
    pixels = cam.world2cam(points)
    # the version below gives (roughly) the same result but is much faster
    pixels = cam.world2cam_fast(points)

reproject_perspective
---------------------

Often, you want to be able to *undistort* a specific region of an image. This is equivalent to
computing parameters for an imaginary perspective camera which captures the region of interest
and backprojecting each pixel captured by the imaginary camera onto the fisheye image plane to
compute a mapping.

.. code-block:: python

    import cv2
    from scipy.spatial.transform import Rotation

    # must already have some image
    img_fisheye = cv2.imread('fisheye-img.png')
    # example: provide any number of points (i.e. four corners) for the region
    # of interest. a best fit perspective camera will reproject this region (and possibly more)
    region_of_interest_px = np.array([
        543.3, 334.9,
        876.5, 334.9,
        543.3, 650.4,
        876.5, 650.4
    ]).reshape(-1, 2)
    perspective_img = cam.reproject_perspective(
        img_fisheye,
        region_of_interest_px,
        img_width=400 # alternatively, specify only img_height
                      # or specify both (aspect ratio may change)
    )
    # region of interest can also be specified in 3D space, i.e. vectors or points which must
    # be included in the image
    region_of_interest_world = cam.cam2world(region_of_interest_px)
    perspective_img = cam.reproject_perspective(
        img_fisheye,
        region_of_interest_world,
        img_width=400
    )
    # you can also apply a rotation to the perspective camera
    perspective_img = cam.reproject_perspective(
        img_fisheye,
        region_of_interest_world,
        img_width=400,
        rotation=Rotation.from_euler('y', 45.0, degrees=True).as_matrix() # 3x3 rotation
    )
