Projection
==========

The camera uses a right-handed coordinate system as defined below:

.. image:: media/coord-system.png
  :alt: A depiction of the camera's coordinate system in 3D. It is a right-handed system with the x-axis pointing to the right, the Y-axis pointing downwards and the Z-axis pointing forward.

The image plane relates to the sensor plane via an affine transformation defined by the stretch matrix and distortion centre.

.. image:: media/sensor-image-plane.png
  :alt: A graphic showing the relationship between points on the image plane and those on the sensor plane. Points on the sensor plane have the distortion centre subtracted from them and are then multiplied by the scaling matrix.

The world space relates to the sensor plane via a monotonically decreasing polynomial mapping radial pixel distance to the z-coordinate.

.. image:: media/projection-flow.png

**Backprojection** involves computing the ray emanating from the camera for a given pixel in the image. This is done by evaluating a polynomial of
degree four whose coefficients were found during the calibration procedure. First, we transform the pixel coordinates to the sensor plane using the stretch matrix and distortion centre. Second, we evaluate the polynomial to compute the 3D ray.

.. math::
  :name: backprojection

  \begin{align*}
    \rho(u', v') &= \sqrt{{u'}^2 + {v'}^2} \\
    z &= f(u', v') \\
    &= a_0 \rho(u', v') + a_2 \rho^2(u', v') + a_3 \rho^3(u', v') + a_4 \rho^4(u', v')
  \end{align*}

**Projection** involves computing the pixel coordinate corresponding to a 3D point or ray. This requires the use of the inverse function :math:`g(\theta)`, which maps a 3D ray's polar angle to the corresponding radial distance, :math:`\rho(u', v')`, on the sensor plane. The first step is to compute the spherical representation of the ray.

.. math::
  :name: projection-spherical-ray

  \begin{align*}
  \theta &= \frac{\cos^{-1} (z)} {\sqrt{x^2 + y^2 + z^2}} \\
  \phi &= \text{atan2}(y, x)
  \end{align*}

We now compute the coordinates on the sensor plane, which are related to the image coordinates by the stretch matrix and distortion centre.

.. math::
  :name: projection

  \begin{align*}
  \rho &= g(\theta) \\
  \begin{pmatrix} u' & v' \end{pmatrix}^T &= \rho \cdot \begin{pmatrix} \cos(\phi) \sin(\phi) \end{pmatrix}^T
  \end{align*}

Undistortion can be performed by backprojecting pixels lying on an imaginary perspective image plane and projecting these rays back onto the fisheye image.

.. todo:: add side by side showing 'undistortion'


Example
-------

.. todo:: Add side-by-side images showing a distorted person standing upright in the fisheye image and visibly rotated in the perspective image.
.. todo:: fix this part at the bottom a bit so it melds well with the examples

The `reproject_perspective` function provides a convenient method for reprojecting an arbitrary image region or world points into a perspective projection.

Note: Sometimes the perspective projection appears rotated:

This skew is often caused by the camera's extrinsic configuration — e.g., pitching the camera down skews the perspective projection near the edges.

You can correct this by providing a rotation matrix containing the camera's pitch and roll angles to the perspective projection function.

