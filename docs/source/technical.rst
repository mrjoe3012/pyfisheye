Technical Details
=================

This implementation is based on [1]_ and [2]_.

pyfisheye uses trust-region constrained optimization (`trust-constr`) as a replacement for MATLAB’s `lsqlin` function.

This approach converges quickly and supports linear constraints, useful for enforcing monotonicity during calibration.

Note the following:

- The original toolbox enforces a monotonic increasing function; pyfisheye finds parameters for a monotonic decreasing function, resulting in a coordinate system where the positive z-axis is forward (see :ref:`Projection<projection>`).

- During linear optimization, radial distances are normalized by the *image radius* — defined as the maximum distance from the distortion centre to any valid pixel coordinate — to improve numerical stability.

- Intrinsic parameters are unnormalized before nonlinear refinement.

- Instead of a high-degree polynomial for inverse mapping, pyfisheye uses a precomputed lookup table with linear interpolation.

This method is less prone to numerical instability but about 5x slower: on a 640x480 image, the lookup takes ~50 ms versus ~10 ms for polynomial evaluation (tested on an AMD Ryzen 7 5800H).

Polynomial support for inverse mapping will be implemented in a future release (contributions welcome).

References
----------

.. [1] Scaramuzza et al., "Omnidirectional Camera Calibration," IEEE IROS 2006, https://doi.org/10.1109/IROS.2006.282372
.. [2] Urban et al., "Improved omnidirectional camera calibration," ISPRS Journal 2015, https://doi.org/10.1016/j.isprsjprs.2015.06.005