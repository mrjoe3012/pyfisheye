from pyfisheye.internal.utils.check_shapes import check_shapes
import numpy as np
import pytest

@check_shapes({
    'points' : 'N,3',
    'norms' : 'N'
})
def fn(points: np.ndarray, norms: np.ndarray) -> np.ndarray:
    return points * norms[:, None]

def test_check_shapes() -> None:
    fn(np.zeros((100, 3), dtype=np.float64), np.zeros((100,), dtype=np.float64))
    with pytest.raises(ValueError):
        fn(np.zeros((95, 3), dtype=np.float64), np.zeros((100,), dtype=np.float64))
    with pytest.raises(ValueError):
        fn(np.zeros((95, 3), dtype=np.float64), np.zeros((90,), dtype=np.float64))
    with pytest.raises(ValueError):
        fn(np.zeros((95, 2), dtype=np.float64), np.zeros((95,), dtype=np.float64))
