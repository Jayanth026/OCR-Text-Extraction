from src.utils import to_python_type

import numpy as np

def test_to_python_type_numpy_scalars():
    assert isinstance(to_python_type(np.int32(5)), int)
    assert isinstance(to_python_type(np.float64(3.14)), float)

def test_to_python_type_numpy_array():
    arr = np.array([1, 2, 3], dtype=np.int32)
    out = to_python_type(arr)
    assert out == [1, 2, 3]
    assert isinstance(out, list)