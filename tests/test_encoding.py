from polynomial_codifiers.PolyEncoder import PolyEncoder
import numpy as np


def test_encoding():
    g = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=int)
    u1 = np.array([1, 0, 0, 0, 0, 1], dtype=int)
    u2 = np.array([1, 1, 1, 1, 1, 1], dtype=int)
    u3 = np.array([0, 0, 0, 0, 1, 1], dtype=int)

    encoder = PolyEncoder(g)

    assert np.array_equal(encoder.encode(u1), np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int))
    assert np.array_equal(encoder.encode(u2), np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1], dtype=int))
    assert np.array_equal(encoder.encode(u3), np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1], dtype=int))