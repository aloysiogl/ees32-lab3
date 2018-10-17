from polynomial_codifiers.PolyDecoder import PolyDecoder
from polynomial_codifiers.PolyEncoder import PolyEncoder
from polynomial_codifiers.PolyOps import poly_divmod
import numpy as np


def test_division():
    num = np.array([0, 0, 0, 0, 1, 0, 1, 0], dtype=int)
    den = np.array([1, 0, 0], dtype=int)
    res = np.array([0, 0, 0, 0, 0, 0, 1, 1])
    encoder = PolyEncoder(num)

    polymult = np.mod(encoder.encode(den) + res, 2)

    print(polymult)

    print(PolyDecoder.pol_div(polymult, num))
    print(list(PolyDecoder.reduce_degree(polymult, PolyDecoder.degree(polymult))))
    print(list(PolyDecoder.reduce_degree(num, PolyDecoder.degree(num))))
    q, r = poly_divmod(list(PolyDecoder.reduce_degree(polymult, PolyDecoder.degree(polymult))), list(PolyDecoder.reduce_degree(num, PolyDecoder.degree(num))))
    print(PolyDecoder.pol_div(polymult, den))
    print("null")
    print(poly_divmod([], [1, 0, 1]))


if __name__ == "__main__":
    test_division()
