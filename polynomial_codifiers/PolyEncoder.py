import numpy as np


class PolyEncoder:
    def __init__(self, g):
        self.g = g

    def set_generator(self, g):
        self.g = g

    def set_generator_lsb_first(self, g):
        self.g = np.flip(g)

    def encode(self, u):
        """Encodes a sequence of bytes"""
        mult = np.mod(np.polymul(self.g, u), 2)
        len_dif = len(self.g) - len(mult)

        # Asserting valid u
        assert len_dif >= 0

        return np.concatenate([np.array([0 for i in range(len_dif)], dtype=int), mult], 0)
