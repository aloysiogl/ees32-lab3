import numpy as np


class Encoder:
    def __init__(self, P):
        self.G = self.__concatenate(P)

    def setParity(self, P):
        self.G = self.__concatenate(P)

    @staticmethod
    def __concatenate(P):
        """Concatenates identity matrix in the beginning of P"""
        n = P.shape[0]
        return np.concatenate([np.identity(n, dtype=int), P], 1)

    def encode(self, seq):
        """Encodes a sequence of bytes"""
        assert seq.shape == (self.G.shape[0],) or seq.shape == (1, self.G.shape[0])
        return np.mod(np.dot(seq, self.G), 2)
