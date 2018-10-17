import numpy as np
import preprocessing.TransitionAnalyser as ta


class ConvEncoder:
    def __init__(self, table):
        self.table = table

    def encode(self, u):
        """Encodes a sequence of bytes"""
        v = []
        curr_state = 0
        for bit in u:
            output, curr_state = self.table[curr_state][bit]
            v = v + output

        return np.array(v)


if __name__ == '__main__':
    p1 = np.array([1, 1, 0, 1])
    p2 = np.array([1, 0, 1, 1])
    p3 = np.array([1, 1, 1, 1])
    polarr = [p1, p2, p3]
    gen = ta.TransitionAnalyser(polarr)

    codifier = ConvEncoder(gen.table_generate(3))
    u = [0, 0, 0, 0]
    print(codifier.encode(u))
