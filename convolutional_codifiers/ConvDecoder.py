import numpy as np
import preprocessing.TransitionAnalyser as ta


class ConvDecoder:
    def __init__(self, table):
        self.table = table

    def decode(self):
        pass

    def distance_transition(self, initial, trans, seq):
        output, new = self.table[initial][trans]
        cost = 0
        for i in range(self.chunk_size):
            cost += (output[i] + seq[i]) % 2
        return cost


if __name__ == '__main__':
    pass