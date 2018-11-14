import numpy as np
import preprocessing.TransitionAnalyser as ta
from math import log
from convolutional_codifiers.ConvEncoder import ConvEncoder


class ConvDecoder:
    def __init__(self, table, prob):
        self.table = table

        assert len(table) > 0

        self.chunk_size = len(table[0][0][0])

        self.p = prob

    def decode(self, message):
        # Splitting message
        split_message = []
        for j in range(len(message) // self.chunk_size):
            split_message.append(message[j * self.chunk_size:self.chunk_size * (j + 1)])

        # Decoding
        paths = [[] for i in range(len(self.table))]
        weights = [-1 for i in range(len(self.table))]
        weights[0] = 0

        for chunk in split_message:
            current_paths = [[] for i in range(len(self.table))]
            current_weights = [-1 for i in range(len(self.table))]
            for j in range(len(weights)):
                if weights[j] < 0:
                    continue
                for k in range(2):
                    transition_weigth = self.exact_probability(j, k, chunk)
                    final = self.table[j][k][1]
                    if current_weights[final] == -1 or current_weights[final] > transition_weigth + weights[j]:
                        current_weights[final] = transition_weigth + weights[j]
                        current_paths[final] = paths[j] + [k]
            paths = current_paths
            weights = current_weights

        x = weights.index(min(weights))
        return paths[x]

    def hamming_distance(self, initial, trans, seq):
        output, new = self.table[initial][trans]
        cost = 0
        for i in range(self.chunk_size):
            cost += (output[i] + seq[i]) % 2
        return cost

    def exact_probability(self, initial, trans, seq):
        hamming = self.hamming_distance(initial, trans, seq)
        cost = hamming*log(self.p, 10) + (self.chunk_size - hamming)*log(1 - self.p, 10)
        return -cost

    def euclidean_distance(self, initial, trans, seq):
        output, new = self.table[initial][trans]
        cost = 0
        for i in range(self.chunk_size):
            cost += (output[i] - seq[i])**2
        return cost


if __name__ == '__main__':
    p1 = np.array([1, 1, 0, 1])
    p2 = np.array([1, 0, 1, 1])
    p3 = np.array([1, 1, 1, 1])
    polarr = [p1, p2, p3]
    gen = ta.TransitionAnalyser(polarr)

    codifier = ConvEncoder(gen.table_generate(3))
    u = [0, 1, 1, 0]
    print('Encode =', codifier.encode(u))

    decoder = ConvDecoder(gen.table_generate(3), 0.2)
    msg = np.mod(np.array(codifier.encode(u)) + np.array([0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0]), 2)
    print('Sent   =', msg)
    print('Decode =', np.array(decoder.decode(msg)))
