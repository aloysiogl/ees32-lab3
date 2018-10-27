import numpy as np
import preprocessing.TransitionAnalyser as ta
from convolutional_codifiers.ConvEncoder import ConvEncoder


class ConvDecoder:
    def __init__(self, table):
        self.table = table

        assert len(table) > 0

        self.chunk_size = len(table[0][0][0])

    def decode(self, message):
        # Splitting message
        split_message = []
        for i in range(len(message)//self.chunk_size):
            split_message.append(message[i*self.chunk_size:self.chunk_size*(i+1)])

        # Decoding
        paths = [[] for i in range(len(self.table))]
        weigths = [-1 for i in range(len(self.table))]
        weigths[0] = 0

        ######

        ######

        #####
        #Heladio
        #####


    def distance_transition(self, initial, trans, seq):
        output, new = self.table[initial][trans]
        cost = 0
        for i in range(self.chunk_size):
            cost += (output[i] + seq[i]) % 2
        return cost


if __name__ == '__main__':
    p1 = np.array([1, 1, 0, 1])
    p2 = np.array([1, 0, 1, 1])
    p3 = np.array([1, 1, 1, 1])
    polarr = [p1, p2, p3]
    gen = ta.TransitionAnalyser(polarr)

    codifier = ConvEncoder(gen.table_generate(3))
    u = [0, 0, 1, 0]
    print(codifier.encode(u))
    for i in gen.table_generate(3):
        print(i)

    decoder = ConvDecoder(gen.table_generate(3))
    decoder.decode(codifier.encode(u))