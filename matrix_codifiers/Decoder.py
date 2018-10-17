import itertools as it
import numpy as np


class Decoder:
    def __init__(self, P, b):
        self.__ht = self.__concatenate(P)
        self.tot = sum(P.shape)
        self.__map = self.__get_sindrome_map(b)

    def setParity(self, P):
        self.__ht = self.__concatenate(P)
        self.tot = sum(P.shape)

    @staticmethod
    def __concatenate(P):
        """Concatenates identity matrix in the beginning of P"""
        n = P.shape[1]
        return np.concatenate([P, np.identity(n, dtype=int)], 0)

    def __get_sindrome_map(self, b):
        nrows = self.__ht.shape[0]
        ncols = self.__ht.shape[1]

        map = np.zeros((pow(2, ncols), nrows), dtype=bool)

        for k in range(1, b + 1):
            errors = [np.array(list(el), dtype=int) for el in self.__kbits(nrows, k)]
            indexes = [self.__bin2int(np.mod(np.dot(error, self.__ht), 2)) for error in errors]
            for i in range(len(indexes)):
                if (np.count_nonzero(map[indexes[i]])) == 0:
                    map[indexes[i]] = errors[i]

        # Mapping for no errors
        map[0] = np.zeros(nrows)

        return map

    @staticmethod
    def __kbits(n, k):
        """ Generates a list of combinations of n bits with k of them being 1. """

        result = []
        for bits in it.combinations(range(n), k):
            s = ['0'] * n
            for bit in bits:
                s[bit] = '1'
            result.append(''.join(s))
        return result

    @staticmethod
    def __bin2int(x):
        y = 0
        for i, j in enumerate(x):
            y += j << i
        return y

    def decode(self, code):
        assert (code.shape == (self.__ht.shape[0],))
        sindrome = np.mod(np.dot(code, self.__ht), 2)
        error = self.__map[self.__bin2int(sindrome)]

        return (code ^ error)[0:int(self.__ht.shape[0] * 4 / 7)]
