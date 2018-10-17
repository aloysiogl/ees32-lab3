import numpy as np


class Generator:
    def __init__(self, dist):
        self.__n = dist - 2
        self.rows = []
        self.__combinations = [[] for i in range(self.__n)]
        self.__tempComb = [[] for i in range(self.__n)]

    def insert(self, x):
        for el in self.__tempComb:
            el.clear()
        self.__tempComb[0].append(x)

        for c in range(0, self.__n - 1):
            for e in self.__combinations[c]:
                xor = x ^ e
                if np.sum(xor) == 0:
                    return False
                self.__tempComb[c + 1].append(xor)
        for e in self.__combinations[self.__n - 1]:
            if np.sum(x ^ e) == 0:
                return False

        for c in range(len(self.__combinations)):
            self.__combinations[c] = self.__combinations[c] + self.__tempComb[c]
        self.rows.append(x)
        return True
