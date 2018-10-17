from random import random as rand


class Channel:

    def __init__(self, p):
        self.__p = p

    def get_p(self):
        return self.__p

    def set_p(self, p):
        self.__p = p

    def add_noise(self, v):
        """
        Receives a numpy.array of bool v and return a new array representing v with noise applied
        :param v: array of bool
        :return: new array
        """
        idx = 0
        noised_v = list(v)
        for bit in noised_v:
            num = rand()
            bit = int(bit)
            if num < self.get_p():
                bit = (bit+1) % 2
            bit = bool(bit)
            noised_v[idx] = bit
            idx += 1
        return noised_v
