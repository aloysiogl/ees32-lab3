import numpy as np
from math import sqrt


class GaussianChannel:

    def __init__(self, var):
        self.__var = var

    def get_p(self):
        return self.__var

    def set_p(self, p):
        self.__var = p

    def add_noise(self, v):
        """
        Receives a numpy.array of bool v and return a new array representing v with noise applied
        :param v: array of bool
        :return: new array
        """
        noise = np.random.normal(0, sqrt(self.__var), len(v))
        return list(noise+v)


if __name__ == '__main__':
    u = np.random.random_integers(0, 1, 10)
    ch = GaussianChannel(1)
    for i in range(len(u)):
        if u[i] == 0:
            u[i] = -1
    v = ch.add_noise(u)
    for i in range(10):
        print('{:2}'.format(u[i]), v[i])
