import numpy as np


class DecoderHamming:
    def __init__(self):
        self.__ht = np.array([[1, 1, 1],
                              [1, 0, 1],
                              [1, 1, 0],
                              [0, 1, 1],
                              [1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]])

    def decode(self, code):
        assert(code.shape == (7,))
        sindrome = np.mod(np.dot(code, self.__ht), 2)
        error = np.zeros(7, dtype=bool)

        if np.all(sindrome == [1, 1, 1]):
            error[0] = True
        elif np.all(sindrome == [1, 0, 1]):
            error[1] = True
        elif np.all(sindrome == [1, 1, 0]):
            error[2] = True
        elif np.all(sindrome == [0, 1, 1]):
            error[3] = True
        elif np.all(sindrome == [1, 0, 0]):
            error[4] = True
        elif np.all(sindrome == [0, 1, 0]):
            error[5] = True
        elif np.all(sindrome == [0, 0, 1]):
            error[6] = True

        return (code ^ error)[0:4]
