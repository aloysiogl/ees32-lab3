import numpy as np

class DistanceFinder:

    def __init__(self):
        self.inputs = []
        self.g = None
        self.k = None
        self.n = None
        self.dist = None

    def set_g(self, g):
        """Sets the G matix"""
        self.k, self.n = g.shape
        self.inputs = []
        self.g = g
        self.__calculate_distance()

    def __calculate_distance(self):
        for i in range(1, 2**self.k):
            inp = [(i >> x) % 2 for x in range(self.k)]
            self.inputs.append(np.array([inp]))

        dist = np.sum(np.mod(np.dot(self.inputs[0], self.g), 2))

        for inp in self.inputs:
            dist = min(dist, np.sum(np.mod(np.dot(inp, self.g), 2)))

        self.dist = dist

    def get_distance(self):
        return self.dist

