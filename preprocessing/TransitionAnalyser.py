import numpy as np


class TransitionAnalyser:
    def __init__(self, pol_list):
        self.pol_list = pol_list
        self.m = max(map(len, pol_list))

    def set_polynomials(self, pol_list):
        self.pol_list = pol_list
        self.m = max(map(len, pol_list))

    def transition(self, curr, inp):
        curr_array = []
        for i in range(self.m):
            curr_array.append(curr % 2)
            curr = curr//2
        print(curr_array)


if __name__ == "__main__":
    p1 = np.array([1,0,0,1,1,1,1])
    p2 = np.array([1,1,0,1,1,0,1])
    polarr = [p1, p2]
    gen = TransitionAnalyser(polarr)
    TransitionAnalyser.transition(0b110, 1)
