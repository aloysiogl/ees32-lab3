import numpy as np


class SMGenerator:
    def __init__(self, pol_list):
        self.set_polynomials(pol_list)

    def set_polynomials(self, pol_list):
        self.pol_list = pol_list
        print(max(map(len, pol_list)))

    def generate(self):
        pass

if __name__ == "__main__":
    p1 = np.array([1,0,0,1,1,1,1])
    p2 = np.array([1,1,0,1,1,0,1])
    polarr = [p1, p2]
    gen = SMGenerator(polarr)

