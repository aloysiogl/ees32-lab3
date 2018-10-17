import numpy as np


class TransitionAnalyser:
    def __init__(self, pol_list):
        self.pol_list = pol_list
        self.m = max(map(len, pol_list))

    def set_polynomials(self, pol_list):
        self.pol_list = pol_list
        self.m = max(map(len, pol_list))

    def transition(self, curr, inp):
        curr_state = curr

        # Reading current state
        curr_array = []
        for i in range(self.m-1):
            curr_array.append(curr % 2)
            curr = curr//2
        curr_array = list(reversed(curr_array))

        outputs = []

        assert inp // 2 < 1

        # Generating outputs from current state
        for pol in self.pol_list:
            partial_curr = curr_array + [inp]
            outputs.append(np.mod(np.dot(partial_curr, pol), 2))

        # Generating next state
        next = ((curr_state << 1) + inp) % (2**(self.m-1))

        return outputs, next

    def table_generate(self, states):
        table = []
        for i in range(2**states):
            transition0 = self.transition(i, 0)
            transition1 = self.transition(i, 1)
            table.append([transition0, transition1])
        return table


if __name__ == "__main__":
    p1 = np.array([0, 0, 1])
    p2 = np.array([0, 1, 1])
    p3 = np.array([1, 0, 1])
    polarr = [p1, p2, p3]
    gen = TransitionAnalyser(polarr)
    # gen.transition(0b111111, 1)
    for ele in gen.table_generate(2):
        print(gen.table_generate(2).index(ele), ele)
