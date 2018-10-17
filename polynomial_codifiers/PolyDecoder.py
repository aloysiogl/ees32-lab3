import numpy as np
from polynomial_codifiers.PolyEncoder import PolyEncoder
from polynomial_codifiers.PolyOps import poly_divmod
from preprocessing.matrix_generator import kbits


class PolyDecoder:
    def __init__(self, g, codeword_length):
        self.g = g
        self.sindromes = []
        self.update_sindromes(codeword_length)

    def set_generator(self, g):
        self.g = g
        self.update_sindromes()

    def set_generator_lsb_first(self, g):
        self.g = np.flip(g)
        self.update_sindromes()

    def update_sindromes(self, codeword_length):
        sind_map = {}

        for i in range(1, codeword_length + 1):
            errors = [[int(y) for y in list(x)] for x in kbits(codeword_length, i)]
            for err in errors:
                err_key = tuple(self.calc_sindrome(err))
                if np.count_nonzero(err_key) == 0:
                    continue
                if err_key not in sind_map or (sind_map[err_key][0] == 0 and sind_map[err_key][1] == i):
                    sind_map[err_key] = (err[len(err) - 1], i)
        for key in sind_map:
            if sind_map[key][0] == 1:
                self.sindromes.append(np.array(key))

    def calc_sindrome(self, message):
        sindrome = self.reduce_degree(self.pol_div(message, self.g)[1], self.degree(self.g))
        return sindrome

    def decode(self, message):
        g = self.reduce_degree(self.g, self.degree(self.g))

        sindrome = self.calc_sindrome(message)
        message_changer = np.array([0 for i in range(len(message) - 1)] + [1], dtype=int)
        rotations = 0

        # While sindrome is not all zeros
        while np.count_nonzero(sindrome) != 0 :
            # Check if sindrome is in set of sindromes
            found = False
            for sind in self.sindromes:
                if np.array_equal(sindrome, sind):
                    message = np.mod(message + message_changer, 2)

                    sindrome = self.calc_sindrome(message)
                    found = True
                    break
            if found:
                continue
            # If sindrome not found in set
            message = self.rotate(message, -1)
            sindrome = self.rotate(sindrome, -1)
            rotations += 1
            if sindrome[0] == 1:
                sindrome = np.mod(sindrome + g, 2)
                sindrome[0] = 0

        return self.reduce_degree(self.pol_div(self.rotate(message, rotations), self.g)[0],
                                  len(self.g) - self.degree(g) + 1)

    @staticmethod
    def degree(p):
        for i in range(len(p)):
            if p[i] != 0:
                return len(p) - i
        return 0

    @staticmethod
    def reduce_degree(p, d):
        arr = [0 for i in range(d)]
        for i in range(d):
            arr[d - i - 1] = p[len(p) - i - 1]
        return np.array(arr)

    @staticmethod
    def rotate(arr, rot):
        new_arr = [arr[(i - rot) % len(arr)] for i in range(len(arr))]
        return np.array(new_arr)

    @staticmethod
    def pol_div(n, d):
        # def leading(p):
        #     for i in range(len(p)):
        #         if p[i] != 0:
        #             return len(p) - i - 1
        #
        # def shift(p, ind):
        #     aws = np.array([0 for i in range(len(p))], dtype=int)
        #
        #     for i in range(len(p)):
        #         if (i - ind) >= 0:
        #             aws[i - ind] = p[i]
        #     return aws
        #
        # len_dif = len(n) - len(d)
        #
        # assert len_dif >= 0
        #
        # d = np.concatenate([np.array([0 for i in range(len_dif)], dtype=int), d], 0)
        #
        # q = np.array([0 for i in range(len(n))], dtype=int)
        # r = np.array(n)
        #
        # while np.count_nonzero(r) != 0 and leading(r) >= leading(d):
        #     t = shift(d, leading(r) - leading(d))
        #     q[len(q) - (leading(r) - leading(d)) - 1] = 1
        #     r = np.mod(r - t, 2)

        if np.count_nonzero(n) == 0:
            q1 = np.array([0 for i in range(len(n))], dtype=int)
            r1 = np.array([0 for i in range(len(n))], dtype=int)
        else:
            q1, r1 = poly_divmod(list(PolyDecoder.reduce_degree(n, PolyDecoder.degree(n))),
                                 list(PolyDecoder.reduce_degree(d, PolyDecoder.degree(d))))
            len_dif = len(n) - len(q1)
            q1 = np.mod(np.concatenate([np.array([0 for i in range(len_dif)], dtype=int), q1], 0), 2)
            len_dif = len(n) - len(r1)
            r1 = np.mod(np.concatenate([np.array([0 for i in range(len_dif)], dtype=int), r1], 0), 2)
        # assert np.array_equal(q,q1)
        # assert np.array_equal(r,r1)

        return q1, r1


if __name__ == "__main__":
    gen = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    encoder = PolyEncoder(gen)
    decoder = PolyDecoder(gen, codeword_length=8)

    message = encoder.encode(np.array([0, 0, 1]))

    message = np.mod(message + np.array([0, 0, 0, 0, 0, 0, 0, 0]), 2)
    print(decoder.sindromes)
    print('codeword', message)
    print('begin decoding')
    decoded = decoder.decode(message)
    print('decoded', decoded)