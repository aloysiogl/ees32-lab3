# -*- coding: utf-8 -*-
import six, sys

# needed for utf-encoding on python 2:
if six.PY2:
    reload(sys)
    sys.setdefaultencoding('utf8')
import matplotlib.pyplot as plt
import numpy as np
import time

from scipy.stats import norm
from math import sqrt
from Channel import Channel
from matrix_codifiers.DecoderHamming import DecoderHamming
from matrix_codifiers.EncoderHamming import EncoderHamming
from preprocessing.MatrixReader import MatrixReader
from preprocessing.poly_generator import octal2poly
from preprocessing.TransitionAnalyser import TransitionAnalyser
from polynomial_codifiers.PolyEncoder import PolyEncoder
from polynomial_codifiers.PolyDecoder import PolyDecoder
from convolutional_codifiers.ConvEncoder import ConvEncoder
from convolutional_codifiers.ConvDecoder import ConvDecoder

# Script which generates N random bits and simulates a random channel with probabilities ranging from 0.5 to 10e-6.
# It then plots a graph comparing different encoding processes.
N = 1000

# Definition for polynomial codifier
chosen_matrices = [5]

# Definition for convolutional codifier
chosen_polynomials = [([[1, 3], [1, 5], [1, 7]], 3),
                      ([[2, 5], [3, 3], [3, 7]], 4),
                      ([[1, 1, 7], [1, 2, 7], [1, 5, 5]], 6)]

# Plotting types
plot_normal = True
plot_hamming = False
plot_cyclic = False
plot_conv = False

# Reading matrices
reader = MatrixReader()
reader.read()


def normal_process(codes, channels):
    outputs = [None] * len(channels)

    for c in range(len(channels)):
        outputs[c] = channels[c].add_noise(np.array(codes))

    return outputs


def hamming_process(codes, channels):
    # Encoding
    hamming_codes = []
    for c in range(len(codes) // 4):
        hamming_codes.append(np.array(codes[c * 4:c * 4 + 4]))

    hamming_encoder = EncoderHamming()
    encodes = [hamming_encoder.encode(code) for code in hamming_codes]

    # Channeling
    outputs = [None] * len(channels)
    for c in range(len(channels)):
        outputs[c] = np.array([channels[c].add_noise(code) for code in encodes])

    # Decoding
    hamming_decoder = DecoderHamming()
    for c in range(len(channels)):
        outputs[c] = np.array([hamming_decoder.decode(code) for code in outputs[c]])
        outputs[c] = outputs[c].flatten()

    return outputs


def cyclic_process(index, codes, channels):
    # Encoding
    cyclic_codes = []
    gen = np.flip(np.array(reader.get_matrix(index)[0]))
    n = len(gen) - PolyDecoder.degree(gen) + 1
    poly_encoder = PolyEncoder(gen)
    for c in range(len(codes) // n):
        cyclic_codes.append(np.array(codes[c * n:c * n + n]))

    encodes = [poly_encoder.encode(code) for code in cyclic_codes]

    # Channeling
    outputs = [None] * len(channels)
    for c in range(len(channels)):
        outputs[c] = np.array([channels[c].add_noise(code) for code in encodes])

    # Decoding
    poly_decoder = PolyDecoder(gen, len(gen))

    for c in range(len(channels)):
        print("processing...")
        outputs[c] = np.array([poly_decoder.decode(code) for code in outputs[c]])
        outputs[c] = outputs[c].flatten()
    print("over")
    return outputs


def convolutional_process(table, codes, channels):
    # Encoding
    conv_encoder = ConvEncoder(table)
    encode = conv_encoder.encode(codes)

    # Channeling
    outputs = [None] * len(channels)

    for c in range(len(channels)):
        outputs[c] = np.array(channels[c].add_noise(np.array(encode, dtype=int)), dtype=int)

    # Decoding
    alpha = 1
    for c in range(len(channels)):
        print('Iteracao {:02}/{}'.format(alpha, len(channels)))
        alpha += 1
        conv_decoder = ConvDecoder(table, channels[c].get_p())
        outputs[c] = np.array(conv_decoder.decode(outputs[c]))

    return outputs


def p_map(eb_n0s, ratio):
    ps = []
    for eb_n0 in eb_n0s:
        ps.append(1 - norm.cdf(sqrt(2 * eb_n0 / ratio)))
    return ps


if __name__ == "__main__":
    t = time.time()

    # Generating random codes
    codes = np.rint(np.random.random_sample(N)).astype(bool)

    # Generating channels with different noises to plot a graph
    ei_n0 = [0, 0.35416315, 0.82118721, 1.35277173, 2.10894229, 2.70594722,
             3.3174483, 4.1419075, 4.77476785, 5.41378309, 6.26609665, 6.91554181,
             7.56835261, 8.43569456, 9.09464674, 9.75571048, 10.63242365]

    channels = [Channel(p) for p in p_map(ei_n0, 1)]

    # Generating outputs without encoding, with hamming encoding and with our encoding
    if plot_normal:
        normal_outputs = normal_process(codes, channels)
    if plot_hamming:
        hamming_outputs = hamming_process(codes, channels)
    if plot_cyclic:
        cyclic_outputs = [cyclic_process(i, codes, channels) for i in chosen_matrices]
    if plot_conv:
        # Generating chosen matrices for convolutional output
        graph_matrices = []
        for tup in chosen_polynomials:
            polynomials = [octal2poly(pol, tup[1]) for pol in tup[0]]
            analyser = TransitionAnalyser(polynomials)
            graph_matrices.append(analyser.table_generate(tup[1]))

        # Generating convolutional outputs
        iteracao = 1
        convolutional_outputs = []
        for matrix in graph_matrices:
            print('PROCESSO: {}/{}'.format(iteracao, 3))
            convolutional_outputs.append(convolutional_process(matrix, codes, channels))
            iteracao += 1
        # convolutional_outputs = [convolutional_process(matrix, codes, channels) for matrix in graph_matrices]

    # Comparing outputs and plotting a graph
    normal_ps = []
    hamming_ps = []
    if plot_cyclic:
        cyclic_ps = [[] for p in range(len(cyclic_outputs))]
    if plot_conv:
        convolutional_ps = [[] for p in range(len(convolutional_outputs))]
    for c in range(len(channels)):
        if plot_normal:
            normal_ps.append(1 - np.count_nonzero(normal_outputs[c] == codes) / N)
        if plot_hamming:
            hamming_ps.append(1 - np.count_nonzero(hamming_outputs[c] == codes) / N)
        if plot_cyclic:
            for i in range(len(cyclic_outputs)):
                cyclic_ps[i].append((1 - np.count_nonzero(cyclic_outputs[i][c] == codes) / N))
        if plot_conv:
            for i in range(len(convolutional_outputs)):
                assert len(convolutional_outputs[i][c]) == len(codes)
                convolutional_ps[i].append((1 - np.count_nonzero(convolutional_outputs[i][c] == codes) / N))

    if plot_normal:
        normal_ps = np.log(normal_ps) / np.log(10)
    if plot_hamming:
        hamming_ps = np.log(hamming_ps) / np.log(10)
    if plot_cyclic:
        for i in range(len(cyclic_ps)):
            cyclic_ps[i] = np.log(cyclic_ps[i]) / np.log(10)
    if plot_conv:
        for i in range(len(convolutional_ps)):
            convolutional_ps[i] = np.log(convolutional_ps[i]) / np.log(10)

    print("Time taken:", time.time() - t, "s")
    fig, ax = plt.subplots()
    plt.xlim([0, 10])
    plt.xlabel("EI/N0")
    plt.ylabel("log(Probabilidade de erro de bit)")
    if plot_normal:
        plt1 = plt.plot(ei_n0, normal_ps, label="Não codificado")
    if plot_hamming:
        plt2 = plt.plot(ei_n0, hamming_ps, label="Hamming")
    if plot_cyclic:
        plt_cycl = []
        for i in range(len(cyclic_ps)):
            plt_cycl.append(plt.plot(ei_n0, cyclic_ps[i], label=reader.get_name(chosen_matrices[i])))
    if plot_conv:
        plt_conv = []
        for i in range(len(convolutional_ps)):
            plt_conv.append(plt.plot(ei_n0, convolutional_ps[i], label="Polinomio " + str(i)))
    ax.legend()
    plt.show()
