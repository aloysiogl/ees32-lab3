# -*- coding: utf-8 -*-
import six, sys

# needed for utf-encoding on python 2:
if six.PY2:
    reload(sys)
    sys.setdefaultencoding('utf8')
import matplotlib.pyplot as plt
import numpy as np
import time

from Channel import Channel
from matrix_codifiers.Decoder import Decoder
from matrix_codifiers.Encoder import Encoder
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
N = 100080

# Definition for polynomial codifier
chosen_matrices = [5]

# Definition for convolutional codifier
chosen_polynomials = [([[1, 3], [1, 5], [1, 7]], 3),
                      ([[2, 5], [3, 3], [3, 7]], 4),
                      ([[1, 1, 7], [1, 2, 7], [1, 5, 5]], 6)]

# Plotting types
plot_normal = True
plot_hamming = True
plot_cyclic = False
plot_conv = False
plot_improved = True

# Defining generator matrices

P6 = np.array([[1, 1, 1, 0, 0, 0],
               [1, 1, 0, 1, 0, 0],
               [1, 1, 0, 0, 1, 0],
               [1, 1, 0, 0, 0, 1],
               [1, 0, 1, 1, 0, 0],
               [1, 0, 1, 0, 1, 0],
               [1, 0, 1, 0, 0, 1],
               [1, 0, 0, 1, 1, 0]])

P9 = np.array([[1, 1, 1, 1, 0, 0, 0, 0, 0],
               [1, 1, 0, 0, 1, 1, 0, 0, 0],
               [1, 1, 0, 0, 0, 0, 1, 1, 0],
               [1, 0, 1, 0, 1, 0, 1, 0, 0],
               [1, 0, 1, 0, 0, 1, 0, 0, 1],
               [1, 0, 0, 1, 0, 1, 0, 1, 0],
               [1, 0, 0, 1, 0, 0, 1, 0, 1],
               [1, 0, 0, 0, 1, 0, 0, 1, 1],
               [0, 1, 1, 0, 0, 0, 0, 1, 1],
               [0, 1, 0, 1, 1, 0, 0, 0, 1],
               [0, 1, 0, 0, 0, 1, 1, 0, 1],
               [1, 1, 1, 1, 1, 1, 1, 1, 1]])

P12 = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                [1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
                [1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                [1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
                [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0],
                [1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
                [1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
                [1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0],
                [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0],
                [1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1],
                [1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1],
                [1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1]])

P15 = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                [1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                [1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                [1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1],
                [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0],
                [1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0]])

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


def improved_process(P, codes, channels):
    improved_codes = []
    n = P.shape[0]
    # print(n)
    # exit()
    assert len(codes) % n == 0
    for c in range(len(codes) // n):
        improved_codes.append(np.array(codes[c * n:c * n + n]))

    # Encoding
    improved_encoder = Encoder(P)
    encodes = [improved_encoder.encode(code) for code in improved_codes]

    # Channeling
    outputs = [None] * len(channels)
    for c in range(len(channels)):
        outputs[c] = np.array([channels[c].add_noise(code) for code in encodes])

    # Decoding
    n = P.shape[1] // 3
    improved_decoder = Decoder(P, n + 1)
    for c in range(len(channels)):
        outputs[c] = np.array([improved_decoder.decode(code) for code in outputs[c]])
        print("processing_imporved...")
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
    conv_decoder = ConvDecoder(table)
    for c in range(len(channels)):
        outputs[c] = np.array(conv_decoder.decode(outputs[c]))

    return outputs


if __name__ == "__main__":
    t = time.time()

    # Generating random codes
    codes = np.rint(np.random.random_sample(N)).astype(bool)

    # Generating channels with different noises to plot a graph
    ps = [0.5, 0.2, 0.1, 5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5, 5e-6, 2e-6]
    channels = [Channel(p) for p in ps]

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
        convolutional_outputs = [convolutional_process(matrix, codes, channels) for matrix in graph_matrices]
    if plot_improved:
        improved_outputs6 = improved_process(P6, codes, channels)
        improved_outputs9 = improved_process(P9, codes, channels)
        improved_outputs12 = improved_process(P12, codes, channels)
        improved_outputs15 = improved_process(P15, codes, channels)

    # Comparing outputs and plotting a graph
    normal_ps = []
    hamming_ps = []
    improved_ps6 = []
    improved_ps9 = []
    improved_ps12 = []
    improved_ps15 = []
    if plot_cyclic:
        cyclic_ps = [[] for p in range(len(cyclic_outputs))]
    if plot_conv:
        convolutional_ps = [[] for p in range(len(convolutional_outputs))]
    for c in range(len(channels)):
        if plot_normal:
            normal_ps.append(1 - np.count_nonzero(normal_outputs[c] == codes) / N)
        if plot_hamming:
            hamming_ps.append(1 - np.count_nonzero(hamming_outputs[c] == codes)/N)
        if plot_cyclic:
            for i in range(len(cyclic_outputs)):
                cyclic_ps[i].append((1 - np.count_nonzero(cyclic_outputs[i][c] == codes) / N))
        if plot_conv:
            for i in range(len(convolutional_outputs)):
                assert len(convolutional_outputs[i][c]) == len(codes)
                convolutional_ps[i].append((1 - np.count_nonzero(convolutional_outputs[i][c] == codes) / N))
        if plot_improved:
            improved_ps6.append(1 - np.count_nonzero(improved_outputs6[c] == codes) / N)
            improved_ps9.append(1 - np.count_nonzero(improved_outputs9[c] == codes) / N)
            improved_ps12.append(1 - np.count_nonzero(improved_outputs12[c] == codes) / N)
            improved_ps15.append(1 - np.count_nonzero(improved_outputs15[c] == codes) / N)
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
    if plot_improved:
        improved_ps6 = np.log(improved_ps6) / np.log(10)
        improved_ps9 = np.log(improved_ps9) / np.log(10)
        improved_ps12 = np.log(improved_ps12) / np.log(10)
        improved_ps15 = np.log(improved_ps15) / np.log(10)
    ps = np.log(ps) / np.log(10)

    print("Time taken:", time.time() - t, "s")
    fig, ax = plt.subplots()
    plt.xlim([0, -6])
    plt.xlabel("log(p)")
    plt.ylabel("log(Probabilidade de erro de bit)")
    if plot_normal:
        plt1 = plt.plot(ps, normal_ps, label="Não codificado")
    if plot_hamming:
        plt2 = plt.plot(ps, hamming_ps, label="Hamming")
    if plot_cyclic:
        plt_cycl = []
        for i in range(len(cyclic_ps)):
            plt_cycl.append(plt.plot(ps, cyclic_ps[i], label=reader.get_name(chosen_matrices[i])))
    if plot_conv:
        plt_conv = []
        for i in range(len(convolutional_ps)):
            plt_conv.append(plt.plot(ps, convolutional_ps[i], label="Polinomio " + str(i)))
    if plot_improved:
        plt3 = plt.plot(ps, improved_ps6, label="Código 14x6")
        plt4 = plt.plot(ps, improved_ps9, label="Código 21x9")
        plt5 = plt.plot(ps, improved_ps12, label="Código 28x12")
        plt6 = plt.plot(ps, improved_ps15, label="Código 35x15")
    ax.legend()
    plt.show()
