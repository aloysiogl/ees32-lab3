# -*- coding: utf-8 -*-
import numpy as np
import os
import re

mask = len("preprocessing")
directory = os.path.dirname(os.path.abspath(__file__))[:-mask]+"/matrizes"


class MatrixReader:

    def __init__(self):
        self.matrices = {}
        self.names = {}

    def read(self):
        for filename in os.listdir(directory):
            if filename.endswith(".csv"):
                file = open(directory + "/" + filename)
                lines = file.readlines()
                array = np.array(list(map(lambda line: list(map(int, line.strip('\n').split(","))), lines)))
                key = re.sub(r'.*{', '{', filename)
                numb = int(filename.partition(":")[0])
                self.matrices[numb] = array
                self.names[numb] = "Cíclico " + key[:-4]
                file.close()

    def get_matrix(self, index):
        return self.matrices[index]

    def get_matrices_indexes(self):
        return self.matrices

    def get_name(self, index):
        return self.names[index]