import os
from preprocessing.DistanceFinder import DistanceFinder
from preprocessing.MatrixReader import MatrixReader

mask = len("preprocessing")
directory = os.path.dirname(os.path.abspath(__file__))[:-mask]+"/matrizes"

if __name__ == "__main__":

    reader = MatrixReader()
    reader.read()
    finder = DistanceFinder()
    choices_map = {}

    for index in reader.get_matrices_indexes():
        array = reader.get_matrix(index)
        finder.set_g(array)
        key = reader.get_name(index)
        if key in choices_map:
            if finder.get_distance() > choices_map[key][0]:
                choices_map[key] = (finder.get_distance(), index)
        else:
            choices_map[key] = (finder.get_distance(), index)
        print("Distance : " + str(finder.get_distance()))
        print("Matrix:")
        print(array)

    # Printing the files chosen
    print("Chosen matrices: ")
    choices = sorted(choices_map.values(), key=lambda choice: int(choice[1]))

    for choice in choices:
        print("File: " + str(choice[1]) + " distance: " + str(choice[0]))
