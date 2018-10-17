import numpy as np

error = np.zeros(7, dtype=np.bool)
error[0] = 1
Ht = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=bool)
print(error.shape)
print(Ht.shape)

print(np.dot(error, Ht))