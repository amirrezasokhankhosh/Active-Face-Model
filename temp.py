import numpy as np

a = np.array([[1, 0], [0, -1]])
b = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])



print( (a @ b.T).T)