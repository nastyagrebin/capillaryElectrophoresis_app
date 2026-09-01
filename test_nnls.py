import numpy as np
from scipy.optimize import nnls
A = np.random.rand(10, 3)
A[:, 0] = 1e-250

max_A = np.max(A, axis=0)
A[:, max_A < 1e-10] = 0.0

b = np.random.rand(10)
x, rnorm = nnls(A, b)
print('x:', x)
