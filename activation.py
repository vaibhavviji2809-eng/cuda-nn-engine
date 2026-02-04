import numpy as np
from numba import cuda

@cuda.jit
def relu_forward(X, Y):
    i, j = cuda.grid(2)
    if i < X.shape[0] and j < X.shape[1]:
        v = X[i, j]
        Y[i, j] = v if v > 0 else 0.0
