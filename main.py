import numpy as np
from numba import cuda

from matmul import matmul_cpu, matmul_naive, matmul_tiled, TPB
from activation import relu_forward

if __name__ == "__main__":

    M = K = N = 512

    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)

    A_d = cuda.to_device(A)
    B_d = cuda.to_device(B)
    C_d = cuda.device_array((M, N), dtype=np.float32)

    threads = (TPB, TPB)
    blocks = ((N + TPB - 1) // TPB, (M + TPB - 1) // TPB)

    matmul_tiled[blocks, threads](A_d, B_d, C_d)
    C = C_d.copy_to_host()

    X_d = cuda.to_device(C)
    Y_d = cuda.device_array_like(C)

    relu_forward[blocks, threads](X_d, Y_d)
    Y = Y_d.copy_to_host()

    print("Forward pass done. Output shape:", Y.shape)
