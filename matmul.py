import os
os.environ['CUDA_HOME'] = os.environ.get('CONDA_PREFIX', '')
os.environ['NUMBA_CUDA_PTX_TARGET'] = '8.2'

import numpy as np
import time
from numba import cuda, float32
print(f"Numba Target PTX: {os.environ.get('NUMBA_CUDA_PTX_TARGET')}")

# CPU MATRIX MULTIPLICATION
# -----------------------------
def matmul_cpu(A, B):
    M, K = A.shape
    K, N = B.shape
    C = np.zeros((M, N), dtype=np.float32)
    for i in range(M):
        for j in range(N):
            s = 0.0
            for k in range(K):
                s += A[i, k] * B[k, j]
            C[i, j] = s
    return C


# -----------------------------
# NAIVE CUDA KERNEL
# -----------------------------
@cuda.jit
def matmul_naive(A, B, C):
    row, col = cuda.grid(2)

    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.0
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp


# -----------------------------
# TILED CUDA KERNEL (SHARED MEMORY)
# -----------------------------
TPB = 16

@cuda.jit
def matmul_tiled(A, B, C):
    sA = cuda.shared.array((TPB, TPB), dtype=float32)
    sB = cuda.shared.array((TPB, TPB), dtype=float32)

    x, y = cuda.grid(2)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    tmp = 0.0
    tiles = (A.shape[1] + TPB - 1) // TPB

    for t in range(tiles):

        colA = t * TPB + tx
        rowB = t * TPB + ty

        if y < A.shape[0] and colA < A.shape[1]:
            sA[ty, tx] = A[y, colA]
        else:
            sA[ty, tx] = 0.0

        if rowB < B.shape[0] and x < B.shape[1]:
            sB[ty, tx] = B[rowB, x]
        else:
            sB[ty, tx] = 0.0

        cuda.syncthreads()

        for k in range(TPB):
            tmp += sA[ty, k] * sB[k, tx]

        cuda.syncthreads()

    if y < C.shape[0] and x < C.shape[1]:
        C[y, x] = tmp


# -----------------------------
# MAIN EXECUTION
# -----------------------------
if __name__ == "__main__":

    M = K = N = 512

    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)

    print("Running CPU version...")
    t0 = time.time()
    C_cpu = matmul_cpu(A, B)
    cpu_time = time.time() - t0
    print(f"CPU Time: {cpu_time:.4f} seconds")

    A_d = cuda.to_device(A)
    B_d = cuda.to_device(B)
    C_d = cuda.device_array((M, N), dtype=np.float32)

    threads = (TPB, TPB)
    blocks = ((N + TPB - 1) // TPB, (M + TPB - 1) // TPB)

    print("Running Naive CUDA version...")
    start = cuda.event()
    end = cuda.event()
    start.record()
    matmul_naive[blocks, threads](A_d, B_d, C_d)
    end.record()
    end.synchronize()
    naive_time = cuda.event_elapsed_time(start, end) / 1000
    C_naive = C_d.copy_to_host()
    print(f"Naive CUDA Time: {naive_time:.4f} seconds")

    print("Running Tiled CUDA version...")
    start = cuda.event()
    end = cuda.event()
    start.record()
    matmul_tiled[blocks, threads](A_d, B_d, C_d)
    end.record()
    end.synchronize()
    tiled_time = cuda.event_elapsed_time(start, end) / 1000
    C_tiled = C_d.copy_to_host()
    print(f"Tiled CUDA Time: {tiled_time:.4f} seconds")

    print("Verifying correctness...")
    print("Naive correct:", np.allclose(C_cpu, C_naive, atol=1e-3))
    print("Tiled correct:", np.allclose(C_cpu, C_tiled, atol=1e-3))

    print("\nSpeedups:")
    print(f"Naive CUDA speedup: {cpu_time / naive_time:.2f}x")
    print(f"Tiled CUDA speedup: {cpu_time / tiled_time:.2f}x")

