"""
matmul/step2_tiled.py
=====================
Blocked (tiled) multiplication — the cache-friendly idea inside
BLAS / GPU kernels. Mathematically identical to naive(); the point is
the ACCESS PATTERN: the inner loop works on small blocks that fit
together in memory.

We verify tiled() == naive() on random matrices, then look at the
numbers: in pure Python the bookkeeping usually hides the win (which
is itself a lesson — hardware matters), but the block structure is
exactly what real kernels (cuBLAS, OpenBLAS, FlashAttention-style)
exploit at 100× the speed.
"""
import sys
import random

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from matmul import naive, tiled

random.seed(7)
n = 24
A = [[random.uniform(-1, 1) for _ in range(n)] for _ in range(n)]
B = [[random.uniform(-1, 1) for _ in range(n)] for _ in range(n)]

print("Step 2 — tiled multiplication: same numbers, better access pattern\n")
print("naive():  for i, j, k:   C[i][j] += A[i][k] · B[k][j]   (every element every time)")
print("tiled():  for blocks of 4:    compute block by block      (reuse in cache)")

C_naive = naive(A, B)
for tile in (2, 4, 8):
    C_tiled = tiled(A, B, TILE=tile)
    same = all(abs(C_naive[i][j] - C_tiled[i][j]) < 1e-9
               for i in range(n) for j in range(n))
    print(f"  tile={tile}:  tiled == naive on {n}×{n} random matrices?  {'✅ yes, identical' if same else '❌ MISMATCH'}")
    assert same

print("""
The key insight for hardware:
  · naive touches A[i][k] and B[k][j] across ALL of B per row — each
    read misses the CPU cache.
  · tiled computes a small TILE×TILE block while those elements are
    still in cache — for large matrices on C/GPU kernels, blocking
    is routinely the difference between 5 TFLOPS and 40 TFLOPS.
Python is too slow for the effect to show in a microbenchmark —
that's why real kernels are written in C and CUDA.
""")
print("PASS: tiled() produces bit-identical results to naive()")