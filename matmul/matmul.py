"""
matmul/matmul.py
================
Matrix multiplication — the workhorse of every AI model.

A transformer is *long chains of these loops*:

    Q @ Kᵀ        attention scores       (tokens × tokens)
    softmax @ V   context vectors        (tokens × tokens) · (tokens × dim)
    X @ W_q/k/v   query / key / value projections
    h @ W_out     feed-forward layers

This module implements the classic algorithms in pure Python (no numpy)
with a tiny FLOP counter so you can SEE why O(n³) matters — and why
GPU kernels spend decades optimizing this single loop.

Functions
---------
naive(A, B)          triple loop — the math everyone learns first     O(n³)
tiled(A, B, size)    blocked loop — the cache-friendly BLAS/GPU idea
strassen(A, B)       divide & conquer — 7 multiplications, not 8     ~O(n^2.807)
matvec(A, x)         matrix × vector — what inference actually RUNS
transpose(M)         flip rows ↔ columns
count_flops(n)       exact (multiplications, additions) in naive on n×n
strassen_muls(n)     exact elementary multiplications of Strassen
"""


def naive(A, B):
    """
    The mathematical definition, as a triple loop:

        C[i][j] = Σ_k  A[i][k] · B[k][j]

    A is (n × p), B is (p × m)  →  C is (n × m).
    """
    n, p, m = len(A), len(B), len(B[0])
    C = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            s = 0.0
            for k in range(p):
                s += A[i][k] * B[k][j]
            C[i][j] = s
    return C


def tiled(A, B, TILE=4):
    """
    Blocked multiplication — the idea behind high-performance BLAS/GPU
    kernels. Mathematically identical to naive(), but it processes the
    matrices in small TILE×TILE blocks so the values the inner loop
    touches sit close together in memory (cache-friendly).

    In interpreted Python the bookkeeping overhead usually hides the
    win (CPU caches are small and Python is slow anyway), which is why
    real kernels are written in C/CUDA. On real hardware, blocking is
    often the difference between 5 TFLOPS and 40 TFLOPS.
    """
    n, p, m = len(A), len(B), len(B[0])
    C = [[0.0] * m for _ in range(n)]
    for i in range(0, n, TILE):
        for j in range(0, m, TILE):
            for k in range(0, p, TILE):
                for ii in range(i, min(i + TILE, n)):
                    for jj in range(j, min(j + TILE, m)):
                        for kk in range(k, min(k + TILE, p)):
                            C[ii][jj] += A[ii][kk] * B[kk][jj]
    return C


def _mat_add(A, B):
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def _mat_sub(A, B):
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def _next_pow2(n):
    k = 1
    while k < n:
        k <<= 1
    return k


def strassen(A, B):
    """
    Divide & conquer (Strassen, 1969): split each matrix into 4 blocks
    and compute C with 7 block-products instead of 8. 8 → 7 may not
    sound like much, but recursion makes it n^2.807 instead of n³.

    Only square sizes; non-powers-of-two are zero-padded internally.
    """
    n = len(A)
    if n <= 2:
        return naive(A, B)
    k = _next_pow2(n)
    if k != n:                               # pad to a power of two
        Ap = [[0.0] * k for _ in range(k)]
        Bp = [[0.0] * k for _ in range(k)]
        for i in range(n):
            for j in range(n):
                Ap[i][j] = A[i][j]
                Bp[i][j] = B[i][j]
        C = strassen(Ap, Bp)
        return [[C[i][j] for j in range(n)] for i in range(n)]

    h = n // 2
    a = [row[:h] for row in A[:h]];   b = [row[h:] for row in A[:h]]
    c = [row[:h] for row in A[h:]];   d = [row[h:] for row in A[h:]]
    e = [row[:h] for row in B[:h]];   f = [row[h:] for row in B[:h]]
    g = [row[:h] for row in B[h:]];   hh = [row[h:] for row in B[h:]]

    p1 = strassen(a,     _mat_sub(f, hh))
    p2 = strassen(_mat_add(a, b), hh)
    p3 = strassen(_mat_add(c, d), e)
    p4 = strassen(d,     _mat_sub(g, e))
    p5 = strassen(_mat_add(a, d), _mat_add(e, hh))
    p6 = strassen(_mat_sub(b, d), _mat_add(g, hh))
    p7 = strassen(_mat_sub(a, c), _mat_add(e, f))

    C00 = _mat_add(_mat_sub(_mat_add(p5, p4), p2), p6)
    C01 = _mat_add(p1, p2)
    C10 = _mat_add(p3, p4)
    C11 = _mat_sub(_mat_sub(_mat_add(p5, p1), p3), p7)

    top = [C00[i] + C01[i] for i in range(h)]
    bot = [C10[i] + C11[i] for i in range(h)]
    return top + bot


def matvec(A, x):
    """
    Matrix × vector — the operation inference actually RUNS.

        out[i] = Σ_j  A[i][j] · x[j]

    Attention as matvecs (what GPT does per query token):
        scores  = matvec(K, q)                 # K: (tokens × d)
        weights = softmax(scores / √d)
        context = matvec(transpose(V), weights)  # V: (tokens × d)
    """
    return [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A))]


def transpose(M):
    """Flip rows ↔ columns: M[row][col] → T[col][row]."""
    return [[M[j][i] for j in range(len(M))] for i in range(len(M[0]))]


# ── FLOP counting: an instrumented number to make O(n³) concrete ──────

class _Fake:
    """A number that counts every multiplication / addition it does."""
    muls = 0
    adds = 0

    def __init__(self, v=0.0):
        self.v = v

    def __mul__(self, o):
        _Fake.muls += 1
        return _Fake(self.v * o.v)

    def __rmul__(self, o):
        _Fake.muls += 1
        return _Fake(self.v * o)

    def __add__(self, o):
        _Fake.adds += 1
        return _Fake(self.v + o.v)

    def __radd__(self, o):
        _Fake.adds += 1
        return _Fake(self.v + o)


def count_flops(n):
    """
    Run naive() on fake n×n matrices and return
    (multiplications, additions) = (n³, n³) — the O(n³) made exact.
    """
    _Fake.muls = _Fake.adds = 0
    A = [[_Fake(1.0) for _ in range(n)] for _ in range(n)]
    B = [[_Fake(1.0) for _ in range(n)] for _ in range(n)]
    naive(A, B)
    return _Fake.muls, _Fake.adds


def strassen_muls(n):
    """Exact elementary multiplications for a power-of-two n×n:
    M(1) = 1,  M(n) = 7 · M(n/2)   ⇒   M(n) = n^log2(7) ≈ n^2.807
    """
    if n == 1:
        return 1
    return 7 * strassen_muls(n // 2)
