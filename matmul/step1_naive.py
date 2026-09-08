"""
matmul/step1_naive.py
=====================
The triple loop — the mathematical definition of matrix multiplication.

    C[i][j] = Σ_k  A[i][k] · B[k][j]

We walk through a real 2×3 · 3×2 example, then make O(n³) concrete
with exact flop counts.
"""
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from matmul import naive, count_flops


def show(A, name):
    print(f"{name}:")
    for row in A:
        print("   " + " ".join(f"{v:5.1f}" for v in row))


A = [[1, 2, 3],
     [4, 5, 6]]
B = [[7, 8],
     [9, 10],
     [11, 12]]

print("Step 1 — the O(n³) triple loop\n")
print("Definition:  C[i][j] = A[i][0]·B[0][j] + A[i][1]·B[1][j] + A[i][2]·B[2][j]\n")
show(A, "A (2×3)")
show(B, "B (3×2)")

C = naive(A, B)
show(C, "C = A·B (2×2)")

# verify cell (0,0) by hand
manual = A[0][0] * B[0][0] + A[0][1] * B[1][0] + A[0][2] * B[2][0]
print(f"\nC[0][0] by hand = 1·7 + 2·9 + 3·11 = {manual}   (code says {C[0][0]})")
assert abs(manual - C[0][0]) < 1e-9, "mismatch!"

print("\nWhy GPU kernels obsess over this loop — exact flop counts (n×n):")
print("   n     multiplications   additions      total flops")
for n in (64, 128, 256, 512, 1024):
    muls = adds = n ** 3
    print(f"  {n:>4}    {muls:>13,}  {adds:>11,}   {muls + adds:>13,}")

# verify the formula against the instrumented counter at a small size
muls_small, adds_small = count_flops(16)
assert (muls_small, adds_small) == (16 ** 3, 16 ** 3)
print("\n[verified] the instrumented counter at n=16 reports exactly 16³ = 4096 "
      "multiplications + 4096 additions")

print("""
Growth: doubling n multiplies the work by 8 (2³ = 8).
That O(n³) is exactly why GPUs with thousands of tiny multipliers
exist — a 8k×8k attention score matrix is 8k³ = 5×10¹¹ flops per head.
""")
print("PASS: naive() matches the hand-computed cell")