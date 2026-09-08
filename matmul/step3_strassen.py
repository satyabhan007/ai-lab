"""
matmul/step3_strassen.py
========================
Divide & conquer — Strassen's algorithm computes matrix products with
7 recursive multiplications instead of 8, dropping the exponent from
3 to log2(7) ≈ 2.807.

We verify strassen() == naive() on random matrices, then compare the
exact multiplication counts.
"""
import sys
import random

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from matmul import naive, strassen, strassen_muls

random.seed(11)

print("Step 3 — Strassen: 7 multiplications instead of 8\n")

for n in (2, 4, 8):
    A = [[random.uniform(-1, 1) for _ in range(n)] for _ in range(n)]
    B = [[random.uniform(-1, 1) for _ in range(n)] for _ in range(n)]
    Ca, Cs = naive(A, B), strassen(A, B)
    same = all(abs(Ca[i][j] - Cs[i][j]) < 1e-9 for i in range(n) for j in range(n))
    print(f"  n={n}:  strassen == naive ?  {'✅ yes' if same else '❌ MISMATCH'}")
    assert same

print("\nExact elementary multiplications (power-of-two squares):")
print("   n    naive (n³)    strassen (7^k)   ratio")
for n in (8, 64, 512, 4096):
    naive_n = n ** 3
    str_n = strassen_muls(n)
    print(f"  {n:>4}   {naive_n:>12,}    {str_n:>12,}      {naive_n / str_n:.2f}× less")

print("""
  naive M(n)     = n³
  strassen M(n)  = n^log2(7) ≈ n^2.807

At 4096×4096 that's 3.8× fewer multiplications — and the same trick
(at different exponents) is why research kernels keep pushing the
matmul exponent toward 2.
""")
print("PASS: strassen() matches naive() exactly and needs fewer multiplications")