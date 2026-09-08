# 🧮 Matrix Multiplication — What Every GPU Is Actually Doing

> **The frame:** everything since micrograd has been *lists of numbers flowing
> through loops*. This chapter is about the single loop that powers it all —
> and why your GPU is basically a million tiny calculators chained together.

**Recommended path:** `step1_naive.py` → `step2_tiled.py` → `step3_strassen.py`
→ `step4_matvec.py`. Run each from the `matmul/` folder.

---

## 1. The one formula

```
C[i][j] =  A[i][0]·B[0][j] + A[i][1]·B[1][j] + … + A[i][k]·B[k][j]
```

One output cell = **dot product** of row `i` of A and column `j` of B.
That's it. Everything below is just *how to compute all those dots faster*.

### The spreadsheet analogy
Think of A as a price list (`items × categories`) and B as a conversion
table (`categories × metrics`). Multiplying them produces a new table where
every item has every metric — each cell is "sum of (item×category × category×metric)".

### The BOOK analogy for O(n³)
You want to know how similar every pair of books is. Comparing 10 books =
100 pairs; comparing 100 books = 10,000 pairs. **Doubling the books
quadruples the pairs** and for matrix multiply, doubling the size
**octuples the work** (2³ = 8). That's O(n³).

---

## 2. Why "naive" is not enough

For an `n×n` matrix, naive matrix multiply does **n³ multiplications and
n³ additions**. Real numbers:

| n (context tokens) | total flops |
|---|---|
| 1,024 | ~2.1×10⁹ |
| 8,192 | ~1.1×10¹² |
| 65,536 | ~5.6×10¹⁴ |

A modern GPU does ~10¹⁴ flops/second. So even a single 65k-token attention
matrix takes seconds **per head** — and GPT has ~96 heads × ~8 layers.
This one formula explains: long-context pricing, KV caches, Flash Attention,
and why GPU vendors keep buying faster chips.

---

## 3. The three algorithms you meet here

### naive — the definition
Triple loop, exactly as written in any textbook. Correct, simple, slow.

### tiled — the cache trick
Instead of touching every element of B for every row of A, work in small
blocks that all fit in the CPU/GPU's fast memory at once. The numbers are
identical — the **access pattern** is what changes. Real kernels (BLAS,
cuBLAS, FlashAttention) live and die by this. In pure Python the effect is
hidden by interpreter overhead — which is itself the lesson: *hardware
matters as much as math*.

### strassen — the exponent trick
Split each matrix into 4 blocks. Naively computing the 4 output blocks needs
8 block-products. Strassen found how to do it with 7 by reusing one sum.
Recursively, that changes the exponent from 3 to **log₂(7) ≈ 2.807**.
At n=4096 that's ~4× fewer multiplications (before constant-factor
engineering takes away some of the win).

---

## 4. The bridge to attention (step4)

The entire Phase 4 attention head, for one query token, is exactly:

```
scores  = matvec(K, q)                K is (tokens × d)
weights = softmax(scores / √d)
context = matvec(Vᵀ, weights)         V is (tokens × d)
```

So when you watched the attention weights in the playground, you were
watching a **GPU run matrix multiplications**. Nothing else. A transformer
is a carefully-orchestrated sequence of these loops, and this chapter is
where "gradients" (Phase 1), "tokens" (Phase 2) and "attention" (Phase 4)
all meet the metal.

---

## 5. Homework

- [ ] Change `naive()` to compute only the **upper triangle** of C when A==B
      (saves ~half the work — the symmetry trick real libraries use).
- [ ] In `step2_tiled.py`, try `TILE=64` on a 200×200 matrix. Why does the
      pure-Python timing barely change? (Hint: Python overhead vs cache.)
- [ ] Verify Strassen's 7 products by hand on two 2×2 matrices in step3.
- [ ] In `step4_matvec.py`, swap the "it" embedding to the WIDE version and
      confirm `street` wins — the same flip you saw in the playground.

---

## Next stop

The Foundation layer is complete: **autograd → tokenizer → matmul**.
The "Core Models" phase (full multi-head transformer on real text) builds
directly on attention/ + matmul/ — that's the natural next chapter.