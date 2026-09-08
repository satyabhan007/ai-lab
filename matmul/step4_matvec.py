"""
matmul/step4_matvec.py
======================
Matrix × vector — the operation inference actually RUNS.

The payoff that ties Phase 4 and Phase 5 together: the entire
attention head we built in attention/ is just a handful of matvecs.

We re-derive 'it' -> animal using ONLY matvec + softmax, i.e. the
exact code path a GPU executes for a single query token.
"""
import sys
import math
import os

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'attention'))

from matmul import matvec, transpose
from attention import softmax

print("Step 4 — attention IS matrix multiplication\n")

# Toy 4-dim embeddings from the attention chapter
E = {
    'the':    [0.0, 0.0, 0.0, 0.0],
    'animal': [0.9, 0.0, 0.0, 0.2],
    "didn't": [0.1, 0.8, 0.0, 0.0],
    'cross':  [0.2, 0.9, 0.3, 0.0],
    'street': [0.0, 0.0, 0.9, 0.0],
    'because':[0.0, 0.4, 0.0, 0.0],
    'it':     [0.7, 0.0, 0.0, 0.8],      # the query vector = "…it was too TIRED"
}

words = ['the', 'animal', "didn't", 'cross', 'the', 'street', 'because', 'it']
q_it  = E['it']

print("1) keys matrix  K = rows = every token's embedding")
K = [E[w] for w in words]
d = len(q_it)

print("2) scores = matvec(K, q_it)   — the ONLY multiply loop")
scores = matvec(K, q_it)                     # matrix × vector!
print(f"   scores = {[f'{s:.2f}' for s in scores]}")

print("\n3) scale by 1/√d  (same stability trick as attention.py)")
scaled = [s / math.sqrt(d) for s in scores]
w = softmax(scaled)
print(f"   softmax weights = {[f'{p*100:.1f}%' for p in w]}")

print("\n4) context = matvec(transpose(V), weights)   — another matvec")
V = K                                          # value matrix = same embeddings
ctx = matvec(transpose(V), w)
animal_w, street_w = w[1], w[5]
print(f"   'it' attends: animal {animal_w*100:.1f}%  vs  street {street_w*100:.1f}%")
win = 'ANIMAL 🐫' if animal_w > street_w else 'STREET 🛣️'
print(f"\n   → 'it' = {win}  (same answer as attention/step2)")

print("""
Recap of what the GPU runs for ONE query token:
    scores  = matvec(K, q)              tokens × d    multiplies
    weights = softmax(scores / √d)      tokens         (no multiplies for softmax)
    context = matvec(Vᵀ, weights)       d × tokens     multiplies
Total ≈ 2·tokens·d multiply-accumulates — linear in context length
for one token. The O(n²) story from the playground is simply this
loop repeated for EVERY query token.
""")
print("PASS: attention computed with nothing but matvec + softmax")