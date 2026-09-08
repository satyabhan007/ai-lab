"""
attention/step1_similarity.py
=============================
STEP 1 — MEANING AS DIRECTION: embeddings, dot product, cosine similarity

Before attention can "pay attention", words must become numbers. Not any
numbers — VECTORS where similar meanings point in similar directions.

You will learn:
  1. what an embedding is (a word → a point in "meaning space")
  2. why the dot product is a similarity meter
  3. why cosine similarity ignores length and keeps direction
  4. how a tiny piece of arithmetic solves an analogy ("king" - "man" + "woman")

Run:  python step1_similarity.py
"""

import os, sys, math, random

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attention import dot, cosine_similarity, norm      # noqa: E402

random.seed(7)

print("═" * 62)
print("  STEP 1 — MEANING AS DIRECTION (embeddings & similarity)")
print("═" * 62)

# ── 1 · hand-crafted embeddings ──────────────────────────────────────
print("""
— 1 · words as points in meaning space —

Real models LEARN their embeddings. We craft 5 dimensions by hand so you
can see the meaning:   [ animal,  size,  pet-ness,  speed,  water-life ]""")
EMB = {
    "cat":     [0.9, 0.2, 0.9, 0.5, 0.0],
    "kitten":  [0.9, 0.1, 0.9, 0.5, 0.0],   # a cat, but smaller
    "dog":     [0.9, 0.5, 0.9, 0.5, 0.0],
    "whale":   [0.9, 1.0, 0.0, 0.3, 1.0],
    "rocket":  [0.0, 0.8, 0.0, 1.0, 0.0],
    "banana":  [0.0, 0.2, 0.1, 0.0, 0.0],
}
for w, v in EMB.items():
    print(f"     {w:<8} {v}")

# ── 2 · dot product = similarity meter ───────────────────────────────
print("""
— 2 · the dot product: multiply matching dimensions, add up —

   cat · kitten = 0.9·0.9 + 0.2·0.1 + 0.9·0.9 + 0.5·0.5 + 0·0
Big overlap in the SAME dimensions → big score. That is all 'similarity' is.""")
pairs = [("cat", "kitten"), ("cat", "dog"), ("cat", "whale"),
         ("cat", "rocket"), ("cat", "banana"), ("rocket", "whale")]
print()
for a, b in pairs:
    print(f"     dot({a:<7}, {b:<7}) = {dot(EMB[a], EMB[b]):5.2f}")

# ── 3 · cosine: direction beats length ───────────────────────────────
print("""
— 3 · the length trap, and the cosine fix —

'kitten' scores HIGHER than 'dog' partly because its numbers are... similar.
But a big whale (size=1.0) has a LONGER vector — is it more 'cat-like'? No!
Cosine similarity divides by both lengths: only DIRECTION survives.""")
long_whale = [x * 5 for x in EMB["whale"]]        # same direction, 5× longer
print(f"     dot(cat, whale)          = {dot(EMB['cat'], EMB['whale']):5.2f}")
print(f"     dot(cat, whale×5)        = {dot(EMB['cat'], long_whale):5.2f}   ← inflates!")
print(f"     cos(cat, whale)          = {cosine_similarity(EMB['cat'], EMB['whale']):5.2f}")
print(f"     cos(cat, whale×5)        = {cosine_similarity(EMB['cat'], long_whale):5.2f}   ← unchanged ✓")
print(f"""
     (|whale×5| = {norm(long_whale):.1f} vs |whale| = {norm(EMB['whale']):.1f} — length grew,
      direction didn't. Cosine = {cosine_similarity(EMB['cat'], EMB['whale']) * 100:.0f}% identical, dot = misleading.)""")

# ── 4 · the famous analogy trick ─────────────────────────────────────
print("""
— 4 · vector arithmetic: king − man + woman ≈ queen —

With learned embeddings, relationships become DIRECTIONS:
  gender ≈ queen − king ≈ woman − man.  Solve an analogy by walking the arrow.""")
ANALOGY = {
    "man":   [0.1, 0.9, 0.0, 0.2],
    "woman": [0.1, 0.9, 1.0, 0.2],
    "king":  [0.9, 0.8, 0.0, 0.7],
    "queen": [0.9, 0.8, 1.0, 0.7],
}
target = [ANALOGY["king"][i] - ANALOGY["man"][i] + ANALOGY["woman"][i] for i in range(4)]
scores = {w: cosine_similarity(target, v) for w, v in ANALOGY.items()}
best = max(scores, key=scores.get)
print(f"     king − man + woman = [{', '.join(f'{x:+.1f}' for x in target)}]")
for w, s in sorted(scores.items(), key=lambda kv: -kv[1]):
    marker = "  ← CLOSEST" if w == best else ""
    print(f"     cos(result, {w:<6}) = {s:+.3f}{marker}")
analogy_ok = best == "queen"

# ── 5 · connection to attention ──────────────────────────────────────
print("""
— 5 · why this powers attention —

Attention does exactly ONE thing with these vectors: it compares them.
  query · key  →  'how relevant are you to me?'
Every relevance score in GPT is a dot product. You just built the meter.""")

print("\n" + "═" * 62)
print(f"  RESULT: analogy solved → '{best}'  [{'PASS' if analogy_ok else 'FAIL'}]")
print("  NEXT → step2_self_attention.py: turn scores into attention weights")
print("═" * 62)
sys.exit(0 if analogy_ok else 1)
