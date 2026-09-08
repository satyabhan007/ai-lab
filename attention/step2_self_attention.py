"""
attention/step2_self_attention.py
=================================
STEP 2 — SELF-ATTENTION: every word looks at every word

Step 1 gave us a similarity meter (dot product). Now we build the full
attention mechanism with it — the same 4 moves GPT makes billions of times:

    1. SCORE    every word pairs with every word      (q·k / √d)
    2. MASK     forbid looking into the future        (causal mask)
    3. SOFTMAX  scores → percentages that sum to 1    (who to look at)
    4. MIX      new meaning = weighted average        (context vector)

Run:  python step2_self_attention.py
"""

import os, sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attention import (        # noqa: E402
    softmax, scale_scores, attention_weights, attend, positional_encoding,
)

print("═" * 62)
print("  STEP 2 — SELF-ATTENTION (score → mask → softmax → mix)")
print("═" * 62)

# ── the sentence ─────────────────────────────────────────────────────
SENTENCE = ["the", "animal", "didn't", "cross", "the", "street", "because", "it"]
# 4-dim toy embeddings: [animal-ness, action-ness, place-ness, state-ness]
E = {
    "the":     [0.1, 0.0, 0.0, 0.0],
    "animal":  [0.9, 0.1, 0.0, 0.3],
    "didn't":  [0.0, 0.8, 0.0, 0.4],
    "cross":   [0.0, 0.9, 0.2, 0.0],
    "street":  [0.0, 0.1, 0.9, 0.0],
    "because": [0.0, 0.2, 0.0, 0.7],
    "it":      [0.5, 0.0, 0.2, 0.5],
}
embs = [E[w] for w in SENTENCE]
D = len(embs[0])

print(f"""
— 0 · the sentence (the classic Winograd puzzle) —

   {" ".join(SENTENCE)}  … what does 'it' refer to?
By the end of this file the attention weights will TELL you.""")

# ── 1 · SCORE ────────────────────────────────────────────────────────
print(f"""
— 1 · SCORE: query · key / √d —

Self-attention = every word uses ITS OWN embedding as the query and tries
it against every key. We scale by √d = √{D} so big vectors can't
accidentally dominate the softmax that follows.""")
q_it = [e + p for e, p in zip(embs[-1], positional_encoding(len(embs) - 1, D))]
keys = [[e + p for e, p in zip(x, positional_encoding(j, D))]
        for j, x in enumerate(embs)]
scores = scale_scores(q_it, keys)
print("     scaled scores from 'it' to every word:")
for w, s in zip(SENTENCE, scores):
    print(f"       {w:<8} {s:+.3f}  {'█' * max(1, int((s + 1) * 6))}")

# ── 2 · MASK ─────────────────────────────────────────────────────────
print("""
— 2 · MASK: no peeking at the future —

GPT is trained to predict the NEXT word. If position i could see positions
>i it would just... copy the answer. So future keys are set to −∞, which
softmax turns into EXACTLY 0%. Here 'it' is last, so nothing changes —
but watch what the mask does to 'cross' (position 3):""")
w_cross = attention_weights(keys[3], keys, causal_pos=3)
print("     'cross' attends to: ["
      + ", ".join(f"{w}:{p:.2f}" for w, p in zip(SENTENCE, w_cross) if p > 0.001)
      + "]  (future words: exactly 0)")

# ── 3 · SOFTMAX ──────────────────────────────────────────────────────
print("""
— 3 · SOFTMAX: scores → a budget of 100% —

Softmax exponentiates and normalizes. Negative scores are fine: they just
become small percentages. The row ALWAYS sums to 1.000 — a budget the
word can spend on looking at its neighbours.""")
w_it = attention_weights(q_it, keys)
print("     attention weights for 'it':")
for w_, p in sorted(zip(SENTENCE, w_it), key=lambda t: -t[1]):
    if p > 0.001:
        print(f"       {w_:<8} {p * 100:5.1f}%  {'▓' * int(p * 40)}")
print(f"       sum = {sum(w_it):.6f}   ← always exactly 1")

# ── 4 · MIX ──────────────────────────────────────────────────────────
print("""
— 4 · MIX: the new meaning of 'it' —

context('it') = Σ weightᵢ · valueᵢ. The word walks around the sentence,
collecting meaning from whomever it trusts.""")
ctx, _ = attend(q_it, keys, keys)
print(f"     old 'it'      = [{', '.join(f'{x:+.2f}' for x in q_it)}]")
print(f"     new 'it'      = [{', '.join(f'{x:+.2f}' for x in ctx)}]")
animal_w, street_w = w_it[1], w_it[5]
resolve_ok = animal_w > street_w
print(f"""
     'it' attended to 'animal' {animal_w * 100:.1f}% vs 'street' {street_w * 100:.1f}%
     → the model resolves the pronoun to the ANIMAL. Attention IS the answer.""")

# ── 5 · temperature on attention ─────────────────────────────────────
print("""
— 5 · temperature: same scores, different focus —

The same dial that controls ChatGPT's creativity controls attention focus:""")
for T in (0.3, 1.0, 3.0):
    wt = softmax(scores[:4], temperature=T)      # only past words
    top = SENTENCE[wt.index(max(wt))]
    print(f"     T={T:<3} → top focus: {top:<8} ({max(wt) * 100:.0f}%), "
          f"spread {max(wt) - min(w for w in wt if w > 0):.2f}")

print("\n" + "═" * 62)
print(f"  RESULT: pronoun resolved by attention  [{'PASS' if resolve_ok else 'FAIL'}]")
print("  NEXT → step3_mini_transformer.py: train this pipeline end-to-end")
print("═" * 62)
sys.exit(0 if resolve_ok else 1)
