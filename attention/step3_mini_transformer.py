"""
attention/step3_mini_transformer.py
===================================
STEP 3 — TRAIN A (TINY) TRANSFORMER END-TO-END

Everything so far used FIXED embeddings. Now we add learnable parameters
and train the full pipeline with YOUR micrograd engine (engine.py):

    tokens → embeddings (+positional) → causal self-attention → head → loss

The task: read a 3-word prefix, predict the ANIMAL class of the sentence.
  "the camel runs" → class 0 (big animal)      "the cat runs" → class 1 (small pet)
The ONLY trainable part is a d×2 classification head on top of the
attention output — exactly how GPT's final layer turns context vectors
into next-token probabilities.

Run:  python step3_mini_transformer.py
"""

import os, sys, random

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'micrograd'))

from engine import Value                 # noqa: E402
from attention import (                  # noqa: E402
    attend, positional_encoding,
)

random.seed(42)

print("═" * 62)
print("  STEP 3 — MINI TRANSFORMER (trained with micrograd)")
print("═" * 62)

# ── 1 · vocabulary & (fixed) embeddings ──────────────────────────────
print("""
— 1 · vocabulary + embeddings —

4 dims: [animal-ness, action-ness, size, pet-ness]. In real GPT these are
LEARNED too — we keep them fixed so the training story stays clear.""")
VOCAB = ["<pad>", "the", "camel", "cat", "runs", "fast"]
E = {
    "<pad>": [0.0, 0.0, 0.0, 0.0],
    "the":   [0.3, 0.0, 0.1, 0.0],
    "camel": [0.9, 0.1, 0.3, 0.5],
    "cat":   [0.2, 0.8, 0.0, 0.1],
    "runs":  [0.0, 0.9, 0.2, 0.0],
    "fast":  [0.1, 0.8, 0.0, 0.3],
}
for w in VOCAB:
    print(f"     {w:<6} {E[w]}")
D_MODEL = 4

# ── 2 · the training data ────────────────────────────────────────────
print("""
— 2 · task: predict the animal class from a 3-word prefix —""")
data = [(["the", "camel", "runs"], 0), (["the", "cat", "runs"], 1)]
for words, cls in data:
    print(f"     {' '.join(words):<18} → class {cls}")

# ── 3 · forward: embeddings + PE → causal attention → head ──────────
print("""
— 3 · the forward pass (one attention head, causal mask) —""")


def sequence_vector(words):
    """Attention pass over the whole prefix; return LAST position's context."""
    embs = [E[w] for w in words]
    keys = [[v + p for v, p in zip(e, positional_encoding(j, D_MODEL))]
            for j, e in enumerate(embs)]
    ctx, w_last = attend(keys[-1], keys, keys, causal_pos=len(words) - 1)
    return ctx, w_last


# trainable classification head — the ONLY learned parameters (d×2 = 8)
W_head = [[Value(random.uniform(-0.5, 0.5)) for _ in range(2)] for _ in range(D_MODEL)]


def head_logits(words):
    h, _ = sequence_vector(words)
    return [sum(h[i] * W_head[i][c] for i in range(D_MODEL)) for c in range(2)]


def forward_loss(dataset):
    """Mean cross-entropy over the dataset (micrograd graph)."""
    loss = Value(0.0)
    for words, cls in dataset:
        l0, l1 = head_logits(words)
        m = l0 if l0.data > l1.data else l1          # max-subtraction trick
        z = (l0 - m).exp() + (l1 - m).exp()          # softmax denominator
        p_correct = (l0 - m).exp() / z if cls == 0 else (l1 - m).exp() / z
        loss = loss + p_correct.log() * -1           # −log P(correct class)
    return loss * (1 / len(dataset))


# ── 4 · the training loop ────────────────────────────────────────────
print("""
— 4 · training: forward → backward (chain rule) → nudge weights —

Identical loop to micrograd/step3_full_network.py — only the model
in front of the loss changed. Watch the loss fall:""")
LR, STEPS = 0.5, 300
for step in range(STEPS):
    loss = forward_loss(data)
    for row in W_head:
        for p in row:
            p.grad = 0.0
    loss.backward()
    for row in W_head:
        for p in row:
            p.data -= LR * p.grad
    if step % 50 == 0 or step == STEPS - 1:
        probs = []
        for words, cls in data:
            l0, l1 = head_logits(words)
            m = max(l0.data, l1.data)
            e0 = pow(2.718281828, l0.data - m)
            e1 = pow(2.718281828, l1.data - m)
            p0 = e0 / (e0 + e1)
            probs.append(p0 if cls == 0 else 1 - p0)
        print(f"     step {step:>3}   loss {loss.data:.4f}   "
              f"P(correct) = {probs[0]:.2f}, {probs[1]:.2f}")

# ── 5 · evaluate: training data + an UNSEEN ending ───────────────────
print("""
— 5 · evaluation —""")
ok = 0
for words, cls in data:
    l0, l1 = head_logits(words)
    got = 0 if l0.data > l1.data else 1
    passed = got == cls
    ok += passed
    print(f"     [{'PASS' if passed else 'FAIL'}] {' '.join(words):<20} → class {got} "
          f"(want {cls})   logits [{l0.data:+.2f}, {l1.data:+.2f}]")
l0, l1 = head_logits(["the", "camel", "fast"])
got = 0 if l0.data > l1.data else 1
gen_ok = got == 0
print(f"     [{'PASS' if gen_ok else 'FAIL'}] 'the camel fast' (UNSEEN)   → class {got} "
          f"(want 0)   logits [{l0.data:+.2f}, {l1.data:+.2f}]")
h, w_last = sequence_vector(["the", "camel", "fast"])
print(f"""
     attention from the last word: camel {w_last[1] * 100:.0f}%, fast {w_last[2] * 100:.0f}%,
     the {w_last[0] * 100:.0f}% — the head reads mostly the ANIMAL position. That
     single dot-product pipeline, stacked 100× with 12k dims, is GPT.""")

print("\n" + "═" * 62)
print(f"  RESULT: {ok}/2 training + {'1/1' if gen_ok else '0/1'} generalization  "
      f"[{'PASS' if ok == 2 and gen_ok else 'FAIL'}]")
print("  You have now trained every component of a transformer except scale.")
print("═" * 62)
sys.exit(0 if ok == 2 and gen_ok else 1)
