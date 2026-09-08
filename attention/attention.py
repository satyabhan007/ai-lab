"""
attention/attention.py
======================
Self-attention building blocks — the heart of every transformer
(GPT, Claude, Gemini, Llama...). Pure Python, zero dependencies.

The pipeline of ONE attention head:

    tokens → embeddings (+ position info) → scores → softmax → weighted mix
             "what does each word mean?"    "who       "how much
                                             should     should I
                                             I look     look at
                                             at?"       each one?"

Why "attention"? Every word looks at every other word and asks:
"how relevant are you to what I am?" — then builds its new meaning
as a weighted average of the relevant ones.
"""

import math

# ── tiny vector helpers (lists of floats — no numpy, on purpose) ─────

def dot(a, b):
    """Dot product: the similarity meter. Bigger = more similar."""
    return sum(x * y for x, y in zip(a, b))


def vec_add(a, b):
    return [x + y for x, y in zip(a, b)]


def vec_scale(a, s):
    return [x * s for x in a]


def norm(a):
    """Vector length (Euclidean)."""
    return math.sqrt(dot(a, a))


def cosine_similarity(a, b):
    """
    Dot product, rescaled to [-1, +1] so length doesn't matter —
    only DIRECTION (i.e. pure meaning, not magnitude).
      +1 = identical direction   0 = unrelated   -1 = opposite
    """
    na, nb = norm(a), norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return dot(a, b) / (na * nb)


# ── softmax: scores → probabilities ──────────────────────────────────

def softmax(scores, temperature=1.0):
    """
    Turn raw similarity scores into percentages that sum to exactly 1.

    temperature < 1  → sharper  (winner takes almost everything)
    temperature > 1  → flatter  (more spread out, more "creative")

    The `max(scores)` subtraction is the famous numerical-stability
    trick: exp() overflows on big numbers, but exp(s - max) can't,
    and the final probabilities are mathematically identical.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    m = max(scores)
    exps = [math.exp((s - m) / temperature) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def scale_scores(query, keys):
    """
    Scaled dot-product scores (the 'attention is all you need' formula
    core): score(q, k) = q·k / sqrt(d). Dividing by sqrt(d) keeps
    scores in a sane range so softmax doesn't saturate.
    """
    d = len(query)
    return [dot(query, k) / math.sqrt(d) for k in keys]


# ── one attention head ───────────────────────────────────────────────

def attention_weights(query, keys, temperature=1.0, causal_pos=None):
    """
    How much `query` (word i) should look at each key (word j).

    causal_pos=i  → causal (GPT-style) masking: position i may look at
                    positions 0..i ONLY. Future words are masked with
                    -infinity → exactly 0% probability after softmax.
                    That single rule is what lets GPT train on all
                    positions in parallel without cheating.
    """
    scores = scale_scores(query, keys)
    if causal_pos is not None:
        scores = [
            s if j <= causal_pos else float("-inf")
            for j, s in enumerate(scores)
        ]
    return softmax(scores, temperature)


def attend(query, keys, values, temperature=1.0, causal_pos=None):
    """
    The full head: weights · values = the new meaning of `query`.
    Returns (context_vector, weights) so you can SEE the weights.
    """
    w = attention_weights(query, keys, temperature, causal_pos)
    out = [0.0] * len(values[0])
    for weight, v in zip(w, values):
        for i, x in enumerate(v):
            out[i] += weight * x
    return out, w


# ── positional encoding: teaching the order of words ─────────────────

def positional_encoding(pos, dim):
    """
    Sinusoidal positions (Vaswani et al. 2017). Each position gets a
    unique wave fingerprint; similar positions have similar patterns,
    so the model can learn "next to", "2 before", etc.

    Even dimensions → sin, odd dimensions → cos, wavelength growing
    from 2π up to 10000·2π (a multi-resolution clock).
    """
    return [
        math.sin(pos / (10000 ** (2 * i / dim))) if i % 2 == 0
        else math.cos(pos / (10000 ** (2 * i / dim)))
        for i in range(dim)
    ]
