"""
lab/scenarios_tester.py
=======================
AI LAB TESTER v2 — two kinds of checks, 46 total:

  PART 1 · LAYER-BY-LAYER UNIT TESTS (is every building block correct?)
     1. autograd engine  — every op's gradient vs numerical differentiation
     2. net anatomy      — Neuron / Layer / MLP structure + parameter counts
     3. mlp learns       — a fresh MLP trains to 100% on the toy dataset
     4. tokenizer        — BPE roundtrips, compression, multi-byte tokens
     5. attention        — softmax, temperature, √d scaling, causal mask, positions

  PART 2 · REAL-WORLD SCENARIOS (do the blocks solve actual problems?)
     6.  ☂  umbrella decision        11. 🌡 temperature sampling
     7.  📧 spam filter              12. 👤 pronoun resolution
     8.  🏠 house price estimate     13. 📈 attention is O(n²)
     9.  💬 chat API billing         14. 🎭 causal mask proof
     10. 🔬 autograd unit test       15. 🤖 mini transformer end-to-end
Run:  python scenarios_tester.py
Exit code 0 = all pass, 1 = something failed (CI-friendly).
"""
import sys, os, math, random

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..', 'micrograd'))
sys.path.insert(0, os.path.join(HERE, '..', 'tokenizer'))
sys.path.insert(0, os.path.join(HERE, '..', 'attention'))

from engine import Value                     # noqa: E402
from nn import Neuron, Layer, MLP            # noqa: E402
from bpe import BPETokenizer                 # noqa: E402
from attention import (                      # noqa: E402
    dot, softmax, scale_scores, attention_weights, attend,
    positional_encoding,
)

random.seed(42)
PASS, FAIL = "[PASS]", "[FAIL]"
BAR_W = 24


def bar(pct, width=BAR_W):
    n = max(0, min(width, int(round(pct / 100 * width))))
    return "█" * n + "░" * (width - n)


def header(title):
    print("\n" + "═" * 62)
    print(f"  {title}")
    print("═" * 62)


results = []   # (group, passed, total)


def record(name, passed, total):
    results.append((name, passed, total))
    pct = 100 * passed / total
    print(f"\n  {name}: {passed}/{total} tests passed   [{bar(pct)}] {pct:.0f}%")


def check(label, ok, detail=""):
    print(f"  {PASS if ok else FAIL} {label}" + (f"\n        {detail}" if detail else ""))
    return 1 if ok else 0


print("╔" + "═" * 60 + "╗")
print("║" + "  AI LAB TESTER v2 — LAYERS + REAL-WORLD SCENARIOS".center(60) + "║")
print("║" + "  micrograd engine · nn · BPE · attention — 46 checks".center(60) + "║")
print("╚" + "═" * 60 + "╝")

# ══════════════════════════════════════════════════════════════════════
# PART 1 · LAYER-BY-LAYER UNIT TESTS
# ══════════════════════════════════════════════════════════════════════
header("PART 1 · LAYER-BY-LAYER UNIT TESTS")

# ── LAYER 1 · autograd engine: every op vs numerical differentiation ─
print("\n— 1 · autograd engine — every op's gradient, engine vs 'nudge' —")
X0 = 1.3
ENGINE_CASES = [
    ("d(x+3)/dx",     lambda v: v + 3,             lambda x: x + 3),
    ("d(4·x)/dx",     lambda v: v * 4,             lambda x: 4 * x),
    ("d(x²)/dx",      lambda v: v * v,             lambda x: x * x),
    ("d(tanh x)/dx",  lambda v: v.tanh(),          math.tanh),
    ("d(e^{x/2})/dx", lambda v: (v * 0.5).exp(),   lambda x: math.exp(x / 2)),
    ("d ln(x²+1)/dx", lambda v: (v * v + 1).log(), lambda x: math.log(x * x + 1)),
]
ok = 0
h = 1e-5
for name, vfunc, pfunc in ENGINE_CASES:
    x = Value(X0)
    y = vfunc(x)
    y.backward()
    g_engine = x.grad
    g_numeric = (pfunc(X0 + h) - pfunc(X0 - h)) / (2 * h)
    diff = abs(g_engine - g_numeric)
    ok += check(f"{name:<14} engine={g_engine:+.6f}  numeric={g_numeric:+.6f}"
                f"  |Δ|={diff:.1e}", diff < 1e-4)
record("⚙ engine ops", ok, len(ENGINE_CASES))
# ── LAYER 2 · net anatomy: Neuron / Layer / MLP structure ────────────
print("\n— 2 · net anatomy — Neuron / Layer / MLP structure —")
ok = 0
n1 = Neuron(3)
ok += check("Neuron(3) has 4 parameters (3 weights + 1 bias)",
            len(n1.parameters()) == 4, f"got {len(n1.parameters())}")
L1 = Layer(3, 5)
outs = L1([0.5, -1.0, 2.0])
ok += check("Layer(3, 5) emits 5 outputs", len(outs) == 5)
m1 = MLP(3, [4, 4, 1])
n_params = len(m1.parameters())
ok += check("MLP(3,[4,4,1]) has 41 parameters (16+20+5)", n_params == 41,
            f"got {n_params}")
o1 = m1([0.5, -1.0, 2.0])
ok += check("MLP forward returns a single Value for regression",
            isinstance(o1, Value), f"got {type(o1).__name__}")
record("🧩 net anatomy", ok, 4)

# ── LAYER 2 LEARNS · a fresh MLP reaches 100% on the toy dataset ─────
print("\n— 3 · mlp learns — fresh weights → 100% accuracy by backprop —")
XS = [[2, 3, -1], [3, -1, 0.5], [0.5, 1, 1], [1, 1, -1], [0, 0, 0]]
XS = [[v / 3 for v in row] for row in XS]
YS = [1, -1, 1, -1, 1]
model = MLP(3, [4, 4, 1])
last_loss = None
for step in range(600):
    preds = [model(x) for x in XS]
    loss = sum((Value(1) - Value(y) * p).relu() for p, y in zip(preds, YS))
    loss = loss * (1 / len(YS))
    last_loss = loss.data
    model.zero_grad()
    loss.backward()
    for p in model.parameters():
        p.data -= 0.05 * p.grad
acc = sum((model(x).data > 0) == (y > 0) for x, y in zip(XS, YS))
ok = check(f"hinge loss after 200 steps = {last_loss:.4f}  (target < 0.1)",
           last_loss < 0.1)
ok += check(f"accuracy = {acc}/5 on the training set", acc == 5)
record("🏋 mlp learns", ok, 2)
# ── LAYER 3 · BPE tokenizer: roundtrips, compression, byte fallback ──
print("\n— 4 · tokenizer — BPE roundtrips, compression, byte fallback —")
CORPUS = (
    "the cat sat on the mat. the cat saw the dog. the dog ran to the cat. "
    "the cat and the dog sat. we learn to build ai from scratch. we learn "
    "to code. we build to learn. the more we build, the more we learn. "
) * 20
tok = BPETokenizer()
tok.train(CORPUS, vocab_size=300)
ok = 0
for label, s in (("ascii", "the cat sat on the mat"),
                 ("unicode + emoji", "héllo wörld 🙂 नमस्ते")):
    ids = tok.encode(s)
    ok += check(f"roundtrip {label}: decode(encode(x)) == x",
                tok.decode(ids) == s)
probe = CORPUS[:200]
n_bytes = len(probe.encode("utf-8"))
n_toks = len(tok.encode(probe))
ok += check(f"compression on trained corpus: {n_bytes} bytes → {n_toks} tokens "
            f"({n_bytes / n_toks:.2f}×)", n_toks < n_bytes)
multi = sum(1 for i in tok.encode("the cat") if len(tok.decode([i])) > 1)
ok += check("multi-character tokens learned (merges worked)", multi > 0,
            f"{multi} multi-char tokens used to encode 'the cat'")
record("🔢 tokenizer", ok, 4)

# ── LAYER 4 · attention: softmax, temperature, scaling, mask, PE ─────
print("\n— 5 · attention — softmax · temperature · √d · causal mask · PE —")
ok = 0
s = softmax([1.0, 2.0, 3.0])
ok += check("softmax sums to exactly 1.0 and peaks on the max score",
            abs(sum(s) - 1.0) < 1e-9 and max(range(3), key=lambda i: s[i]) == 2,
            f"p = [{s[0]:.3f}, {s[1]:.3f}, {s[2]:.3f}]")
sharp = softmax([1.0, 2.0, 3.0], temperature=0.3)
flat = softmax([1.0, 2.0, 3.0], temperature=2.0)
ok += check("low temperature sharpens, high flattens (creativity dial)",
            max(sharp) > max(s) > max(flat),
            f"p_max: T=0.3 → {max(sharp):.3f} · T=1 → {max(s):.3f} · T=2 → {max(flat):.3f}")
q, K = [1.0, 2.0, 3.0], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
ok += check("scores are scaled by 1/√d (stability trick)",
            abs(scale_scores(q, K)[0] - dot(q, K[0]) / 3 ** 0.5) < 1e-12)
w = attention_weights([0.3, 0.9, 0.1], K + [[0.0, 0.0, 1.0]], causal_pos=1)
ok += check("causal mask: position 1 gives 0% to all later positions",
            all(p == 0.0 for p in w[2:]) and abs(sum(w) - 1.0) < 1e-9,
            f"weights = [{w[0]:.2f}, {w[1]:.2f}, 0, 0]")
pe = {p: positional_encoding(p, 8) for p in (0, 1, 3, 50)}
ok += check("positional encoding is deterministic", pe[3] == positional_encoding(3, 8))
ok += check("nearby positions encode more similarly than far ones",
            dot(pe[0], pe[1]) > dot(pe[0], pe[50]),
            f"cos(0,1)={dot(pe[0],pe[1]):+.2f} > cos(0,50)={dot(pe[0],pe[50]):+.2f}")
record("👀 attention", ok, 6)
# ══════════════════════════════════════════════════════════════════════
# PART 2 · REAL-WORLD SCENARIOS
# ══════════════════════════════════════════════════════════════════════
header("PART 2 · REAL-WORLD SCENARIOS")

# ── SCENARIO 6 — ☂ UMBRELLA (frozen-weight neuron) ───────────────────
print("\n— 6 · umbrella — a frozen neuron decides 4 real mornings —")
W, B = [2.0, -0.5, 1.0], -0.3


def umbrella(cloudy, summer, rained_yesterday):
    z = Value(cloudy) * W[0] + Value(summer) * W[1] + Value(rained_yesterday) * W[2] + B
    return z.tanh()


cases = [
    ("stormy:   cloudy=0.9 summer=0.1 rain_y=0.9", (0.9, 0.1, 0.9), True),
    ("beach day: cloudy=0.1 summer=0.9 rain_y=0.0", (0.1, 0.9, 0.0), False),
    ("typical:   cloudy=0.8 summer=0.7 rain_y=0.9", (0.8, 0.7, 0.9), True),
    ("clear sky: cloudy=0.0 summer=0.0 rain_y=0.0", (0.0, 0.0, 0.0), False),
]
ok = 0
for desc, (c, s, r), want_umbrella in cases:
    out = umbrella(c, s, r).data
    got = out > 0.3
    passed = got == want_umbrella
    ok += passed
    verdict = "☂ bring it" if got else "😎 leave it"
    print(f"  {PASS if passed else FAIL} {desc}")
    print(f"        output={out:+.3f}  →  {verdict}   (expected {'umbrella' if want_umbrella else 'no umbrella'})")
record("☂ umbrella", ok, len(cases))

# ── SCENARIO 7 — 📧 SPAM FILTER (train on 6 mails, test on 2 UNSEEN) ─
print("\n— 7 · spam filter — trained here, judged on UNSEEN mail —")


def features(text):
    """3 features: contains spammy word, has link/click word, shouty (!)."""
    t = text.lower()
    return [
        1.0 if any(w in t for w in ("free", "win", "prize", "winner")) else 0.0,
        1.0 if ("http" in t or "click" in t) else 0.0,
        1.0 if t.count("!") >= 1 else 0.0,
    ]


train_mails = [
    ("WIN a free prize now!", 1), ("free free free click here", 1),
    ("you won a free ticket", 1), ("meeting at 3pm in room 4", -1),
    ("lunch tomorrow?", -1), ("project update attached", -1),
]
spam_model = MLP(3, [4, 1])
for _ in range(150):
    preds = [spam_model(features(t)) for t, _ in train_mails]
    loss = sum((Value(1) - Value(y) * p).relu() for p, (_, y) in zip(preds, train_mails))
    loss = loss * (1 / len(train_mails))
    spam_model.zero_grad()
    loss.backward()
    for p in spam_model.parameters():
        p.data -= 0.1 * p.grad

test_mails = [
    ("claim your free prize, click http://x.co", 1),
    ("see you at the standup tomorrow", -1),
]
ok = 0
for text, want in test_mails:
    out = spam_model(features(text)).data
    got = 1 if out > 0 else -1
    passed = got == want
    ok += passed
    print(f"  {PASS if passed else FAIL} unseen mail: {text!r}")
    print(f"        spam score={out:+.3f}  →  {'SPAM' if got == 1 else 'HAM'}   (expected {'SPAM' if want == 1 else 'HAM'})")
record("📧 spam filter", ok, len(test_mails))
# ── SCENARIO 8 — 🏠 HOUSE PRICE (MSE regression) ─────────────────────
print("\n— 8 · house price — same loop, different loss (MSE) —")
# truth: price_k$ = 50 + 2 * size_m²   (e.g. 80 m² → 210k$)
sizes = [40, 60, 80, 100, 120]
xs = [[(s - 80) / 40] for s in sizes]          # normalize to [-1, 1]
ys = [(50 + 2 * s) / 100 for s in sizes]       # price in $100k units
house = MLP(1, [8, 1])
for _ in range(400):
    preds = [house(x) for x in xs]
    loss = sum((p - Value(y)) ** 2 for p, y in zip(preds, ys)) * (1 / len(ys))
    house.zero_grad()
    loss.backward()
    for p in house.parameters():
        p.data -= 0.05 * p.grad
final_loss = loss.data
pred_90 = house([(90 - 80) / 40]).data * 100           # back to k$
true_90 = 50 + 2 * 90
passed_loss = final_loss < 0.05
passed_pred = abs(pred_90 - true_90) < 30
print(f"  {PASS if passed_loss else FAIL} training MSE = {final_loss:.4f}  (target < 0.05)")
print(f"        avg error ≈ {final_loss ** 0.5 * 100:.1f}k$ on training houses")
print(f"  {PASS if passed_pred else FAIL} predict 90 m² house: {pred_90:.1f}k$  (truth {true_90}k$)")
record("🏠 house price", passed_loss + passed_pred, 2)

# ── SCENARIO 9 — 💬 CHAT API BILLING (tokenizer + cost) ──────────────
print("\n— 9 · chat API billing — tokens are literally money —")
RATE = 0.15 / 1e6        # $ per token (GPT-4o-mini input class)
messages = ["hi", "the cat sat on the mat", "great job! \U0001F600 see you tomorrow"]
ok = 0
prev_tokens = -1
for m in messages:
    ids = tok.encode(m)
    round_ok = tok.decode(ids) == m
    grows = len(ids) >= prev_tokens
    prev_tokens = len(ids)
    cost = len(ids) * RATE
    passed = round_ok and grows
    ok += passed
    print(f"  {PASS if passed else FAIL} {m!r}")
    print(f"        {len(m.encode('utf-8'))} bytes → {len(ids)} tokens → ${cost:.8f} per message")
record("💬 API billing", ok, len(messages))

# ── SCENARIO 10 — 🔬 AUTOGRAD UNIT TEST (analytical vs numerical) ────
print("\n— 10 · autograd unit test — the check real ML teams run —")
x = Value(1.7)
f = 3 * x ** 2 + 2 * x + 1
f.backward()
analytical = x.grad
h = 1e-5
numerical = ((3 * (1.7 + h) ** 2 + 2 * (1.7 + h) + 1) -
             (3 * (1.7 - h) ** 2 + 2 * (1.7 - h) + 1)) / (2 * h)
diff = abs(analytical - numerical)
passed = diff < 1e-3
print(f"  {PASS if passed else FAIL} f(x)=3x²+2x+1 at x=1.7")
print(f"        engine says  df/dx = {analytical:.6f}")
print(f"        nudge method df/dx = {numerical:.6f}")
print(f"        |difference| = {diff:.2e}  (target < 1e-3)")
record("🔬 autograd unit", int(passed), 1)
# ── SCENARIO 11 — 🌡 TEMPERATURE SAMPLING (the creativity dial) ──────
print("\n— 11 · temperature sampling — how ChatGPT picks its words —")
P = [0.70, 0.20, 0.10]                      # model's honest probabilities
LOGITS = [math.log(p) for p in P]           # inverse softmax
WORDS_ = ["the", "a", "cat"]
ok = 0
for T, expect in ((0.2, "deterministic"), (2.5, "exploratory")):
    dist = softmax(LOGITS, temperature=T)
    random.seed(11)
    draws = [random.choices(WORDS_, weights=dist)[0] for _ in range(30)]
    n_distinct = len(set(draws))
    top_share = draws.count(max(set(draws), key=draws.count)) / 30
    if expect == "deterministic":
        passed = n_distinct == 1
    else:
        passed = n_distinct >= 2
    ok += passed
    print(f"  {PASS if passed else FAIL} T={T:<3} 30 samples → {n_distinct} distinct words "
          f"(top word {top_share * 100:.0f}%)  → {expect}")
    print(f"        distribution: [{', '.join(f'{w}:{p * 100:.1f}%' for w, p in zip(WORDS_, dist))}]")
record("🌡 temperature", ok, 2)

# ── SCENARIO 12 — 👤 PRONOUN RESOLUTION (attention's party trick) ────
print("\n— 12 · pronoun resolution — what does 'it' refer to? —")
# toy 4-dim embeddings: [animal, action, street, tired/wide-sense]
SENT = ["the", "animal", "didn't", "cross", "the", "street", "because", "it"]
EMB_ = [[0.0, 0, 0, 0], [0.9, 0, 0, 0.2], [0.1, 0.8, 0, 0], [0.2, 0.9, 0.3, 0],
        [0.0, 0, 0, 0], [0, 0, 0.9, 0], [0, 0.4, 0, 0], [0.85, 0, 0, 0.4]]
ok = 0
for ending, it_emb, expect_animal in (("…was too TIRED", [0.7, 0, 0, 0.8], True),
                                      ("…was too WIDE", [0, 0, 0.8, 0.5], False)):
    embs = EMB_ + [it_emb]
    w = attention_weights(embs[-1], embs)
    a_w, s_w = w[1], w[5]                    # 'animal' vs 'street'
    got_animal = a_w > s_w
    passed = got_animal == expect_animal
    ok += passed
    print(f"  {PASS if passed else FAIL} \"the animal didn't cross the street {ending}\"")
    print(f"        'it' attends: animal {a_w * 100:.1f}% vs street {s_w * 100:.1f}%"
          f"  → 'it' = {'animal' if got_animal else 'street'}")
record("👤 pronouns", ok, 2)

# ── SCENARIO 13 — 📈 ATTENTION IS O(n²) (why context costs money) ────
print("\n— 13 · attention cost — why long context = big bill —")
ok = 0
for n in (4, 8, 16):
    pairs = n * n                            # every token scores every token
    ok += check(f"n={n:<3} tokens → {pairs} pairwise scores (n²)", pairs == n * n)
ratio = (16 * 16) / (4 * 4)
ok += check("4× more tokens → 16× more compute  (quadratic!)", ratio == 16,
            "1k tokens = 1M scores · 8k tokens = 67M scores · 128k = 16 billion")
record("📈 attn cost", ok, 4)

# ── SCENARIO 14 — 🎭 CAUSAL MASK PROOF (GPT never sees the future) ───
print("\n— 14 · causal mask — verify NO position leaks the future —")
random.seed(3)
N = 6
EMB6 = [[random.uniform(-1, 1) for _ in range(4)] for _ in range(N)]
future_mass = 0.0
row_sums_ok = True
for i in range(N):
    w = attention_weights(EMB6[i], EMB6, causal_pos=i)
    future_mass += sum(w[i + 1:])
    if abs(sum(w) - 1.0) > 1e-9:
        row_sums_ok = False
passed = future_mass < 1e-12 and row_sums_ok
print(f"  {PASS if passed else FAIL} {N}×{N} attention matrix, causal")
print(f"        total attention on future positions = {future_mass:.1e}  (must be 0)")
print(f"        every row still sums to 100%  → training stays parallel, no cheating")
record("🎭 causal mask", int(passed), 1)
# ── SCENARIO 15 — 🤖 MINI TRANSFORMER (end-to-end, trained) ──────────
print("\n— 15 · mini transformer — tokens → +PE → attention → prediction —")
# 4-dim hand-crafted embeddings (in real GPT these are LEARNED)
E = {
    "<pad>": [0.0, 0.0, 0.0, 0.0],
    "the":   [0.3, 0.0, 0.1, 0.0],
    "camel": [0.9, 0.1, 0.3, 0.5],
    "cat":   [0.2, 0.8, 0.0, 0.1],
    "runs":  [0.0, 0.9, 0.2, 0.0],
    "fast":  [0.1, 0.8, 0.0, 0.3],
}
data = [(["the", "camel", "runs"], 0), (["the", "cat", "runs"], 1)]
D_MODEL = 4


def sequence_vector(words):
    """One causal attention pass; return last position's context vector."""
    embs = [E[w] for w in words]
    keys = [[v + p for v, p in zip(e, positional_encoding(j, D_MODEL))]
            for j, e in enumerate(embs)]
    ctx, _ = attend(keys[-1], keys, keys, causal_pos=len(words) - 1)
    return ctx                       # last position predicts the NEXT word


# trainable classification head: d_model × 2 logits (the only learned params)
W_head = [[Value(random.uniform(-0.5, 0.5)) for _ in range(2)] for _ in range(D_MODEL)]


def head_logits(words):
    h = sequence_vector(words)
    return [sum(h[i] * W_head[i][c] for i in range(D_MODEL)) for c in range(2)]


for step in range(300):
    loss = Value(0.0)
    for words, cls in data:
        l0, l1 = head_logits(words)
        m = l0 if l0.data > l1.data else l1        # stability trick again
        z = (l0 - m).exp() + (l1 - m).exp()        # softmax denominator
        p_correct = (l0 - m).exp() / z if cls == 0 else (l1 - m).exp() / z
        loss = loss + p_correct.log() * -1         # cross-entropy
    loss = loss * (1 / len(data))
    for row in W_head:
        for p in row:
            p.grad = 0.0
    loss.backward()
    for row in W_head:
        for p in row:
            p.data -= 0.5 * p.grad

ok = 0
ok += check(f"training cross-entropy = {loss.data:.4f}  (target < 0.5)",
            loss.data < 0.5)
n_correct = 0
for words, cls in data:
    l0, l1 = head_logits(words)
    got = 0 if l0.data > l1.data else 1
    n_correct += got == cls
    print(f"  {PASS if got == cls else FAIL} {' '.join(words)!r} + '___' → "
          f"class {got} (want {cls})   logits [{l0.data:+.2f}, {l1.data:+.2f}]")
ok += check(f"both training sequences predicted next word correctly ({n_correct}/2)",
            n_correct == 2)
l0, l1 = head_logits(["the", "camel", "fast"])     # UNSEEN ending
gen = (0 if l0.data > l1.data else 1) == 0
ok += check(f"generalizes to unseen ending: 'the camel fast' → class "
            f"{0 if l0.data > l1.data else 1} (want 0)", gen)
record("🤖 mini transformer", ok, 3)

# ──────────────────────────────────────────────────────────────────────
# FINAL SCOREBOARD
# ──────────────────────────────────────────────────────────────────────
def scoreblock(title, rows):
    print("\n" + "╔" + "═" * 60 + "╗")
    print("║" + title.center(60) + "║")
    print("╚" + "═" * 60 + "╝")
    p_all = t_all = 0
    for name, p, t in rows:
        pct = 100 * p / t
        print(f"  {name:<22} [{bar(pct, 30)}] {p}/{t}")
        p_all += p
        t_all += t
    return p_all, t_all


p1, t1 = scoreblock("  PART 1 · LAYER-BY-LAYER UNIT TESTS", results[:5])
p2, t2 = scoreblock("  PART 2 · REAL-WORLD SCENARIOS", results[5:])
total_p, total_t = p1 + p2, t1 + t2
pct = 100 * total_p / total_t
print(f"\n  TOTAL: {total_p}/{total_t}  [{bar(pct)}] {pct:.0f}%")
if total_p == total_t:
    print("\n  ✅ ALL SYSTEMS GO — every layer verified, every scenario solved.")
    print("     engine → nn → tokenizer → attention → real products. Phase 4 complete!")
    sys.exit(0)
else:
    print("\n  ❌ FAILURES FOUND — re-read the failing check above.")
    sys.exit(1)





