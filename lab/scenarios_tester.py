"""
lab/scenarios_tester.py
=======================
REAL-WORLD SCENARIOS TESTER — everything you built in phases 1 & 2,
tested against 5 real-world situations with built-in PASS/FAIL checks.

  Scenario 1  ☂  Umbrella decision      (frozen-weight neuron)
  Scenario 2  📧 Spam filter            (train an MLP, test on UNSEEN mail)
  Scenario 3  🏠 House price estimate   (MSE regression)
  Scenario 4  💬 Chat API billing       (tokenizer + cost calculator)
  Scenario 5  🔬 Autograd unit test     (analytical vs numerical gradient)

Run:  python scenarios_tester.py
Exit code 0 = all pass, 1 = something failed (CI-friendly).
"""

import sys, os, random

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..', 'micrograd'))
sys.path.insert(0, os.path.join(HERE, '..', 'tokenizer'))

from engine import Value          # noqa: E402
from nn import MLP                # noqa: E402
from bpe import BPETokenizer      # noqa: E402

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


results = []   # (scenario, passed, total)


def record(name, passed, total):
    results.append((name, passed, total))
    pct = 100 * passed / total
    print(f"\n  {name}: {passed}/{total} tests passed   [{bar(pct)}] {pct:.0f}%")


# ════════════════════════════════════════════════════════════════════
print("╔" + "═" * 60 + "╗")
print("║" + "  AI LAB — REAL-WORLD SCENARIOS TESTER".center(60) + "║")
print("║" + "  phases 1+2 under test: micrograd engine · nn · BPE tokenizer".center(60) + "║")
print("╚" + "═" * 60 + "╝")

# ──────────────────────────────────────────────────────────────────────
# SCENARIO 1 — ☂ UMBRELLA (frozen-weight neuron)
# ──────────────────────────────────────────────────────────────────────
header("SCENARIO 1 ☂  Umbrella decision (frozen neuron)")
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
record("☂ Umbrella", ok, len(cases))

# ──────────────────────────────────────────────────────────────────────
# SCENARIO 2 — 📧 SPAM FILTER (train on 6 mails, test on 2 UNSEEN)
# ──────────────────────────────────────────────────────────────────────
header("SCENARIO 2 📧  Spam filter (train → test on unseen mail)")


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
model = MLP(3, [4, 1])
for _ in range(150):
    preds = [model(features(t)) for t, _ in train_mails]
    loss = sum((Value(1) - Value(y) * p).relu() for p, (_, y) in zip(preds, train_mails)) * (1 / len(train_mails))
    model.zero_grad()
    loss.backward()
    for p in model.parameters():
        p.data -= 0.1 * p.grad

test_mails = [
    ("claim your free prize, click http://x.co", 1),
    ("see you at the standup tomorrow", -1),
]
ok = 0
for text, want in test_mails:
    out = model(features(text)).data
    got = 1 if out > 0 else -1
    passed = got == want
    ok += passed
    print(f"  {PASS if passed else FAIL} unseen mail: {text!r}")
    print(f"        spam score={out:+.3f}  →  {'SPAM' if got == 1 else 'HAM'}   (expected {'SPAM' if want == 1 else 'HAM'})")
record("📧 Spam filter", ok, len(test_mails))

# ──────────────────────────────────────────────────────────────────────
# SCENARIO 3 — 🏠 HOUSE PRICE (MSE regression)
# ──────────────────────────────────────────────────────────────────────
header("SCENARIO 3 🏠  House price estimate (MSE regression)")
# truth: price_k$ = 50 + 2 * size_m2   (e.g. 80 m² → 210k$)
sizes = [40, 60, 80, 100, 120]
xs = [[(s - 80) / 40] for s in sizes]                 # normalize to [-1, 1]
ys = [(50 + 2 * s) / 100 for s in sizes]              # price in $100k units
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
record("🏠 House price", passed_loss + passed_pred, 2)

# ──────────────────────────────────────────────────────────────────────
# SCENARIO 4 — 💬 CHAT API BILLING (tokenizer + cost)
# ──────────────────────────────────────────────────────────────────────
header("SCENARIO 4 💬  Chat API billing (tokens = money)")
CORPUS = (
    "the cat sat on the mat. the cat saw the dog. the dog ran to the cat. "
    "the cat and the dog sat. we learn to build ai from scratch. we learn "
    "to code. we build to learn. the more we build, the more we learn. "
) * 20
tok = BPETokenizer()
tok.train(CORPUS, vocab_size=300)

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

# ──────────────────────────────────────────────────────────────────────
# SCENARIO 5 — 🔬 AUTOGRAD UNIT TEST (analytical vs numerical)
# ──────────────────────────────────────────────────────────────────────
header("SCENARIO 5 🔬  Autograd unit test (the check real ML teams run)")
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
record("🔬 Autograd check", int(passed), 1)

# ──────────────────────────────────────────────────────────────────────
# FINAL SCOREBOARD
# ──────────────────────────────────────────────────────────────────────
total_p = sum(p for _, p, _ in results)
total_t = sum(t for _, _, t in results)
print("\n" + "╔" + "═" * 60 + "╗")
print("║" + "  FINAL SCOREBOARD".center(60) + "║")
print("╚" + "═" * 60 + "╝")
for name, p, t in results:
    pct = 100 * p / t
    print(f"  {name:<22} [{bar(pct, 30)}] {p}/{t}")
pct = 100 * total_p / total_t
print(f"\n  TOTAL: {total_p}/{total_t}  [{bar(pct)}] {pct:.0f}%")
if total_p == total_t:
    print("\n  ✅ ALL SYSTEMS GO — your phase-1 & phase-2 engines pass every real-world test.")
    sys.exit(0)
else:
    print("\n  ❌ FAILURES FOUND — re-read the failing scenario above.")
    sys.exit(1)