"""
MICROGRAD INTERACTIVE EXPLAINER
================================
Run this file. It walks you through each concept one at a time.
After each section, it PAUSES and waits for you.

Type your question at any PAUSE prompt — it will be saved to QUESTIONS.md
Press Enter alone to move to the next section.

Usage:
    python explainer.py
"""

import sys
import os
import time

# ─── Setup ────────────────────────────────────────────────────────────────────

QUESTIONS_FILE = os.path.join(os.path.dirname(__file__), 'QUESTIONS.md')
session_questions = []

def pause(section_id, section_title):
    """Pause and collect questions from the user."""
    print()
    print("─" * 60)
    q = input(f"  ❓  Any question about '{section_title}'?\n"
              f"      (Press Enter to continue, or type your question): ").strip()
    if q:
        entry = f"\n### [{section_id}] {section_title}\n**Q:** {q}\n**A:** *(to be answered)*\n"
        session_questions.append(entry)
        # Save immediately
        with open(QUESTIONS_FILE, 'a', encoding='utf-8') as f:
            f.write(entry)
        print(f"\n  ✓ Saved to QUESTIONS.md — I'll answer it when you ask!\n")
    print()

def show(text):
    """Print with a tiny delay so it feels like it's being written."""
    for line in text.split('\n'):
        print(line)

def run(code_str, namespace):
    """Execute code and return locals."""
    exec(code_str, namespace)

def banner(title):
    print()
    print("╔" + "═" * 58 + "╗")
    print("║  " + title.center(56) + "  ║")
    print("╚" + "═" * 58 + "╝")

def section(num, title):
    print()
    print(f"  ── SECTION {num}: {title} " + "─" * max(1, 45 - len(title)))
    print()


# ─── Add engine.py to path ────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

# ─── Write fresh QUESTIONS header ─────────────────────────────────────────────
with open(QUESTIONS_FILE, 'a', encoding='utf-8') as f:
    from datetime import datetime
    f.write(f"\n\n# Session: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")


# ══════════════════════════════════════════════════════════════════════════════
banner("MICROGRAD EXPLAINER — Learn by Running")
# ══════════════════════════════════════════════════════════════════════════════

show("""
  Welcome! This explainer walks through micrograd one concept at a time.

  Each section:
    1. Explains the idea in plain words
    2. Shows and RUNS the actual code
    3. Pauses for your questions

  Your questions are saved to QUESTIONS.md automatically.
  You can ask me about them anytime.
""")

input("  → Press Enter to begin...\n")


# ══════════════════════════════════════════════════════════════════════════════
section(1, "The Problem — Why Do We Need Gradients?")
# ══════════════════════════════════════════════════════════════════════════════

show("""
  Imagine a simple machine:

      output = weight × input

  If input = 3.0 and weight = 2.0 → output = 6.0
  We WANT output to be 9.0.

  Should we make the weight BIGGER or SMALLER?

  Answer: bigger. Because output = weight × 3.0
          So more weight → more output.

  The gradient tells us this AUTOMATICALLY.
  For ANY function. For ANY number of weights.

  That's the entire point of micrograd.
""")

pause("S1", "The Problem — Why Gradients?")


# ══════════════════════════════════════════════════════════════════════════════
section(2, "What is a Gradient? — The Slope Idea")
# ══════════════════════════════════════════════════════════════════════════════

show("""
  A gradient = the slope of a curve at a point.

  Example: y = x²

       y
       │         *
       │       *   *
       │     *       *
       │   *           *
       │ *               *
       └──────────────────── x
           minimum at x=0

  At x=3: slope = 2x = 6  (positive → curve going up to the right)
  At x=-3: slope = -6     (negative → curve going up to the left)
  At x=0: slope = 0       (flat → we're at the minimum)

  To minimize y: move x in the OPPOSITE direction of the slope.
  Slope is positive? → decrease x.
  Slope is negative? → increase x.
  Slope is zero? → you're at the minimum. Done.

  Neural network training = finding the minimum of the LOSS function.
  Gradient descent = moving downhill step by step.
""")

pause("S2", "What is a Gradient?")


# ══════════════════════════════════════════════════════════════════════════════
section(3, "The Value Class — Wrapping Numbers")
# ══════════════════════════════════════════════════════════════════════════════

show("""
  Every number in micrograd is wrapped in a Value object.

  A Value stores:
    .data  → the actual number
    .grad  → the gradient (starts at 0, filled by .backward())

  That's the whole idea.
""")

show("  CODE:")
show("""
    from engine import Value

    a = Value(2.0, label='a')
    print(a)
    print("data:", a.data)
    print("grad:", a.grad)   # 0.0 — not computed yet
""")

ns = {}
from engine import Value
a = Value(2.0, label='a')
print(f"  a        = {a}")
print(f"  a.data   = {a.data}")
print(f"  a.grad   = {a.grad}  ← not computed yet\n")

pause("S3", "The Value Class")


# ══════════════════════════════════════════════════════════════════════════════
section(4, "Operations — Building the Computation Graph")
# ══════════════════════════════════════════════════════════════════════════════

show("""
  When you do arithmetic on Value objects, they DON'T just compute a number.
  They build a GRAPH — a record of what operations happened.

  Every new Value remembers:
    - Which Values created it  (_children)
    - Which operation          (_op)
    - How to backprop through it (_backward)
""")

show("  CODE:")
show("""
    a = Value(2.0, label='a')
    b = Value(3.0, label='b')
    c = a * b              # c = 6.0, remembers: (a, b), op='*'
    d = a + b              # d = 5.0, remembers: (a, b), op='+'
    e = c + d              # e = 11.0

    # The graph:
    #   a ──┬──[×]──→ c ──┐
    #   b ──┘              [+]──→ e
    #   a ──┬──[+]──→ d ──┘
    #   b ──┘
""")

a = Value(2.0, label='a')
b = Value(3.0, label='b')
c = a * b
d = a + b
e = c + d

print(f"  a = {a.data}")
print(f"  b = {b.data}")
print(f"  c = a * b = {c.data}")
print(f"  d = a + b = {d.data}")
print(f"  e = c + d = {e.data}")
print()
print(f"  c._children = {set(v.label for v in c._children)}  (a and b created c)")
print(f"  c._op       = '{c._op}'")
print(f"  e._children = {set(v.label or str(v.data) for v in e._children)}")

pause("S4", "Building the Computation Graph")


# ══════════════════════════════════════════════════════════════════════════════
section(5, "Backward Pass — Computing Gradients Automatically")
# ══════════════════════════════════════════════════════════════════════════════

show("""
  .backward() walks the graph in REVERSE and computes .grad for every node.

  The math for addition  z = a + b:
    dz/da = 1  →  a.grad += z.grad × 1
    dz/db = 1  →  b.grad += z.grad × 1
    (gradient passes through unchanged)

  The math for multiplication  z = a × b:
    dz/da = b  →  a.grad += z.grad × b.data
    dz/db = a  →  b.grad += z.grad × a.data
    (each input's gradient = the OTHER input's value × incoming gradient)
""")

show("  CODE — running e.backward():")

a = Value(2.0, label='a')
b = Value(3.0, label='b')
c = a * b
d = a + b
e = c + d

print(f"  Before backward: a.grad={a.grad}, b.grad={b.grad}")
e.backward()
print(f"  After  backward: a.grad={a.grad}, b.grad={b.grad}")
print()
print("  Manual check:")
print("  e = a*b + a + b")
print(f"  de/da = b + 1 = {b.data} + 1 = {b.data + 1}  →  a.grad={a.grad} {'✓' if a.grad == b.data+1 else '✗'}")
print(f"  de/db = a + 1 = {a.data} + 1 = {a.data + 1}  →  b.grad={b.grad} {'✓' if b.grad == a.data+1 else '✗'}")

pause("S5", "The Backward Pass")


# ══════════════════════════════════════════════════════════════════════════════
section(6, "The Chain Rule — How Gradients Flow Through Layers")
# ══════════════════════════════════════════════════════════════════════════════

show("""
  The Chain Rule is the mathematical rule that makes backprop work.

  If you have:   final = f( g( x ) )
  Then:          d(final)/dx = d(final)/d(g) × d(g)/dx

  In English: the gradient at x = (gradient arriving from above)
                                 × (local derivative at this node)

  Example with 3 nodes in a chain:
    a → [×2] → b → [×3] → c → [×4] → d

    d = a × 2 × 3 × 4 = a × 24
    dd/da = 24

  How it flows back:
    d.grad = 1.0                       (seed)
    c.grad = d.grad × 4 = 4.0         (local deriv of ×4 is 4)
    b.grad = c.grad × 3 = 12.0        (local deriv of ×3 is 3)
    a.grad = b.grad × 2 = 24.0        (local deriv of ×2 is 2)

  Each step MULTIPLIES the incoming gradient by the local derivative.
  That's the chain rule.
""")

a = Value(5.0, label='a')
b = a * 2
c = b * 3
d = c * 4

d.backward()

print(f"  a = {a.data}")
print(f"  b = a×2 = {b.data}")
print(f"  c = b×3 = {c.data}")
print(f"  d = c×4 = {d.data}")
print()
print(f"  a.grad = {a.grad}  ← expected: 2×3×4 = 24  {'✓' if a.grad==24 else '✗'}")
print(f"  b.grad = {b.grad}  ← expected: 3×4 = 12")
print(f"  c.grad = {c.grad}  ← expected: 4")

pause("S6", "The Chain Rule")


# ══════════════════════════════════════════════════════════════════════════════
section(7, "Activation Functions — Why tanh and ReLU Exist")
# ══════════════════════════════════════════════════════════════════════════════

show("""
  Without activation functions:
    output = w2 × (w1 × x + b1) + b2
           = (w2×w1) × x + (w2×b1 + b2)
           = W × x + B          ← STILL just a straight line!

  No matter how many layers — if no activation — the whole network
  collapses into a single linear function. Useless for any real problem.

  With tanh activation:
    output = tanh(w × x + b)    ← curved, expressive

  Stack multiple curved functions → can approximate any shape.
""")

import math

show("  Comparing raw vs tanh:")
print()
print(f"  {'Input':>8}  {'Raw (no act)':>14}  {'tanh output':>14}")
print("  " + "─" * 42)
for x in [-3, -2, -1, 0, 1, 2, 3]:
    raw = x * 1.0
    t = math.tanh(x)
    print(f"  {x:>8}  {raw:>14.4f}  {t:>14.4f}")

print()
print("  Notice: tanh always stays between -1 and +1")
print("  ReLU = max(0, x):  negative → 0,  positive → unchanged")

a = Value(-1.5)
print()
print(f"  a = {a.data}")
print(f"  a.tanh() = {a.tanh().data:.4f}")
print(f"  a.relu() = {a.relu().data:.4f}  ← clamped to 0 because a < 0")

pause("S7", "Activation Functions")


# ══════════════════════════════════════════════════════════════════════════════
section(8, "A Neuron — The Building Block")
# ══════════════════════════════════════════════════════════════════════════════

show("""
  One neuron does exactly this:

    output = tanh( w1×x1 + w2×x2 + ... + wn×xn + b )

  The weights (w) and bias (b) are LEARNED during training.
  The inputs (x) are your data — they don't change.

  The neuron learns to recognize one specific pattern in the input.
""")

show("  CODE — manually building one neuron from scratch:")

x1 = Value(2.0, label='x1')
x2 = Value(3.0, label='x2')

w1 = Value(0.5,  label='w1')   # initialized, will be learned
w2 = Value(-0.3, label='w2')
b  = Value(0.1,  label='b')

# Forward pass
dot = w1*x1 + w2*x2 + b
output = dot.tanh()

print(f"  x1={x1.data}, x2={x2.data}")
print(f"  w1={w1.data}, w2={w2.data}, b={b.data}")
print()
print(f"  dot = w1×x1 + w2×x2 + b")
print(f"      = {w1.data}×{x1.data} + {w2.data}×{x2.data} + {b.data}")
print(f"      = {w1.data*x1.data} + {w2.data*x2.data} + {b.data}")
print(f"      = {dot.data:.4f}")
print(f"  tanh(dot) = {output.data:.4f}  ← neuron output")

output.backward()
print()
print("  After backward — gradients on weights:")
print(f"  w1.grad = {w1.grad:.4f}  ← how much does output change if w1 changes?")
print(f"  w2.grad = {w2.grad:.4f}")
print(f"  b.grad  = {b.grad:.4f}")

pause("S8", "A Single Neuron")


# ══════════════════════════════════════════════════════════════════════════════
section(9, "The Training Loop — Making the Neuron Learn")
# ══════════════════════════════════════════════════════════════════════════════

show("""
  Training = repeating 4 steps until the network is good:

    Step 1: FORWARD  — compute output and loss
    Step 2: ZERO     — reset all gradients to 0
    Step 3: BACKWARD — compute all gradients
    Step 4: UPDATE   — nudge weights in direction that reduces loss

  The loss measures: how wrong are we right now?
  We use MSE: loss = (output - target)²

  Goal: train the neuron to output +1.0 when input is [2.0, 3.0]
""")

import random
random.seed(0)

# Fresh start
w1 = Value(random.uniform(-1,1), label='w1')
w2 = Value(random.uniform(-1,1), label='w2')
b  = Value(0.0, label='b')

x = [Value(2.0), Value(3.0)]
target = 1.0
lr = 0.1

print(f"  Initial: w1={w1.data:.4f}, w2={w2.data:.4f}, b={b.data:.4f}")
print()
print(f"  {'Step':>5}  {'Output':>10}  {'Loss':>10}  {'w1':>9}  {'w2':>9}")
print("  " + "─" * 50)

for step in range(15):
    # Step 1: Forward
    out = (w1*x[0] + w2*x[1] + b).tanh()
    loss = (out - target) ** 2

    # Step 2: Zero gradients
    w1.grad = 0.0
    w2.grad = 0.0
    b.grad  = 0.0

    # Step 3: Backward
    loss.backward()

    # Step 4: Update
    w1.data -= lr * w1.grad
    w2.data -= lr * w2.grad
    b.data  -= lr * b.grad

    print(f"  {step:>5}  {out.data:>10.6f}  {loss.data:>10.6f}  {w1.data:>9.5f}  {w2.data:>9.5f}")

print()
print(f"  Target was {target:.1f}")
print(f"  Final output: {out.data:.6f}")
print(f"  Error: {abs(out.data - target):.6f}")

pause("S9", "The Training Loop")


# ══════════════════════════════════════════════════════════════════════════════
section(10, "Full MLP — Multiple Neurons, Multiple Layers")
# ══════════════════════════════════════════════════════════════════════════════

show("""
  A single neuron can only learn one pattern.
  A layer of neurons can learn MULTIPLE patterns from the same input.
  Multiple layers can combine patterns → learn complex concepts.

  MLP(3, [4, 4, 1]) means:
    Input:    3 numbers
    Layer 1:  4 neurons (each reads all 3 inputs)
    Layer 2:  4 neurons (each reads all 4 outputs from layer 1)
    Output:   1 neuron  (reads all 4 from layer 2)
""")

from nn import MLP

random.seed(42)
model = MLP(n_inputs=3, layer_sizes=[4, 4, 1])

print(f"  Model: {len(model.parameters())} total parameters (weights + biases)")
print()

X = [
    [2.0,  3.0, -1.0],   # → +1
    [3.0, -1.0,  0.5],   # → -1
    [0.5,  1.0,  1.0],   # → +1
    [1.0,  1.0, -1.0],   # → -1
]
y_true = [1.0, -1.0, 1.0, -1.0]

def total_loss(model, X, y_true):
    y_pred = [model(xi) for xi in X]
    losses = [(1 - yi*yp).relu() for yi, yp in zip(y_true, y_pred)]
    data_loss = sum(losses) * (1.0/len(losses))
    reg = 0.0001 * sum(p**2 for p in model.parameters())
    acc = sum(1 for yi, yp in zip(y_true, y_pred) if (yi>0)==(yp.data>0)) / len(y_true)
    return data_loss + reg, acc

print(f"  {'Step':>5}  {'Loss':>10}  {'Accuracy':>10}")
print("  " + "─" * 32)

for step in range(80):
    loss, acc = total_loss(model, X, y_true)
    model.zero_grad()
    loss.backward()
    for p in model.parameters():
        p.data -= 0.05 * p.grad
    if step % 10 == 0 or step == 79:
        print(f"  {step:>5}  {loss.data:>10.6f}  {acc*100:>9.1f}%")

print()
print("  Final predictions:")
for i, (xi, yi) in enumerate(zip(X, y_true)):
    pred = model(xi)
    ok = "✓" if (yi>0)==(pred.data>0) else "✗"
    print(f"    Sample {i+1}: pred={pred.data:+.4f}  target={yi:+.0f}  {ok}")

pause("S10", "Full MLP Training")


# ══════════════════════════════════════════════════════════════════════════════
banner("ALL SECTIONS COMPLETE")
# ══════════════════════════════════════════════════════════════════════════════

print("""
  What you just walked through:

    S1 — Why gradients exist
    S2 — What a gradient is (the slope idea)
    S3 — The Value class (wrapping numbers)
    S4 — Computation graph (recording operations)
    S5 — Backward pass (computing gradients)
    S6 — Chain rule (how gradients flow)
    S7 — Activation functions (tanh, relu)
    S8 — One neuron (dot product + activation)
    S9 — Training loop (4 steps, repeated)
    S10 — Full MLP (100% accuracy from scratch)
""")

if session_questions:
    print(f"  Your questions this session ({len(session_questions)}):")
    for q in session_questions:
        line = [l for l in q.strip().split('\n') if l.startswith('**Q:**')]
        if line:
            print(f"    · {line[0].replace('**Q:** ', '')}")
    print()
    print(f"  All saved to: QUESTIONS.md")
    print(f"  Paste any question to the chat and I'll answer in detail.")
else:
    print("  No questions logged this session.")
    print("  If anything was unclear, ask me directly in chat.")

print()
print("  NEXT STEP:")
print("  → Run each step file individually and modify the numbers:")
print("    python step1_just_values.py")
print("    python step2_one_neuron.py")
print("    python step3_full_network.py")
print()
