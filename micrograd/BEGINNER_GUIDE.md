# The Amateur's Guide to Micrograd & AI Engineering

### Every section of REPORT.md explained in plain English — no prior math or ML knowledge needed

> **How to read this:** Read top to bottom, like a story. Each section builds on the last.
> Everything maps to real code in this folder: `engine.py`, `nn.py`, `step1..3`.
>
> **🎨 Prefer pictures over numbers?** Open **`VISUAL_GUIDE.md`** in this same folder —
> every analogy and concept below is also drawn as a diagram there.
>
> **Difficulty markers used throughout:**
>
> - 🟢 **Basic** — everyday language, analogies, tiny numbers. Everyone should get these.
> - 🟡 **Intermediate** — needs a little thought, real numbers worked out.
> - 🔴 **Advanced** — deeper details. Skip on first read; come back later.

---

## Part 0 — The Whole Story in One Paragraph 🟢

A neural network is just a **pile of adjustable numbers (weights)** combined with your data using
simple math (+, ×, tanh). Training means: make a guess, measure how wrong you are (the **loss**),
then use **gradients** (a formula that says "nudge this number up or down") to fix every weight a
little bit. Repeat thousands of times. `engine.py` (~200 lines) implements the "compute every
nudge automatically" part. That's it. GPT-4 does the *same thing*, just with billions of numbers
on GPUs.

---

## Part 0.5 — Five Everyday Analogies (Read These First!) 🟢

Before any math, here is the entire training process in five things you already know:

### Analogy 1 — The Shower Thermostat (learning rate & overshoot)

You step into a shower that's too cold. You turn the hot knob. But by how much?

- Turn it a **tiny** bit → you'll stand there shivering for 20 minutes *(learning rate too small)*
- Crank it **fully** → SCALDING. Now you crank cold fully. Freezing. Scalding. You oscillate
  forever and never land on comfortable *(learning rate too big — divergence!)*
- Turn it a **small sensible** amount each time → warm shower in a few adjustments *(just right)*

A neural network does exactly this with every weight, every step. The learning rate IS the
"how far do I turn the knob per adjustment" setting.

### Analogy 2 — Tasting the Soup (the loss + the loop)

You're making soup. You taste it (⇒ measure the loss). "Too bland" (high loss). You add a
**pinch** of salt (⇒ small weight update). Taste again. Less bland (lower loss). Repeat until
delicious (loss ≈ 0).

Notice you never compute the perfect salt amount on paper. You use **feedback**: taste → adjust →
taste. Training a network is cooking with a very precise "tongue" (the gradient) that tells you
not just "too bland" but exactly "add 0.32 grams of salt."

### Analogy 3 — Golf in Thick Fog (gradient descent)

You're on a hilly golf course in fog so thick you can't see the hole (the lowest point). But you
CAN feel the slope under your feet. Strategy: always step in the direction that feels *downhill*.
Small steps. Eventually you arrive at the bottom — without ever *seeing* the destination.
That is literally gradient descent: `step_direction = −slope`.

### Analogy 4 — The Blame Meeting (backpropagation)

A company loses money (high loss). The CEO asks each department: "how much of this is YOUR fault?"
Each department blames its sub-teams, proportional to how much each sub-team contributed to the
department's decisions. Eventually every individual employee gets a blame score.

- The **loss** = the total money lost.
- **Backward pass** = the blame flowing down the org chart, department by department.
- **The chain rule** = the blame gets *multiplied* by each manager's "influence" on the way down.
- **The weights** = the employees. High-blame employees change their behavior the most.

### Analogy 5 — Currency Exchange (the chain rule)

Exchange $100 → Euros at rate 0.9, then Euros → Rupees at rate 90.

```
$100 → €90 → ₹8100
```

How sensitive are the final Rupees to each rate?

- Nudge the USD→EUR rate by 0.01 → each dollar yields 0.01 fewer Euros →
  Rupees change by 0.01 × 90 = **0.9 per dollar**.
- Nudge the EUR→INR rate by 1 → each Euro yields 1 fewer Rupee →
  Rupees change by 0.9 × 1 = **0.9 per dollar** (0.9 Euros per dollar × 1).
- Total sensitivity of Rupees to Dollars = 0.9 × 90 = **81** = the PRODUCT
  of the per-step rates.

**The total sensitivity is the PRODUCT of the step-by-step sensitivities.**
That's the chain rule: rates along the path multiply.

---

## 1. What Is AI Engineering? (User vs Builder) 🟢

- An **AI user** calls APIs, writes prompts, imports libraries. Like *driving a car*.
- An **AI engineer** builds the engine, the transmission, the fuel system. Like *building the car*.

Why build from scratch? Because after you build a tiny engine yourself, every big framework stops
being magic. PyTorch stops being a black box and becomes "micrograd with tensors."

**Your project = building the engine.** `engine.py` IS the engine.

---

## 2–3. The Roadmap & Starter Pack (What to Learn, In Order) 🟢

Think of it like martial arts belts:

| Belt | Project | One-line Why |
| ------ | --------- | -------------- |
| 🥇 White | **Autograd engine** (you did this!) | Teaches how machines *learn*. Everything rests on it. |
| 🥈 | Tokenizer (BPE) | Turns text into numbers — all models eat numbers, not words. |
| 🥉 | Matrix multiplication | The single math operation 95% of AI compute is spent on. |
| 🟢 | Transformer | The architecture behind ChatGPT, Claude, Gemini. |
| 🟢 | Small Language Model | Train a tiny GPT yourself. |
| 🔵 | ReAct Agent | The Thought → Action → Observation loop behind AI agents. |
| 🟣 | LoRA / RLHF / Quantization | How models get fine-tuned, aligned, and shrunk. |
| 🟤 | Production systems | Inference servers, distributed training, MoE. |

The **starter pack** = Autograd + Transformer + ReAct Agent. These 3 cover ~80% of concepts.

**Prerequisites (honestly):** Python classes & functions, basic NumPy, and knowing
`d/dx(x²) = 2x`. That's genuinely all the math you need to start.

---

## 4. The Core Problem — Why Gradients Exist (The Knob Story) 🟢

### 4.1 One knob, one problem

```
output = weight × input

weight = 2.0   ← the knob (you can turn it)
input  = 3.0   ← the data (fixed, cannot touch)

output = 6.0
```

Someone says: *"It should output 9.0."* You turn the knob to 3.0. Done. Easy — with ONE knob.

### 4.2 The real problem: a wall of knobs

A neural network has thousands to **billions** of knobs, all tangled together. You can't brute-force
try combinations (more possibilities than atoms in the universe). You need a formula that, **for
each knob individually**, says: *"turn this one up a little"* or *"turn that one down a little."*

**That formula is the gradient.**

🟢 **Kitchen version:** imagine seasoning a 20-dish banquet. Tasting and adjusting each dish one at
a time, re-tasting after every change, would take forever — flavors interact. You'd want a magic
tasting spoon that, in ONE pass, tells you for ALL 20 dishes: "more salt here, less lemon there,
+1 tsp sugar there." Backpropagation is the magic tasting spoon.

### 4.3 What is a derivative, really? (zero-calculus intro) 🟢

Forget symbols. A derivative answers ONE question:

> *"If I change the input a tiny bit, how much does the output change?"*

- **Straight line** `y = 3x`: increase x by 1 → y increases by exactly 3. Always. Slope = 3. Derivative = 3.
- **Curve** `y = x²`: the steepness CHANGES as you move. At x=2, nudging x by 0.1 gives
  4.41 − 4 = 0.41 ≈ slope 4. At x=3: 9.61 − 9 = 0.61 ≈ slope 6. The pattern: slope = 2x.
  That's the famous rule `d/dx(x²) = 2x`. You just derived it by nudging. That's ALL calculus
  does — automate the nudging.

### 4.4 One number for "how wrong are we?" (the Loss) 🟢

```
Loss = (prediction − target)²

prediction=0.3, target=1.0  →  Loss = (0.7)² = 0.49   ← very wrong
prediction=0.9, target=1.0  →  Loss = (0.1)² = 0.01   ← almost right
prediction=1.0, target=1.0  →  Loss = 0              ← perfect
```

🟢 **Archery version:** prediction = where your arrow lands, target = the bullseye. Loss = squared
distance from center. Missing by 7 cm is punished 49× worse than missing by 1 cm — as it should be.

**Why squared?**

1. Errors can't be negative (missing left by 1 and right by 1 shouldn't cancel to "perfect").
2. Big mistakes hurt much more than small ones.

🟡 **A hidden bonus (advanced):** squaring also gives a *smooth* derivative. The alternative
`Loss = |error|` has a sharp KINK at zero — its derivative jumps from −1 to +1, which breaks
smooth learning. Squared error's derivative `2×error` glides smoothly through zero.

### 4.5 Finding the nudge by hand (just once, to see it) 🟡

With `output = w1×x1 + w2×x2 + b` (x1=2, x2=3, w2=−0.3, b=0.1, target=1.0):

```
w1=0.5   → output=0.2 → Loss = 0.64
w1=0.501 → output=0.202 → Loss = 0.636804

Loss got SMALLER when w1 got BIGGER.
Ratio = −0.003196 / 0.001 ≈ −3.2   ← this is the gradient of w1
```

**How to read −3.2:** negative means "increase w1 → loss goes down."
So we update: `w1 = 0.5 − 0.1 × (−3.2) = 0.82` and loss drops from 0.64 → 0.0256. Magic? No — math.

### 4.6 Why we need the automatic version 🟢

That nudge-test works, but you'd have to run the whole network TWICE per knob.
A million knobs = 2 million runs per training step. Impossible.

**Backpropagation** computes ALL knobs' gradients in **one** pass. `engine.py` implements it.
That is the entire point of this project.

🔴 **Advanced — gradient checking:** the "nudge" method is called a *numerical* gradient, and it's
still used today to VERIFY autograd engines:
`f'(x) ≈ (f(x+h) − f(x−h)) / 2h` (the "central difference" — averaging both sides is more
accurate). If analytical gradients ever disagree with numerical ones, you have a bug. Karpathy's
micrograd lecture does exactly this test.

---

## 5. What Is a Gradient? (The Foggy Hill) 🟢

Picture hiking in thick fog, trying to reach the valley (lowest loss):

```
LOSS
 │\
 │ \        ← you are standing at *
 │  \   *
 │   \ / \
 │    \   \
 │     \_  \____
 │      ↑  valley (goal)
 └──────────────→ WEIGHT
```

You can't see the valley, but you CAN feel the slope under your feet:

- Slope tilts **up** to your right (gradient **positive**) → walk **left** (decrease weight)
- Slope tilts **down** to your right (gradient **negative**) → walk **right** (increase weight)
- Slope is **flat** (gradient **zero**) → you're at the bottom. Done!

That's exactly why the update rule is a **subtraction**:

```
weight = weight − learning_rate × gradient
```

Subtracting the slope always walks you downhill. A gradient is nothing mystical — it's just
**"if I nudge this number a tiny bit, how much does the loss change, and in which direction?"**

🟡 **Full worked example, every number shown** (`y = x²` at x=3):

```
x = 3.000 → y = 9.000000
x = 3.001 → y = 9.006001     (nudge x by +0.001)

change in y = 0.006001
change in x = 0.001
gradient ≈ 0.006001 / 0.001 = 6.001 ≈ 6.0

Formula check: d/dx(x²) = 2x = 2×3 = 6 ✓
Meaning: "if x grows a little, y grows 6× as fast" — and since y=loss here,
         we should DECREASE x to decrease the loss.
```

---

## 6. The Big Picture (The Loop Everything Lives In) 🟢

```
   ① FORWARD        ② LOSS           ③ BACKWARD          ④ UPDATE
   run the network  measure error    assign blame        fix the weights
   ────────────→   ────────────→   ─────────────────→  ────────────────→
   input → guess    (guess−truth)²   d(loss)/d(weight)    w −= lr × grad
                                                          │
                                              repeat until loss ≈ 0
```

🟢 Soup version: **Cook → Taste → Figure out which ingredients caused the blandness → Adjust → Repeat.**

One sentence: **Predict → Measure error → Figure out whose fault it is → Fix them slightly → Repeat.**
Everything else in this guide is just the details of these four boxes.

---

## 7–8. The Value Class — The Entire Engine (`engine.py`) 🟡

**The trick:** replace every plain number with a smart object that remembers where it came from.

### 8.1 What's inside a Value

When you write `a = Value(2.0)`, here's what lives inside that object:

| Field | Plain English | Initial value |
| ------- | --------------- | --------------- |
| `.data` | the actual number | 2.0 |
| `.grad` | "how much should I change?" — the blame score | 0.0 (unknown yet) |
| `._backward` | a stored recipe: "when someone asks, here's how I pass blame to my parents" | do nothing |
| `._children` | which Values created me? | `{}` (I'm a leaf — no parents) |
| `._op` | which operation made me? | `''` (none — I was born directly) |

### 8.2 What happens when you compute

When you write `c = a * b` (with a=2, b=3):

```
c.data      = 6.0            ← 2.0 × 3.0
c._children = {a, b}         ← "I was made from a and b"
c._op       = '*'            ← "via multiplication"
c._backward = <function>     ← "when blame arrives at me,
                               I'll give a × b.data and b × a.data back"
```

🟢 **Delivery-slip analogy:** every operation fills out a delivery slip attached to its result:
*"I was born from these two packages (children), via this process (op), and if anyone sends blame
back to me, forward it like THIS (backward rule)."* At the end of the day, `loss.backward()` just
reads every slip and forwards every package. No node needs to understand global math — each one
only knows its own tiny local rule.

### 8.3 Why store a FUNCTION (a "closure") on the object? 🟡

Because each operation has a DIFFERENT blame-passing rule. `+` passes blame through unchanged; `*`
swaps-and-multiplies; tanh uses `1−out²`. Storing the rule *on each node* means `.backward()`
later doesn't need to know ANY math — it just asks every node: "pass the blame back, your way."
This split (each op knows only its local rule + a generic walker) is the core design of every
autograd system ever written, including PyTorch.

---

## 9–10. Every Operation & Its Blame Rule 🟡

Each operation needs 2 things: how to compute forward, and how to send gradient backward.
All the backward rules are just the operation's **derivative** multiplied by the incoming blame.

| Operation | Forward | Backward rule (plain English) | Concrete example (incoming grad = 1) |
| ----------- | --------- | ------------------------------- | -------------------------------------- |
| **add** `a+b` | sum | blame **passes through unchanged** to both | a=2, b=3 → a.grad=1, b.grad=1 |
| **mul** `a×b` | product | **swap values**: a gets grad×b, b gets grad×a | a=2, b=3 → a.grad=3, b.grad=2 |
| **pow** `aⁿ` | power | power rule: grad × n × a^(n−1) | a=3, n=2 → a.grad=6 |
| **tanh** `tanh(a)` | squash to (−1,1) | grad × (1 − output²) | a=0.5 → a.grad=0.787 |
| **relu** `max(0,a)` | cut negatives | if a>0: pass through; if a≤0: **block** | a=−1 → a.grad=0 (dead) |
| **exp** `eᵃ` | e to the power | grad × output (e^x is its own derivative!) | a=1 → a.grad=2.718 |
| **log** `ln(a)` | natural log | grad × 1/a | a=2 → a.grad=0.5 |
| **sub** `a−b` | via `a + (−b)` | +1 to a, −1 to b | a=5,b=3 → a.grad=1, b.grad=−1 |
| **div** `a/b` | via `a × b⁻¹` | 1/b to a, −a/b² to b | a=6,b=2 → a.grad=0.5, b.grad=−1.5 |

**The one pattern to remember:** *backward rule = local derivative × incoming gradient.*

### 10.1 Intuition for each rule 🟢

- **Addition is a copy machine.** `z = a + b` — z is just a and b glued together, so whatever
  blame z gets, a and b each get a full copy. (Two kids split nothing — each keeps the whole
  responsibility, because each contributed "the whole of itself.")
- **Multiplication splits blame by the OTHER side's size.** `z = a × b`. If b is enormous (1000),
  a tiny change in a changes z enormously → a deserves big blame (×1000). If b is tiny (0.001),
  changing a barely matters → small blame. **Your blame = incoming × the other guy's value.**
- **Division is multiplication in disguise** (pizza version): `z = a/b` is "a pizzas cut among b
  people." More people (bigger b) → smaller slices → so blame toward b is NEGATIVE
  (growing b shrinks z): b gets `−a/b²`.
- **tanh blame shrinks near the edges.** tanh squashes into (−1, 1), so near ±1 it's *flat* —
  wiggling the input does almost nothing there → blame `1 − out²` → nearly 0 at the edges.
- **ReLU is a bouncer.** Positive inputs walk right through untouched. Negative inputs are
  thrown out and get NO message back (blame = 0).

### 10.2 Why `+=` and not `=` 🔴

Because a Value can be used in MANY places (like `a` appearing in both `a*b` and `a+b`). Blame
arrives from every path it's on, and we must **add up** all the blame. Using `=` would keep only
the last path and silently throw away the rest. Forgetting `+=` is the #1 beginner bug in
autograd engines. (It's also why PyTorch gradients accumulate between steps — and why
`zero_grad()` must exist. Sections 11 and 21 revisit this.)

---

## 11–12. Backpropagation — The Full Algorithm (`backward()` in engine.py) 🟡

Three steps, exactly as coded in `engine.py` lines 165–189:

**Step 1 — Topological sort (line the dominos up in order).**
A depth-first walk that lists every node *after* all its parents. Result: leaves first, output last.
Why needed: we must process blame top-down — a node can only pass blame it has already received.

🟢 **Getting-dressed analogy:** you cannot put on shoes before socks. A topological sort is the
universe's way of listing "socks, then shoes." Backprop runs the list in REVERSE: *undress from
the shoes down* — blame starts at the output (shoes) and works down to the inputs (socks).

**Step 2 — Seed the output.**
`self.grad = 1.0`. Loss's blame for itself is 1 by definition ("loss changing with respect to loss
changes 1-to-1"). This is the first domino.

**Step 3 — Walk in reverse, calling each node's `_backward()`.**
Every node pushes its blame to its parents, using its own rule. By the end, every leaf
(every weight) has its gradient. One pass. All gradients. Done.

### 12.1 Full hand-trace: `e = (a×b) + (a+b)` with a=2, b=3 🟡

```
GRAPH:   a=2 ──┬── [×] ──→ c=6 ──┐
               └── [+] ──→ d=5 ──┴── [+] ──→ e=11

FORWARD:  c = 2×3 = 6      d = 2+3 = 5      e = 6+5 = 11

SEED:     e.grad = 1.0

BACKWARD (reverse order):
  e's rule (+):  c.grad += 1.0        d.grad += 1.0
  d's rule (+):  a.grad += 1.0        b.grad += 1.0
  c's rule (×):  a.grad += 1×3=3.0    b.grad += 1×2=2.0

TOTALS:   a.grad = 1 + 3 = 4.0        b.grad = 1 + 2 = 3.0

CHECK:    e = ab + a + b  →  de/da = b+1 = 4 ✓   de/db = a+1 = 3 ✓
```

Notice `a.grad` got contributions from **two paths** (through c AND through d). That's exactly
why we accumulate with `+=`.

🔴 **Why the order matters (the bug you'd hit otherwise):** if you processed `a` before `c`, then
`a`'s blame would be computed from `c.grad` while `c.grad` is still 0 — a silently wrong answer.
Reverse topological order guarantees every node's blame is FULLY collected before it forwards it.
PyTorch has the same invariant internally.

---

## 13. The Chain Rule (How Blame Travels) 🟡

If final = f(g(x)), the chain rule says:

```
d(final)/dx  =  d(final)/d(g)  ×  d(g)/dx
                └── incoming ──┘  └─ local ─┘
                     blame            derivative
```

🟢 Recall the currency analogy: total sensitivity = product of the per-step rates.
🟢 Or gears: if gear A spins 3× gear B, and gear B spins 4× gear C, then A spins 12× gear C.
Rates multiply along the path.

A 3-node chain, x=5:

```
x ──[×2]──→ a=10 ──[×3]──→ b=30 ──[×4]──→ c=120

c.grad = 1.0            (seed)
b.grad = 1.0 × 4 = 4.0  (local deriv of the ×4 node is 4)
a.grad = 4.0 × 3 = 12.0
x.grad = 12.0 × 2 = 24.0

Sanity check: c = 24x, so dc/dx = 24 ✓
```

**At every node: multiply incoming blame by the local derivative. Repeat across the graph.**
That's the chain rule. Backprop = the chain rule + bookkeeping.

---

## 14. Activation Functions (Why tanh & ReLU Exist) 🟡

**The problem they solve:** without a non-linear function sandwiched between layers, stacking
layers is pointless:

```
Layer1: y = w1x + b1
Layer2: z = w2(w1x + b1) + b2 = (w2w1)x + (w2b1+b2)   ← still a straight line!
```

100 layers of straight lines = one straight line. Useless for curved patterns.
An activation function adds the "bend."

🟢 **Origami analogy:** folding paper (activations) lets straight creases build curved shapes.
Unfold everything (compose all the linear layers) and you just have... a flat sheet.

### 14.1 tanh — the smooth squasher 🟢

- Maps any number into **(−1, 1)**. tanh(0)=0, big positives → +1, big negatives → −1.
- Gradient = `1 − output²`. Great in the middle (≈1), nearly zero when saturated at ±1.
- Used in your MLP's hidden layers (small networks, smooth learning).

```
tanh(−3)=−0.995   tanh(−1)=−0.762   tanh(0)=0   tanh(1)=0.762   tanh(3)=0.995

gradient at output 0     → 1 − 0²     = 1.000   (wide open, learning fast)
gradient at output 0.762 → 1 − 0.762² = 0.419   (decent)
gradient at output 0.995 → 1 − 0.995² = 0.010   (saturated, learning stalls)
```

### 14.2 ReLU — the simple gate 🟢

- `max(0, x)`: negatives become 0, positives pass through. One comparison — blazing fast.
- Gradient is exactly 1 (positive side) or 0 (negative side).
- **The "dying ReLU" trap:** if a neuron's inputs are *always* negative, its gradient is *always*
  0 → it never updates → permanently dead. (Fix: Leaky ReLU `max(0.01x, x)` — a tiny drip of
  blame keeps the neuron technically alive.)

🔴 **Advanced notes:**

- Why is ReLU so popular in deep nets? (a) The forward pass is one comparison — no exp/tanh
  computation; (b) gradient is exactly 1 on the positive side, so blame doesn't shrink as it
  crosses many layers (tanh's `1−out² < 1` always shrinks it — multiply 0.4 ten times and you
  get ~0.0001: the "vanishing gradient" problem); (c) sparse outputs (many zeros) are cheap.
- Modern LLMs mostly use GELU/SwiGLU — smooth cousins of ReLU. Same idea, nicer curvature.
- Softmax (coming in the Transformer project) converts raw scores into probabilities that sum
  to 1 — you'll meet it when the network must "choose a word."

### 14.3 Which one when? 🟢

```
tanh     → small networks' hidden layers (yours)
ReLU     → deep networks (fast, but watch for dead neurons)
linear   → output layer for regression / raw scores (yours uses this too)
```

🔴 **Why is YOUR output layer linear (see `nn.py` line 116)?** tanh would cap the output at
(−1, 1). But hinge loss *wants* scores beyond ±1 to express confidence (pred=+1.5 = "very sure"),
and regression wants any number at all. So: hidden layers get the bend, output layer stays raw.

---

## 15–16. What Is a Neuron? (Biology → Math → Code) 🟢

**Biology:** a brain cell gets many input signals through dendrites; if the total is strong
enough, it fires. Strengths of connections *vary* — learning literally = connections strengthening.

**Artificial version, exactly:**

```
output = activation( w1×x1 + w2×x2 + ... + wn×xn + b )
                      └──────── weighted sum ───────┘└─bias─┘
```

### 16.1 Worked example (matches `step2_one_neuron.py`) 🟢

```
inputs:  x1=2.0, x2=0.0
weights: w1=−3.0, w2=1.0      ← LEARNABLE (start random)
bias:    b=6.88               ← LEARNABLE

dot    = (2×−3) + (0×1) + 6.88 = 0.88
output = tanh(0.88) = 0.707
```

Run the file and you'll see exactly these numbers, plus every gradient.

### 16.2 The umbrella analogy (a neuron deciding "take an umbrella?") 🟢

```
input:  is it cloudy?        x1=0.8   weight w1=+2.0   ← strongly predicts rain
input:  is it summer?        x2=0.7   weight w2=−0.5   ← summer rain is less likely
input:  was yesterday rainy? x3=0.9   weight w3=+1.0
bias:                                 b=−0.3           ← baseline reluctance

sum = 0.8×2.0 − 0.7×0.5 + 0.9×1.0 − 0.3 = 1.85
tanh(1.85) = 0.952  ≈  "BRING THE UMBRELLA"   (verified: tanh(1.85)=0.9518)
```

The **weights** are the neuron's learned opinions: "how much does this input matter?"
Training is how those opinions got formed. A neuron never changes its inputs or its math —
only its opinions (weights).

🟡 **What is the bias, intuitively?** It's the neuron's default mood with zero evidence.
With all inputs = 0, output = activation(b). High bias = cheerful by default; negative bias =
suspicious by default. Training tunes the mood. That's why every neuron gets one.

---

## 17–18. Layers & MLP (Wiring Neurons Together) 🟡

**Layer** = several neurons all reading the SAME input, each with its own weights,
each free to learn a different pattern.

🟢 **Panel of judges analogy:** one judge might have a narrow view; a panel of 4 judges each
watching the same performance, each with their own criteria, then a final judge (output layer)
summarizes the panel. That's `Layer(3→4)` followed by `Layer(4→1)`.

**MLP (Multi-Layer Perceptron)** = layers stacked: each layer's output becomes the next
layer's input. Built in `nn.py`:

```
model = MLP(3, [4, 4, 1])     ← from step3_full_network.py

INPUT(3) ──→ HIDDEN(4, tanh) ──→ HIDDEN(4, tanh) ──→ OUTPUT(1, linear)

PARAMETER COUNT (why 41?):
  Hidden layer 1: 4 neurons × (3 weights + 1 bias) = 16
  Hidden layer 2: 4 neurons × (4 weights + 1 bias) = 20
  Output layer:   1 neuron  × (4 weights + 1 bias) =  5
                                                   ─────
  total                                             41  ← every one gets a gradient
```

**Why stack layers?** Hierarchy:

```
Layer 1 learns: raw features      (edges, word patterns)
Layer 2 learns: combinations      (shapes, phrases)
Layer 3+ learns: high concepts    (faces, intent)
```

🟢 **Kitchen analogy:** layer 1 = prep cooks (chop, dice — raw features), layer 2 = line cooks
(combine into sauces), layer 3 = the chef (recognizes the dish, makes the call). Each level works
on what the previous level produced.

🔴 **The symmetry problem (why random init matters):** if all neurons started with identical
weights, they'd all compute the same output and receive identical gradients — forever identical.
A "panel of judges" where all judges are clones is as useful as one judge. Random starting weights
break the tie so neurons specialize. (`nn.py` line 35: `random.uniform(-1, 1)`.)

---

## 19–20. Loss Functions (Measuring Wrongness) 🟢

Loss = one number: **"how wrong, right now?"** Training's whole goal is pushing it to 0.

### 20.1 MSE — Mean Squared Error (for predicting numbers) 🟢

```
loss = (prediction − target)²

pred=0.3, target=1.0 → 0.49   (bad)
pred=0.9, target=1.0 → 0.01   (good)
pred=1.0, target=1.0 → 0.00   (perfect)
```

🟢 Archery: squared distance from the bullseye. Off by 7 cm hurts 49× more than off by 1 cm.

### 20.2 Hinge — for predicting a CLASS (+1 or −1) 🟢

```
loss = max(0, 1 − y_true × y_pred)
```

🟢 **Bouncer analogy:** hinge loss is a bouncer with a strict guest list. You're only "safe"
(0 loss) if you're on the right side **AND** confident (margin > 1). Loitering near the door
(small correct prediction) still gets some loss. Wrong side? Big loss.

```
true=+1, pred=+1.5 → 1−1.5 = −0.5 → max(0,−0.5) = 0     ✓ confident & correct
true=+1, pred=+0.3 → 1−0.3 =  0.7 → 0.7                  ✓ right but shy
true=+1, pred=−0.8 → 1+0.8 =  1.8 → 1.8                  ✗ confidently wrong
```

Rule of thumb: **regression → MSE · classification → hinge (or cross-entropy later).**

🔴 **Why not MSE for classification?** With classes ±1 and tanh outputs capped at (−1, 1), a
correct prediction can never satisfy MSE perfectly (|pred| < 1 < needed distance)... and MSE
keeps pushing even when the class answer is already right. Hinge says "correct and confident
enough? we're DONE" — exactly the behavior classification needs.

---

## 21–22. The Training Loop — THE 4 Steps (`step2_one_neuron.py` lines 109–128) 🟢

Every network, every framework, forever — these 4 steps on repeat:

```python
for step in range(20):
    # 1) FORWARD: guess + measure
    o    = (w1*x[0] + w2*x[1] + b).tanh()   # builds the graph as a side effect
    loss = (o - target) ** 2

    # 2) ZERO: wipe old gradients
    w1.grad = w2.grad = b.grad = 0.0        # CRITICAL, see below

    # 3) BACKWARD: assign blame
    loss.backward()                          # fills every .grad automatically

    # 4) UPDATE: each knob nudges downhill
    w1.data -= 0.1 * w1.grad
    w2.data -= 0.1 * w2.grad
    b.data  -= 0.1 * b.grad
```

**Why each step exists:**

- **① Forward** — runs the math AND secretly builds the computation graph (every `+`,`*`,`tanh`
  recorded its children + backward rule). Step ③ will walk exactly that graph.
- **② Zero** — gradients **accumulate** (`+=`). If you skip this, step 2's gradient =
  step1's + step2's. After 10 steps you're updating on a meaningless pile of history.
  This is the most common real-world bug (in PyTorch too: `optimizer.zero_grad()`).
  🟢 **Whiteboard analogy:** gradients are notes written on a shared whiteboard. Before each new
  lesson you ERASE the board. Forget to erase, and today's notes are written over yesterday's —
  after a week you're taking advice from a smear.
- **③ Backward** — one call: topo-sort, seed 1.0, reverse walk. Every weight gets blame.
- **④ Update** — `data −= lr × grad`. Positive grad → decrease w. Negative grad → increase w.
  Either way, downhill.

🔴 **Why update `.data` directly, not through operations?** Because the update is
*bookkeeping, not part of the network*. If you wrote `w = w − lr*w.grad` as Value operations,
you'd pollute the graph and grow memory forever. We only mutate `.data` — the graph is rebuilt
fresh next forward pass. (PyTorch wraps this in `torch.no_grad()` for the same reason.)

---

## 23–24. Gradient Descent Visualized (The Walk Downhill) 🟢

Minimize `Loss = w²` (bowl shape, minimum at w=0). Gradient = 2w.

```
Loss
9│   *                   *
4│     *               *
1│       *           *
0│          *__*__*          ← minimum (flat, grad=0)
 └─────────────────── w
   −3 −2 −1  0  1  2  3

Standing at w=−2: grad = −4 (downhill → right)
   w = −2 − 0.1×(−4) = −1.6   ✓ moved right, closer to 0
Standing at w=+2: grad = +4 (downhill → left)
   w = +2 − 0.1×(+4) = +1.6   ✓ moved left
Standing at w=0:  grad = 0
   w stays. Converged.        ✓
```

### 24.1 The learning rate (lr) is the step size — Goldilocks matters 🟢

```
lr = 1.0    → 3 → −3 → 3 → −3 ...   BOUNCES FOREVER (too big — overshoots the valley)
lr = 0.001  → 2.994 → 2.988 → ...   CRAWLS (too small — would need 10,000 steps)
lr = 0.1    → 2.4 → 1.92 → 1.54 …   GLIDES DOWN (just right)
```

🟢 This is the shower knob from Analogy 1. Crank too hard → scalding → freezing → scalding.
Nudge too gently → cold shower forever.

🟡 **The math of the bounce (why lr=1.0 never converges on w²):** starting at w=3,
`w_new = w − 1.0 × 2w = −w`. So 3 → −3 → 3 → −3 … perfectly sideways forever. At lr=1.1:
`w_new = w − 2.2w = −1.2w` → 3 → −3.6 → 4.32 → … the steps GROW — divergence. The rule of
thumb: for loss = w², lr < 1.0 converges, lr ≥ 1.0 never does.

🔴 **Local vs global minima:** real loss landscapes are not one clean bowl — they're mountain
ranges with many valleys. Gradient descent finds *a* valley, not necessarily the deepest one.
Amazingly, in high-dimensional networks this mostly works fine (most valleys are nearly as good).
Tricks like momentum, learning-rate decay, and restarts help explore better.

---

## 25. Concrete Examples (7 Worked Problems, Basic → Advanced)

### Example 1 🟢 — Gradient of a polynomial

`y = 3x² + 2x + 1` at x=4. By the power rule: `dy/dx = 6x + 2 = 26`.

```python
x = Value(4.0)
y = 3*x**2 + 2*x + 1
y.backward()
print(x.grad)   # 26.0 ✓
```

Numerical check: y(4.001) − y(4.000) = 57.026003 − 57.0 = 0.026 → /0.001 = 26.0 ✓

### Example 2 🟡 — Two weights, one full training step

`output = w1×2 + w2×3`, target = 10, start w1=1, w2=1 → output=5, loss=(5−10)²=25.

```
d(loss)/d(output) = 2×(output−target) = 2×(5−10) = −10     ← local rule of the square
d(output)/d(w1)   = 2    →  w1.grad = −10 × 2 = −20        ← ×-rule: blame × other input
d(output)/d(w2)   = 3    →  w2.grad = −10 × 3 = −30

update (lr=0.05):  w1 = 1 − 0.05×(−20) = 2.0
                   w2 = 1 − 0.05×(−30) = 2.5

new output = 2×2 + 2.5×3 = 11.5    (overshot past 10 — but...)
new loss   = (11.5 − 10)² = 2.25   ← down from 25! One step, 11× better.
```

Watch the blame flow: −10 came from the squaring node; each weight multiplied it by its own
input (2 and 3) — the swap-rule of multiplication in action.

### Example 3 🟢 — Why zero_grad() matters (with soup)

```python
w = Value(1.0)
for step in range(3):
    loss = w * 2          # true gradient is ALWAYS 2
    loss.backward()
    print(w.grad)
# Step 0: 2.0   ← correct
# Step 1: 4.0   ← WRONG (accumulated 2+2)
# Step 2: 6.0   ← WRONG (2+2+2)
```

🟢 Soup version: you tasted once ("needs salt"), added salt, and next taste you're still reading
the OLD note "needs salt" on top of the new one — so you add double salt forever. Erase the note
(`w.grad = 0.0`) before each backward.

### Example 4 🟡 — Learning rate effect on `(w×2 − 10)²` starting at w=0

Gradient = `8w − 40`. The update is exact, so we can compute every step precisely:
lr=0.05 gives the recursion `w ← 0.6w + 2`; lr=0.001 gives `w ← 0.992w + 0.04`.

```
lr = 0.5 (too big):   w: 0 → 20 → −40 → 140 → …      OVERSHOOTS and DIVERGES (|w| grows)
lr = 0.05 (good):     step 10:  w=4.970,  loss=0.004
                      step 25:  w=4.99999, loss≈0    ← fully converged ✓
lr = 0.001 (tiny):    step 100: w=2.761,  loss=20.1  ← not even halfway after 100 steps
                      needs ≈ 570 steps just to reach loss 0.01 (25 steps were
                      enough at lr=0.05 — a 23× penalty for a 50× smaller lr)
```

### Example 5 🟡 — Division gradient, fully by hand

`z = a/b` with a=6, b=2 → z=3. Suppose z.grad = 1.

```
a.grad += z.grad × (1/b)     = 1 × 0.5  =  0.5    (more numerator → bigger result)
b.grad += z.grad × (−a/b²)   = 1 × −1.5 = −1.5    (more denominator → smaller result)

Numerical check for b: z(b=2.001) = 6/2.001 = 2.99850
  Δz = −0.00150, Δb = 0.001 → slope = −1.5 ✓
```

### Example 6 🟡 — tanh vs ReLU: the dying neuron

```
pre-activation = −2.0

tanh:  output = tanh(−2) = −0.964
       gradient = 1 − (−0.964)² = 0.071   → small but ALIVE → still learns

ReLU:  output = max(0, −2) = 0
       gradient = 0                        → DEAD → never learns
```

If a ReLU neuron's inputs are negative for ALL training samples, it can never recover.
tanh always keeps a sliver of gradient. That's the "dying ReLU" problem from section 14.

### Example 7 🔴 — Chain rule through a square (the full chain, manually)

`y = (3x + 2)²` at x=1. Two operations: inner `u = 3x+2 = 5`, outer `y = u² = 25`.

```
chain rule:  dy/dx = (dy/du) × (du/dx) = 2u × 3 = 2×5×3 = 30

micrograd path:
  x ──[×3]──→ 3x ──[+2]──→ u=5 ──[**2]──→ y=25
  y.grad=1 → u.grad = 1 × 2u = 10 → (3x).grad = 10 × 1 → x.grad = 10 × 3 = 30 ✓

numerical check: y(1.001) = (5.003)² = 25.030009 → Δy/Δx = 0.030009/0.001 ≈ 30 ✓
```

---

## 26. Training Results — How to Read Them 🟢

**Gradient correctness** — `e = a*b + (a+b)`, a=2, b=3 →
`a.grad=4.0`, `b.grad=3.0`, matching calculus exactly ✓

**One neuron** (`step2_one_neuron.py`, train output→1.0 for input [2,3]):

```
Step 0:  output=0.0997  loss=0.8106    ← random guess
Step 1:  output=0.9889  loss=0.0001    ← almost all learning in ONE step (lr=0.1)
Step 19: output=0.9892  loss=0.0001    ← converged (slowly polishing now)
```

🟢 Notice it prints **"Still learning..."** — the final output is 0.9892, and the success check
demands being within 0.01 of the target 1.0. It's off by 0.0108. Totally normal: convergence
*asymptotes* — the last 1% takes disproportionately long (the tanh gradient is small near 1.0,
see section 14.1). More steps, or a slightly higher lr, would close the gap.

**Full MLP, 41 parameters** (`step3_full_network.py`, **5 samples**, binary classes).
Loss = hinge, **averaged** over samples, plus a tiny L2 penalty `0.0001×Σw²`.
These are the REAL numbers from running the current file:

```
Step 0:  loss=1.0114  accuracy=40%   ← 2 of 5 correct (worse than a coin flip!)
Step 10: loss=0.7867  accuracy=60%   ← learning
Step 20: loss=0.5914  accuracy=80%   ← getting there
Step 30: loss=0.4027  accuracy=100%  ← all 5 correct
Step 99: loss=0.0016  accuracy=100%  ← confidently correct

Final predictions (all 5 correct ✓):
  [2.0,  3.0, -1.0] → +1.22  (target +1)      [3.0, -1.0,  0.5] → -1.89  (target -1)
  [0.5,  1.0,  1.0] → +2.04  (target +1)      [1.0,  1.0, -1.0] → -1.02  (target -1)
  [0.0,  0.0,  0.0] → +1.05  (target +1)  ← all-zero sample: every weight's
                                            contribution is 0, so only the BIAS
                                            could learn this one — and it did!
```

🟢 What you're watching: loss falls *smoothly* (the optimizer is healthy), while accuracy jumps
in 20% steps (5 samples → each flipped prediction moves it by exactly one fifth) as predictions
cross the ±1 decision boundary. Loss is the smooth compass; accuracy is the choppy scoreboard.
Train on loss, report accuracy.

Loss fell smoothly, accuracy climbed to 100% — **using ~200 lines of pure Python, zero ML
libraries.** That's the same algorithm GPT-4's training uses, at microscopic scale.

---

## 27. Homework — How to Actually Learn This 🟢

Doing these is the difference between "I read about it" and "I own it":

1. **Rewrite `__mul__` from memory.** Delete it, write it back. If you can, you understand
   closures + swap-the-values rule. Mastery checkpoint.
2. **Learning rate lab** (step2): try lr=1.0 (predict: divergence!), lr=0.001 (count steps to
   converge), lr=0.1. Watch the loss column. This builds intuition no book can give.
3. **Add sample `[0,0,0] → +1`** (step3). Does it still hit 100%? More samples = harder problem.
   ✅ **This one is already done** — the current `step3_full_network.py` (line 32) contains
   exactly this 5th sample, and the network still reaches 100% (see section 26).
   🔴 With input [0,0,0], every weight's contribution is 0 — only the BIAS can move that
   prediction. Interesting: it tests whether your bias can do its job alone.
4. **Manual trace** of `c=a*b; d=c**2; e=d+a` with a=3, b=2 — on paper first, then verify by
   running the ready-made `homework4.py`. Its actual, verified output is:
   `a.grad=25.0, b.grad=36.0, c.grad=12.0, d.grad=1.0` —

   ```
   e = d + a:            d.grad += 1,   a.grad += 1        (d.grad=1)
   d = c**2:             c.grad += 1 × 2×c = 2×6 = 12      (c.grad=12)
   c = a×b:              a.grad += 12×2 = 24, b.grad += 12×3 = 36
   totals:  a.grad = 1 + 24 = 25,  b.grad = 36,  c.grad = 12,  d.grad = 1
   analytic check: e = (ab)² + a = a²b² + a → de/da = 2ab² + 1 = 24+1 = 25 ✓
                                                    de/db = 2a²b = 36 ✓
   ```

🟢 **Study tips that actually work:**

- Trace examples on paper FIRST, code SECOND. Paper forces the chain rule into your head.
- After each run, change ONE thing and predict the outcome before running. Wrong predictions
  are the fastest way to learn.
- Explain backprop out loud to a rubber duck / friend / pet. If you stall, that's the gap.

---

## 28. Micrograd → PyTorch (You Already Know PyTorch) 🟡

```
micrograd                              PyTorch
─────────────────────────────────────────────────────────────
Value(3.0)                       →  torch.tensor(3.0, requires_grad=True)
a + b, a * b, a ** 2             →  identical operators
a.tanh(), a.relu()               →  torch.tanh(a), torch.relu(a)
loss.backward()                  →  loss.backward()          (same name!)
p.grad                           →  p.grad                   (same name!)
p.data -= lr * p.grad            →  with torch.no_grad(): p -= lr * p.grad
model.zero_grad()                →  optimizer.zero_grad()
one number at a time             →  whole tensors / batches at once
~200 lines Python                →  optimized C++/CUDA, billions of numbers
```

**Same algorithm. Same math. Same names.** The ONLY differences are scale (tensors vs scalars)
and speed (GPU kernels vs Python). When you read PyTorch code, mentally translate it to
micrograd and it will make sense.

🔴 **Extra translations you'll meet soon:** `nn.Linear(3, 4)` = your `Layer(3, 4)` with linear
activation; `nn.Tanh()` = your `.tanh()`; `MSELoss()` = your `(pred − target)**2` averaged over
samples; `optim.SGD(model.parameters(), lr=0.1)` = your step-④ loop; `loss.item()` = `loss.data`.

🔴 **Batches & regularization (both live in your step3!):** the loss there AVERAGES the hinge
loss over all 5 samples before one backward — that's "full-batch" gradient descent. It also adds
`0.0001 × Σw²` (**L2 regularization**): a small extra charge for large weights that keeps the
network honest and stops any single weight from exploding. Real training uses mini-batches
(e.g., 32 samples): same 4 steps, just fewer samples per step — a trade of gradient *quality*
for *speed*.

---

## 29. Pocket Glossary (25 Terms That Matter Most) 🟢

| Term | One-liner |
| ------ | ----------- |
| **Weight / Bias** | Adjustable knob. Bias = a knob not attached to any input (default mood). |
| **Parameter** | Any knob: `len(model.parameters())` = how many knobs you have (41 here). |
| **Gradient** | Per-knob blame score: "which way + how hard to turn me." |
| **Loss** | One number for total wrongness. Goal: → 0. |
| **Learning rate** | Step size. Big → bounce. Small → crawl. |
| **Forward pass** | Run inputs through → get prediction (and secretly build the graph). |
| **Backward pass** | Walk graph in reverse → fill every knob's `.grad`. |
| **Backpropagation** | That reverse walk. Chain rule + topological order + `+=`. |
| **Chain rule** | Incoming blame × local derivative, at every node. |
| **Topological sort** | Line nodes up so parents come before children (socks before shoes). |
| **Activation (tanh/ReLU)** | The non-linear "bend" that makes layers meaningful. |
| **zero_grad** | Wipe yesterday's blame before today's (or blame piles up). |
| **Dying ReLU** | Neuron stuck at negative inputs → zero gradient forever → never learns. |
| **Vanishing gradient** | Blame shrinking to ~0 across many layers (why tanh struggles in deep nets). |
| **Epoch vs Step** | Step = one forward→zero→backward→update round. Epoch = one full pass over ALL data. |
| **Batch / mini-batch** | How many samples per step. 5 here; 32–1024 in real life. |
| **Overfitting** | Memorizing the practice exam (training data) but failing the real one (new data). |
| **L2 regularization** | Extra loss term `λ×Σw²` — rewards small weights → simpler, more honest model. |
| **Symmetry problem** | Identical starting weights → identical neurons forever. Random init breaks it. |
| **Local vs global minimum** | A valley vs THE deepest valley. Descent finds *a* valley; usually fine. |
| **Leaf node** | Value created directly (input or weight), not by an operation. |
| **Computational graph** | The web of Values + ops built during the forward pass. |
| **Numerical gradient** | The nudge-and-measure approximation; used to check autograd for bugs. |
| **Tensor** | PyTorch's Value, but a whole grid of numbers at once. |
| **Autograd** | The umbrella name for all of the above: automatic gradient computation. |

---

## 30. What's Next 🟢

1. **This week:** do all 4 homework exercises. Rebuild `Value` from a blank file.
2. **Weeks 3–4:** Karpathy's "GPT from scratch" video → build a tiny Transformer.
3. **Then:** nanoGPT training, then the ReAct agent loop.
4. Your files: run `python step1_just_values.py` → `step2_one_neuron.py` → `step3_full_network.py`
   in order. Each prints its own explanation as it runs.

> **The one idea to keep forever:** *Training = repeatedly nudging every knob a tiny bit
> downhill, where "downhill" comes from the gradient, and the gradient comes from the chain
> rule applied backward through the graph.* Everything above is that sentence, expanded.

---

## 31. CAPSTONE — Retrain It Yourself (The Complete Recipe) 🟢🟡

Everything above compresses into ONE skill: **starting from a blank file and retraining a
network from nothing.** Here is the exact recipe, with a checkpoint after every stage.
If you can do all 6 stages without looking anything up, you have genuinely mastered this.

### Stage 1 — Rebuild the engine (blank file → `Value`)

Write the class in THIS order (each step only uses the ones before it):

```
[1] __init__   → store .data, .grad=0.0, ._backward = do-nothing, ._children, ._op
[2] __add__    → forward: sum.     backward: BOTH parents += out.grad
[3] __mul__    → forward: product. backward: self.grad   += out.grad * other.data
                                     other.grad += out.grad * self.data   (SWAP!)
[4] __pow__    → forward: a**n.    backward: out.grad * n * a**(n-1)
[5] tanh       → forward: tanh(a). backward: out.grad * (1 - out**2)
[6] mirrors    → __neg__, __sub__, __truediv__, __radd__, __rmul__ (one-liners each)
[7] backward() → topo sort (DFS, append each node after its children)
                 → seed self.grad = 1.0
                 → for node in reversed(topo_order): node._backward()
```

✅ **Checkpoint 1:** `x = Value(3.0); (x**2).backward()` → `x.grad == 6.0`.
✅ **Checkpoint 2:** `e = a*b + (a+b)` with a=2, b=3 → `a.grad=4.0, b.grad=3.0`.
✅ **Checkpoint 3:** run `homework4.py` (a=3, b=2) → `a=25, b=36, c=12, d=1`. (Verified!)

### Stage 2 — Rebuild the network (blank file → Neuron / Layer / MLP)

```
Neuron(n)  → w = [Value(random.uniform(-1,1)) for _ in range(n)]; b = Value(0.0)
             __call__(x):  return (sum(wi*xi) + b).tanh()
             parameters(): return w + [b]
Layer(..)  → a list of Neurons; each one reads the SAME input
MLP(..)    → a list of Layers; hidden layers tanh, OUTPUT layer linear
             zero_grad(): p.grad = 0.0 for every parameter
             parameters(): every p from every layer
```

✅ **Checkpoint:** `len(MLP(3, [4, 4, 1]).parameters())` → exactly **41**.

### Stage 3 — Data + loss

```
X      → your samples (each a list of numbers)
y_true → correct answer per sample (here: +1 or -1)
loss   = average over samples of (1 - y_true*y_pred).relu()      ← hinge
        (+ optional 0.0001 * Σ p**2                              ← L2, keeps weights tame)
```

✅ **Checkpoint:** an untrained model should give loss around ~1.0. If it's huge (100+),
something is broken; if it's exactly 0 before training, your targets leaked into the model.

### Stage 4 — THE loop (4 lines that never change, in any framework)

```python
for step in range(100):
    loss, acc = compute_loss(model, X, y_true)   # 1) forward
    model.zero_grad()                            # 2) zero
    loss.backward()                              # 3) backward
    for p in model.parameters():                 # 4) update
        p.data -= 0.05 * p.grad
```

✅ **Checkpoint:** loss falls almost every step; accuracy climbs toward 100%.

### Stage 5 — Tune it (symptom → knob)

| Symptom | First knob to turn | Why |
| --------- | -------------------- | ----- |
| Loss bounces / explodes | lower lr (0.05 → 0.01) | steps overshoot the valley (shower knob!) |
| Loss barely moves | raise lr or train more steps | steps too timid |
| Loss stuck above 0 | more neurons/layers/steps | model too weak for the pattern |
| Loss ≈ 0 but predictions weak | more data / stronger L2 | memorizing instead of learning |

### Stage 6 — Retrain on YOUR OWN data

Swap in your own `X` and `y_true` (same shapes) — e.g., "is this number above 0.5?" with labels
±1. Keep everything else identical and retrain. You will then have personally done the complete
cycle every ML engineer does: **data → model → loss → loop → tune → retrain.**

🔴 **The three silent killers when retraining from scratch** (in the order people hit them):

1. Forgot `zero_grad()` → loss plateaus, then behaves chaotically.
2. Used `=` instead of `+=` in a backward closure → wrong gradients whenever a Value is reused.
3. Walked the topo list FORWARD instead of `reversed()` → gradients computed from zeros.

---

## Appendix A — Derivatives Cheat Sheet (Learn These 8, That's Enough) 🟡

A derivative = **slope** = "how much does output change per tiny input change?"

| Function | Derivative | Memory hook |
| ---------- | ----------- | ------------- |
| `c` (constant) | 0 | Flat line — no slope |
| `x` | 1 | Slope of y=x is 1 everywhere |
| `c×f(x)` | `c×f'(x)` | Stretching vertically stretches the slope |
| `f + g` | `f' + g'` | Slopes just add |
| `xⁿ` | `n×x^(n−1)` | "Bring the power down, reduce it by one" |
| `eˣ` | `eˣ` | The only function that is its own slope |
| `ln(x)` | `1/x` | Log grows fast, then slower and slower |
| `tanh(x)` | `1 − tanh(x)²` | Derivative written in terms of the OUTPUT |
| `max(0,x)` (ReLU) | 1 if x>0, else 0 | A switch, not a curve |

**The chain rule (the big one):** `(f(g(x)))' = f'(g(x)) × g'(x)` — outer derivative × inner
derivative. In micrograd terms: *incoming blame × local derivative, at every node.*

---

## Appendix B — Amateur FAQ 🟢🟡

**Q: Do I need to be good at calculus?**
No. You need ONE idea (slope) and the 8 rules in Appendix A. Micrograd itself will teach you
the rest better than any calculus course, because you'll *see* every rule in code.

**Q: Why do weights start RANDOM instead of all zero?**
Zero (or identical) weights make every neuron compute the same thing and receive the same
gradient — they stay clones forever. Randomness breaks the tie (see section 18, symmetry problem).

**Q: What if the loss goes UP during training?**
Two usual suspects: (1) learning rate too big — it leapt over the valley (turn down the shower
knob); (2) you forgot `zero_grad()` and gradients piled up. Check ② before blaming ④.

**Q: Why not just solve for the perfect weights directly with algebra?**
For real networks the equations are nonlinear in millions of variables with no closed-form
solution. Iterative descent is the only practical universal method — the same reason you tune
soup by tasting rather than solving flavor chemistry equations.

**Q: Is 100% accuracy on 5 samples impressive?**
It proves the *mechanism* works. With so few samples the network can memorize them — real ML
needs many more samples and a held-out test set to prove genuine learning (see overfitting).

**Q: What's the difference between a step and an epoch?**
Step = one round of the 4-step loop (using whatever batch you fed it). Epoch = the loop run
enough times to have seen every training sample once. 5 samples, full-batch → 1 step = 1 epoch.

**Q: Is this really how ChatGPT works inside?**
The learning core, yes: forward → loss → backward → update, at massive scale. The differences
are the architecture (Transformer with attention instead of an MLP), the data (trillions of
tokens), the loss (next-word prediction instead of MSE), and the hardware (thousands of GPUs).
The gradient mechanics you built in `engine.py` are the same.

**Q: Why does `backward()` need a special node order (topological sort)?**
Because blame must arrive at a node BEFORE it can forward blame onward. Processing in the wrong
order would compute gradients from zeros — silently wrong answers (see section 12, advanced note).

**Q: Where do the backward RULES come from — do I have to memorize them?**
No memorizing. Each rule is just the operation's derivative (Appendix A) written in code, scaled
by the incoming gradient. If you can differentiate the forward operation, you can invent its
backward rule on the spot.
