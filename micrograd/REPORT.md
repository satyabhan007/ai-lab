# Micrograd & AI Engineering — Master Learning Report
### Author: Satya · Session: September 6, 2026 · Status: Active

---

## Table of Contents

| # | Section | What It Covers |
|---|---------|---------------|
| 1 | [What Is AI Engineering](#1-what-is-ai-engineering) | The field, the mindset, why build from scratch |
| 2 | [The Full Roadmap](#2-the-full-learning-roadmap) | All 20+ projects in dependency order |
| 3 | [The Starter Pack](#3-the-starter-pack) | The 3 projects that cover 80% of AI engineering |
| 4 | [The Core Problem](#4-the-core-problem--why-gradients-exist) | Why gradients exist, the knob analogy |
| 5 | [What Is a Gradient](#5-what-is-a-gradient) | Slope, hill, direction — with numbers |
| 6 | [Mindmap: Big Picture](#6-mindmap--the-big-picture) | Full neural network learning loop |
| 7 | [The Value Class](#7-the-value-class--the-entire-engine) | Line-by-line with diagrams |
| 8 | [Mindmap: Value Object](#8-mindmap--the-value-object) | What every field stores and why |
| 9 | [Operations & Backward Rules](#9-operations--their-backward-rules) | Every op with formula and example |
| 10 | [Mindmap: Gradient Flow](#10-mindmap--gradient-flow-through-operations) | Visual derivation of every backward rule |
| 11 | [Backpropagation](#11-backpropagation--the-full-algorithm) | Step-by-step with manual trace |
| 12 | [Mindmap: Backward Pass](#12-mindmap--the-backward-pass-traced) | Full graph traced by hand |
| 13 | [The Chain Rule](#13-the-chain-rule) | How gradients travel through layers |
| 14 | [Activation Functions](#14-activation-functions) | tanh vs ReLU — why they exist |
| 15 | [What Is a Neuron](#15-what-is-a-neuron) | Biology → math → code |
| 16 | [Mindmap: Neuron](#16-mindmap--one-neuron) | Visual diagram + real-world analogy |
| 17 | [Layers and MLP](#17-layers-and-mlp) | Architecture with parameter counts |
| 18 | [Mindmap: MLP Architecture](#18-mindmap--mlp-architecture) | Full network diagram |
| 19 | [Loss Functions](#19-loss-functions) | MSE, Hinge — with worked numbers |
| 20 | [Mindmap: Loss Functions](#20-mindmap--loss-functions) | Visual comparison |
| 21 | [The Training Loop](#21-the-training-loop--4-steps) | 4 steps with deep explanation |
| 22 | [Mindmap: Training Loop](#22-mindmap--the-4-step-training-loop) | Flowchart |
| 23 | [Gradient Descent](#23-gradient-descent--visualized) | Visual with curve, learning rate |
| 24 | [Mindmap: Gradient Descent](#24-mindmap--gradient-descent) | Visual on loss curve |
| 25 | [Concrete Examples](#25-concrete-examples) | 5 worked problems |
| 26 | [Training Results](#26-training-results) | Actual output from the code |
| 27 | [Homework](#27-homework) | 4 exercises with guidance |
| 28 | [Micrograd → PyTorch](#28-micrograd--pytorch) | Mapping from scratch to production |
| 29 | [Glossary](#29-complete-glossary) | 40 terms in plain English |
| 30 | [Resources](#30-resources--references) | Videos, papers, schedule |

---

---

# 1. What Is AI Engineering

## The Core Distinction

Most people learn to USE AI tools. AI Engineers learn to BUILD them.

| AI User | AI Engineer |
|---------|-------------|
| Calls the ChatGPT API | Builds the inference server |
| Uses a RAG library | Builds the vector database and retrieval pipeline |
| Fine-tunes a model with a script | Builds the training loop from scratch |
| Writes prompts | Builds the tokenizer that processes them |
| Uses PyTorch | Understands what PyTorch does internally |

## Why Build From Scratch?

> Using a library is like driving a car.
> Building from scratch is like building the engine.
> You only TRULY understand the car after you've built the engine.
> After that, driving is trivial.

When you build the autograd engine yourself (micrograd), you permanently understand:
- What a gradient is and why it works
- How every deep learning framework (PyTorch, JAX, TensorFlow) operates internally
- Why certain bugs happen and how to fix them
- What every training hyperparameter actually does

You cannot get this from reading documentation.

---

---

# 2. The Full Learning Roadmap

The 50+ AI engineering projects form a dependency tree.
Learn them in this order — each one builds on the previous.

## Tier 1 — Foundation (Do These First)

| # | Project | Why First |
|---|---------|-----------|
| 1 | **Autograd Engine** (Micrograd) | Gradients power everything. No understanding without this. |
| 2 | **Tokenizer** (BPE) | All text models start with tokenization. Text → Numbers. |
| 3 | **Matrix Multiplication kernel** | Everything runs on matmul. Understand what hardware executes. |

## Tier 2 — Core Models

| # | Project | Why |
|---|---------|-----|
| 4 | **Transformer from scratch** | The dominant architecture. Non-negotiable. |
| 5 | **Small Language Model (SLM)** | Apply transformer end-to-end with actual training. |
| 6 | **Softmax kernel optimization** | Numerical stability — why values don't explode. |
| 7 | **Flash Attention kernel** (CUDA) | Now you understand why it was invented. |

## Tier 3 — Agents and Reasoning

| # | Project | Why |
|---|---------|-----|
| 8 | **Chain of Thought Reasoner** | Structured prompting and multi-step reasoning. |
| 9 | **ReAct Agent loop** | Tool use + reasoning. The pattern behind every AI agent. |
| 10 | **Function Calling router** | How agents decide which tool to call. |
| 11 | **Structured Output parser** (CFGs) | Reliable JSON/schema extraction from models. |

## Tier 4 — Memory and Retrieval

| # | Project | Why |
|---|---------|-----|
| 12 | **Vector Database** (HNSW index) | The data structure under all RAG systems. |
| 13 | **Embedding model** | What creates the vectors that go into the database. |
| 14 | **RAG pipeline** | Combine: embeddings + vector DB + LLM. |
| 15 | **Graph RAG system** | Upgrade from flat retrieval to structured knowledge. |

## Tier 5 — Training and Optimization

| # | Project | Why |
|---|---------|-----|
| 16 | **LoRA trainer** | Most practical fine-tuning method used in production today. |
| 17 | **DPO loss function** | Simpler than PPO, widely used for model alignment. |
| 18 | **RLHF pipeline** (PPO) | How models learn from human feedback. |
| 19 | **Quantization library** (Int8/FP4) | How large models shrink to run on smaller hardware. |
| 20 | **KV Cache paging system** (vLLM) | How inference servers manage GPU memory efficiently. |

## Tier 6 — Production and Scale

Speculative Decoding → Distributed Training (FSDP) → MoE routing → Inference Server in C++/Rust

---

---

# 3. The Starter Pack

> These 3 projects cover 80% of what matters in AI engineering today.

```
Project 1: Autograd Engine  →  understand how neural networks LEARN
Project 2: Transformer      →  understand how language MODELS work
Project 3: ReAct Agent      →  understand how AI AGENTS operate
```

All other projects are specializations built on top of these three.

**Minimum Prerequisites:**
- Python at intermediate level (functions, classes, list comprehensions)
- NumPy matrix ops: `@` (matmul), `reshape`, broadcasting
- Basic calculus: knowing that `d/dx(x²) = 2x` is enough to start

---

---

# 4. The Core Problem — Why Gradients Exist

## The Knob Problem — Explained From Zero

### Step 1: Think of the Simplest Possible Machine

Forget neural networks for a moment.
Imagine a machine that does ONE thing:

```
output = weight × input

weight = 2.0   (a knob you can turn)
input  = 3.0   (your data — fixed, you cannot change this)

output = 2.0 × 3.0 = 6.0
```

Now someone tells you: **"The correct output should be 9.0, not 6.0."**

What do you do?

You need to adjust the **weight** (the knob).
But which direction? Bigger or smaller?

```
Current:   weight=2.0 → output=6.0   (target=9.0, we're too LOW)
Try:       weight=3.0 → output=9.0   ← correct! we needed BIGGER weight
```

Simple — because the relationship is obvious.

But what if you have 1,000,000 weights all connected together?
You cannot try every combination. You need a FORMULA that tells you,
for each weight, whether to go bigger or smaller — and by exactly how much.

**That formula is the gradient.**

---

### Step 2: What "Loss" Means

Before adjusting anything, we need one number that says: **"How wrong are we right now?"**

This is the **Loss**.

```
prediction = what the network currently outputs
target     = what the correct answer is

Loss = (prediction - target)²

Example:
  prediction = 0.3
  target     = 1.0
  Loss = (0.3 - 1.0)² = (-0.7)² = 0.49

Another try:
  prediction = 0.9   (better!)
  target     = 1.0
  Loss = (0.9 - 1.0)² = (-0.1)² = 0.01   ← much smaller

Perfect prediction:
  prediction = 1.0
  target     = 1.0
  Loss = (1.0 - 1.0)² = 0.0   ← zero loss = perfect
```

**Goal: Make Loss as close to 0 as possible.**

Why squaring? Two reasons:
1. Always positive — can't have negative error
2. Large errors are penalized MORE than small ones (0.7² = 0.49, but 0.1² = 0.01)

---

### Step 3: The Actual Problem With Multiple Weights

A real neural network has many weights simultaneously.
Here's a concrete example with 3 weights:

```
Formula: output = w1×x1 + w2×x2 + b

x1 = 2.0, x2 = 3.0   (input data — fixed)
w1 = 0.5              (weight 1 — LEARNABLE)
w2 = -0.3             (weight 2 — LEARNABLE)
b  = 0.1              (bias — LEARNABLE)

Computation:
  output = (0.5 × 2.0) + (-0.3 × 3.0) + 0.1
         =  1.0        +  (-0.9)       + 0.1
         =  0.2

Target: 1.0
Loss = (0.2 - 1.0)² = 0.64     ← we're wrong
```

Now: we have 3 adjustable values (w1, w2, b).
We need to know for EACH of them:

```
For w1: "If I increase w1 by 0.001, does Loss go up or down, and by how much?"
For w2: "If I increase w2 by 0.001, does Loss go up or down, and by how much?"
For b:  "If I increase b  by 0.001, does Loss go up or down, and by how much?"
```

These three answers are the three **gradients**:
- `d(Loss)/d(w1)` — gradient of Loss with respect to w1
- `d(Loss)/d(w2)` — gradient of Loss with respect to w2
- `d(Loss)/d(b)`  — gradient of Loss with respect to b

---

### Step 4: Trying It By Hand (Brute Force — Just Once)

Let's manually compute `d(Loss)/d(w1)` by nudging w1:

```
Original:
  w1 = 0.5
  output = 0.5×2.0 + (-0.3)×3.0 + 0.1 = 0.2
  Loss   = (0.2 - 1.0)² = 0.64

Nudge w1 by +0.001:
  w1 = 0.501
  output = 0.501×2.0 + (-0.3)×3.0 + 0.1 = 0.202
  Loss   = (0.202 - 1.0)² = 0.636804

Change in Loss = 0.636804 - 0.64 = -0.003196
Change in w1   = 0.001

Gradient ≈ Change in Loss / Change in w1
          = -0.003196 / 0.001
          = -3.196 ≈ -3.2
```

**What does -3.2 mean?**

```
Gradient is NEGATIVE = -3.2

This means: if w1 increases, Loss DECREASES
            → we should INCREASE w1 to reduce Loss

Update: w1 = w1 - learning_rate × gradient
           = 0.5  - 0.1 × (-3.2)
           = 0.5  + 0.32
           = 0.82
```

Let's verify:
```
New w1 = 0.82
output = 0.82×2.0 + (-0.3)×3.0 + 0.1 = 1.64 - 0.9 + 0.1 = 0.84
Loss   = (0.84 - 1.0)² = 0.0256   ← down from 0.64! 
```

**Loss dropped from 0.64 to 0.0256 in ONE step!** That's what gradients do.

---

### Step 5: Why We Can't Always Do This By Hand

The nudge method (above) works but it's slow:
- For each weight, you run the ENTIRE network TWICE (original + nudged)
- 1 million weights = 2 million forward passes per training step
- 100,000 training steps = 200,000,000,000 forward passes

**Totally impractical.**

The Chain Rule lets us compute ALL gradients in ONE backward pass.
That is what `loss.backward()` does.
That is why backpropagation was a breakthrough.
That is what micrograd implements.

---

### Step 6: The Full Network View

Here is what a real (small) network looks like:

```
                         ┌──────────────────────────┐
  Input: [2.0, 3.0] ──→  │  w1 = 0.5   (LEARNABLE)  │
                         │  w2 = -0.3  (LEARNABLE)  │
                         │  b  = 0.1   (LEARNABLE)  │ ──→ Output: 0.3
                         └──────────────────────────┘
                                      │
                              True answer: 1.0
                                      │
                          Loss = (0.3 - 1.0)² = 0.49
                                      │
                            We need 3 gradients:
                            d(Loss)/d(w1) = ?  ← how to fix w1
                            d(Loss)/d(w2) = ?  ← how to fix w2
                            d(Loss)/d(b)  = ?  ← how to fix b
                                      │
                          loss.backward() computes all 3
                          in one pass, automatically.
```

---

### Step 7: The Update — Closing the Loop

Once we have gradients, we apply them:

```
learning_rate = 0.1   (small step — don't overshoot)

w1 = w1 - learning_rate × d(Loss)/d(w1)
w2 = w2 - learning_rate × d(Loss)/d(w2)
b  = b  - learning_rate × d(Loss)/d(b)
```

Why subtract? Because:
- Positive gradient → Loss increases as weight increases → decrease weight
- Negative gradient → Loss decreases as weight increases → increase weight
- Subtracting gradient always moves in the direction that REDUCES loss

Then we predict again. Loss is smaller.
We repeat. Loss gets smaller again.
After enough steps, Loss ≈ 0 and the network predicts correctly.

**This entire loop is neural network training. Micrograd is the mechanism that makes it automatic.**

---

## Summary: The Knob Problem in 4 Lines

```
1. Network makes a prediction with current weights
2. We measure how wrong it is (Loss)
3. We compute which direction to nudge EACH weight to reduce Loss (Gradients)
4. We nudge all weights slightly in that direction (Gradient Descent)
→ Repeat until Loss ≈ 0
```

## Why You Cannot Ignore This

Every neural network — GPT-4, Gemini, Claude — is trained using this exact mechanism.
The only difference between micrograd and PyTorch is scale:

- Micrograd: one number at a time, pure Python, ~150 lines
- PyTorch: billions of numbers at once, highly optimized C++/CUDA, ~500,000 lines

**The concept is identical. Learn it once, understand everything.**

---

---

# 5. What Is a Gradient

## The Slope Analogy

Imagine standing on a hillside in complete fog. Your goal is to reach the valley (the lowest point = minimum loss).

```
LOSS
  │
  │\         You are here ←──→ gradient tells you
  │ \                         which way is downhill
  │  \       *
  │   \     / \
  │    \   /   \
  │     \_/     \
  │      ↑       \____
  │   VALLEY         (you want to get here)
  └───────────────────────→ WEIGHT VALUE
```

- **Gradient is positive** → the hill rises to the right → move LEFT (decrease weight)
- **Gradient is negative** → the hill falls to the right → move RIGHT (increase weight)
- **Gradient is zero** → you are at the bottom → done!

Update rule: `weight = weight - learning_rate × gradient`

## Gradient From a Simple Example

```python
# y = x²
# dy/dx = 2x

x = Value(3.0)
y = x ** 2      # y = 9
y.backward()
print(x.grad)   # 6.0  ← = 2×3 = 6 ✓
```

**Numerical verification** (no calculus needed):
```
x = 3.000 → y = 9.000000
x = 3.001 → y = 9.006001  (nudge x by +0.001)

Change in y = 0.006001
Change in x = 0.001
Ratio = 0.006001 / 0.001 = 6.001 ≈ 6 ✓
```

The gradient is just: "how sensitive is the output to a tiny change in the input?"

---

---

# 6. Mindmap — The Big Picture

```
                     NEURAL NETWORK TRAINING LOOP
                                  │
         ┌────────────────────────┼────────────────────────┐
         │                        │                        │
   FORWARD PASS             LOSS FUNCTION            BACKWARD PASS
   (run the network)        (measure error)           (fix weights)
         │                        │                        │
   Feed input through        Compare prediction       Compute d(loss)/d(w)
   every layer in order      to true answer           for every weight
   (input→L1→L2→output)      e.g. (pred-true)²        using chain rule
         │                        │                        │
   Produces one number        Produces one number      Produces one gradient
   (the prediction)           (the loss)               per weight
                                                            │
                                                    GRADIENT DESCENT
                                                    w = w - lr × gradient
                                                            │
                                                    Repeat until loss ≈ 0
```

**The cycle in one sentence:**
> Predict → Measure error → Figure out whose fault it is → Fix them slightly → Repeat

---

---

# 7. The Value Class — The Entire Engine

The ENTIRE autograd engine is the `Value` class in [`engine.py`](engine.py).
Everything else is built on top of it.

## What It Stores

```python
class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data      = float(data)    # THE NUMBER
        self.grad      = 0.0            # THE GRADIENT (starts unknown)
        self._backward = lambda: None   # HOW TO BACKPROP (filled by ops)
        self._children = set(_children)# WHAT CREATED THIS NODE
        self._op       = _op            # WHICH OPERATION (+, *, tanh...)
        self.label     = label          # OPTIONAL NAME for printing
```

**When you write `a = Value(2.0, label='a')`:**

```
┌────────────────────────────────┐
│  Value: 'a'                    │
│  ─────────────────────────     │
│  .data      = 2.0              │  ← the actual number
│  .grad      = 0.0              │  ← not computed yet
│  ._backward = (empty function) │  ← no operation, so nothing to backprop
│  ._children = { }              │  ← no parents — this is a leaf node
│  ._op       = ''               │  ← no operation that created this
└────────────────────────────────┘
```

**After `c = a * b`:**

```
┌────────────────────────────────┐
│  Value (unnamed)               │
│  ─────────────────────────     │
│  .data      = 6.0              │  ← 2.0 × 3.0
│  .grad      = 0.0              │  ← not computed yet
│  ._backward = <function>       │  ← stores the rule: how does c's grad
│                                │    flow back to a and b?
│  ._children = {a, b}           │  ← created from a and b
│  ._op       = '*'              │  ← via multiplication
└────────────────────────────────┘
```

---

---

# 8. Mindmap — The Value Object

```
                            Value(3.0)
                                 │
           ┌─────────────────────┼─────────────────────┐
           │                     │                     │
         .data                 .grad              ._backward
           │                     │                     │
       the actual            the gradient         a closure (function)
       number stored         of LOSS w.r.t.       that pushes gradient
       here = 3.0            THIS node            one step backward
                             starts at 0.0
                             filled by             ._children
                             .backward()           │
                                                who created this?
                                                e.g. {a, b} if c=a+b

─────────────────────────────────────────────────────────────────
LEAF NODE (input, created directly):        INTERMEDIATE NODE (created by op):
  Value(3.0)                                  c = a * b
  .data = 3.0                                 .data = a.data × b.data
  .grad = 0.0  (filled later)                 .grad = 0.0  (filled later)
  ._backward = do nothing                     ._backward = push grad to a and b
  ._children = {}                             ._children = {a, b}
  ._op = ''                                   ._op = '*'
─────────────────────────────────────────────────────────────────
```

---

---

# 9. Operations & Their Backward Rules

For each operation, there are two things:
1. The **forward rule** — how to compute the output value
2. The **backward rule** — how to propagate the gradient back to inputs

## Addition: `z = a + b`

```
Forward:   z.data = a.data + b.data

Backward rule (from calculus):
  dz/da = 1    →   a.grad += z.grad × 1
  dz/db = 1    →   b.grad += z.grad × 1

In English: gradient flows UNCHANGED through addition.
If z.grad = 5.0, then a and b each receive 5.0.
```

**Code:**
```python
def _backward():
    self.grad  += out.grad      # × 1 implicit
    other.grad += out.grad      # × 1 implicit
```

## Multiplication: `z = a × b`

```
Forward:   z.data = a.data × b.data

Backward rule:
  dz/da = b.data   →   a.grad += z.grad × b.data
  dz/db = a.data   →   b.grad += z.grad × a.data

In English: to get a's gradient, multiply incoming grad by b's VALUE (not b's grad).
```

**Example:**
```
a=2, b=3, z=6, z.grad=1.0
→ a.grad = 1.0 × 3 = 3.0
→ b.grad = 1.0 × 2 = 2.0
```

**Code:**
```python
def _backward():
    self.grad  += out.grad * other.data
    other.grad += out.grad * self.data
```

## Power: `z = aⁿ`

```
Forward:   z.data = a.data ** n

Backward rule (power rule from calculus):
  dz/da = n × a^(n-1)   →   a.grad += z.grad × n × a.data^(n-1)
```

**Example:**
```
a=3, n=2, z=9, z.grad=1.0
→ a.grad = 1.0 × 2 × 3¹ = 6.0
```

## tanh: `z = tanh(a)`

```
Forward:   z.data = tanh(a.data)

Backward rule:
  dz/da = 1 - tanh(a)²  =  1 - z.data²

In English: the local derivative depends on the OUTPUT of tanh (z), not the input.
```

**Example:**
```
a=0.5, z=tanh(0.5)=0.462, z.grad=1.0
→ a.grad = 1.0 × (1 - 0.462²) = 1.0 × 0.787 = 0.787
```

## ReLU: `z = max(0, a)`

```
Forward:   z.data = max(0, a.data)

Backward rule:
  if a.data > 0:  a.grad += z.grad   (gradient passes through)
  if a.data ≤ 0:  a.grad += 0        (gradient is BLOCKED)
```

**Example A (positive input):**
```
a=2.0, z=2.0, z.grad=1.0
→ a.grad = 1.0 ✓  (gradient flows through)
```

**Example B (negative input — dead ReLU):**
```
a=-1.0, z=0.0, z.grad=1.0
→ a.grad = 0.0  (gradient completely blocked — neuron is "dead")
```

---

---

# 10. Mindmap — Gradient Flow Through Operations

```
OPERATION      FORMULA         BACKWARD RULE               EXAMPLE (z.grad=1.0)
─────────────────────────────────────────────────────────────────────────────────
ADD            z = a + b       a.grad += z.grad × 1        a=2, b=3:
                               b.grad += z.grad × 1        → a.grad=1.0, b.grad=1.0

MUL            z = a × b       a.grad += z.grad × b.data   a=2, b=3:
                               b.grad += z.grad × a.data   → a.grad=3.0, b.grad=2.0

POW            z = aⁿ          a.grad += z.grad            a=3, n=2:
                                        × n × a^(n-1)      → a.grad=6.0

NEG            z = -a          a.grad += z.grad × -1       a=5:
               (= a × -1)                                  → a.grad=-1.0

SUB            z = a - b       a.grad += z.grad × 1        a=5, b=3:
               (= a + (-b))    b.grad += z.grad × -1       → a.grad=1.0, b.grad=-1.0

DIV            z = a / b       a.grad += z.grad × 1/b      a=6, b=2:
               (= a × b⁻¹)    b.grad += z.grad × -a/b²    → a.grad=0.5, b.grad=-1.5

TANH           z = tanh(a)     a.grad += z.grad            a=0.5, z=0.462:
                                        × (1 - z²)         → a.grad=0.787

RELU           z = max(0, a)   a>0: a.grad += z.grad       a=2.0: a.grad=1.0
                               a≤0: a.grad += 0            a=-1.0: a.grad=0.0

EXP            z = eᵃ          a.grad += z.grad × z.data   a=1.0, z=2.718:
                                                           → a.grad=2.718

LOG            z = ln(a)       a.grad += z.grad × 1/a      a=2.0:
                                                           → a.grad=0.5
─────────────────────────────────────────────────────────────────────────────────
```

**The pattern:** Every backward rule is just the derivative of that operation, multiplied by the incoming gradient (chain rule).

---

---

# 11. Backpropagation — The Full Algorithm

Backpropagation is the algorithm that computes ALL gradients in one pass through the graph.

## The Three Steps

### Step 1: Topological Sort

Visits all nodes using depth-first search.
Appends each node AFTER all its children → leaves first, output last.

```python
def build_topo(node):
    if id(node) not in visited:
        visited.add(id(node))
        for child in node._children:
            build_topo(child)       # recurse into children first
        topo_order.append(node)     # then append self
```

Result for `e = a*b + (a+b)` where `a=2, b=3`:
```
topo_order = [a, b, c, d, e]   (leaves first)
reversed   = [e, d, c, b, a]   (output first)
```

### Step 2: Seed the Gradient

```python
self.grad = 1.0
```

`d(Loss)/d(Loss) = 1` — by definition. We differentiate Loss w.r.t. itself.

### Step 3: Walk in Reverse, Call `_backward`

```python
for node in reversed(topo_order):
    node._backward()
```

Each call pushes the gradient one step backward through the graph.

## Why Topological Order Matters

A node's `_backward()` uses `out.grad` (the gradient coming from above).
That gradient must be FULLY COMPUTED before we call `_backward()`.

```
Wrong order:  a → c → e (forward order)
  When we try to compute a.grad, e.grad hasn't been computed yet!
  a.grad += c.grad × ... but c.grad is still 0!

Correct order: e → d → c → b → a (reverse topological)
  e.grad = 1.0  (seeded)
  d.grad = f(e.grad)  ← e.grad is ready ✓
  c.grad = f(e.grad)  ← e.grad is ready ✓
  b.grad = f(c.grad, d.grad)  ← both ready ✓
  a.grad = f(c.grad, d.grad)  ← both ready ✓
```

---

---

# 12. Mindmap — The Backward Pass Traced

## Computation: `e = (a × b) + (a + b)` with `a=2, b=3`

```
GRAPH STRUCTURE:
   a=2  b=3
    │    │
    ├────┤
    │    │
   [×]  [+]        op='*'       op='+'
    │    │
   c=6  d=5
    │    │
    └────┘
      [+]          op='+'
       │
      e=11

─────────────────────────────────────────────────────
FORWARD PASS (left to right, building the graph):
  c = 2 × 3 = 6
  d = 2 + 3 = 5
  e = 6 + 5 = 11

─────────────────────────────────────────────────────
SEED:
  e.grad = 1.0   (d(Loss)/d(Loss) = 1)

─────────────────────────────────────────────────────
BACKWARD PASS (right to left, calling _backward):

Node e (op='+', children={c,d}):
  c.grad += e.grad × 1 = 1.0
  d.grad += e.grad × 1 = 1.0

Node d (op='+', children={a,b}):
  a.grad += d.grad × 1 = 1.0    ← ACCUMULATES with +=
  b.grad += d.grad × 1 = 1.0    ← ACCUMULATES with +=

Node c (op='*', children={a,b}):
  a.grad += c.grad × b.data = 1.0 × 3 = 3.0  ← ACCUMULATES
  b.grad += c.grad × a.data = 1.0 × 2 = 2.0  ← ACCUMULATES

─────────────────────────────────────────────────────
FINAL GRADIENTS:
  a.grad = 1.0 + 3.0 = 4.0
  b.grad = 1.0 + 2.0 = 3.0

VERIFICATION:
  e = a×b + a + b
  de/da = b + 1 = 3 + 1 = 4  ✓
  de/db = a + 1 = 2 + 1 = 3  ✓
─────────────────────────────────────────────────────
```

**Key insight:** `a` appears in TWO places (in both `c=a*b` and `d=a+b`).
Each path adds to `a.grad` separately. That's why we use `+=` not `=`.

---

---

# 13. The Chain Rule

## What It Is

The chain rule connects the gradient of a composed function back to each input.

```
If: final = f( g( x ) )
Then: d(final)/dx = d(final)/d(g)  ×  d(g)/dx
                  = (incoming grad) × (local derivative)
```

## In a 3-Node Chain

```
x ──→ [×2] ──→ a ──→ [×3] ──→ b ──→ [×4] ──→ c

c = x × 2 × 3 × 4 = x × 24
dc/dx = 24

How gradients flow backward:
  c.grad = 1.0                     (seed)
  b.grad = c.grad × 4 = 4.0        (local deriv of ×4 node is 4)
  a.grad = b.grad × 3 = 12.0       (local deriv of ×3 node is 3)
  x.grad = a.grad × 2 = 24.0       (local deriv of ×2 node is 2)

Result: x.grad = 1 × 4 × 3 × 2 = 24 ✓
```

**At every node:** multiply incoming gradient by local derivative.
**That's the chain rule.** Applied repeatedly across the entire graph.

## Code Verification

```python
x = Value(5.0)
a = x * 2
b = a * 3
c = b * 4
c.backward()
print(x.grad)   # 24.0 ✓
```

---

---

# 14. Activation Functions

## The Problem Without Them

Without activation functions, stacking layers achieves nothing:

```
Layer 1: output = w1 × x + b1
Layer 2: output = w2 × (w1 × x + b1) + b2
                = (w2×w1) × x + (w2×b1 + b2)
                = W × x + B           ← STILL just a straight line

No matter how many layers you stack — without activations —
the entire network collapses into y = Wx + B.
A linear function cannot learn nonlinear patterns.
```

## tanh

```
z = tanh(pre_activation)

Properties:
  Output range: (-1, 1)
  tanh(0) = 0        (centered)
  tanh(∞) = 1
  tanh(-∞) = -1

Values:
  tanh(-3) = -0.995
  tanh(-1) = -0.762
  tanh(0)  =  0.000
  tanh(1)  =  0.762
  tanh(3)  =  0.995

Backward:
  d(tanh)/dx = 1 - tanh(x)²
  At x=0: gradient = 1.0      (maximum, best gradient flow)
  At x=3: gradient = 1 - 0.995² = 0.010  (saturated, small gradient)
```

## ReLU

```
z = max(0, pre_activation)

Properties:
  Output range: [0, ∞)
  Fast to compute (just a comparison)
  Can "die" (negative inputs → zero gradient forever)

Values:
  relu(-3) = 0
  relu(-1) = 0
  relu(0)  = 0
  relu(1)  = 1
  relu(3)  = 3

Backward:
  d(relu)/dx = 1 if x > 0
             = 0 if x ≤ 0

The dying ReLU problem:
  If a neuron consistently receives negative inputs,
  its gradient is always 0 → weights never update → neuron is dead
```

## When to Use Each

```
tanh  → use in hidden layers of small networks (like our MLP)
ReLU  → use in deep networks (faster training, but watch for dying neurons)
Linear (no activation) → use in output layer for regression/raw scores
```

---

---

# 15. What Is a Neuron

## From Biology to Code

**Biological neuron:**
- Receives signals from many dendrites (inputs)
- If total signal exceeds a threshold, it fires (activates)
- Sends signal through axon to the next neurons

**Artificial neuron:**
- Receives numbers (inputs)
- Computes weighted sum
- Passes through activation function
- Outputs one number

## The Computation

```
output = activation( w1×x1 + w2×x2 + ... + wn×xn + b )
                     ╰──────────────────────────────╯
                              dot product
```

## Worked Example

```
Inputs:    x1=2.0, x2=3.0, x3=-1.0
Weights:   w1=0.5, w2=-0.3, w3=0.8   ← LEARNED (start random)
Bias:      b=0.1                       ← LEARNED (start at 0)

Dot product:
  = (0.5 × 2.0) + (-0.3 × 3.0) + (0.8 × -1.0) + 0.1
  =  1.0        +  (-0.9)       +  (-0.8)       + 0.1
  = -0.6

Activation (tanh):
  tanh(-0.6) = -0.537

Neuron output: -0.537
```

## Gradient on Weights After Backward

```
If we compute loss and call loss.backward():
  w1.grad = how much output changes if w1 changes
          = output_grad × d(output)/d(w1)
          = output_grad × x1 × (1 - tanh²)

This tells us: increase or decrease w1 to reduce loss?
```

---

---

# 16. Mindmap — One Neuron

```
                           ONE NEURON
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
           INPUTS            WEIGHTS            OUTPUT
              │                 │                 │
         x1, x2, ..., xn    w1, w2, ..., wn   activation(dot + b)
         (your data)        (LEARNED)          (one number, −1 to 1 for tanh)
         doesn't change     adjusted by         represents the neuron's
         during training    gradient descent    "opinion" on the input
```

## Real-World Analogy — Umbrella Decision Neuron

```
Input 1: Is it cloudy?         x1=0.8    w1=+2.0  (HIGH importance)
Input 2: Is it summer?         x2=0.7    w2=-0.5  (inversely related)
Input 3: Was yesterday rainy?  x3=0.9    w3=+1.0  (medium importance)
bias:                                    b=-0.3   (learned threshold)

Dot product = 0.8×2.0 + 0.7×(-0.5) + 0.9×1.0 + (-0.3)
            = 1.6 - 0.35 + 0.9 - 0.3
            = 1.85

tanh(1.85) = 0.956   ← close to +1, bring umbrella!

The WEIGHTS are what the neuron learned from past rainy/not-rainy days.
Higher w1 = neuron learned that clouds strongly predict rain.
Training adjusts these weights via gradient descent.
```

---

---

# 17. Layers and MLP

## What Is a Layer?

A layer is **many neurons all reading the same input**, each learning different patterns.

```python
Layer(n_inputs=3, n_neurons=4)
```

```
Input: [x1, x2, x3]
          │   │   │
    ┌─────┼───┼───┼─────────────────────────────┐
    │                                            │
    │  Neuron 1 (w=[0.5,-0.3,0.8], b=0.1)  → out1   learns pattern A
    │  Neuron 2 (w=[-0.2,0.9,-0.1], b=0.0) → out2   learns pattern B
    │  Neuron 3 (w=[0.7,0.2,0.5], b=-0.1)  → out3   learns pattern C
    │  Neuron 4 (w=[-0.4,-0.6,0.3], b=0.2) → out4   learns pattern D
    │                                            │
    └────────────────────────────────────────────┘

Output: [out1, out2, out3, out4]
```

## What Is an MLP?

Layers stacked sequentially. Output of each layer is input to the next.

```python
model = MLP(n_inputs=3, layer_sizes=[4, 4, 1])
```

```
LAYER    SIZE     ACTIVATION    PARAMETERS
───────────────────────────────────────────────────────
Input    3        —             0 (inputs, not learned)
Layer 1  4        tanh          4 × (3+1) = 16
Layer 2  4        tanh          4 × (4+1) = 20
Output   1        linear        1 × (4+1) =  5
─────────────────────────────────────────────────────
TOTAL PARAMETERS: 41
```

**Why multiple layers?**

```
Layer 1 learns: raw features (edges in an image, word patterns in text)
Layer 2 learns: combinations of features (shapes, phrases)
Layer 3+ learns: high-level concepts (faces, sentences, intent)
```

---

---

# 18. Mindmap — MLP Architecture

```
         INPUT             HIDDEN 1          HIDDEN 2          OUTPUT
       (3 values)         (4 neurons)       (4 neurons)       (1 neuron)

       [x1]─────────→ [N1]──┐
       [x2]─────────→ [N2]──┤──→[N5]──┐
       [x3]─────────→ [N3]──┤    [N6]──┤──→ [N9] ──→ prediction
                        [N4]──┘    [N7]──┤
                                   [N8]──┘

Every N has its own weights.
Every connection has a weight.
All weights learned by gradient descent.

PARAMETER COUNT:
  N1,N2,N3,N4: each reads 3 inputs → 3 weights + 1 bias = 4 each → 16 total
  N5,N6,N7,N8: each reads 4 inputs → 4 weights + 1 bias = 5 each → 20 total
  N9:          reads 4 inputs       → 4 weights + 1 bias = 5       →  5 total
                                                                    ─────────
                                                          TOTAL:      41
```

---

---

# 19. Loss Functions

## What Is Loss?

Loss = a single number answering: **"How wrong are we right now?"**

```
Goal: Loss = 0 (perfect predictions)
Reality: Loss starts high, decreases with training
```

## Mean Squared Error (MSE)

Used for regression (predicting a continuous number).

```
loss = (prediction - target)²

Example 1: pred=0.3, target=1.0
  loss = (0.3 - 1.0)² = (-0.7)² = 0.49

Example 2: pred=0.9, target=1.0
  loss = (0.9 - 1.0)² = (-0.1)² = 0.01  ← much better

Example 3: pred=1.0, target=1.0
  loss = (1.0 - 1.0)² = 0.0  ← perfect
```

The squaring serves two purposes:
1. Always positive (can't have negative error)
2. Penalizes large errors more heavily than small ones

## Hinge Loss

Used for classification (predicting +1 or -1).

```
loss = max(0,  1 - y_true × y_pred)

Case 1: correct AND confident
  y_true=+1, y_pred=+1.5
  1 - (1 × 1.5) = -0.5 → max(0, -0.5) = 0     ← no loss!

Case 2: correct but uncertain
  y_true=+1, y_pred=+0.3
  1 - (1 × 0.3) = 0.7  → max(0, 0.7) = 0.7    ← some loss

Case 3: completely wrong
  y_true=+1, y_pred=-0.8
  1 - (1 × -0.8) = 1.8 → max(0, 1.8) = 1.8    ← large loss!

Case 4: correct, other class
  y_true=-1, y_pred=-1.5
  1 - (-1 × -1.5) = -0.5 → max(0, -0.5) = 0   ← no loss!
```

---

---

# 20. Mindmap — Loss Functions

```
                          LOSS FUNCTION
                                │
                ┌───────────────┴───────────────┐
                │                               │
          REGRESSION                      CLASSIFICATION
          predict a number                predict a category
                │                               │
         Mean Squared Error               Hinge Loss
         (pred - target)²               max(0, 1 - y×pred)
                │                               │
         pred=0.3, target=1.0:          pred=+1.5, true=+1:
         loss = (0.3-1.0)² = 0.49       loss = max(0,-0.5) = 0 ✓
                                        pred=-0.5, true=+1:
                                        loss = max(0,1.5) = 1.5 ✗

LOSS OVER TRAINING TIME:

Loss
 │
5│ ●
4│   ●
3│     ●
2│       ●
1│         ●  ●
0│              ●─────●────●────●
 └────────────────────────────────→ Training Steps
```

---

---

# 21. The Training Loop — 4 Steps

Every framework. Every network. Every training run. These 4 steps, repeated.

```python
for step in range(n_steps):

    # ─── STEP 1: FORWARD PASS ──────────────────────────────────────────
    y_pred = model(x)               # run input through every layer
    loss   = loss_function(y_pred, y_true)   # compute scalar loss

    # ─── STEP 2: ZERO GRADIENTS ────────────────────────────────────────
    model.zero_grad()               # reset ALL param.grad to 0.0
                                    # CRITICAL: prevents gradient accumulation

    # ─── STEP 3: BACKWARD PASS ─────────────────────────────────────────
    loss.backward()                 # computes grad for every parameter
                                    # automatically via chain rule

    # ─── STEP 4: GRADIENT DESCENT UPDATE ───────────────────────────────
    for p in model.parameters():
        p.data -= learning_rate * p.grad   # move each weight downhill
```

## Why Each Step Exists

### Step 1 — Forward Pass

Builds the computation graph as a side effect.
Every operation records its children and backward function.
This graph is what `.backward()` traverses in Step 3.

### Step 2 — Zero Gradients

Gradients accumulate with `+=`. Old gradients from the previous step persist.
If you skip `zero_grad()`:

```
Step 1: w.grad = 2.0   (correct)
Step 2: w.grad = 5.0   (WRONG: 2.0 from step1 + 3.0 from step2)
Step 3: w.grad = 8.5   (completely wrong: history of all steps)
```

The gradient must reflect ONLY the current step.

### Step 3 — Backward Pass

One call propagates the gradient through the ENTIRE graph.
Every parameter gets its gradient computed automatically.

### Step 4 — Update

The rule `p.data -= lr × p.grad` moves the weight downhill.

```
grad=+4.0 → w -= lr × (+4.0) → w decreases → loss decreases ✓
grad=-2.0 → w -= lr × (-2.0) → w increases → loss decreases ✓
grad=0.0  → w -= lr × 0      → w unchanged  → at minimum already
```

---

---

# 22. Mindmap — The 4-Step Training Loop

```
              ┌─────────────────────────────┐
              │   Initialize random weights │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │       STEP 1: FORWARD       │
              │  prediction = model(input)  │
              │  loss = f(prediction, true) │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │     STEP 2: ZERO GRADS      │
              │   for p in parameters():   │
              │       p.grad = 0.0          │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │      STEP 3: BACKWARD       │
              │     loss.backward()         │
              │  (fills every param.grad)   │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │       STEP 4: UPDATE        │
              │   for p in parameters():   │
              │     p.data -= lr × p.grad   │
              └──────────────┬──────────────┘
                             │
                     ┌───────▼────────┐
                     │  loss < target?│
                     └───────┬────────┘
                           ╱   ╲
                         YES    NO
                          │      │
                        STOP   go to Step 1
```

---

---

# 23. Gradient Descent — Visualized

## On a Simple Curve

```
Loss = w²          (minimum at w=0)

Loss
 │
9│        *               *
 │      *   *           *   *
4│    *       *       *       *
 │  *           *   *           *
1│                *
0│                  ← minimum
 └─────────────────────────────── w
  -3  -2  -1   0   1   2   3

At w=-2: gradient = 2×(-2) = -4 (negative slope, hill goes down to the right)
  update: w = -2 - lr × (-4) = -2 + 0.4 = -1.6   ← moved right ✓

At w=+2: gradient = 2×(+2) = +4 (positive slope, hill goes down to the left)
  update: w = +2 - lr × (+4) = +2 - 0.4 = +1.6   ← moved left ✓

At w=0: gradient = 0  (minimum)
  update: w = 0 - lr × 0 = 0   ← no change, done ✓
```

## Learning Rate Effect

```
Too large (lr=1.0):
  w=3 → w=3-1×6=-3 → w=-3-1×(-6)=3 → bouncing forever!

Too small (lr=0.001):
  w=3 → w=2.994 → w=2.988 → ... → needs thousands of steps

Just right (lr=0.1):
  w=3 → 2.4 → 1.92 → 1.54 → 1.23 → ... → converges smoothly
```

---

---

# 24. Mindmap — Gradient Descent

```
LOSS vs WEIGHT VALUE CURVE

Loss │
     │  *                           *
     │    *                       *
     │      *                   *
     │        *               *
     │          *           *
     │            *       *
     │              *───*
     └────────────────────────────── weight
                        ↑
                    MINIMUM

If you are LEFT of minimum:
  gradient < 0  (downward slope to the right)
  update: w -= lr × (negative) = w increases → moves RIGHT toward minimum ✓

If you are RIGHT of minimum:
  gradient > 0  (upward slope to the right)
  update: w -= lr × (positive) = w decreases → moves LEFT toward minimum ✓

LEARNING RATE CONTROLS STEP SIZE:
  Large lr → takes big steps → may jump over minimum
  Small lr → takes tiny steps → accurate but slow
  Ideal lr → converges in reasonable time without bouncing
```

---

---

# 25. Concrete Examples

## Example 1 — Gradient of a Polynomial

**Problem:** `y = 3x² + 2x + 1`. Find `dy/dx` at `x=4`.

**By calculus:** `dy/dx = 6x + 2 = 26`

**By micrograd:**
```python
x = Value(4.0)
y = 3*x**2 + 2*x + 1
y.backward()
print(x.grad)   # 26.0 ✓
```

**Numerical verification:**
```
x=4.000 → y = 3(16) + 8 + 1 = 57.000
x=4.001 → y = 3(16.008) + 8.002 + 1 = 57.026
Δy/Δx = 0.026/0.001 = 26.0 ✓
```

---

## Example 2 — Two Weights, One Training Step

**Setup:** `output = w1×2 + w2×3`, target = 10

**Initial:** `w1=1.0, w2=1.0` → `output=5`, `loss=(5-10)²=25`

**Backward trace:**
```
d(loss)/d(output) = 2×(output-target) = 2×(5-10) = -10

d(output)/d(w1)   = 2   →   w1.grad = -10 × 2 = -20
d(output)/d(w2)   = 3   →   w2.grad = -10 × 3 = -30
```

**Update (lr=0.05):**
```
w1 = 1.0 - 0.05×(-20) = 1.0 + 1.0 = 2.0
w2 = 1.0 - 0.05×(-30) = 1.0 + 1.5 = 2.5
```

**New output:** `2.0×2 + 2.5×3 = 4 + 7.5 = 11.5` (overshot but improving)

---

## Example 3 — Why zero_grad() Matters

```python
w = Value(1.0)

# WITHOUT zero_grad():
for step in range(3):
    loss = w * 2       # d(loss)/dw should always be 2
    loss.backward()
    print(w.grad)
# Step 0: 2.0   ← correct
# Step 1: 4.0   ← WRONG (accumulated: 2+2)
# Step 2: 6.0   ← WRONG (accumulated: 2+2+2)

# WITH zero_grad():
w = Value(1.0)
for step in range(3):
    w.grad = 0.0       ← reset before backward
    loss = w * 2
    loss.backward()
    print(w.grad)
# Step 0: 2.0   ✓
# Step 1: 2.0   ✓
# Step 2: 2.0   ✓
```

---

## Example 4 — Learning Rate Effect

```
Problem: minimize loss = (w×2 - 10)²  starting at w=0

lr = 0.5  (too large):
  Step 0: w=0  → loss=100 → grad=-40 → w=20  OVERSHOOT
  Step 1: w=20 → loss=900 → bouncing, diverges

lr = 0.05 (good):
  Step 0:  w=0.0  → loss=100.0
  Step 5:  w=3.2  → loss=21.2
  Step 15: w=4.7  → loss=0.36
  Step 25: w=4.95 → loss=0.01  ← converged ✓

lr = 0.001 (too small):
  Step 0:   w=0.00 → loss=100.0
  Step 100: w=0.78 → loss=70.4  ← barely moved
  Would need 10,000+ steps
```

---

## Example 5 — tanh vs ReLU — The Dying Neuron

```
Input to neuron: pre_activation = -2.0

With tanh:
  output = tanh(-2.0) = -0.964
  gradient = 1 - (-0.964)² = 1 - 0.929 = 0.071
  → small but NON-ZERO → neuron still learns

With ReLU:
  output = max(0, -2.0) = 0.0
  gradient = 0.0
  → ZERO → no gradient → weights never update → neuron permanently dead

If all training inputs to a neuron are negative:
  ReLU neuron = dead (never recovers)
  tanh neuron = alive (always has some gradient)

This is called the "dying ReLU" problem.
Solution: use Leaky ReLU: max(0.01x, x) → small negative gradient preserved.
```

---

---

# 26. Training Results

## Step 1 — Gradient Verification

```
a = Value(2.0), b = Value(3.0)
e = a*b + (a+b) = 11.0

After e.backward():
  a.grad = 4.0   ← de/da = b+1 = 4 ✓
  b.grad = 3.0   ← de/db = a+1 = 3 ✓
```

## Step 2 — One Neuron Training

```
Goal: neuron output → 1.0 for input [2.0, 3.0]

 Step      Output      Loss         w1          w2
─────────────────────────────────────────────────────
    0    0.099668    0.810598    0.8566      0.2348
    5    0.988997    0.000121    0.8570      0.2356
   10    0.989070    0.000119    0.8575      0.2363
   19    0.989198    0.000117    0.8584      0.2375

Output converges toward 1.0. Loss decreasing each step. ✓
```

## Step 3 — Full MLP (41 parameters)

```
Dataset: 4 samples, 3 features, binary classification (+1 or -1)
Network: MLP(3 → [4, 4, 1])

 Step      Loss        Accuracy
─────────────────────────────────
    0    1.014010       50%    ← random guessing
   10    0.653518       75%    ← learning
   20    0.219246      100%    ← got all 4 correct
   30    0.088498      100%    ← more confident
   40    0.005319      100%    ← very confident
   99    0.001432      100%    ← highly confident

Final predictions:
  [2.0,  3.0, -1.0] → pred=+1.0201  target=+1  ✓
  [3.0, -1.0,  0.5] → pred=-1.7564  target=-1  ✓
  [0.5,  1.0,  1.0] → pred=+1.2068  target=+1  ✓
  [1.0,  1.0, -1.0] → pred=-1.0829  target=-1  ✓

100% accuracy. Built with 150 lines of pure Python. No libraries. ✓
```

---

---

# 27. Homework

## Exercise 1 — Rewrite `__mul__` From Memory

Open [`engine.py`](engine.py). Delete the `__mul__` method (lines ~68–83).

Write it back WITHOUT looking at the original:
```python
def __mul__(self, other):
    # 1. Wrap other if it's a plain number
    # 2. Create out = Value(...)
    # 3. Define _backward that updates self.grad and other.grad
    # 4. Attach _backward to out
    # 5. Return out

    # Remember:
    # z = a × b
    # dz/da = b  → a.grad += z.grad × b.data
    # dz/db = a  → b.grad += z.grad × a.data
```

You have mastered backprop when you can do this without looking.

## Exercise 2 — Learning Rate Exploration

In `step2_one_neuron.py`, change:
- `learning_rate = 1.0` → what happens?
- `learning_rate = 0.001` → how many steps to get close to 1.0?
- `learning_rate = 0.1` → how does it compare to 0.05?

## Exercise 3 — Add a 5th Sample

In `step3_full_network.py`, add:
```python
X.append([0.0, 0.0, 0.0])
y_true.append(1.0)
```
Does the network still reach 100%? Why might it take more or fewer steps?

## Exercise 4 — Manual Backward Trace

For this expression:
```python
a = Value(3.0)
b = Value(2.0)
c = a * b        # c = 6
d = c ** 2       # d = 36
e = d + a        # e = 39
```

On paper, trace `e.backward()` step by step.
Find `a.grad`, `b.grad`, `c.grad`, `d.grad`.

Then verify with code:
```python
e.backward()
print(a.grad, b.grad, c.grad, d.grad)
```

---

---

# 28. Micrograd → PyTorch

```
MICROGRAD                          PYTORCH
───────────────────────────────────────────────────────────────────────
Value(3.0)                   →   torch.tensor(3.0, requires_grad=True)
a + b                        →   a + b                    (identical)
a * b                        →   a * b                    (identical)
a ** 2                       →   a ** 2                   (identical)
a.tanh()                     →   torch.tanh(a)
a.relu()                     →   torch.relu(a)
loss.backward()              →   loss.backward()           (identical)
p.grad                       →   p.grad                    (identical)
p.data -= lr * p.grad        →   with torch.no_grad():
                                     p -= lr * p.grad
model.parameters()           →   model.parameters()        (identical)
model.zero_grad()            →   optimizer.zero_grad()     (same idea)
._backward closure           →   autograd Function         (same concept)
one number at a time         →   millions of numbers at once (tensor)
```

**The only real difference:** PyTorch operates on tensors (matrices, batches).
The algorithm is identical. The math is identical. The concept is identical.

Once you understand micrograd, PyTorch is just micrograd with:
- Tensors instead of scalars (batch processing)
- CUDA kernels (GPU acceleration)
- Optimized C++ backend (speed)
- Pre-built layers (Conv2d, LSTM, Transformer, etc.)

---

---

# 29. Complete Glossary

| Term | Plain English Definition |
|------|--------------------------|
| **Weight** | A learnable number (knob). Starts random, adjusted during training. |
| **Bias** | Extra learnable offset added after dot product. Like `b` in `y=mx+b`. |
| **Parameter** | Any learnable value = weight or bias. `model.parameters()` returns all. |
| **Gradient** | The slope of the loss w.r.t. a weight. Tells you which direction reduces loss. |
| **grad** | Short for gradient. Stored in `.grad` on every Value/tensor. |
| **Backpropagation** | Algorithm computing ALL gradients at once, flowing from output to inputs. |
| **Chain rule** | `d(f∘g)/dx = f'(g(x)) × g'(x)`. How gradients multiply at each node. |
| **Forward pass** | Running input through network left to right. Produces prediction. |
| **Backward pass** | Running gradient right to left. Fills `.grad` for every parameter. |
| **Loss** | One number: how wrong are we? Goal: minimize to 0. |
| **MSE** | Mean Squared Error: `(pred - target)²`. Common regression loss. |
| **Hinge loss** | `max(0, 1 - y×pred)`. Common classification loss. 0 when correct and confident. |
| **Gradient descent** | Iterative weight update: `w = w - lr × gradient`. Moves downhill. |
| **Learning rate** | Step size for gradient descent. Too big → bounces. Too small → slow. |
| **zero_grad()** | Reset all `.grad` to 0 before each backward. Prevents accumulation. |
| **Epoch / Step** | One full cycle: forward + loss + zero_grad + backward + update. |
| **Activation function** | Non-linearity applied after dot product (tanh, relu). Prevents collapse to linear. |
| **tanh** | Activation: maps to (-1, 1). Smooth, centered. Used in our MLP hidden layers. |
| **ReLU** | Activation: `max(0, x)`. Fast. Can die (zero gradient for negative inputs). |
| **Linear activation** | No activation function. Used in output layer for regression. |
| **Dying ReLU** | Neuron with always-negative inputs → always-zero gradient → never learns. |
| **Neuron** | Unit computing `activation(w·x + b)`. The building block of networks. |
| **Dot product** | `w1×x1 + w2×x2 + ... + wn×xn`. What a neuron computes before activation. |
| **Layer** | Many neurons reading the same input. Each learns a different pattern. |
| **MLP** | Multi-Layer Perceptron. Neurons organized in sequential layers. |
| **Computational graph** | Directed graph of Value nodes connected by operations. Backprop walks this. |
| **Topological sort** | Order nodes so each appears before its dependents. Required for backprop. |
| **Topological order** | Leaves first, output last. Reversed = output first, leaves last (for backprop). |
| **Autograd** | Automatic differentiation. Computes gradients automatically via chain rule. |
| **L2 regularization** | Penalty `λ×Σw²` added to loss. Prevents weights from growing too large. |
| **Overfitting** | Network memorizes training data but fails on new data. |
| **Leaf node** | Value created directly (not by an operation). e.g. `Value(2.0)`. Inputs and weights. |
| **_backward closure** | A function stored per node that pushes gradient one step back. |
| **Accumulation** | Gradients use `+=` not `=` because a value can be used multiple times. |
| **PyTorch** | Production autograd framework. Same concept as micrograd, at massive scale. |
| **Tensor** | A multi-dimensional array. PyTorch's `Value` — but for batches of numbers. |
| **Transformer** | Architecture using self-attention. Powers GPT, Claude, Gemini. Next project. |
| **ReAct** | Reasoning + Acting. Agent pattern: Thought → Action → Observation loop. |
| **RAG** | Retrieval-Augmented Generation. Fetch documents, feed to LLM for answers. |
| **Embedding** | Vector representing meaning of text. Similar meaning → similar vectors. |

---

---

# 30. Resources & References

## Core Learning Resources

| Resource | Format | Priority | Link |
|----------|--------|----------|------|
| Micrograd lecture (Karpathy) | Video 2h | **MUST WATCH** | `youtube.com/watch?v=VMj-3S1tku0` |
| Micrograd source code | Python ~150 lines | **MUST READ** | `github.com/karpathy/micrograd` |
| GPT from scratch (Karpathy) | Video 4h | After micrograd | `youtube.com/watch?v=kCc8FmEb1nY` |
| nanoGPT | Python | After GPT video | `github.com/karpathy/nanoGPT` |
| Attention Is All You Need | Paper | After nanoGPT | `arxiv.org/abs/1706.03762` |
| ReAct paper | Paper | After Transformer | `arxiv.org/abs/2210.03629` |

## Files in This Project

```
micrograd/
├── engine.py                ← The autograd engine (150 lines, study this)
├── nn.py                    ← Neuron, Layer, MLP built on engine
├── step1_just_values.py     ← Run first: gradients and chain rule
├── step2_one_neuron.py      ← Run second: training a single neuron
├── step3_full_network.py    ← Run third: full MLP, 100% accuracy
├── explainer.py             ← Interactive explainer with question saving
├── AI_ENGINEERING_NOTES.md  ← Session Q&A notes
├── MINDMAPS.md              ← Visual mindmaps
├── DEEP_EXPLANATION.md      ← Concept deep dives
└── REPORT.md (this file)    ← The complete master report
```

## How to Run Everything

```powershell
cd "d:\test\account rotate\account_rotator\micrograd"
$env:PYTHONIOENCODING='utf-8'

# Step-by-step lessons:
python step1_just_values.py
python step2_one_neuron.py
python step3_full_network.py

# Interactive explainer with question saving:
python explainer.py
```

## Weekly Study Schedule

| Week | Task | Daily Time |
|------|------|-----------|
| 1 | Watch Karpathy micrograd video | 2h |
| 2 | Rewrite micrograd from a blank file | 2h |
| 3–4 | Watch Karpathy GPT video, code along (pause often) | 2h |
| 5–6 | Train nanoGPT on a small text dataset | 1–2h |
| 7 | Read ReAct paper, build basic tool loop with any LLM | 2h |
| 8 | Add 3 real tools (search, calculator, code runner) | 2h |

---

*Report compiled: 2026-09-06 · Next milestone: Transformer from scratch (Week 3)*

<!-- watcher-test 20:53:50 -->
