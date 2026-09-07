# AI Engineering — Explained in Micro Detail
> For someone starting from absolute zero. No assumptions.

---

# PART 1: What is "Learning" in a Neural Network?

## The REAL question

Before any code, answer this:

> If you adjust a knob on a machine, and the machine gets slightly better at its job —
> how do you know WHICH WAY to turn the knob?

That's the entire problem. Neural networks have **millions of knobs** (called weights).
Micrograd tells you **which way to turn each one**, automatically.

---

## Start with the simplest possible "machine"

Forget neural networks. Forget AI. Look at this:

```
y = 2 * x
```

If `x = 3`, then `y = 6`.

**Question:** If I want `y` to be `10` instead of `6`, should I make `x` bigger or smaller?

**Answer:** Bigger. Because `y = 2*x`, so more `x` → more `y`.

That intuition — "which direction to change `x` to change `y`" — is a **gradient**.

```
dy/dx = 2      ← the gradient
```

It says: "for every 1 unit increase in x, y increases by 2."

This is all calculus is. You learned the core idea. Now scale it to millions of variables.

---

# PART 2: What is a Gradient? (With NO math symbols)

## The slope of a hill

Imagine you're standing on a hillside.

```
        /\
       /  \
      /    \
_____/      \_____
```

- If you're on the LEFT side: the ground rises as you go right → **positive slope**
- If you're on the RIGHT side: the ground falls as you go right → **negative slope**
- At the BOTTOM of the valley: flat → **slope = 0**

The **gradient** is just: "how steep is the ground right where I'm standing, and which direction does it slope?"

In a neural network:
- The "hill" is the **loss function** (how wrong the network is)
- Your position on the hill is the **current weight value**
- The gradient tells you: **which direction to move the weight to go downhill** (reduce loss)

---

## The gradient in code (step by step)

```python
a = Value(3.0)   # a = 3
b = a * a        # b = a² = 9

b.backward()

print(a.grad)    # prints: 6.0
```

Why `6.0`?

```
b = a²
db/da = 2a = 2 * 3 = 6
```

In English: "If `a` increases by 0.001, then `b` increases by 0.001 * 6 = 0.006"

Let's verify:
```python
# Before: a=3.0, b=9.0
# After nudging a by 0.001: a=3.001, b=3.001² = 9.006001
# Change in b: 9.006001 - 9.0 = 0.006001 ≈ 0.001 * 6 ✓
```

---

# PART 3: What is the `Value` class? (Line by line)

Open [`engine.py`](engine.py). Let's go through every part.

## Part 3a: `__init__`

```python
def __init__(self, data, _children=(), _op='', label=''):
    self.data = float(data)    # the actual number stored here
    self.grad = 0.0            # gradient starts at 0 — we haven't computed it yet
    self._backward = lambda: None  # empty function — filled in by each operation
    self._children = set(_children)  # what Values were used to create this one?
    self._op = _op             # just for debugging: '+', '*', 'tanh', etc.
    self.label = label         # optional name: 'w1', 'x', etc.
```

When you write:
```python
a = Value(3.0, label='a')
```

You get an object that:
- Holds the number `3.0`
- Has a gradient of `0.0` (unknown yet)
- Remembers nothing (no children, no operation)

```
a
┌──────────────────┐
│ data = 3.0       │
│ grad = 0.0       │
│ _children = {}   │
│ _op = ''         │
└──────────────────┘
```

---

## Part 3b: `__mul__` — multiplication

```python
def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), '*')
```

**Line 1:** If you write `a * 3`, the `3` is a plain Python int.
We wrap it in `Value(3)` so we can compute gradients through it.

**Line 2:** Create a new `Value` whose:
- `data` = `a.data * other.data` (forward pass — actual multiplication)
- `_children` = `(self, other)` — it remembers its parents!
- `_op` = `'*'` — for debugging

So after `c = a * b`:
```
    a           b
┌────────┐  ┌────────┐
│ data=2 │  │ data=3 │
└────┬───┘  └───┬────┘
     │           │
     └─────┬─────┘
           ↓  op='*'
        c
   ┌────────────┐
   │ data = 6   │
   │ grad = 0.0 │
   │ children={a,b} │
   └────────────┘
```

Now the `_backward` function:

```python
    def _backward():
        self.grad  += out.grad * other.data
        other.grad += out.grad * self.data
    out._backward = _backward
```

This is the **derivative of multiplication**:

```
If c = a * b
Then:
  dc/da = b    (how much does c change if a changes? → by b)
  dc/db = a    (how much does c change if b changes? → by a)
```

But we don't want `dc/da`, we want `d(Loss)/da`. That's where `out.grad` comes in.

**The Chain Rule:**
```
d(Loss)/da = d(Loss)/dc * dc/da
           = out.grad   *   b
```

So: `self.grad += out.grad * other.data`

**The `+=` is important:**
A variable can be used multiple times in a computation.
Each time it's used, the gradient **accumulates** (adds up).
```python
# e = a + a  (a used twice)
# de/da = 1 + 1 = 2
# Each path adds 1 to a.grad
```

---

## Part 3c: `backward()` — the main function

```python
def backward(self):
    topo_order = []
    visited = set()

    def build_topo(node):
        if id(node) not in visited:
            visited.add(id(node))
            for child in node._children:
                build_topo(child)
            topo_order.append(node)

    build_topo(self)
    self.grad = 1.0

    for node in reversed(topo_order):
        node._backward()
```

### Why do we need "topological order"?

Look at this graph:
```
  a    b
   \  /
    c = a*b
    |
    d = c + 1
    |
   Loss
```

To compute `d(Loss)/da`, we need:
1. First compute `d(Loss)/dd`
2. Then `d(Loss)/dc` using `d(Loss)/dd`
3. Then `d(Loss)/da` using `d(Loss)/dc`

**We must go from output → inputs.** Each node needs its OWN gradient computed
before it can compute its children's gradients.

Topological sort guarantees this order.

### Step by step through `backward()`:

**Step 1: `build_topo(self)`**

Visits all nodes using recursion (depth-first search).
Appends each node AFTER visiting its children.

Result: `[a, b, c, d, Loss]` (leaves first, output last)

**Step 2: `self.grad = 1.0`**

`d(Loss)/d(Loss) = 1` — always. By definition.
We differentiate Loss with respect to itself. It's 1.

**Step 3: `for node in reversed(topo_order):`**

Walk backwards: `Loss → d → c → b → a`

At each node, call `node._backward()` which pushes the gradient one step back.

---

# PART 4: What is a Neuron?

## The biology (simplified)

A real neuron in your brain:
- Receives signals from many other neurons (inputs)
- If the total signal is strong enough, it fires (activates)
- Sends its signal forward to the next neurons

An artificial neuron does the same thing with numbers.

## The math of one neuron

```
inputs:   x1=2.0, x2=3.0
weights:  w1=0.5, w2=-0.3   ← these are LEARNED
bias:     b=0.1             ← also learned

step 1: dot product = (w1*x1) + (w2*x2) + b
                    = (0.5*2.0) + (-0.3*3.0) + 0.1
                    = 1.0 + (-0.9) + 0.1
                    = 0.2

step 2: activation  = tanh(0.2) = 0.197   ← squishes to (-1, 1)
```

## Why the activation function?

Without tanh/relu:
```
output = w1*x1 + w2*x2 + b
```
This is just a straight line. No matter how many neurons you stack,
the whole network is still just... a straight line.
Useless for complex patterns.

With tanh:
```
output = tanh(w1*x1 + w2*x2 + b)
```
Now it's curved. Stack enough of these → any shape possible.

```
tanh input:   ...-3  -2  -1   0   1   2   3...
tanh output:  ...-1  -0.96 -0.76  0  0.76  0.96  1...
             always stays between -1 and +1
```

relu is even simpler:
```
relu(x) = max(0, x)
        = x if x > 0
        = 0 if x <= 0
```

---

# PART 5: What is a Layer and MLP?

## Layer = many neurons reading the same input

```python
Layer(n_inputs=3, n_neurons=4)
```

```
Input: [x1, x2, x3]
         │   │   │
    ┌────┼───┼───┼────┐
    │  Neuron 1       │ → out1
    │  Neuron 2       │ → out2
    │  Neuron 3       │ → out3
    │  Neuron 4       │ → out4
    └─────────────────┘
```

Each neuron has its OWN weights. Same input, different learned responses.

## MLP = stack of layers

```python
model = MLP(n_inputs=3, layer_sizes=[4, 4, 1])
```

```
Input (3)
    │
Layer 1 — 4 neurons (hidden) — tanh activation
    │
Layer 2 — 4 neurons (hidden) — tanh activation
    │
Layer 3 — 1 neuron  (output) — linear (no activation)
    │
Output (single number)
```

The final number is the prediction.

---

# PART 6: What is Loss?

Loss = a single number that says **"how wrong are you right now?"**

## Mean Squared Error (MSE)

```
prediction = 0.3
target      = 1.0

loss = (prediction - target)²
     = (0.3 - 1.0)²
     = (-0.7)²
     = 0.49
```

After learning:
```
prediction = 0.98
target      = 1.0

loss = (0.98 - 1.0)² = 0.0004   ← much smaller!
```

Loss = 0 means perfect. We're trying to minimize it.

## Hinge Loss (used in step3)

```
loss per sample = max(0,  1 - y_true * y_pred)
```

- If `y_true = +1` and `y_pred = +2.0`:  `1 - (1 * 2.0) = -1 → max(0, -1) = 0`  (correct, no loss)
- If `y_true = +1` and `y_pred = -0.5`: `1 - (1 * -0.5) = 1.5 → max(0, 1.5) = 1.5`  (wrong, big loss)
- If `y_true = -1` and `y_pred = -1.5`: `1 - (-1 * -1.5) = 1 - 1.5 = -0.5 → max(0, -0.5) = 0` (correct)

---

# PART 7: The Training Loop — The Exact 4 Steps

Every training loop in every framework (PyTorch, TensorFlow, JAX) does these 4 steps:

```python
for step in range(100):

    # STEP 1: Forward pass — run the network, get predictions, compute loss
    y_pred = model(x)
    loss = (y_pred - y_true) ** 2

    # STEP 2: Zero gradients — CRITICAL, do this before every backward
    model.zero_grad()

    # STEP 3: Backward pass — compute ALL gradients automatically
    loss.backward()

    # STEP 4: Update weights — gradient descent
    for p in model.parameters():
        p.data -= learning_rate * p.grad
```

### Why zero_grad() matters

Gradients accumulate with `+=`. If you forget to zero them:

```
Step 1: w1.grad = 2.0
Step 2: w1.grad = 2.0 + 3.0 = 5.0  ← WRONG, should be 3.0
Step 3: w1.grad = 5.0 + 1.0 = 6.0  ← even more wrong
```

You'd be updating weights based on history of ALL previous steps, not just the current one.

### What is learning_rate?

It controls HOW BIG a step you take in the direction of the gradient.

```
learning_rate = 0.1

w1.grad = 4.0          ← gradient says: nudge this way
w1.data -= 0.1 * 4.0  ← take a small step: move by 0.4, not 4.0
```

Too big → overshoot, bounce around, never converge
Too small → takes forever

Typical values: `0.001` to `0.1`

---

# PART 8: How to Run the Code

## Setup

Open a terminal in the `micrograd/` folder:

```bash
# Windows PowerShell
$env:PYTHONIOENCODING='utf-8'   ← fixes arrow symbols (←) in output
```

## Run Step 1 — Just Values

```bash
python step1_just_values.py
```

**What it shows:**
- Two values `a=2, b=3`
- Their sum and product
- After `.backward()`, how gradients flow
- Manual math check that the gradients are correct

**Read the output carefully.** Every number should make sense from the math.

## Run Step 2 — One Neuron

```bash
python step2_one_neuron.py
```

**What it shows:**
- One neuron computing `tanh(w1*x1 + w2*x2 + b)`
- The exact gradient at each node
- A training loop showing the neuron learning to output ~1.0

**Watch the `output` column** — it starts near 0, climbs toward 1.0 as weights update.

## Run Step 3 — Full Network

```bash
python step3_full_network.py
```

**What it shows:**
- A 3-layer MLP with 41 total parameters
- Training on 4 samples (classify +1 or -1)
- Loss decreasing, accuracy climbing to 100%

```
Step  0:  loss=1.014010  accuracy=50%   ← random guessing
Step 20:  loss=0.219246  accuracy=100%  ← got it!
Step 40:  loss=0.005319  accuracy=100%  ← very confident now
```

---

# PART 9: Your Homework (In Micro Detail)

### Exercise 1 — Delete and rewrite `__mul__`

Open `engine.py`. Delete lines 68-83 (the `__mul__` method). Write it back:

```python
def __mul__(self, other):
    # 1. wrap other if it's not a Value
    # 2. create out = Value(self.data * other.data, ...)
    # 3. define _backward that updates self.grad and other.grad
    # 4. attach _backward to out
    # 5. return out
```

The derivative of `z = a * b` with respect to:
- `a` is `b`
- `b` is `a`

Don't look. Just write it.

### Exercise 2 — Reach 0.999 in step2

In `step2_one_neuron.py`, change the number of training steps.
Try `range(50)`, `range(100)`, `range(500)`.

What happens when you set `learning_rate = 1.0`?
What happens with `learning_rate = 0.001`?

### Exercise 3 — Add a 5th sample to step3

In `step3_full_network.py`, add:
```python
X.append([0.0, 0.0, 0.0])
y_true.append(1.0)
```

Does the network still reach 100% accuracy?
Why might it take more or fewer steps?

### Exercise 4 — Draw the graph on paper

For this expression:
```python
a = Value(2.0)
b = Value(3.0)
c = a * b        # c = 6
d = a + b        # d = 5
e = c + d        # e = 11
```

Draw boxes and arrows on paper:
```
a → [*] → c
b ↗      ↘
          [+] → e
a → [+] → d
b ↗
```

Now trace `.backward()` by hand:
- `e.grad = 1.0`
- `c.grad += e.grad * 1 = 1.0`  (addition passes gradient straight through)
- `d.grad += e.grad * 1 = 1.0`
- `a.grad += c.grad * b.data = 1.0 * 3.0 = 3.0`  (from the * branch)
- `b.grad += c.grad * a.data = 1.0 * 2.0 = 2.0`
- `a.grad += d.grad * 1 = 1.0`  (from the + branch, accumulates!)
- `b.grad += d.grad * 1 = 1.0`

Final:
- `a.grad = 3.0 + 1.0 = 4.0` ✓
- `b.grad = 2.0 + 1.0 = 3.0` ✓

---

# PART 10: Vocabulary Glossary

| Term | Plain English meaning |
|------|-----------------------|
| **Weight** | A knob the network adjusts to get better. Starts random, learned via training. |
| **Bias** | An extra adjustable offset. Like the y-intercept in y=mx+b. |
| **Gradient** | The slope — tells you which direction to move a weight to reduce loss. |
| **Backpropagation** | The algorithm that computes gradients for all weights at once, starting from the loss and working backwards. |
| **Chain rule** | How gradients pass through operations: multiply the local gradient by the incoming gradient. |
| **Forward pass** | Running input through the network to get a prediction. Left → right. |
| **Backward pass** | Computing gradients from loss back to inputs. Right → left. |
| **Loss** | A single number: how wrong is the network right now? Lower = better. |
| **Gradient descent** | Updating weights by moving in the direction that reduces loss: `w -= lr * grad`. |
| **Learning rate** | How big a step to take each update. Too big = bounces. Too small = slow. |
| **Epoch / Step** | One full pass through: forward + backward + update. |
| **zero_grad()** | Reset all gradients to 0 before each step. CRITICAL. |
| **Activation function** | Adds non-linearity (tanh, relu) so the network can learn curves, not just lines. |
| **MLP** | Multi-Layer Perceptron — the simplest kind of neural network (stacked layers of neurons). |
| **Parameter** | Any learnable value (weights + biases). `model.parameters()` returns all of them. |
| **Overfitting** | Network memorizes training data but fails on new data. |
| **L2 Regularization** | Penalty on large weights (added to loss) to prevent overfitting. `+= 0.0001 * w²` |
| **Topological sort** | Ordering nodes so each node comes before its parents. Needed for correct backprop order. |

---

*Last updated: 2026-09-06*
*Next: Build the Transformer from scratch*
