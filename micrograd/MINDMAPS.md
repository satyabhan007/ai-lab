# Micrograd — Mindmaps & Examples
> Read this alongside the explainer. Add your comments below each section.

---

## MINDMAP 1 — The Big Picture

```
                        NEURAL NETWORK LEARNING
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
         FORWARD PASS        LOSS FUNCTION       BACKWARD PASS
         (prediction)        (how wrong?)         (fix it)
              │                   │                   │
         Run input           Compare output      Compute gradients
         through all         to true answer      for every weight
         layers              (MSE, Hinge...)     (backpropagation)
              │                   │                   │
         Get a number         Get one number      Get a gradient
         (prediction)         (e.g. 0.73)         per weight
                                                       │
                                               GRADIENT DESCENT
                                               weight -= lr × grad
```

**In plain English:**
> Forward = "What do I predict?"
> Loss = "How wrong am I?"
> Backward = "Which weights caused the mistake?"
> Update = "Adjust those weights slightly"

---

## MINDMAP 2 — The Value Object

```
                         Value(3.0)
                              │
            ┌─────────────────┼─────────────────┐
            │                 │                 │
          .data             .grad           ._backward
          = 3.0             = 0.0           = empty fn
          (the number)    (unknown yet)   (filled when created
                                           by an operation)
                                                 │
                                    ┌────────────┴────────────┐
                                    │                         │
                              If created by            If created by
                               a = Value(3)              c = a * b
                               (leaf node)
                              _backward = do nothing   _backward stores:
                                                       a.grad += c.grad × b.data
                                                       b.grad += c.grad × a.data
```

**Example — What happens when you write `c = a + b`:**

```python
a = Value(2.0)    #  data=2.0  grad=0.0  children={}
b = Value(3.0)    #  data=3.0  grad=0.0  children={}
c = a + b         #  data=5.0  grad=0.0  children={a,b}  op='+'

# c._backward = function that does:
#   a.grad += c.grad × 1
#   b.grad += c.grad × 1
```

---

## MINDMAP 3 — Gradient Flow Through Operations

```
ADDITION:   z = a + b
            ─────────
            Forward:  z.data = a.data + b.data
            Backward: a.grad += z.grad × 1     ← gradient passes unchanged
                      b.grad += z.grad × 1

            EXAMPLE:
            a=2, b=3, z=5, z.grad=1.0
            → a.grad = 1.0  b.grad = 1.0


MULTIPLICATION: z = a × b
            ──────────────
            Forward:  z.data = a.data × b.data
            Backward: a.grad += z.grad × b.data   ← flip! use other's VALUE
                      b.grad += z.grad × a.data

            EXAMPLE:
            a=2, b=3, z=6, z.grad=1.0
            → a.grad = 1.0 × 3 = 3.0
            → b.grad = 1.0 × 2 = 2.0


POWER:      z = aⁿ
            ────────
            Forward:  z.data = a.data ** n
            Backward: a.grad += z.grad × n × a.data^(n-1)

            EXAMPLE: z = a², a=3, z.grad=1.0
            → a.grad = 1.0 × 2 × 3^1 = 6.0


TANH:       z = tanh(a)
            ─────────────
            Forward:  z.data = tanh(a.data)
            Backward: a.grad += z.grad × (1 - z.data²)

            EXAMPLE: a=0.5, tanh(0.5)=0.462, z.grad=1.0
            → a.grad = 1.0 × (1 - 0.462²) = 1.0 × 0.786 = 0.786


RELU:       z = max(0, a)
            ──────────────
            Forward:  z.data = max(0, a.data)
            Backward: a.grad += z.grad  if a.data > 0
                      a.grad += 0       if a.data ≤ 0

            EXAMPLE A: a=2.0, z=2.0, z.grad=1.0
            → a.grad = 1.0  (positive, gradient flows)

            EXAMPLE B: a=-1.0, z=0.0, z.grad=1.0
            → a.grad = 0.0  (negative, gradient blocked)
```

---

## MINDMAP 4 — Backpropagation Step by Step

```
COMPUTATION:  e = (a × b) + (a + b)   with a=2, b=3

GRAPH:
      a=2  b=3
       │    │
       └─[×]┘ ──→ c=6
       │    │
       └─[+]┘ ──→ d=5
                    │
               c+d=[+] ──→ e=11

FORWARD (left to right):
  c = 2×3 = 6
  d = 2+3 = 5
  e = 6+5 = 11

BACKWARD (right to left):
  Step 0:  e.grad = 1.0              (seed — always 1)
  Step 1:  c.grad += e.grad×1 = 1.0  (+ passes grad unchanged)
           d.grad += e.grad×1 = 1.0
  Step 2:  a.grad += c.grad×b = 3.0  (× rule: use other's value)
           b.grad += c.grad×a = 2.0
  Step 3:  a.grad += d.grad×1 = 1.0  (+ rule: passes grad unchanged)
           b.grad += d.grad×1 = 1.0

FINAL GRADIENTS:
  a.grad = 3.0 + 1.0 = 4.0   ← de/da = b+1 = 3+1 = 4 ✓
  b.grad = 2.0 + 1.0 = 3.0   ← de/db = a+1 = 2+1 = 3 ✓
```

---

## MINDMAP 5 — What a Neuron Computes

```
                    INPUTS          WEIGHTS (learned)
                 x1 = 2.0  ───×───  w1 = 0.5  ──→  1.0
                 x2 = 3.0  ───×───  w2 = -0.3 ──→ -0.9
                 x3 = -1.0 ───×───  w3 = 0.8  ──→ -0.8
                                                     │
                                    bias b=0.1  ──→  0.1
                                                     │
                                              SUM = -0.6
                                                     │
                                              tanh(-0.6)
                                                     │
                                            OUTPUT = -0.537
```

**Real-world analogy:**
```
Deciding whether to bring an umbrella:

Input 1: Is it cloudy?       (x1 = 0.8)  weight: HIGH importance (w1 = 2.0)
Input 2: Is it hot outside?  (x2 = 0.3)  weight: LOW importance (w2 = 0.1)
Input 3: Was yesterday rainy? (x3 = 0.9) weight: MEDIUM (w3 = 1.0)

Dot product = 0.8×2.0 + 0.3×0.1 + 0.9×1.0 = 1.6 + 0.03 + 0.9 = 2.53
tanh(2.53) = 0.987  → bring umbrella!

The WEIGHTS are what the neuron learned from past experience.
Training adjusts the weights so the neuron makes better decisions.
```

---

## MINDMAP 6 — Layer and MLP Architecture

```
MLP(n_inputs=3, layer_sizes=[4, 4, 1])

INPUT LAYER            HIDDEN LAYER 1       HIDDEN LAYER 2     OUTPUT
(your data)            (4 neurons)          (4 neurons)        (1 neuron)

[x1]──────────────→  [N1]─┐             [N5]─┐
[x2]──────────────→  [N2]─┤──→ 4 vals──→[N6]─┤──→ 4 vals──→  [N9]──→ prediction
[x3]──────────────→  [N3]─┤             [N7]─┤
                     [N4]─┘             [N8]─┘

Each Ni has its own weights.
N1: reads [x1,x2,x3] → learns one pattern
N2: reads [x1,x2,x3] → learns different pattern
...

Parameter count:
  Layer 1: 4 neurons × (3 inputs + 1 bias) = 16 parameters
  Layer 2: 4 neurons × (4 inputs + 1 bias) = 20 parameters
  Layer 3: 1 neuron  × (4 inputs + 1 bias) =  5 parameters
  TOTAL: 41 parameters
```

---

## MINDMAP 7 — Loss Functions

```
LOSS = a single number measuring: "how wrong are we?"

         PREDICTION vs TRUTH
               │
      ┌────────┴────────┐
      │                 │
  REGRESSION         CLASSIFICATION
  (predict a number)  (predict a category)
      │                 │
  Mean Squared Error   Hinge Loss
  loss=(pred-target)²  loss=max(0, 1-y×pred)
      │                 │
  Example:             Example:
  pred=0.3, target=1   pred=+0.8, true=+1:
  loss=(0.3-1)²=0.49   loss=max(0,1-1×0.8)=0.2

                        pred=-0.8, true=+1:  ← wrong direction
                        loss=max(0,1-1×-0.8)=1.8  ← big loss!

                        pred=+1.5, true=+1:  ← correct AND confident
                        loss=max(0,1-1×1.5)=0  ← no loss at all
```

---

## MINDMAP 8 — Gradient Descent Visualized

```
LOSS vs WEIGHT VALUE

    Loss
     │
   5 │   *                           *
   4 │     *                       *
   3 │       *                   *
   2 │         *               *
   1 │           *           *
   0 │              *─────*
     └──────────────────────────────── weight value
     -3   -2   -1    0    1    2    3
                     ↑
                  MINIMUM (best weight)

If we're at weight = -2.0:
  gradient = negative  → slope going down to the right
  → move RIGHT (increase weight)
  → w = w - lr × (negative) = w + something  ← increases w ✓

If we're at weight = +2.0:
  gradient = positive  → slope going down to the left
  → move LEFT (decrease weight)
  → w = w - lr × (positive) = w - something  ← decreases w ✓

If we're at weight = 0.0:
  gradient = 0  → flat, we're at the minimum
  → no update
  → w = w - lr × 0 = w  ← stays the same ✓
```

---

## MINDMAP 9 — The 4-Step Training Loop

```
                    START: random weights
                           │
              ┌────────────▼────────────┐
              │     STEP 1: FORWARD     │
              │                         │
              │   prediction = model(x) │
              │   loss = (pred-y)²      │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │   STEP 2: ZERO GRADS    │
              │                         │
              │   for each param:       │
              │     param.grad = 0      │   ← CRITICAL: prevents accumulation
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │    STEP 3: BACKWARD     │
              │                         │
              │   loss.backward()       │   ← chain rule, automatic
              │   (fills all .grad)     │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │    STEP 4: UPDATE       │
              │                         │
              │   for each param:       │
              │     p.data -= lr×p.grad │   ← gradient descent
              └────────────┬────────────┘
                           │
                     loss small enough?
                      YES          NO
                       │            │
                     DONE      go back to STEP 1
```

---

## MINDMAP 10 — How This Connects to PyTorch

```
Micrograd                          PyTorch equivalent
─────────────────────────────────────────────────────────
Value(3.0)                     torch.tensor(3.0, requires_grad=True)
a + b                          a + b                (same!)
a * b                          a * b                (same!)
a.tanh()                       torch.tanh(a)
a.relu()                       torch.relu(a)
loss.backward()                loss.backward()      (same!)
p.grad                         p.grad               (same!)
p.data -= lr*p.grad            with torch.no_grad():
                                   p -= lr*p.grad
model.parameters()             model.parameters()   (same!)
model.zero_grad()              optimizer.zero_grad() (same idea)
Value._backward closure        autograd Function     (same idea)
scalar computation             tensor computation
                               (batches of numbers at once)
```

**The ONLY real difference:** PyTorch operates on Tensors (matrices of numbers)
instead of single scalars. The concept is identical.

---

## CONCRETE EXAMPLES — Each Concept

---

### Example A: Gradient of a Simple Function

**Problem:** `y = 3x² + 2x + 1`. What is `dy/dx` at `x=4`?

**By calculus:** `dy/dx = 6x + 2 = 6×4 + 2 = 26`

**By micrograd:**
```python
x = Value(4.0)
y = 3*x**2 + 2*x + 1
y.backward()
print(x.grad)   # 26.0 ✓
```

**Verify by nudging:**
```
x=4.000: y = 3(16) + 2(4) + 1 = 57
x=4.001: y = 3(16.008) + 2(4.001) + 1 = 57.026
change in y / change in x = 0.026 / 0.001 = 26.0 ✓
```

---

### Example B: Two Weights, Loss, Gradient

**Problem:** Train `output = w1*2 + w2*3` to equal `10`

```python
w1 = Value(1.0)   # initial guess
w2 = Value(1.0)

# With w1=1, w2=1: output = 2+3 = 5. Target = 10.
# Loss = (5-10)² = 25. Too high.

for step in range(20):
    output = w1*2 + w2*3
    loss = (output - 10) ** 2

    w1.grad = 0; w2.grad = 0
    loss.backward()

    w1.data -= 0.05 * w1.grad
    w2.data -= 0.05 * w2.grad

# After training: output ≈ 10, loss ≈ 0
```

**What happens inside step 0:**
```
output = w1×2 + w2×3 = 5.0
loss   = (5.0 - 10.0)² = 25.0

loss.backward():
  dloss/d(output) = 2×(output-target) = 2×(5-10) = -10
  dloss/d(w1)     = dloss/d(output) × d(output)/d(w1)
                  = -10 × 2 = -20
  dloss/d(w2)     = -10 × 3 = -30

update:
  w1 = 1.0 - 0.05×(-20) = 1.0 + 1.0 = 2.0
  w2 = 1.0 - 0.05×(-30) = 1.0 + 1.5 = 2.5

new output = 2.0×2 + 2.5×3 = 4.0 + 7.5 = 11.5  ← overshot! but learning
```

---

### Example C: Why zero_grad() Matters

```python
w = Value(1.0)

# WRONG — without zero_grad():
for step in range(3):
    loss = w * 2
    loss.backward()
    print(f"Step {step}: w.grad = {w.grad}")
    # Step 0: w.grad = 2.0  ← correct
    # Step 1: w.grad = 4.0  ← WRONG: should be 2.0, but accumulated
    # Step 2: w.grad = 6.0  ← getting worse

# RIGHT — with zero_grad():
w = Value(1.0)
for step in range(3):
    loss = w * 2
    w.grad = 0.0           ← reset before each backward
    loss.backward()
    print(f"Step {step}: w.grad = {w.grad}")
    # Step 0: w.grad = 2.0 ✓
    # Step 1: w.grad = 2.0 ✓
    # Step 2: w.grad = 2.0 ✓
```

---

### Example D: Learning Rate Effect

```
Same problem: minimize loss = (w*2 - 10)²
Starting at w=0

Learning rate = 0.5  (TOO LARGE):
  Step 0: w=0    → loss=100  → update → w=10
  Step 1: w=10   → loss=100  → update → w=0   ← bouncing!
  Step 2: w=0    → loss=100  → never converges

Learning rate = 0.05  (GOOD):
  Step 0: w=0.0  → loss=100.0
  Step 1: w=1.0  → loss=64.0
  Step 2: w=1.8  → loss=40.96
  ...
  Step 20: w≈4.9 → loss≈0.01  ✓

Learning rate = 0.001  (TOO SMALL):
  Step 0: w=0.0  → loss=100.0
  Step 10: w=0.2 → loss=92.16  ← barely moving
  Would need 1000+ steps to converge
```

---

### Example E: tanh vs ReLU in a Neuron

```
Input: x = -2.0
Weights: w=1.0, b=0.0
Pre-activation: w×x + b = -2.0

With tanh:   output = tanh(-2.0) = -0.964   ← negative but alive
  gradient:  1 - (-0.964)² = 0.071          ← small but non-zero ✓

With ReLU:   output = max(0, -2.0) = 0.0    ← completely zero
  gradient:  0.0                             ← DEAD! no gradient flows

This is the "dying ReLU" problem.
A neuron with negative input becomes permanently dead with ReLU.
tanh never fully dies.
```

---

## YOUR QUESTIONS LOG
> Write your questions here as you read. Or type them in the chat directly.

```
Q1: ___________________________________________________________

Q2: ___________________________________________________________

Q3: ___________________________________________________________

Q4: ___________________________________________________________

Q5: ___________________________________________________________
```

---

## WHAT TO DO NEXT

**Step 1:** Read Mindmap 1 → understand the big loop

**Step 2:** Read Mindmap 4 → trace the backward pass on paper

**Step 3:** Read Example A → run it in Python yourself

**Step 4:** Read Example C → understand why zero_grad() exists

**Step 5:** Run the interactive explainer:
```powershell
cd "d:\test\account rotate\account_rotator\micrograd"
$env:PYTHONIOENCODING='utf-8'
python explainer.py
```

**Step 6:** Ask any unclear concept directly in the chat. Reference the mindmap
number (e.g. "Mindmap 4 — why does a.grad accumulate with +=?")

---

*Last updated: 2026-09-06*

<!-- watcher test: 20:49:01 -->


<!-- auto-push test: 20:51:41 -->
