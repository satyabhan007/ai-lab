"""
micrograd/step2_one_neuron.py
=============================
LESSON 2: What does ONE neuron actually do?

A neuron is just:
  output = activation(w1*x1 + w2*x2 + ... + b)

We'll build this manually first (no nn.py),
then verify it with nn.Neuron.

Key question: What does the gradient of the LOSS w.r.t. the WEIGHT mean?
"""

import sys, os, math
sys.path.insert(0, os.path.dirname(__file__))
from engine import Value


print("=" * 60)
print("STEP 2: One neuron, manual computation")
print("=" * 60)

# ── Inputs (these are data, NOT learned)
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')

# ── Weights (these ARE learned — they start random, we improve them)
w1 = Value(-3.0, label='w1')
w2 = Value(1.0,  label='w2')

# ── Bias (also learned)
b  = Value(6.8813735870195432, label='b')   # chosen so tanh gives nice output

print(f"\nInputs:  x1={x1.data}, x2={x2.data}")
print(f"Weights: w1={w1.data}, w2={w2.data}")
print(f"Bias:    b={b.data:.4f}")

# ── Forward pass: dot product
x1w1 = x1 * w1          # x1*w1
x2w2 = x2 * w2          # x2*w2
x1w1x2w2 = x1w1 + x2w2  # x1*w1 + x2*w2
n = x1w1x2w2 + b         # + bias = the "pre-activation"
o = n.tanh()             # apply tanh activation = neuron output

print(f"\nForward pass:")
print(f"  x1*w1 = {x1w1.data:.4f}")
print(f"  x2*w2 = {x2w2.data:.4f}")
print(f"  dot + bias (n) = {n.data:.4f}")
print(f"  tanh(n) = output (o) = {o.data:.4f}")

# ── Backward pass
o.backward()

print(f"\nBackward pass (gradients):")
print(f"  do/dn  = 1 - tanh(n)^2 = {1 - o.data**2:.4f}")
print(f"  w1.grad = {w1.grad:.4f}  ← if w1 increases by 0.001, output changes by {w1.grad*0.001:.6f}")
print(f"  w2.grad = {w2.grad:.4f}")
print(f"  x1.grad = {x1.grad:.4f}")
print(f"  x2.grad = {x2.grad:.4f}")
print(f"  b.grad  = {b.grad:.4f}")


print("\n" + "=" * 60)
print("STEP 2b: The gradient is a DIRECTION TO IMPROVE")
print("=" * 60)

print("""
Think of gradient descent as hill-climbing in REVERSE:
  - If w1.grad is POSITIVE → increasing w1 INCREASES the output
  - To MINIMIZE loss, we move w1 in the NEGATIVE gradient direction
  - Update rule: w1 = w1 - learning_rate * w1.grad
""")

print("Before update:")
print(f"  w1 = {w1.data:.4f},  w1.grad = {w1.grad:.4f}")

learning_rate = 0.1
w1.data -= learning_rate * w1.grad
w2.data -= learning_rate * w2.grad
b.data  -= learning_rate * b.grad

print(f"After one gradient descent step (lr={learning_rate}):")
print(f"  w1 = {w1.data:.4f}")
print(f"  w2 = {w2.data:.4f}")
print(f"  b  = {b.data:.4f}")


print("\n" + "=" * 60)
print("STEP 2c: A complete training loop on ONE neuron")
print("=" * 60)
print("""
Goal: Train a single neuron to output ~1.0 when input is [2.0, 3.0]

Loss = (output - target)^2   ← Mean Squared Error for 1 sample
""")

# Fresh weights
w1 = Value(0.5, label='w1')
w2 = Value(-0.3, label='w2')
b  = Value(0.0, label='b')

x = [Value(2.0), Value(3.0)]   # input
target = 1.0                    # what we want the neuron to output

print(f"{'Step':>5}  {'output':>10}  {'loss':>10}  {'w1':>8}  {'w2':>8}")
print("-" * 50)

for step in range(20):
    # Forward pass
    o = (w1 * x[0] + w2 * x[1] + b).tanh()

    # Loss: how far are we from target?
    loss = (o - target) ** 2

    # Zero gradients before backward (critical!)
    w1.grad = 0.0
    w2.grad = 0.0
    b.grad  = 0.0

    # Backward pass: compute gradients
    loss.backward()

    # Gradient descent: update weights
    lr = 0.1
    w1.data -= lr * w1.grad
    w2.data -= lr * w2.grad
    b.data  -= lr * b.grad

    print(f"{step:>5}  {o.data:>10.6f}  {loss.data:>10.6f}  {w1.data:>8.4f}  {w2.data:>8.4f}")

print(f"\nTarget was: {target}")
print(f"Final output: {o.data:.6f}")
print(f"The neuron learned! ✓" if abs(o.data - target) < 0.01 else "Still learning...")
