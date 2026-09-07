"""
micrograd/step3_full_network.py
================================
LESSON 3: A full MLP trained on a real classification problem.

Dataset: 4 samples, binary classification (output +1 or -1)
Network: MLP with 3 inputs → [4, 4, 1]
Loss:    Hinge loss (max(0, 1 - y*ypred)) — classic SVM loss

After running this, you've implemented backpropagation from scratch
and trained a neural network. This is EXACTLY what PyTorch does
internally — you just built it yourself.
"""

import sys, os, random
sys.path.insert(0, os.path.dirname(__file__))
from engine import Value
from nn import MLP

random.seed(42)

print("=" * 60)
print("STEP 3: Training an MLP on a toy dataset")
print("=" * 60)

# ── Dataset: 4 inputs, each with 3 features; labels are +1 or -1
X = [
    [2.0,  3.0, -1.0],   # label +1
    [3.0, -1.0,  0.5],   # label -1
    [0.5,  1.0,  1.0],   # label +1
    [1.0,  1.0, -1.0],   # label -1
    [0.0, 0.0, 0.0],    # label +1
]
y_true = [1.0, -1.0, 1.0, -1.0, 1.0]

# ── Build the model
model = MLP(n_inputs=3, layer_sizes=[4, 4, 1])
print(f"\nModel: {model}")
print(f"Total parameters: {len(model.parameters())}")


def compute_loss(model, X, y_true):
    """
    Hinge loss:  loss = sum( max(0, 1 - y_true * y_pred) )

    - When prediction is correct AND confident: loss = 0
    - When prediction is wrong or unsure: loss > 0
    """
    # Forward pass on all samples
    y_pred = [model(xi) for xi in X]

    # Hinge loss per sample
    losses = [(1 - yi * yp).relu() for yi, yp in zip(y_true, y_pred)]

    # Average loss + L2 regularization (prevents weights from exploding)
    data_loss = sum(losses) * (1.0 / len(losses))
    l2_reg = 0.0001 * sum(p**2 for p in model.parameters())
    total_loss = data_loss + l2_reg

    # Accuracy (how many are correct?)
    correct = sum(1 for yi, yp in zip(y_true, y_pred) if (yi > 0) == (yp.data > 0))
    accuracy = correct / len(y_true)

    return total_loss, y_pred, accuracy


print("\n{'Step':>5}  {'Loss':>10}  {'Accuracy':>10}")
print("-" * 35)

# ── Training loop
for step in range(100):

    # 1. Forward pass + compute loss
    loss, y_pred, accuracy = compute_loss(model, X, y_true)

    # 2. Zero all gradients  ← CRITICAL: must do this before every backward
    model.zero_grad()

    # 3. Backward pass — computes grad for every parameter
    loss.backward()

    # 4. Gradient descent — update every parameter
    learning_rate = 0.05
    for p in model.parameters():
        p.data -= learning_rate * p.grad

    if step % 10 == 0 or step == 99:
        print(f"{step:>5}  {loss.data:>10.6f}  {accuracy*100:>9.1f}%")


print("\n--- Final Predictions ---")
for i, (xi, yi) in enumerate(zip(X, y_true)):
    pred = model(xi)
    correct = "✓" if (yi > 0) == (pred.data > 0) else "✗"
    print(f"  Sample {i+1}: pred={pred.data:+.4f}  target={yi:+.0f}  {correct}")


print("\n" + "=" * 60)
print("WHAT YOU JUST BUILT (from scratch):")
print("=" * 60)
print("""
engine.py   ← Automatic differentiation (autograd)
  Value.__add__, __mul__, __pow__  = forward pass ops
  Value._backward closures         = gradient formulas
  Value.backward()                 = topological sort + chain rule

nn.py       ← Neural network primitives
  Neuron    = dot product + activation
  Layer     = list of neurons
  MLP       = stack of layers

step3 (this file):
  loss function   = how wrong are we?
  zero_grad()     = reset before each step
  loss.backward() = compute ALL gradients
  p.data -= lr * p.grad  = gradient descent

This is PyTorch's core loop. Every framework works this way.
""")
