"""
micrograd/step1_just_values.py
==============================
LESSON 1: What is a Value? What is a gradient?

Run this file.
Read every print statement.
Make sure every number makes sense to you before moving to step2.

Key question to answer: WHY is a.grad = 6.0?
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from engine import Value


print("=" * 60)
print("STEP 1: Basic arithmetic on Value objects")
print("=" * 60)

# Two inputs
a = Value(2.0, label='a')
b = Value(3.0, label='b')

print(f"\na = {a}")
print(f"b = {b}")

# Forward pass — build the computation graph
c = a * b        # c = 2.0 * 3.0 = 6.0
print(f"\nc = a * b = {c}")

d = a + b        # d = 2.0 + 3.0 = 5.0
print(f"d = a + b = {d}")

e = c + d        # e = 6.0 + 5.0 = 11.0
print(f"e = c + d = {e}")

print("\nBefore backward — all gradients are 0.0")
print(f"  a.grad = {a.grad}")
print(f"  b.grad = {b.grad}")
print(f"  c.grad = {c.grad}")
print(f"  d.grad = {d.grad}")
print(f"  e.grad = {e.grad}")


print("\n--- calling e.backward() ---\n")
e.backward()


print("After backward — gradients tell us: if I nudge this value slightly,")
print("how much does e change?")
print()
print(f"  e.grad = {e.grad}  ← dE/dE = 1 always (by definition)")
print(f"  c.grad = {c.grad}  ← dE/dC = 1 (c flows into e via addition)")
print(f"  d.grad = {d.grad}  ← dE/dD = 1 (d flows into e via addition)")
print()
print(f"  a.grad = {a.grad}  ← dE/dA = ?  (a appears in BOTH c=a*b and d=a+b)")
print(f"  b.grad = {b.grad}  ← dE/dB = ?")

print("\n--- Manual check (the math) ---")
print("e = c + d = (a*b) + (a+b) = a*b + a + b")
print()
print("dE/dA = b + 1 = 3 + 1 = 4  →", 3 + 1, "✓" if a.grad == 4.0 else "✗")
print("dE/dB = a + 1 = 2 + 1 = 3  →", 2 + 1, "✓" if b.grad == 3.0 else "✗")


print("\n" + "=" * 60)
print("STEP 1b: Using powers and negation")
print("=" * 60)

x = Value(4.0, label='x')

# x^2
x_sq = x ** 2
x_sq.backward()
print(f"\nx = {x.data}")
print(f"x^2 = {x_sq.data}")
print(f"d(x^2)/dx = {x.grad}  ← expected: 2*x = {2*x.data}")
print("✓" if x.grad == 8.0 else f"✗ got {x.grad}")

x.grad = 0.0  # reset

# -x = x * -1
neg_x = -x
neg_x.backward()
print(f"\n-x = {neg_x.data}")
print(f"d(-x)/dx = {x.grad}  ← expected: -1")
print("✓" if x.grad == -1.0 else f"✗ got {x.grad}")


print("\n" + "=" * 60)
print("EXERCISE: Try to predict the gradient before running")
print("=" * 60)
print("""
  p = Value(3.0)
  q = Value(-2.0)
  r = p * q + p**2

  What is p.grad after r.backward()?
  Hint: r = p*q + p^2
        dr/dp = q + 2*p = -2 + 6 = 4
""")

p = Value(3.0, label='p')
q = Value(-2.0, label='q')
r = p * q + p**2
r.backward()
print(f"  p.grad = {p.grad}  ← {'✓ correct!' if p.grad == 4.0 else '✗'}")
print(f"  q.grad = {q.grad}  ← dr/dq = p = 3.0 {'✓' if q.grad == 3.0 else '✗'}")
