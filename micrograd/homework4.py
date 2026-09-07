import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from engine import Value

a = Value(3.0, label='a')
b = Value(2.0, label='b')
c = a * b; c.label = 'c'
d = c ** 2; d.label = 'd'
e = d + a; e.label = 'e'

e.backward()

print(f"{a.grad=}")
print(f"{b.grad=}")
print(f"{c.grad=}")
print(f"{d.grad=}")
