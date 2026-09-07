"""
micrograd/engine.py
===================
The ENTIRE autograd engine lives here.

Core idea:
  Every number is wrapped in a Value object.
  Every operation (+, *, **) creates a new Value AND remembers:
    - what inputs created it   (_children)
    - how to compute the gradient back through it  (_backward)

When you call .backward() on the final output, it walks the graph
in reverse (topological order) and accumulates .grad on every node.

That's the whole thing.
"""


class Value:
    """
    A single scalar value in the computational graph.

    Attributes
    ----------
    data     : the actual number  (float)
    grad     : gradient of the final output with respect to this value
               starts at 0.0; filled in by .backward()
    _backward: a closure that pushes gradient one step back
    _children: the Value nodes that produced this one
    _op      : the operation that produced this node (for debugging)
    label    : optional name for printing
    """

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = float(data)
        self.grad = 0.0              # starts at zero — no gradient yet

        # Internal bookkeeping
        self._backward = lambda: None   # default: do nothing
        self._children = set(_children)
        self._op = _op
        self.label = label

    # ------------------------------------------------------------------
    # Forward pass operations
    # Each returns a new Value AND defines how to backprop through it
    # ------------------------------------------------------------------

    def __add__(self, other):
        """
        z = self + other

        Derivative rule:
          dL/d(self)  = dL/dz * 1   (gradient flows through addition unchanged)
          dL/d(other) = dL/dz * 1
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # dL/d(self) += dL/dz * 1
            self.grad  += out.grad
            # dL/d(other) += dL/dz * 1
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward
        return out


    def __pow__(self, exponent):
        """
        z = self ** exponent   (exponent must be a plain int or float)

        Derivative rule:
          dL/d(self) = dL/dz * exponent * self.data**(exponent - 1)
        """
        assert isinstance(exponent, (int, float)), "exponent must be int or float"
        out = Value(self.data ** exponent, (self,), f'**{exponent}')

        def _backward():
            self.grad += out.grad * exponent * (self.data ** (exponent - 1))

        out._backward = _backward
        return out

    def relu(self):
        """
        z = max(0, self)

        Derivative rule:
          dL/d(self) = dL/dz   if self.data > 0
                    = 0        if self.data <= 0
        """
        out = Value(max(0, self.data), (self,), 'ReLU')

        def _backward():
            self.grad += out.grad * (1.0 if self.data > 0 else 0.0)

        out._backward = _backward
        return out

    def tanh(self):
        """
        z = tanh(self)

        Derivative rule:
          dL/d(self) = dL/dz * (1 - tanh(self)^2)
                     = dL/dz * (1 - z^2)
        """
        import math
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += out.grad * (1 - t**2)

        out._backward = _backward
        return out

    def exp(self):
        """
        z = e^self
        Derivative: d(e^x)/dx = e^x
        """
        import math
        e = math.exp(self.data)
        out = Value(e, (self,), 'exp')

        def _backward():
            self.grad += out.grad * e

        out._backward = _backward
        return out

    def log(self):
        """
        z = ln(self)
        Derivative: d(ln x)/dx = 1/x
        """
        import math
        assert self.data > 0, "log of non-positive number"
        out = Value(math.log(self.data), (self,), 'log')

        def _backward():
            self.grad += out.grad * (1.0 / self.data)

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Backward pass  —  the magic
    # ------------------------------------------------------------------

    def backward(self):
        """
        Compute gradients for all nodes in the graph via backpropagation.

        Steps:
          1. Topological sort  (each node appears before its children)
          2. Seed this node's gradient to 1.0  (dL/dL = 1)
          3. Walk reversed order calling each node's _backward()
        """
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

    # ------------------------------------------------------------------
    # Operator mirrors
    # ------------------------------------------------------------------

    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __rsub__(self, other): return Value(other) + (-self)
    def __rtruediv__(self, other): return Value(other) * self**-1
    def __sub__(self, other): return self + (-other)
    def __neg__(self): return self * -1
    def __truediv__(self, other): return self * other**-1

    def __repr__(self):
        label = f"'{self.label}' " if self.label else ''
        return f"Value({label}data={self.data:.4f}, grad={self.grad:.4f})"
