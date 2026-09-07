"""
micrograd/nn.py
===============
A tiny neural network library built ON TOP of engine.py.

Nothing here is magic — every Neuron, Layer, and MLP
is just Value objects wired together with operations
you already understand from engine.py.

Architecture built here:
  Neuron  →  one dot product + activation
  Layer   →  list of Neurons
  MLP     →  list of Layers (Multi-Layer Perceptron)
"""

import random
from engine import Value


class Neuron:
    """
    One artificial neuron.

    Does:  output = activation( w1*x1 + w2*x2 + ... + wn*xn + b )

    Parameters
    ----------
    n_inputs   : how many inputs this neuron receives
    activation : 'relu' | 'tanh' | 'linear'
                 'linear' means no activation (raw dot product)
    """

    def __init__(self, n_inputs, activation='tanh'):
        # Random weights in [-1, 1] for each input
        self.w = [Value(random.uniform(-1, 1), label=f'w{i}') for i in range(n_inputs)]
        # Bias — one per neuron
        self.b = Value(0.0, label='b')
        self.activation = activation

    def __call__(self, x):
        """
        Forward pass: compute the neuron's output given inputs x.

        x is a list of Values (or plain numbers, they get auto-wrapped).
        """
        # Dot product: sum of w_i * x_i
        dot = sum(wi * xi for wi, xi in zip(self.w, x)) + self.b

        # Apply activation
        if self.activation == 'tanh':
            return dot.tanh()
        elif self.activation == 'relu':
            return dot.relu()
        else:
            return dot   # linear — no activation

    def parameters(self):
        """Return all learnable parameters (weights + bias)."""
        return self.w + [self.b]

    def __repr__(self):
        return f"Neuron(inputs={len(self.w)}, activation={self.activation})"


class Layer:
    """
    One layer = a list of Neurons all reading the same inputs.

    Parameters
    ----------
    n_inputs  : number of inputs each neuron receives
    n_neurons : how many neurons in this layer
    """

    def __init__(self, n_inputs, n_neurons, activation='tanh'):
        self.neurons = [Neuron(n_inputs, activation) for _ in range(n_neurons)]

    def __call__(self, x):
        """Forward pass: run all neurons, return their outputs as a list."""
        outputs = [neuron(x) for neuron in self.neurons]
        # If only one neuron, return a scalar directly (cleaner for output layer)
        return outputs[0] if len(outputs) == 1 else outputs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

    def __repr__(self):
        return f"Layer({self.neurons})"


class MLP:
    """
    Multi-Layer Perceptron.

    Parameters
    ----------
    n_inputs    : number of input features
    layer_sizes : list of ints — how many neurons in each layer
                  e.g. [4, 4, 1] means two hidden layers of 4, one output neuron

    Example
    -------
    model = MLP(3, [4, 4, 1])
    # Reads 3 inputs, has 2 hidden layers of 4 neurons each, outputs 1 value
    """

    def __init__(self, n_inputs, layer_sizes):
        # Build all layer sizes: [n_inputs, sz1, sz2, ..., output_sz]
        sizes = [n_inputs] + layer_sizes

        self.layers = []
        for i in range(len(layer_sizes)):
            is_output_layer = (i == len(layer_sizes) - 1)
            # Output layer: linear (no activation) — common for regression
            # Hidden layers: tanh
            activation = 'linear' if is_output_layer else 'tanh'
            self.layers.append(Layer(sizes[i], sizes[i+1], activation))

    def __call__(self, x):
        """Forward pass: feed input through every layer sequentially."""
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        """All parameters across all layers."""
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        """Reset all gradients to 0 before each backward pass."""
        for p in self.parameters():
            p.grad = 0.0

    def __repr__(self):
        return f"MLP(layers={self.layers}, total_params={len(self.parameters())})"
