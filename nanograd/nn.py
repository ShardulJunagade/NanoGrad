import random
from .engine import Value

class Module:
  def zero_grad(self):
    for p in self.parameters():
      p.grad = 0

  def parameters(self):
    return []


class Neuron(Module):
  """
  A simple neuron with a fixed number of inputs and a given activation function.
  Args:
    nin (int): Number of inputs.
    activation (str): Activation function. Can be 'relu', 'sigmoid', 'leaky_relu', 'tanh' or 'linear'.
  """
  def __init__(self, nin, activation='linear'):
    self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
    self.b = Value(0)
    self.activation = activation

  def __call__(self, x):
    act = sum((wi*xi for wi, xi in zip(self.w, x)), 0.0) + self.b
    if self.activation == 'relu':
      out = act.relu()
    elif self.activation == 'sigmoid':
      out = act.sigmoid()
    elif self.activation == 'leaky_relu':
      out = act.leaky_relu()
    elif self.activation == 'tanh':
      out = act.tanh()
    elif self.activation == 'linear':
      out = act
    else:
      raise ValueError(f"Unknown activation function: {self.activation}")
    return out
    
  def parameters(self):
    return self.w + [self.b]
  
  def __repr__(self):
    return f"{self.activation.capitalize()}Neuron({len(self.w)})"


class Layer(Module):
  """
  A layer of neurons.
  Args:
    nin (int): Number of inputs.
    nout (int): Number of neurons.
    activation (str): Activation function. Can be 'relu', 'sigmoid', 'leaky_relu', 'tanh' or 'linear'.
  """
  def __init__(self, nin, nout, activation='relu'):
    self.neurons = [Neuron(nin, activation) for _ in range(nout)]
  
  def __call__(self, x):
    outs = [n(x) for n in self.neurons]
    return outs[0] if len(outs) == 1 else outs

  def parameters(self):
    return [p for n in self.neurons for p in n.parameters()]
  
  def __repr__(self):
    return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
  """
  A multi-layer perceptron.
  Args:
    nin (int): Number of inputs.
    nouts (list of int): Number of neurons in each layer.
    activations (list of str): Activation function for each layer. Can be 'relu', 'sigmoid', 'leaky_relu', 'tanh' or 'linear'.
  """
  def __init__(self, nin, nouts, activations=None):
    sz = [nin] + nouts
    if activations is None:
      activations = ['relu'] * len(nouts)
      activations[-1] = 'linear'
    self.layers = [Layer(sz[i], sz[i+1], activations[i]) for i in range(len(nouts))]

  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x
  
  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]
  
  def __repr__(self):
    return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
