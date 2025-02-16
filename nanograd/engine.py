import math

class Value:
  """
  Stores a single scalar value and its gradient.
  Args:
      data (float): The value of the scalar.
      _children (tuple): The children of the node.
      _op (str): The operation that produced this node.
  Attributes:
      data (float): The value of the scalar.
      grad (float): The gradient of the scalar.
      _backward (callable): The backward function.
      _prev (set): The children of the node.
      _op (str): The operation that produced this node.
  """
  def __init__(self, data, _children=(), _op=''):
    self.data = data
    self.grad = 0.0
    self._backward = lambda: None
    self._prev = set(_children)
    self._op = _op


  def __repr__(self):
    return f"Value(data={self.data}, grad={self.grad})"

  def __hash__(self):
    return id(self)


  def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')
    
    def _backward():
      self.grad += out.grad
      other.grad += out.grad
    out._backward = _backward

    return out


  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), '*')
    
    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    out._backward = _backward
    
    return out


  def __pow__(self, other):
    assert isinstance(other, (int, float)), "only supporting int/float for now"
    out = Value(self.data**other, (self,), f'**{other}')

    def _backward():
      self.grad += other * self.data**(other-1) * out.grad
    out._backward = _backward

    return out
  

  def exp(self):
    out = Value(math.exp(self.data), (self,), 'exp')

    def _backward():
      self.grad += math.exp(self.data) * out.grad
    out._backward = _backward

    return out
  

  def log(self):
    out = Value(math.log(self.data), (self,), 'log')
    
    def _backward():
      self.grad += (1 / self.data) * out.grad
    out._backward = _backward

    return out


  def tanh(self):
    x = self.data
    t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
    out = Value(t, (self,), 'tanh')

    def _backward():
      self.grad += (1 - t**2) * out.grad
    out._backward = _backward
    
    return out
  

  def relu(self):
    out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
    
    def _backward():
      self.grad += (out.data > 0) * out.grad
    out._backward = _backward

    return out
  

  def sigmoid(self):
    s = 1 / (1 + math.exp(-self.data))
    out = Value(s, (self,), 'sigmoid')

    def _backward():
      self.grad += s * (1 - s) * out.grad
    out._backward = _backward

    return out
  

  def leaky_relu(self, alpha=0.01):
    out = Value(self.data if self.data > 0 else alpha * self.data, (self,), 'Leaky ReLU')
    
    def _backward():
      self.grad += (1 if out.data > 0 else alpha) * out.grad
    out._backward = _backward

    return out

  
  def backward(self):
    topo = []
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
    build_topo(self)

    self.grad = 1.0
    for node in reversed(topo):
      node._backward()


  def __neg__(self):  # -self
    return self * -1
  
  def __radd__(self, other):  # other + self
    return self + other
  
  def __sub__(self, other):  # self - other
    return self + (-other)

  def __rsub__(self, other):  # other - self
    return other + (-self)
  
  def __rmul__(self, other):  # other * self
    return self * other
  
  def __truediv__(self, other): # self / other
    return self * other**-1
  
  def __rtruediv__(self, other): # other / self
    return other * self**-1
  
  def __lt__(self, other):  # self < other
    return self.data < other.data
  
  def __le__(self, other):  # self <= other
    return self.data <= other.data
  
  def __eq__(self, other):  # self == other
    return self.data == other.data
  
  def __gt__(self, other):  # self > other
    return self.data > other.data
  
  def __ge__(self, other):  # self >= other
    return self.data >= other.data
  
  def __ne__(self, other):  # self != other
    return self.data != other.data

  def __abs__(self):  # abs(self)
    return self if self.data >= 0 else -self