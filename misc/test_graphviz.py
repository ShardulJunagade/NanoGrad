import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from nanograd.viz import draw_dot
from nanograd.engine import Value

a = Value(-4.0)
b = Value(5.0)
c = a * b + b**3
d = c.relu()

print(f'{d.data:.4f}')        # Forward pass result: 105.0
d.backward()
print(f'dd/da: {a.grad:.4f}') # Gradient w.r.t. a: 5.0
print(f'dd/db: {b.grad:.4f}') # Gradient w.r.t. b: 71.0

dot = draw_dot(d)
dot.render('misc/computational_graph', format='png', cleanup=True)