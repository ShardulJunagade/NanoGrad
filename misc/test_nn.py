import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import matplotlib.pyplot as plt
import numpy as np
from nanograd.nn import MLP
from nanograd.engine import Value

# Generate some sample data
np.random.seed(0)
X = np.random.randn(100, 2)
y = np.array([1 if x[0] * x[1] > 0 else 0 for x in X])

# Convert data to Value objects
X_val = [[Value(xij) for xij in xi] for xi in X]
y_val = [Value(yi) for yi in y]


mlp = MLP(2, [4, 4, 1], activations=['relu', 'relu', 'sigmoid'])

learning_rate = 0.01
for epoch in range(100):
    # Forward pass
    y_pred = [mlp(xi) for xi in X_val]
    loss = sum((yi_pred - yi)**2 for yi_pred, yi in zip(y_pred, y_val))
    
    # Backward pass
    mlp.zero_grad()
    loss.backward()
    
    # Update parameters
    for p in mlp.parameters():
        p.data -= learning_rate * p.grad
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.data}')




# Accuracy
y_pred_np = np.array([yi_pred.data for yi_pred in y_pred])
accuracy = np.mean((y_pred_np > 0.5) == y)
print(f'Accuracy: {accuracy}')


# Plot the initial dataset and the results
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
# Initial dataset
axs[0].scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', alpha=0.5)
axs[0].set_title('Initial Dataset')
axs[0].set_xlabel('X1')
axs[0].set_ylabel('X2')
# MLP Classification Results
X_np = np.array([[xij.data for xij in xi] for xi in X_val])
y_pred_np = np.array([yi_pred.data for yi_pred in y_pred])
axs[1].scatter(X_np[:, 0], X_np[:, 1], c=y_pred_np > 0.5, cmap='bwr', alpha=0.5)
axs[1].set_title('MLP Classification Results')
axs[1].set_xlabel('X1')
axs[1].set_ylabel('X2')
plt.show()
