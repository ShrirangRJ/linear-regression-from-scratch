import numpy as np

"""
Linear Regression From Scratch
Goal: learn y = 3x + 2 using gradient descent
"""

# -----------------------------
# 1) Create synthetic dataset
# -----------------------------
np.random.seed(42)

X = np.random.rand(100, 1)
y = 3 * X + 2 + np.random.randn(100, 1) * 0.1   # true relationship + noise

# -----------------------------
# 2) Initialize parameters
# -----------------------------
w = np.random.randn(1)
b = 0.0

learning_rate = 0.0001    # stable learning rate
steps = 8000             # more iterations, smaller updates

# -----------------------------
# 3) Training loop
# -----------------------------
for step in range(steps):
    # forward pass (predictions)
    y_pred = X.dot(w) + b

    # mean squared error loss
    loss = ((y_pred - y) ** 2).mean()

    # gradients (derivatives)
    grad_w = (2 / len(X)) * np.sum((y_pred - y) * X)
    grad_b = (2 / len(X)) * np.sum(y_pred - y)

    # gradient descent update
    w -= learning_rate * grad_w
    b -= learning_rate * grad_b

    # print progress occasionally
    if step % 500 == 0:
        print(f"step {step:4d} | loss = {loss:.6f} | w = {w[0]:.4f} | b = {b:.4f}")

# -----------------------------
# 4) Final learned parameters
# -----------------------------
print("\nLearned parameters:")
print("w =", w.item())
print("b =", float(b))
