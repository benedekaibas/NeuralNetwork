import numpy as np

def softmax(logits):
    """Implementing the softmax function."""
    exp_logits = np.exp(logits)
    softmax_output = exp_logits / np.sum(logits)
    return softmax_output

def cross_entropy(y_true, y_pred):
    # implement cross entropy
    return np.sum(y_true * np.log(y_pred))

# example input
X = np.array([0.2, 0.5, 0.8])

# example weights

W = np.array([
    [0.5, -0.2, 0.3],
    [0.1, 0.7, -0.5],
    [-0.3, 0.2, 0.8],
])

# create bias 
b = np.array([0.1, -0.1, 0.2])

# Output example to compare result
y_true = np.array([1, 0, 0]) # this is gonna be the cat

# calculate raw scores (logits)
logits = np.dot(X, W, b)
print(f"Raw scores/output: {logits}")

y_pred = softmax(logits)
print(f"Softmax: {y_pred}")

loss = cross_entropy(y_true, y_pred)
