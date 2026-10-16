# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import OFFLINE

"""Debug 10.1: a hand-built network with three bugs (deliberately broken).

A colleague's NumPy network for the moons data. One bug stops the script with an error; two do not,
and those are worse: the gradient check disagrees and the loss goes UP. Find and fix all three. When
everything is fixed the script prints: gradient check agree: True, loss 0.6714 -> 0.1432, test
accuracy 0.972 (the lesson's numbers).
"""
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=1000, noise=0.25, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def init_params(n_in, n_hidden, rng):
    return {"W1": rng.normal(0, np.sqrt(2 / n_in), size=(n_in, n_hidden)), "b1": np.zeros(n_hidden),
            "W2": rng.normal(0, np.sqrt(1 / n_hidden), size=(1, n_hidden)), "b2": np.zeros(1)}


def forward(params, X):
    z1 = X @ params["W1"] + params["b1"]
    a1 = np.maximum(0, z1)
    p = sigmoid(a1 @ params["W2"] + params["b2"]).ravel()
    return p, (X, z1, a1)


def bce_loss(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def backward(params, cache, p, y):
    X, z1, a1 = cache
    dz2 = ((p - y) / len(y)).reshape(-1, 1)
    grads = {"W2": a1.T @ dz2, "b2": dz2.sum(axis=0)}
    dz1 = dz2 @ params["W2"].T
    grads["W1"] = X.T @ dz1
    grads["b1"] = dz1.sum(axis=0)
    return grads


params = init_params(2, 16, np.random.default_rng(42))
p, cache = forward(params, X_train)
grads = backward(params, cache, p, y_train)

# gradient check on W1[1, 7]
eps, original = 1e-6, params["W1"][1, 7]
params["W1"][1, 7] = original + eps
up = bce_loss(forward(params, X_train)[0], y_train)
params["W1"][1, 7] = original - eps
down = bce_loss(forward(params, X_train)[0], y_train)
params["W1"][1, 7] = original
numeric = (up - down) / (2 * eps)
print(f"gradient check  backprop {grads['W1'][1, 7]:.8f}  numeric {numeric:.8f}  agree: {np.isclose(grads['W1'][1, 7], numeric)}")

history = []
for epoch in range(2000):
    p, cache = forward(params, X_train)
    history.append(bce_loss(p, y_train))
    grads = backward(params, cache, p, y_train)
    for name in params:
        params[name] += 0.5 * grads[name]
print(f"loss {history[0]:.4f} -> {history[-1]:.4f}")
print(f"test accuracy {((forward(params, X_test)[0] > 0.5) == y_test).mean():.3f}")
