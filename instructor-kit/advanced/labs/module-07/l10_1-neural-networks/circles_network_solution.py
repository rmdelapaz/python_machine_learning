# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import OUTPUT

"""Lab 10.1 A (SOLUTION): a hand-built network on a ring, and how many hidden units a ring needs.

Lesson 10.1 bent a boundary around two moons. Here one class sits inside a ring of the other, so a
straight line is hopeless. You write the forward pass and the loss yourself, check the backward pass
numerically on a weight the lesson did not check, train, then ask the width question with a twist:
what is the SMALLEST number of ReLU units that can enclose a ring?
"""
# %% Step 1: the data and the straight-line yardstick (done for you)
import numpy as np
from sklearn.datasets import make_circles
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X, y = make_circles(n_samples=1000, noise=0.1, factor=0.5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
linear = LogisticRegression().fit(X_train, y_train)
print(X_train.shape, X_test.shape, f"inner-ring share {y_train.mean():.2f}")
print(f"logistic regression test accuracy: {linear.score(X_test, y_test):.3f}")


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def init_params(n_in, n_hidden, rng):
    return {"W1": rng.normal(0, np.sqrt(2 / n_in), size=(n_in, n_hidden)), "b1": np.zeros(n_hidden),
            "W2": rng.normal(0, np.sqrt(1 / n_hidden), size=(n_hidden, 1)), "b2": np.zeros(1)}


# %% Step 2: the forward pass (shapes in the comments)
def forward(params, X):
    z1 = X @ params["W1"] + params["b1"]          # (n, 2) @ (2, h) -> (n, h)
    a1 = np.maximum(0, z1)                        # ReLU
    z2 = a1 @ params["W2"] + params["b2"]         # (n, h) @ (h, 1) -> (n, 1)
    return sigmoid(z2).ravel(), (X, z1, a1)


params = init_params(2, 8, np.random.default_rng(42))
p, cache = forward(params, X_train)
print("\nprobabilities shape:", p.shape, " first three:", p[:3].round(3))


# %% Step 3: the loss, and the coin-flip yardstick
def bce_loss(p, y, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


print(f"untrained loss {bce_loss(p, y_train):.4f}   coin flip {np.log(2):.4f}")


# %% Step 4: the backward pass (given) and a numeric check on W2[5, 0]
def backward(params, cache, p, y):
    X, z1, a1 = cache
    dz2 = ((p - y) / len(y)).reshape(-1, 1)
    grads = {"W2": a1.T @ dz2, "b2": dz2.sum(axis=0)}
    dz1 = (dz2 @ params["W2"].T) * (z1 > 0)
    grads["W1"] = X.T @ dz1
    grads["b1"] = dz1.sum(axis=0)
    return grads


grads = backward(params, cache, p, y_train)
eps = 1e-6
original = params["W2"][5, 0]
params["W2"][5, 0] = original + eps
up = bce_loss(forward(params, X_train)[0], y_train)
params["W2"][5, 0] = original - eps
down = bce_loss(forward(params, X_train)[0], y_train)
params["W2"][5, 0] = original
numeric = (up - down) / (2 * eps)
print(f"W2[5, 0]  backprop {grads['W2'][5, 0]:.8f}   numeric {numeric:.8f}   agree: {np.isclose(grads['W2'][5, 0], numeric)}")


# %% Step 5: the training loop (forward, loss, backward, update)
def train(params, X, y, learning_rate=0.5, epochs=2000):
    history = []
    for _ in range(epochs):
        p, cache = forward(params, X)
        history.append(bce_loss(p, y))
        grads = backward(params, cache, p, y)
        for name in params:
            params[name] -= learning_rate * grads[name]
    return history


def test_accuracy(params):
    return ((forward(params, X_test)[0] > 0.5) == y_test).mean()


params = init_params(2, 8, np.random.default_rng(42))
history = train(params, X_train, y_train)
print(f"\n8 hidden units: loss {history[0]:.4f} -> {history[-1]:.4f}, test accuracy {test_accuracy(params):.3f}")

# %% Step 6: how many hidden units does a ring need?
print("\nwidth sweep (2,000 epochs, learning rate 0.5, seed 42)")
for h in [1, 2, 3, 4, 8]:
    net = init_params(2, h, np.random.default_rng(42))
    final = train(net, X_train, y_train)[-1]
    print(f"{h} hidden units: final loss {final:.4f}  test accuracy {test_accuracy(net):.3f}")
print("Each ReLU unit adds one straight edge. Two edges make a wedge; three make a triangle,")
print("the first shape that can close around the inner ring.")

# %% Step 7: read a learning-rate problem from the loss alone
print("\nlearning-rate check (8 hidden units, 500 epochs): loss after 100 / 300 / 500 epochs")
for lr in [0.05, 0.5, 50]:
    h = train(init_params(2, 8, np.random.default_rng(42)), X_train, y_train, learning_rate=lr, epochs=500)
    print(f"learning rate {lr:>5}: {h[99]:.3f}  {h[299]:.3f}  {h[499]:.3f}")

# %% Stretch: save the decision map
import matplotlib.pyplot as plt

xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 200), np.linspace(-1.5, 1.5, 200))
grid = np.column_stack([xx.ravel(), yy.ravel()])
three = init_params(2, 3, np.random.default_rng(42))
train(three, X_train, y_train)
fig, ax = plt.subplots(figsize=(5, 5), layout="constrained")
ax.contourf(xx, yy, forward(three, grid)[0].reshape(xx.shape), levels=[0, 0.5, 1], colors=["#dbe1fb", "#fde2c8"])
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="coolwarm", s=10)
ax.set_title("Three ReLU units: a triangle around the ring")
fig.savefig(OUTPUT / "ring_three_units.png")
print("\nsaved ring_three_units.png")
