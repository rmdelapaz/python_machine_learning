# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import OUTPUT

"""Lab 10.1 A: a hand-built network on a ring, and how many hidden units a ring needs.

Lesson 10.1 bent a boundary around two moons. Here one class sits inside a ring of the other, so a
straight line is hopeless. You write the forward pass and the loss yourself, check the backward pass
numerically on a weight the lesson did not check, train, then answer: what is the SMALLEST number of
ReLU units that can enclose a ring? Work through TODO 1 to 6 and run the file after each one. It runs
as it is: unfinished steps print a reminder.
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


# %% Step 2: the forward pass
# TODO 1: compute z1 = X @ W1 + b1, a1 = ReLU(z1), z2 = a1 @ W2 + b2, p = sigmoid(z2) flattened with
#         .ravel(). Write the shape of each line in a comment, e.g. # (n, 2) @ (2, h) -> (n, h).
#         Return p and the cache (X, z1, a1) for the backward pass.
def forward(params, X):
    return None, None


params = init_params(2, 8, np.random.default_rng(42))
p, cache = forward(params, X_train)
if p is None:
    print("\nTODO 1: write forward()")
else:
    print("\nprobabilities shape:", p.shape, " first three:", p[:3].round(3))


# %% Step 3: the loss
# TODO 2: binary cross-entropy, -mean(y log p + (1 - y) log(1 - p)), with p clipped to [eps, 1 - eps].
#         Then print the untrained loss next to np.log(2). Before you run it: which should be bigger?
def bce_loss(p, y, eps=1e-12):
    return None


if p is not None and bce_loss(p, y_train) is not None:
    print(f"untrained loss {bce_loss(p, y_train):.4f}   coin flip {np.log(2):.4f}")
else:
    print("TODO 2: write bce_loss()")


# %% Step 4: the backward pass (given) and a numeric check
def backward(params, cache, p, y):
    X, z1, a1 = cache
    dz2 = ((p - y) / len(y)).reshape(-1, 1)
    grads = {"W2": a1.T @ dz2, "b2": dz2.sum(axis=0)}
    dz1 = (dz2 @ params["W2"].T) * (z1 > 0)
    grads["W1"] = X.T @ dz1
    grads["b1"] = dz1.sum(axis=0)
    return grads


# TODO 3: check the gradient of the OUTPUT weight W2[5, 0] numerically: nudge it up by eps = 1e-6,
#         compute the loss, nudge it down, compute the loss, put it back, and divide the difference by
#         2 * eps. Print it next to grads["W2"][5, 0] and np.isclose(...) of the two.
print("TODO 3: numeric check of W2[5, 0]")


# %% Step 5: the training loop
# TODO 4: write train(params, X, y, learning_rate=0.5, epochs=2000) that repeats forward, loss (append
#         to a history list), backward, and the update params[name] -= learning_rate * grads[name].
#         Return the history. Then train an 8-unit network and print the first and last loss and the
#         test accuracy (probability > 0.5 counts as class 1).
def train(params, X, y, learning_rate=0.5, epochs=2000):
    return None


print("TODO 4: train 8 hidden units")

# %% Step 6: how many hidden units does a ring need?
# TODO 5: loop over h in [1, 2, 3, 4, 8]: a fresh init_params(2, h, np.random.default_rng(42)), train it,
#         print the final loss and the test accuracy. Predict first: where does the big jump happen, and
#         why that number? (Think of each ReLU unit as one straight edge.)
print("TODO 5: the width sweep")

# %% Step 7: read a learning-rate problem from the loss alone
# TODO 6: with 8 units and 500 epochs, train at learning rates 0.05, 0.5 and 50 and print the loss after
#         100, 300 and 500 epochs (history[99], history[299], history[499]). Name each: timid, fine, wild.
print("TODO 6: the learning-rate check")

print("\nExpected when done: logistic regression 0.488 (a coin flip); untrained loss 0.6927 against 0.6931;")
print("W2[5, 0] backprop and numeric both 0.03266046; 8 units: loss 0.6927 -> 0.0156, test accuracy 0.984;")
print("width sweep 0.648 / 0.752 / 0.992 / 0.984 / 0.984 (the jump comes at THREE units);")
print("learning rates: 0.05 crawls (0.487 at 500), 0.5 reaches 0.045, 50 bounces around 4.")
