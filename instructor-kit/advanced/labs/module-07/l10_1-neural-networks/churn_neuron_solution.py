# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 10.1 B (SOLUTION): one neuron, trained by hand, IS logistic regression.

Lesson 10.1 said that a single neuron with a sigmoid is logistic regression. Prove it on the churn
table: train ONE sigmoid neuron with plain gradient descent in NumPy (you write its gradient, which
is one line), then compare its test AUC and its ten weights with scikit-learn's LogisticRegression.
On the way, see why starting at zero is fine for one neuron but fatal for a hidden layer.
"""
# %% Step 1: the lesson's churn split and scaling (done for you)
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

customers = pd.read_csv(DATA / "customers_churn.csv")
features = ["tenure_months", "orders_12m", "spend_12m", "avg_order_value", "days_since_last_order",
            "returns_12m", "return_rate", "support_tickets_12m", "discount_share", "app_user"]
X_train, X_test, y_train, y_test = train_test_split(
    customers[features], customers["churned"], test_size=0.25, stratify=customers["churned"], random_state=42)
scaler = StandardScaler().fit(X_train)
A_train, A_test = scaler.transform(X_train), scaler.transform(X_test)
b_train, b_test = y_train.to_numpy(), y_test.to_numpy()
print(A_train.shape, A_test.shape)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def bce_loss(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


# %% Step 2: one neuron, starting from all zeros
w = np.zeros(10)
b = 0.0
p = sigmoid(A_train @ w + b)
print(f"start: every probability is {p[0]:.2f}, loss {bce_loss(p, b_train):.4f}")

# %% Step 3: gradient descent; the gradient of one sigmoid neuron with cross-entropy
learning_rate = 1.0
for epoch in range(1, 1001):
    p = sigmoid(A_train @ w + b)                 # forward
    dz = (p - b_train) / len(b_train)            # slope of the loss w.r.t. z: (p - y) / n
    w -= learning_rate * (A_train.T @ dz)        # (10, n) @ (n,) -> (10,)
    b -= learning_rate * dz.sum()
    if epoch in (1, 10, 100, 1000):
        print(f"epoch {epoch:4d}  loss {bce_loss(p, b_train):.4f}")

neuron_auc = roc_auc_score(b_test, sigmoid(A_test @ w + b))

# %% Step 4: the same model, the scikit-learn way
logreg = LogisticRegression(C=1e6, max_iter=1000).fit(A_train, b_train)   # C=1e6: practically no penalty
logreg_auc = roc_auc_score(b_test, logreg.predict_proba(A_test)[:, 1])
print(f"\ntest ROC AUC  neuron {neuron_auc:.4f}   LogisticRegression {logreg_auc:.4f}")
print(f"largest weight difference: {np.abs(logreg.coef_[0] - w).max():.4f}")
print(pd.DataFrame({"neuron": w, "LogisticRegression": logreg.coef_[0]}, index=features).round(2).to_string())

# %% Step 5: why zeros were fine here, and fatal with a hidden layer
W1 = np.zeros((10, 4))                           # four hidden units, all starting at zero
hidden = np.maximum(0, A_train[:3] @ W1)
print("\nfour zero-initialized hidden units, first customer:", hidden[0])
print("Every hidden unit outputs 0 and has slope 0 there, so no unit can ever start learning.")
print("A single neuron has no hidden units to tie: its gradient (p - y) is not zero, so it learns.")
