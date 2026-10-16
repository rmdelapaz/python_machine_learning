# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 10.1 B: one neuron, trained by hand, IS logistic regression.

Lesson 10.1 said that a single neuron with a sigmoid is logistic regression. Prove it on the churn
table: train ONE sigmoid neuron with plain gradient descent in NumPy (you write its gradient, which
is one line), then compare its test AUC and its ten weights with scikit-learn's LogisticRegression.
Work in pairs: one types, the other predicts each printed number out loud. Swap at TODO 3.
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
# TODO 1: w = np.zeros(10) and b = 0.0. Compute p = sigmoid(A_train @ w + b) and print p[0] and the loss.
#         Predict both before you run it.
w = b = None
print("TODO 1: the zero-start neuron")

# %% Step 3: gradient descent
# TODO 2: for 1,000 epochs with learning_rate = 1.0: forward (p), then the gradient of the loss with
#         respect to z, dz = (p - b_train) / len(b_train), then update
#         w -= learning_rate * (A_train.T @ dz) and b -= learning_rate * dz.sum().
#         Print the loss at epochs 1, 10, 100 and 1000. Then neuron_auc = ROC AUC on the test rows.
neuron_auc = None
print("TODO 2: train the neuron")

# %% Step 4: the same model, the scikit-learn way
# TODO 3: LogisticRegression(C=1e6, max_iter=1000) (C=1e6 means practically no penalty) on the same
#         scaled rows. Print both test AUCs, the largest absolute difference between logreg.coef_[0]
#         and w, and a two-column DataFrame of the weights. Which feature pushes hardest toward churn?
print("TODO 3: compare with LogisticRegression")

# %% Step 5: why zeros were fine here
# TODO 4: build W1 = np.zeros((10, 4)) and print np.maximum(0, A_train[:3] @ W1)[0]. Then write two
#         sentences: why can these hidden units never start learning, while the single neuron above did?
print("TODO 4: zeros and a hidden layer")

print("\nExpected when done: start probability 0.50, loss 0.6931; loss 0.3689 at epoch 10 and 0.3416 at")
print("epochs 100 and 1000; test ROC AUC 0.8235 for BOTH models; largest weight difference 0.0022;")
print("days_since_last_order has the largest weight (0.88), then discount_share (0.67).")
