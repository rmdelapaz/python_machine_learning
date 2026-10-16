# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 10.2 A (SOLUTION): the PyTorch workflow on card fraud.

Lesson 10.2 trained a churn network. The payments team asks for a fraud score instead: 50,000 card
payments, under 1% fraud. Same workflow, new data: prepare float32 tensors, write a small nn.Module,
train with mini-batches and early stopping on a validation set, open the test set once, and compare
honestly with logistic regression and gradient boosting.
"""
# %% Step 1: three-way split and preprocessing (fitted on the fitting rows only)
import copy

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

tx = pd.read_csv(DATA / "transactions.csv")
numeric = ["amount", "account_age_days", "hour", "items", "device_new", "billing_shipping_mismatch",
           "country_mismatch", "card_attempts", "velocity_24h"]
X_train_df, X_test_df, y_train_s, y_test_s = train_test_split(
    tx[numeric + ["category"]], tx["is_fraud"], test_size=0.25, stratify=tx["is_fraud"], random_state=42)
X_fit_df, X_val_df, y_fit_s, y_val_s = train_test_split(
    X_train_df, y_train_s, test_size=0.2, stratify=y_train_s, random_state=42)
prep = ColumnTransformer([("num", StandardScaler(), numeric),
                          ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), ["category"])])
A_fit = prep.fit_transform(X_fit_df)
A_val, A_test = prep.transform(X_val_df), prep.transform(X_test_df)
print(f"fit {A_fit.shape}, val {A_val.shape}, test {A_test.shape}")
print(f"frauds: fit {y_fit_s.sum()}, val {y_val_s.sum()}, test {y_test_s.sum()}")


# %% Step 2: float32 tensors with targets shaped (n, 1)
def as_tensors(A, y_s):
    return (torch.tensor(A, dtype=torch.float32),
            torch.tensor(y_s.to_numpy(), dtype=torch.float32).unsqueeze(1))


X_fit, y_fit = as_tensors(A_fit, y_fit_s)
X_val, y_val = as_tensors(A_val, y_val_s)
X_test, y_test = as_tensors(A_test, y_test_s)
print(X_fit.dtype, tuple(y_fit.shape))


# %% Step 3: a small module that returns logits
class FraudNet(nn.Module):
    def __init__(self, n_features, n_hidden=32):
        super().__init__()
        self.hidden = nn.Linear(n_features, n_hidden)
        self.output = nn.Linear(n_hidden, 1)

    def forward(self, x):
        return self.output(torch.relu(self.hidden(x)))      # logits: the loss applies the sigmoid


torch.manual_seed(42)
model = FraudNet(n_features=X_fit.shape[1])
print("parameters:", sum(p.numel() for p in model.parameters()))

# %% Step 4: train with early stopping on the validation loss
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
torch.manual_seed(42)
loader = DataLoader(TensorDataset(X_fit, y_fit), batch_size=256, shuffle=True)

best_loss, best_state, best_epoch = float("inf"), None, 0
for epoch in range(1, 31):
    model.train()
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        val_loss = loss_fn(model(X_val), y_val).item()
    if val_loss < best_loss:
        best_loss, best_state, best_epoch = val_loss, copy.deepcopy(model.state_dict()), epoch
    elif epoch - best_epoch >= 5:
        break
model.load_state_dict(best_state)
print(f"stopped after epoch {epoch}; best epoch {best_epoch}, validation loss {best_loss:.4f}")

# %% Step 5: open the test set once, and compare on the same rows and features
model.eval()
with torch.no_grad():
    net_scores = torch.sigmoid(model(X_test)).squeeze(1).numpy()
results = {"neural network (PyTorch)": net_scores}
for name, clf in [("logistic regression", LogisticRegression(max_iter=1000)),
                  ("gradient boosting (defaults)", HistGradientBoostingClassifier(random_state=42))]:
    results[name] = clf.fit(A_fit, y_fit_s).predict_proba(A_test)[:, 1]

print("\nmodel                          test ROC AUC  average precision")
for name, scores in results.items():
    print(f"{name:30s} {roc_auc_score(y_test_s, scores):12.3f}  {average_precision_score(y_test_s, scores):17.3f}")

# %% Step 6: what the fraud team would see: the 100 riskiest test payments
top100 = pd.Series(net_scores, index=X_test_df.index).nlargest(100).index
print(f"\nfrauds among the network's 100 riskiest payments: {y_test_s.loc[top100].sum()} "
      f"(a random 100 would hold {100 * y_test_s.mean():.1f})")
