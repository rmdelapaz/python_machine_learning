# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 10.2 A: the PyTorch workflow on card fraud.

Lesson 10.2 trained a churn network. The payments team asks for a fraud score instead: 50,000 card
payments, under 1% fraud. Same workflow, new data: prepare float32 tensors, write a small nn.Module,
train with mini-batches and early stopping on a validation set, open the test set once, and compare
honestly with logistic regression and gradient boosting. Work through TODO 1 to 5; the file runs as it
is, and unfinished steps print a reminder.
"""
# %% Step 1: three-way split and preprocessing (done for you)
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
# TODO 1: return two tensors: the features as float32, and the targets as float32 with shape (n, 1)
#         (hint: .unsqueeze(1)). Then build X_fit/y_fit, X_val/y_val and X_test/y_test and print the
#         dtype of X_fit and the shape of y_fit.
def as_tensors(A, y_s):
    return None, None


X_fit, y_fit = as_tensors(A_fit, y_fit_s)
if X_fit is None:
    print("TODO 1: float32 tensors")


# %% Step 3: a small module that returns logits
# TODO 2: a FraudNet(nn.Module) with nn.Linear(n_features, 32), ReLU, nn.Linear(32, 1). No sigmoid at
#         the end: BCEWithLogitsLoss applies it. Seed with torch.manual_seed(42), create it with
#         n_features=X_fit.shape[1] and print the number of parameters. Predict the count first.
print("TODO 2: FraudNet")

# %% Step 4: train with early stopping on the validation loss
# TODO 3: BCEWithLogitsLoss, Adam with lr=0.003, torch.manual_seed(42) then a shuffled DataLoader with
#         batch_size=256. Up to 30 epochs: train (model.train(), zero_grad, loss, backward, step), then the
#         validation loss under model.eval() and torch.no_grad(). Keep copy.deepcopy(model.state_dict())
#         whenever it is the best so far; stop after 5 epochs without improvement; load the best state.
print("TODO 3: train with early stopping")

# %% Step 5: open the test set once, and compare
# TODO 4: probabilities for the test rows (sigmoid of the logits, in eval mode, under no_grad). Fit
#         LogisticRegression(max_iter=1000) and HistGradientBoostingClassifier(random_state=42) on A_fit,
#         y_fit_s. Print test ROC AUC and average precision for all three. Which would you hand to the
#         fraud team, and what would you try before deciding?
print("TODO 4: the honest comparison")

# %% Step 6: the 100 riskiest test payments
# TODO 5: how many real frauds are among the network's 100 highest scores? Compare with 100 * fraud rate.
print("TODO 5: the top 100")

print("\nExpected when done: 30,000 / 7,500 / 12,500 rows with 263 / 66 / 109 frauds; float32 and (30000, 1);")
print("673 parameters; stopped after epoch 24, best epoch 19, validation loss 0.0245; test ROC AUC network")
print("0.855 (AP 0.483), logistic regression 0.844 (0.456), default boosting 0.825 (0.399); 50 frauds")
print("among the 100 riskiest against 0.9 at random. (Training numbers may vary slightly on your machine.)")
