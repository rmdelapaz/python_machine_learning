# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 2.2 B: how big is the leak?

Three steps, each done twice: once BEFORE cross-validation (on all training rows) and once INSIDE a
pipeline (refitted on every fold). The question is not "is it wrong?" (the first way always is) but
"how much does it lie?" Work in pairs: before each run, the one who is not typing predicts the gap
between the two numbers out loud. The file runs as it is: unfinished steps print a reminder.
"""
# %% Step 1: the same split as Lab A (done for you)
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline

tx = pd.read_csv(DATA / "transactions.csv")
tx["amount_dollars"] = tx["amount"].round().astype(int).astype(str)      # "449", "25", ...
num_cols = ["amount", "account_age_days", "hour", "items", "device_new",
            "billing_shipping_mismatch", "country_mismatch", "card_attempts", "velocity_24h"]
train, test = train_test_split(tx, test_size=0.3, stratify=tx["is_fraud"], random_state=42)
y_train, y_test = train["is_fraud"], test["is_fraud"]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
counts = train["amount_dollars"].value_counts()
print(f"{len(counts)} distinct dollar amounts in train; {(counts <= 5).sum()} of them occur 5 times or fewer")

# %% Step 2: a step that learns from the TARGET, with few rows per group: "fraud rate of this amount"
from sklearn.preprocessing import TargetEncoder

# Done for you: the rate computed with pandas on ALL training rows, then cross-validated.
rate = y_train.groupby(train["amount_dollars"]).mean()
X_rate = pd.DataFrame({"amount_fraud_rate": train["amount_dollars"].map(rate)})
leaky = cross_val_score(LogisticRegression(), X_rate, y_train, cv=cv, scoring="roc_auc").mean()
print(f"target rate  before CV {leaky:.3f}")
# TODO 1: the honest version. honest_pipe = make_pipeline(TargetEncoder(cv=cv), LogisticRegression());
#         cross-validate it on train[["amount_dollars"]] (mean ROC AUC), then fit it on the training rows
#         and print the test ROC AUC on test[["amount_dollars"]]. Which number does the test agree with?
print("TODO 1: TargetEncoder inside a pipeline: CV and test ROC AUC")

# %% Step 3: a step that learns only from X: a scaler fitted on all training rows
from sklearn.preprocessing import StandardScaler

# TODO 2: early = CV ROC AUC of LogisticRegression() on StandardScaler().fit_transform(train[num_cols]);
#         inside = CV ROC AUC of make_pipeline(StandardScaler(), LogisticRegression()) on train[num_cols].
#         Print both with 4 decimals and the difference with 5 (format spec +.5f).
print("TODO 2: scaler before CV vs inside the pipeline")

# %% Step 4: a step that learns nothing: log1p of the amount
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import FunctionTransformer

# TODO 3: before = the same scaled logistic pipeline on train[num_cols].assign(amount=np.log1p(train["amount"]));
#         in_pipe = make_pipeline(make_column_transformer((FunctionTransformer(np.log1p), ["amount"]),
#         remainder="passthrough"), StandardScaler(), LogisticRegression()) on train[num_cols].
#         Print both and the difference. Why is it exactly zero this time?
print("TODO 3: a stateless step before CV vs inside the pipeline")

print("\nExpected when done: 1016 distinct amounts, 434 of them 5 times or fewer; target rate before CV")
print("0.913, inside the pipeline 0.583, test 0.568; scaler 0.8613 both ways (difference -0.00001);")
print("log1p 0.8617 both ways (difference +0.00000).")
