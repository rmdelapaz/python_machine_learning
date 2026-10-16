# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 2.2 B (SOLUTION): how big is the leak?

Three steps, each done twice: once BEFORE cross-validation (on all training rows) and once INSIDE a
pipeline (refitted on every fold). The question is not "is it wrong?" (the first way always is) but
"how much does it lie?" The answer depends on whether the step looks at the target, and on how few
rows each learned number comes from.
"""
# %% Step 1: the same split as Lab A
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
rate = y_train.groupby(train["amount_dollars"]).mean()                    # every training label used
X_rate = pd.DataFrame({"amount_fraud_rate": train["amount_dollars"].map(rate)})
leaky = cross_val_score(LogisticRegression(), X_rate, y_train, cv=cv, scoring="roc_auc").mean()

from sklearn.preprocessing import TargetEncoder

honest_pipe = make_pipeline(TargetEncoder(cv=cv), LogisticRegression())
honest = cross_val_score(honest_pipe, train[["amount_dollars"]], y_train, cv=cv, scoring="roc_auc").mean()
honest_pipe.fit(train[["amount_dollars"]], y_train)
test_auc = roc_auc_score(y_test, honest_pipe.predict_proba(test[["amount_dollars"]])[:, 1])
print(f"target rate  before CV {leaky:.3f}   inside pipeline {honest:.3f}   test {test_auc:.3f}")

# %% Step 3: a step that learns only from X: a scaler fitted on all training rows
from sklearn.preprocessing import StandardScaler

X_scaled_early = StandardScaler().fit_transform(train[num_cols])           # fitted before the folds
early = cross_val_score(LogisticRegression(), X_scaled_early, y_train, cv=cv, scoring="roc_auc").mean()
inside = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression()), train[num_cols], y_train,
                         cv=cv, scoring="roc_auc").mean()
print(f"scaler       before CV {early:.4f}  inside pipeline {inside:.4f}  difference {early - inside:+.5f}")

# %% Step 4: a step that learns nothing: log1p of the amount
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import FunctionTransformer

logged = train[num_cols].assign(amount=np.log1p(train["amount"]))         # done before the folds
before = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression()), logged, y_train,
                         cv=cv, scoring="roc_auc").mean()
log_step = make_column_transformer((FunctionTransformer(np.log1p), ["amount"]), remainder="passthrough")
in_pipe = cross_val_score(make_pipeline(log_step, StandardScaler(), LogisticRegression()), train[num_cols],
                          y_train, cv=cv, scoring="roc_auc").mean()
print(f"log1p        before CV {before:.4f}  inside pipeline {in_pipe:.4f}  difference {before - in_pipe:+.5f}")

# %% Step 5: the verdict
print("\nlearns from y, few rows per value : huge lie  (put it in the pipeline, or drop the idea)")
print("learns from X only                : tiny lie  (still: put it in the pipeline)")
print("learns nothing (stateless)        : no lie    (safe anywhere; the pipeline costs nothing)")
