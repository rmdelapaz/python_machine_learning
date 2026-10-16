# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 13.1 A (SOLUTION): a leak audit you can run on ANY table.

The case study checked one suspicious column by hand. A real project has dozens of columns and
nobody tells you which one is the trap. Here you write one function that screens every column with
the lesson's leak questions, run it on the churn table (one planted leak), then on the fraud table:
a clean audit is a result too.
"""
# %% Step 1: one column alone: how well does it rank, and is it filled for only one class?
import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def one_column_auc(frame, col, target):
    """Cross-validated AUC of a model that sees ONE column (text one-hot encoded, gaps as their own value)."""
    if pd.api.types.is_numeric_dtype(frame[col]):
        prep = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler())
    else:
        prep = make_pipeline(SimpleImputer(strategy="constant", fill_value="(missing)"),
                             OneHotEncoder(handle_unknown="ignore", max_categories=50))
    pipe = make_pipeline(make_column_transformer((prep, [col])), LogisticRegression(max_iter=1000))
    return cross_val_score(pipe, frame[[col]], frame[target], cv=CV, scoring="roc_auc").mean()


def audit(frame, target, skip=()):
    rows = []
    y = frame[target]
    for col in frame.columns.drop([target, *skip]):
        filled = frame[col].notna()
        rows.append({"column": col, "auc_alone": round(one_column_auc(frame, col, target), 3),
                     "filled_if_1": round(filled[y == 1].mean(), 3), "filled_if_0": round(filled[y == 0].mean(), 3),
                     "distinct": frame[col].nunique(), "text": not pd.api.types.is_numeric_dtype(frame[col])})
    out = pd.DataFrame(rows).set_index("column")
    one_class = ((out["filled_if_0"] == 0) | (out["filled_if_1"] == 0)) & (out[["filled_if_0", "filled_if_1"]].max(axis=1) > 0)
    out["flag"] = np.where(one_class, "FILLED FOR ONE CLASS",
                           np.where(out["auc_alone"] >= 0.9, "TOO GOOD",
                                    np.where(out["text"] & (out["distinct"] >= 0.9 * len(frame)), "ID-LIKE", "")))
    return out.drop(columns="text").sort_values("auc_alone", ascending=False)

# %% Step 2: the churn table
customers = pd.read_csv(DATA / "customers_churn.csv")
churn_audit = audit(customers, "churned")
print(churn_audit.to_string())

# %% Step 3: read the flags
print("\nflagged:", churn_audit[churn_audit["flag"] != ""]["flag"].to_dict())
best_honest = churn_audit[churn_audit["flag"] == ""]["auc_alone"].idxmax()
print(f"strongest single honest column: {best_honest} ({churn_audit.loc[best_honest, 'auc_alone']})")
reason = churn_audit.loc["cancellation_reason"]
print(f"cancellation_reason alone: AUC {reason['auc_alone']}, filled for {reason['filled_if_1']:.0%} of churners "
      f"and {reason['filled_if_0']:.0%} of stayers")

# %% Step 4: the fraud table, with the timestamp left out (it is how you split, not a feature)
tx = pd.read_csv(DATA / "transactions.csv")
fraud_audit = audit(tx, "is_fraud", skip=["timestamp"])
print()
print(fraud_audit.to_string())
print("\nflagged:", fraud_audit[fraud_audit["flag"] != ""]["flag"].to_dict())
