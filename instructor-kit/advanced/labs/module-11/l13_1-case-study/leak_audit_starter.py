# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 13.1 A: a leak audit you can run on ANY table.

The case study checked one suspicious column by hand. A real project has dozens of columns and
nobody tells you which one is the trap. Here you write one function that screens every column with
the lesson's leak questions, run it on the churn table (one planted leak), then on the fraud table:
a clean audit is a result too. one_column_auc is done for you; work through TODO 1 to 4.
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
    """One row per column (except the target and `skip`): auc_alone, filled_if_1, filled_if_0, distinct, flag."""
    # TODO 1: for every column, record one_column_auc(...) (rounded to 3), the share of rows where the
    #         column is filled for target 1 and for target 0 (notna().mean()), and nunique().
    # TODO 2: flag = "FILLED FOR ONE CLASS" when one class never has a value but the other does,
    #         else "TOO GOOD" when auc_alone >= 0.9, else "ID-LIKE" for a TEXT column whose values are
    #         (almost) all distinct, else "". Return the table sorted by auc_alone, highest first.
    return None


# %% Step 2: the churn table
customers = pd.read_csv(DATA / "customers_churn.csv")
churn_audit = audit(customers, "churned")
if churn_audit is None:
    print("TODO 1-2: write audit(), then run it on the churn table")
else:
    print(churn_audit.to_string())

# %% Step 3: read the flags
# TODO 3: print the flagged columns, the strongest UNflagged column and its AUC, and what the audit says
#         about cancellation_reason. Which flag is a leak, and which is just a label?
print("TODO 3: read the flags")

# %% Step 4: the fraud table, with the timestamp left out (it is how you split, not a feature)
# TODO 4: run audit(tx, "is_fraud", skip=["timestamp"]) on transactions.csv. Is there a leak?
print("TODO 4: audit the fraud table")

print("\nExpected when done: churn flags cancellation_reason (FILLED FOR ONE CLASS: AUC alone 0.815, filled for 63%")
print("of churners and 0% of stayers) and customer_id (ID-LIKE, AUC 0.500); the strongest honest column is")
print("days_since_last_order (0.715). Fraud: only transaction_id is flagged (ID-LIKE); velocity_24h is the")
print("strongest single column (0.782). No leak there: a clean audit is a finding.")
