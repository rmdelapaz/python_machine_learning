# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 1.3 B (SOLUTION): was C=0.01 really better?

In Session 1 (Lab 1.1 B) logistic regression with C=0.01 scored 0.853 accuracy on the test set against
0.848 for the default C=1.0. One split, one number each. Is the difference real, or is it the wobble
Lesson 1.2 measured? Settle it the Lesson 1.3 way: on the training customers only, with identical
repeated folds, a pipeline, and paired differences.
"""
# %% Step 1: the standard split; the test set stays locked
import pandas as pd
from sklearn.model_selection import train_test_split

customers = pd.read_csv(DATA / "customers_churn.csv")
features = ["tenure_months", "orders_12m", "spend_12m", "avg_order_value", "days_since_last_order",
            "returns_12m", "return_rate", "support_tickets_12m", "discount_share", "app_user"]
X_train, X_test, y_train, y_test = train_test_split(
    customers[features], customers["churned"], test_size=0.25, stratify=customers["churned"], random_state=42
)
print(X_train.shape)

# %% Step 2: the leaky way and the pipeline way, side by side
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X_scaled = StandardScaler().fit_transform(X_train)          # the scaler sees every validation fold
leaky = cross_val_score(LogisticRegression(C=0.01), X_scaled, y_train, cv=cv)
honest = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(C=0.01)), X_train, y_train, cv=cv)
print(f"C=0.01 accuracy, pre-scaled: {leaky.mean():.4f}   in a pipeline: {honest.mean():.4f}")

# %% Step 3: both settings on the same 15 folds, two metrics
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
results = {}
for C in [1.0, 0.01]:
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=C))
    results[C] = cross_validate(pipe, X_train, y_train, cv=rcv, scoring=["accuracy", "roc_auc"])
    print(f"C={C:<5} accuracy {results[C]['test_accuracy'].mean():.3f} +/- {results[C]['test_accuracy'].std():.3f}"
          f"   AUC {results[C]['test_roc_auc'].mean():.3f} +/- {results[C]['test_roc_auc'].std():.3f}")

# %% Step 4: paired differences (C=0.01 minus C=1.0), fold by fold
for metric in ["test_accuracy", "test_roc_auc"]:
    diff = results[0.01][metric] - results[1.0][metric]
    print(f"{metric[5:]:9s} C=0.01 ahead in {(diff > 0).sum():2d} of 15 folds, behind in {(diff < 0).sum():2d}; "
          f"mean difference {diff.mean():+.4f}")

# %% Step 5: a wider sweep, with training scores (is anything overfitting?)
print()
for C in [0.001, 0.01, 0.1, 1.0, 10.0]:
    r = cross_validate(make_pipeline(StandardScaler(), LogisticRegression(C=C)), X_train, y_train, cv=cv,
                       scoring="roc_auc", return_train_score=True)
    print(f"C={C:<6} train AUC {r['train_score'].mean():.3f}   validation AUC {r['test_score'].mean():.3f}")

# %% Step 6: verdict, built from the numbers
acc_diff = results[0.01]["test_accuracy"] - results[1.0]["test_accuracy"]
auc_diff = results[0.01]["test_roc_auc"] - results[1.0]["test_roc_auc"]
acc_word = "ahead of" if acc_diff.mean() > 0 else "behind"
print(f"\nVerdict: on accuracy C=0.01 is {acc_word} C=1.0 by {abs(acc_diff.mean()):.4f} on average "
      f"(ahead in {(acc_diff > 0).sum()} of 15 folds);")
print(f"on AUC the gap is {auc_diff.mean():+.4f}, against a fold-to-fold spread of "
      f"{results[1.0]['test_roc_auc'].std():.3f}: a tie. Session 1's 0.853 vs 0.848 was one split's opinion.")
