# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA, OUTPUT

"""Debug 12.1: a "reproducible" experiment with FOUR bugs (deliberately broken handout).

Two bugs stop the script with an error. Two let it run: one makes the numbers change every time you
run it, one quietly switches off a safety check. Find and fix all four. When everything is fixed,
two runs of the script print exactly the same lines, ending with: reloaded forest, test AUC 0.8637.
"""
import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import mlflow
import pandas as pd
import skops.io as sio
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

customers = pd.read_csv(DATA / "customers_churn.csv")
numeric = ["tenure_months", "orders_12m", "spend_12m", "avg_order_value", "days_since_last_order",
           "returns_12m", "return_rate", "support_tickets_12m", "discount_share", "app_user"]
X_train, X_test, y_train, y_test = train_test_split(customers[numeric], customers["churned"], test_size=0.25,
                                                    stratify=customers["churned"], random_state=42)

mlflow.set_tracking_uri("./mlruns")
mlflow.set_experiment("debug-forest-depth")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = {}
with mlflow.start_run(run_name="forest-depths"):
    for depth in [4, 8]:
        forest = RandomForestClassifier(n_estimators=100, max_depth=depth, n_jobs=-1)
        results[depth] = cross_val_score(forest, X_train, y_train, cv=cv, scoring="roc_auc").mean()
        mlflow.log_param("max_depth", depth)
        mlflow.log_metric("cv_auc", results[depth])
        print(f"max_depth {depth}: CV AUC {results[depth]:.4f}")

best_depth = max(results, key=results.get)
final = RandomForestClassifier(n_estimators=100, max_depth=best_depth, n_jobs=-1, random_state=42).fit(X_train, y_train)
path = OUTPUT / "forest.skops"
sio.dump(final, path)
loaded = sio.load(path, trusted=sio.get_untrusted_types(file=path))
print(f"reloaded forest, test AUC {roc_auc_score(y_test, loaded.predict_proba(X_test)[:, 1]):.4f}")
