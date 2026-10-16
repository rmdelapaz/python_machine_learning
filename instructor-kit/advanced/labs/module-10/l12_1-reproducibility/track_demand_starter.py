# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA, OUTPUT

"""Lab 12.1 A: a lab notebook for the demand forecast.

The lesson tracked churn classifiers. The planning team's weekly demand model (Lessons 3.2 and 5.3) has
the same problem: three people tried "a few things" and nobody knows which settings gave which error.
Put the five ingredients in one place: a CONFIG, a data fingerprint, seeds, one MLflow run per
candidate (SQLite store, in OUTPUT), the winner chosen by QUERYING the store, and one test score.
Steps 1 and 2 are done for you. Work through TODO 1 to 5; run the file after each one. It runs as it
is: unfinished steps print a reminder. (The first MLflow call prints a few INFO lines: that is normal.)
"""
# %% Step 1: one configuration, one fingerprint
import hashlib
import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")      # MLflow 3.17 otherwise prints an unrelated hint
import numpy as np
import pandas as pd

CONFIG = {
    "data_file": "weekly_demand.csv",
    "numeric": ["store_sqft", "list_price", "discount_pct", "price", "promo", "holiday_week",
                "avg_high_temp_f", "week_of_year"],
    "categorical": ["store", "category", "product_id"],
    "target": "units_sold",
    "fit_weeks": "2024 weeks 1-39", "validation_weeks": "2024 weeks 40-53", "test_year": 2025,
    "seed": 42,
    "experiment": "demand-lab",
}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1_000_000), b""):
            digest.update(chunk)
    return digest.hexdigest()


data_hash = file_sha256(DATA / CONFIG["data_file"])
print("data fingerprint:", data_hash[:16])

# %% Step 2: a split by time and three candidates
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

demand = pd.read_csv(DATA / CONFIG["data_file"])
features = CONFIG["numeric"] + CONFIG["categorical"]
y2024 = demand["year"] == 2024
fit = demand[y2024 & (demand["week_of_year"] <= 39)]
val = demand[y2024 & (demand["week_of_year"] >= 40)]
train_all, test = demand[y2024], demand[demand["year"] == CONFIG["test_year"]]
print(f"fit {len(fit):,}  validation {len(val):,}  test {len(test):,}")


def build_pipeline(model):
    prep = ColumnTransformer([("num", StandardScaler(), CONFIG["numeric"]),
                              ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                               CONFIG["categorical"])])
    return Pipeline([("prep", prep), ("model", model)])


seed = CONFIG["seed"]
candidates = {
    "ridge": Ridge(alpha=1.0),
    "forest": RandomForestRegressor(n_estimators=100, min_samples_leaf=3, n_jobs=-1, random_state=seed),
    "hgb": HistGradientBoostingRegressor(random_state=seed),
}

# %% Step 3: one MLflow run per candidate (SQLite store and artifacts inside OUTPUT)
import mlflow

mlflow.set_tracking_uri(f"sqlite:///{OUTPUT / 'mlflow.db'}")     # done for you: never "./mlruns" (refused)
if mlflow.get_experiment_by_name(CONFIG["experiment"]) is None:
    mlflow.create_experiment(CONFIG["experiment"], artifact_location=(OUTPUT / "mlartifacts").as_uri())
mlflow.set_experiment(CONFIG["experiment"])

# TODO 1: for each candidate, ONE run (mlflow.start_run(run_name=name)): fit build_pipeline(model) on
#         `fit`, compute the validation MAE, then log_params(model.get_params()), log_dict(CONFIG,
#         "config.json"), set_tags (estimator name, data_sha256) and log_metric("val_mae", ...).
#         Print each name with its validation MAE.
print("TODO 1: three tracked runs")

# %% Step 4: choose by querying the store, then test once
# TODO 2: runs = mlflow.search_runs(experiment_names=[...], order_by=["metrics.val_mae ASC"]) (ASC: an
#         error, lower is better). Print name, estimator and val_mae; best_name = the first row's
#         "tags.mlflow.runName".
# TODO 3: refit the winner on ALL of 2024 (train_all), compute the 2025 test MAE once, and log it in a
#         run named f"final-{best_name}" with the winner's params and the data fingerprint tag.
print("TODO 2-3: query the store, test once, log the final run")

# %% Step 5: ask the store a question
# TODO 4: search_runs with filter_string="metrics.val_mae < 12": which runs qualify? How many runs are in
#         the store in total (why one more than the candidates)?
print("TODO 4: a filtered query")

# %% Step 6: is it reproducible? The same seed twice, then a different seed
from sklearn.base import clone
# TODO 5: for "hgb" and "forest": the validation MAE with clone(model).set_params(random_state=s) for
#         s = 42, 42, 7. Which model does not care about its seed here, and why might that be?
print("TODO 5: seeds")

print("\nExpected when done: validation MAE ridge 12.63, forest 11.31, hgb 10.74; the store puts hgb first;")
print("test MAE (2025) 8.21; below 12: ['hgb', 'forest']; 4 runs in the store; hgb 10.7445 for every seed;")
print("forest 11.3146 twice with seed 42 and 11.2821 with seed 7.")
