# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA, OUTPUT

"""Lab 12.1 A (SOLUTION): a lab notebook for the demand forecast.

The lesson tracked churn classifiers. The planning team's weekly demand model (Lessons 3.2 and 5.3) has
the same problem: three people tried "a few things" and nobody knows which settings gave which error.
Put the five ingredients in one place: a CONFIG, a data fingerprint, seeds, one MLflow run per
candidate (SQLite store, in OUTPUT), the winner chosen by QUERYING the store, and one test score.
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

mlflow.set_tracking_uri(f"sqlite:///{OUTPUT / 'mlflow.db'}")
if mlflow.get_experiment_by_name(CONFIG["experiment"]) is None:
    mlflow.create_experiment(CONFIG["experiment"], artifact_location=(OUTPUT / "mlartifacts").as_uri())
mlflow.set_experiment(CONFIG["experiment"])

for name, model in candidates.items():
    with mlflow.start_run(run_name=name):
        pipe = build_pipeline(model).fit(fit[features], fit[CONFIG["target"]])
        val_mae = mean_absolute_error(val[CONFIG["target"]], pipe.predict(val[features]))
        mlflow.log_params(model.get_params())
        mlflow.log_dict(CONFIG, "config.json")
        mlflow.set_tags({"estimator": type(model).__name__, "data_sha256": data_hash})
        mlflow.log_metric("val_mae", val_mae)
        print(f"{name:7s} validation MAE {val_mae:.2f}")

# %% Step 4: choose by querying the store (lowest error first), then test once
runs = mlflow.search_runs(experiment_names=[CONFIG["experiment"]], order_by=["metrics.val_mae ASC"])
print(runs[["tags.mlflow.runName", "tags.estimator", "metrics.val_mae"]].round(2).to_string(index=False))
best_name = runs.iloc[0]["tags.mlflow.runName"]

final = build_pipeline(candidates[best_name]).fit(train_all[features], train_all[CONFIG["target"]])
test_mae = mean_absolute_error(test[CONFIG["target"]], final.predict(test[features]))
with mlflow.start_run(run_name=f"final-{best_name}"):
    mlflow.log_params(candidates[best_name].get_params())
    mlflow.log_metric("test_mae", test_mae)
    mlflow.set_tag("data_sha256", data_hash)
print(f"chosen: {best_name}; test MAE (2025, used once) {test_mae:.2f}")

# %% Step 5: ask the store a question
good = mlflow.search_runs(experiment_names=[CONFIG["experiment"]], filter_string="metrics.val_mae < 12",
                          order_by=["metrics.val_mae ASC"])
print("runs with validation MAE below 12:", good["tags.mlflow.runName"].tolist())
print("runs in the store:", len(mlflow.search_runs(experiment_names=[CONFIG["experiment"]])))

# %% Step 6: is it reproducible? The same seed twice, then a different seed
from sklearn.base import clone


def val_mae_with(name, seed_value):
    model = clone(candidates[name]).set_params(random_state=seed_value)
    pipe = build_pipeline(model).fit(fit[features], fit[CONFIG["target"]])
    return mean_absolute_error(val[CONFIG["target"]], pipe.predict(val[features]))


for name in ["hgb", "forest"]:
    print(f"{name:6s} seed 42 twice: {val_mae_with(name, 42):.4f} and {val_mae_with(name, 42):.4f}; "
          f"seed 7: {val_mae_with(name, 7):.4f}")
