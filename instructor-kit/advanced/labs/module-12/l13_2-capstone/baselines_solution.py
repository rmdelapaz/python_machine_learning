# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Capstone lab 1 (SOLUTION): a second honest baseline for every problem (Milestones 1 to 3).

The starter template gives each problem ONE baseline. A good plan names the bar to beat, and often
there is a second, stronger "no machine learning" answer: what a sensible person would do today with
a spreadsheet. If your model cannot beat THAT, it is not worth deploying. This lab measures one more
baseline per problem, on the same split and metric as the template.
"""
# %% Step 1: the template's baselines (the lesson's five functions, in capstone_start.py next to this file)
sys.path.insert(0, str(pathlib.Path(globals().get("__file__", "lab.py")).resolve().parent))
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, mean_absolute_error, recall_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from capstone_start import STARTERS

problems = {name: start() for name, start in STARTERS.items()}
rows = [{"problem": n, "metric": p["metric"], "baseline": p["baseline"][0], "value": round(p["baseline"][1], 3)}
        for n, p in problems.items()]

# %% Step 2: demand: per product AND store, and "same week last year"
dm = problems["demand"]
train, test = dm["train"], dm["test"]
per_ps = train.groupby(["product_id", "store"])["units_sold"].mean()
pred = per_ps.reindex(pd.MultiIndex.from_frame(test[["product_id", "store"]])).to_numpy()
rows.append({"problem": "demand", "metric": dm["metric"], "baseline": "per product x store mean",
             "value": round(mean_absolute_error(test["units_sold"], pred), 3)})
last_year = train.groupby(["product_id", "store", "week_of_year"])["units_sold"].mean()   # 2024 has two ISO week-1 rows
pred = last_year.reindex(pd.MultiIndex.from_frame(test[["product_id", "store", "week_of_year"]])).to_numpy()
rows.append({"problem": "demand", "metric": dm["metric"], "baseline": "same week last year",
             "value": round(mean_absolute_error(test["units_sold"], pred), 3)})

# %% Step 3: fraud: a one-line rule the team could use today (rank by velocity_24h, ties by amount)
fr = problems["fraud"]
june = fr["test"]
rule = june["velocity_24h"] + june["amount"] / 1e5
rows.append({"problem": "fraud", "metric": fr["metric"], "baseline": "rule: velocity_24h",
             "value": round(average_precision_score(june["is_fraud"], rule), 3)})
held = rule >= np.quantile(rule, 0.99)
rule_catch = int(june["is_fraud"][held].sum())

# %% Step 4: reviews: a keyword rule ("negative if it says not, return or regret")
rv = problems["reviews"]
rtest = rv["test"]
says_bad = rtest["text"].str.contains(r"\b(?:not|return|returned|regret)\b", case=False)
rows.append({"problem": "reviews", "metric": rv["metric"], "baseline": "keyword rule",
             "value": round(accuracy_score(rtest["positive"], (~says_bad).astype(int)), 3)})
rule_neg_recall = recall_score(rtest["positive"], (~says_bad).astype(int), pos_label=0)

# %% Step 5: recommendations: ten random unowned products
rc = problems["recs"]
rng = np.random.default_rng(42)
products = sorted(set(rc["train"]["product_id"]) | set(rc["test"]["product_id"]))
owned = rc["train"].groupby("customer_id")["product_id"].apply(set)
hits = 0
for customer, item in zip(rc["test"]["customer_id"], rc["test"]["product_id"]):
    unowned = [p for p in products if p not in owned[customer]]
    hits += item in set(rng.choice(unowned, 10, replace=False))
rows.append({"problem": "recs", "metric": rc["metric"], "baseline": "random unowned", "value": round(hits / len(rc["test"]), 3)})

# %% Step 6: segments: five random groups (what a silhouette of "no structure" looks like)
sg = problems["segments"]
X = StandardScaler().fit_transform(sg["train"][sg["features"]])
random_labels = rng.integers(0, 5, len(X))
rows.append({"problem": "segments", "metric": sg["metric"], "baseline": "five random groups",
             "value": round(silhouette_score(X, random_labels, sample_size=3000, random_state=42), 3)})

# %% Step 7: the table every plan should quote
order = {name: i for i, name in enumerate(STARTERS)}
table = pd.DataFrame(rows).sort_values("problem", key=lambda s: s.map(order), kind="stable").set_index(["problem", "baseline"])
print(table.to_string())
print(f"\nfraud rule, top 1% of June ({held.sum()} payments): {rule_catch} of {june['is_fraud'].sum()} frauds")
print(f"reviews keyword rule: negative recall {rule_neg_recall:.3f} (always positive: 0.000)")
