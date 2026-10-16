"""capstone_start.py: the lesson's starter template (Lesson 13.2, "The Starter Template"), as a module.

The labs in this folder import it (`from capstone_start import start_fraud`). The five functions are the
lesson's, unchanged except that they read from `labs_paths.DATA` instead of the relative `data/` folder.
Each one loads one problem's data, makes the split the brief asks for and measures the honest baseline.
"""
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, average_precision_score, mean_absolute_error, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from labs_paths import DATA


def start_demand():
    """Weekly units per product and store. Train 2024, test 2025. Metric: MAE in units."""
    df = pd.read_csv(DATA / "weekly_demand.csv", parse_dates=["week_start"])
    train, test = df[df["year"] == 2024], df[df["year"] == 2025]
    per_product = train.groupby("product_id")["units_sold"].mean()
    baseline = mean_absolute_error(test["units_sold"], test["product_id"].map(per_product))
    return {"train": train, "test": test, "target": "units_sold",
            "metric": "MAE (units)", "baseline": ("per-product mean", baseline)}


def start_fraud():
    """Card payments. Train January to May, test June. Metric: average precision."""
    df = pd.read_csv(DATA / "transactions.csv", parse_dates=["timestamp"])
    is_test = df["timestamp"] >= "2025-06-01"
    train, test = df[~is_test], df[is_test]
    dummy = DummyClassifier(strategy="prior").fit(train[["amount"]], train["is_fraud"])
    baseline = average_precision_score(test["is_fraud"], dummy.predict_proba(test[["amount"]])[:, 1])
    return {"train": train, "test": test, "target": "is_fraud",
            "metric": "average precision", "baseline": ("no model", baseline)}


def start_reviews():
    """Review text, positive (4-5 stars) vs negative (1-2). Metric: accuracy and recall of negatives."""
    df = pd.read_csv(DATA / "reviews.csv")
    df = df[df["rating"] != 3].assign(positive=lambda d: (d["rating"] >= 4).astype(int))
    train, test = train_test_split(df, test_size=0.25, stratify=df["positive"], random_state=42)
    baseline = accuracy_score(test["positive"], np.ones(len(test), dtype=int))
    return {"train": train, "test": test, "target": "positive",
            "metric": "accuracy", "baseline": ("always positive", baseline)}


def start_recs():
    """Customer x product purchases. Leave-last-out test. Metric: hit rate at 10."""
    df = pd.read_csv(DATA / "interactions.csv", parse_dates=["last_purchase_date"])
    df = df.sort_values(["customer_id", "last_purchase_date", "product_id"])
    test = df.groupby("customer_id").tail(1)              # each customer's latest purchase
    train = df.drop(test.index)
    popular = train.groupby("product_id")["customer_id"].count().sort_values(ascending=False)
    owned = train.groupby("customer_id")["product_id"].apply(set)
    hits = 0
    for customer, item in zip(test["customer_id"], test["product_id"]):
        top10 = [p for p in popular.index if p not in owned[customer]][:10]
        hits += item in top10
    return {"train": train, "test": test, "target": "product_id",
            "metric": "hit rate at 10", "baseline": ("most popular unowned", hits / len(test))}


def start_segments():
    """Customer behavior, no target. Metric: silhouette, plus stability and profiles."""
    df = pd.read_csv(DATA / "customers_churn.csv")
    cols = ["orders_12m", "avg_order_value", "discount_share", "return_rate"]
    X = StandardScaler().fit_transform(df[cols])
    baseline = silhouette_score(X, df["loyalty_tier"], sample_size=3000, random_state=42)
    return {"train": df, "test": None, "target": None, "features": cols,
            "metric": "silhouette", "baseline": ("existing loyalty tiers", baseline)}


STARTERS = {"demand": start_demand, "fraud": start_fraud, "reviews": start_reviews,
            "recs": start_recs, "segments": start_segments}
