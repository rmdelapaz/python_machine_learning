# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Find the bugs (Lesson 8.4). This handout is deliberately broken.

A colleague clustered customers by return behavior with DBSCAN and wrote a short report. Two bugs stop
the script with an error; three more run without complaint and make the report wrong. Fix all five.
Fixed, it prints: clusters: 4, noise share: 7.4%, churn among noise customers: 0.083, silhouette of the
clustered customers: 0.268 (and no cluster for the new customer: DBSCAN cannot predict).
"""
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

customers = pd.read_csv(DATA / "customers_churn.csv")
cols = ["returns_12m", "returns_rate", "spend_12m"]
X = customers[cols].to_numpy()                        # spend in dollars, return_rate a fraction

labels = DBSCAN(eps=0.5, min_samples=20).fit_predict(X)
print("clusters:", len(set(labels)))
print(f"noise share: {(labels == -1).mean():.1%}")

noise = labels == -1
print(f"churn among noise customers: {customers.loc[noise, 'churned'].mean():.3f}")
print("silhouette:", round(silhouette_score(X, labels), 3))

model = DBSCAN(eps=0.5, min_samples=20).fit(X)
new_customer = np.array([[9, 0.6, 900.0]])
print("new customer's cluster:", model.predict(new_customer))
