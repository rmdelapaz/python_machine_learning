# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Debug 8.2: a "choose k, then assign newcomers" script with three bugs (deliberately broken handout).

One bug stops the script with an error. Two do not: the script picks a k and assigns new customers, and
both answers are wrong. Fix them one at a time. When all three are fixed it prints
  k chosen by silhouette: 5 (0.479)
  newcomers: [0, 0, 0]   (three discount-heavy shoppers: all Deal Hunters, cluster 0)
"""
# %%
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

customers = pd.read_csv(DATA / "customers_churn.csv")
behavior = ["orders_12m", "avg_order_value", "discount_share", "return_rate"]
log_some = ColumnTransformer(
    [("log", FunctionTransformer(np.log, feature_names_out="one-to-one"), ["orders_12m", "avg_order_value"])],
    remainder="passthrough", verbose_feature_names_out=False,
)
prep = make_pipeline(log_some, StandardScaler()).set_output(transform="pandas")
Z = prep.fit_transform(customers[behavior])

# %% Score every k
scores = {}
for k in range(1, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Z)
    scores[k] = {"inertia": km.inertia_, "silhouette": silhouette_score(Z, km.labels_)}
table = pd.DataFrame(scores).T
best_k = int(table["inertia"].idxmin())
print(f"k chosen by silhouette: {best_k} ({table.loc[best_k, 'silhouette']:.3f})")

# %% Assign three newcomers
final = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(Z)
newcomers = pd.DataFrame({"orders_12m": [5, 6, 4], "avg_order_value": [60.0, 65.0, 70.0],
                          "discount_share": [0.50, 0.55, 0.60], "return_rate": [0.07, 0.08, 0.09]})
new_Z = make_pipeline(log_some, StandardScaler()).set_output(transform="pandas").fit_transform(newcomers)
print("newcomers:", final.predict(new_Z).tolist())
