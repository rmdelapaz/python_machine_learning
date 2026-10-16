# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Debug 11.2: an item-kNN evaluation with FOUR bugs (deliberately broken handout).

Two bugs stop the script with an error; two let it print hit rates you should not believe. Find and
fix all four. When everything is fixed it prints: random 0.047, best sellers 0.209, item-kNN 0.308.
Hint: compare every number with the random baseline and with what the lesson printed.
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

interactions = pd.read_csv(DATA / "interactions.csv").sort_values(["customer_id", "last_purchase_date", "product_id"])
products = pd.read_csv(DATA / "products_ml.csv")
test = interactions.groupby("customer_id").head(1)          # each customer's held-out purchase
train = interactions.drop(test.index)

customer_ids = np.sort(interactions["customer_id"].unique())
product_ids = products["product_id"].to_numpy()
row_of = pd.Series(np.arange(len(customer_ids)), index=customer_ids)
col_of = pd.Series(np.arange(len(product_ids)), index=product_ids)
R = csr_matrix((np.ones(len(train)), (row_of[train["customer_id"]].to_numpy(), col_of[train["product_id"]].to_numpy())),
               shape=(len(customer_ids), len(product_ids)))
owned = R.toarray() > 0
test_col = col_of[test.set_index("customer_id").loc[customer_ids, "product_id"]].to_numpy()


def hit_rate(scores, k=10):
    scores = np.array(scores, dtype=float)
    scores[owned] = -np.inf
    top = np.argsort(scores, axis=1)[:, :k]
    return (top == test_col).any(axis=1).mean()


rng = np.random.default_rng(42)
print(f"random       HR@10 = {hit_rate(rng.random(R.shape)):.3f}")
popularity = np.asarray(R.sum(axis=0)).ravel()
print(f"best sellers HR@10 = {hit_rate(np.tile(popularity, (R.shape[0], 1))):.3f}")

item_sim = cosine_similarity(R)
np.fill_diagonal(item_sim, 0)
print(f"item-kNN     HR@10 = {hit_rate(R @ item_sim):.3f}")
