# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 11.2 A (SOLUTION): is a smarter best-seller list enough?

Before paying for "personalization", the marketing team asks a fair question: what if each customer
simply got the best sellers of THEIR store, or of THEIR favorite activity? Those are cheap, easy to
explain, and personal in a way. Build both segment baselines with the lesson's honest test
(leave-last-out, hit rate at 10, owned products masked) and see where they land between the global
best sellers and item-based collaborative filtering.
"""
# %% Step 1: the leave-last-out split and the matrix (the lesson's recipe)
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

interactions = pd.read_csv(DATA / "interactions.csv").sort_values(["customer_id", "last_purchase_date", "product_id"])
products = pd.read_csv(DATA / "products_ml.csv")
customers = pd.read_csv(DATA / "customers_churn.csv", usecols=["customer_id", "home_store"])

test = interactions.groupby("customer_id").tail(1)
train = interactions.drop(test.index)
customer_ids = np.sort(interactions["customer_id"].unique())
product_ids = products["product_id"].to_numpy()
row_of = pd.Series(np.arange(len(customer_ids)), index=customer_ids)
col_of = pd.Series(np.arange(len(product_ids)), index=product_ids)
R = csr_matrix((np.ones(len(train)), (row_of[train["customer_id"]].to_numpy(), col_of[train["product_id"]].to_numpy())),
               shape=(len(customer_ids), len(product_ids)))
owned = R.toarray() > 0
test_col = col_of[test.set_index("customer_id").loc[customer_ids, "product_id"]].to_numpy()


def top_k(scores, k=10):
    scores = np.array(scores, dtype=float)
    scores[owned] = -np.inf
    return np.argsort(-scores, axis=1)[:, :k]


def hit_rate(scores, k=10):
    return (top_k(scores, k) == test_col[:, None]).any(axis=1).mean()


popularity = np.asarray(R.sum(axis=0)).ravel()
results = {"global best sellers": np.tile(popularity, (R.shape[0], 1))}
print(R.shape, f"global best sellers HR@10 {hit_rate(results['global best sellers']):.3f}")

# %% Step 2: best sellers of the customer's home store
store = customers.set_index("customer_id").loc[customer_ids, "home_store"].to_numpy()
by_store = np.zeros(R.shape)
for s in np.unique(store):
    rows = store == s
    by_store[rows] = np.asarray(R[rows].sum(axis=0)).ravel()      # popularity among that store's customers
results["home-store best sellers"] = by_store
print("stores:", pd.Series(store).value_counts().to_dict())

# %% Step 3: best sellers of the customer's favorite activity
activity = products["primary_activity"].to_numpy()
activities = np.unique(activity)
counts = np.stack([owned[:, activity == a].sum(axis=1) for a in activities], axis=1)   # customers x activities
favorite = activities[counts.argmax(axis=1)]
print("favorite activity:", pd.Series(favorite).value_counts().to_dict())
in_favorite = activity[None, :] == favorite[:, None]                                   # customers x products
results["favorite-activity best sellers"] = popularity * in_favorite + popularity / 1e6   # ties broken by popularity

# %% Step 4: item-based collaborative filtering, for comparison
item_sim = cosine_similarity(R.T)
np.fill_diagonal(item_sim, 0)
results["item-kNN"] = R @ item_sim

# %% Step 5: the scoreboard: accuracy and variety
top20 = np.argsort(-popularity)[:20]
for name, scores in results.items():
    lists = top_k(scores)
    print(f"{name:31s} HR@10 {hit_rate(scores):.3f}   products shown {len(np.unique(lists)):3d}   "
          f"top-20 slots {np.isin(lists, top20).mean():.0%}")

# %% Stretch: who does the activity list fail? Customers whose favorite is "general"
general = favorite == "general"
hits_act = (top_k(results["favorite-activity best sellers"]) == test_col[:, None]).any(axis=1)
hits_knn = (top_k(results["item-kNN"]) == test_col[:, None]).any(axis=1)
print(f"\nfavorite 'general' ({general.sum()} customers): activity list {hits_act[general].mean():.3f}, "
      f"item-kNN {hits_knn[general].mean():.3f}; everyone else: {hits_act[~general].mean():.3f} vs {hits_knn[~general].mean():.3f}")
