# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 11.2 A: is a smarter best-seller list enough?

Before paying for "personalization", the marketing team asks a fair question: what if each customer
simply got the best sellers of THEIR store, or of THEIR favorite activity? Build both segment
baselines with the lesson's honest test and see where they land between the global best sellers and
item-based collaborative filtering. Work through TODO 1 to 5; run the file after each one.
It runs as it is: unfinished steps print a reminder.
"""
# %% Step 1: the leave-last-out split and the matrix (the lesson's recipe, done for you)
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
# TODO 1: build `by_store`, a (3000, 200) array where every row holds the popularity of each product
#         among customers of THAT row's store: for each store s, rows = store == s, and
#         by_store[rows] = R[rows].sum(axis=0) (flattened with np.asarray(...).ravel()).
#         Add it to `results` as "home-store best sellers". Predict first: better than 0.209? ______
print("TODO 1: home-store best sellers")

# %% Step 3: best sellers of the customer's favorite activity
activity = products["primary_activity"].to_numpy()
activities = np.unique(activity)
# TODO 2: counts = a (3000, 6) array: for each activity a, owned[:, activity == a].sum(axis=1).
#         favorite = activities[counts.argmax(axis=1)]; print how many customers have each favorite.
# TODO 3: in_favorite = activity[None, :] == favorite[:, None] (customers x products). The score is
#         popularity * in_favorite + popularity / 1e6 (the tiny second term breaks ties by popularity,
#         so the list is still full when the activity has few unowned products). Add it to `results`.
print("TODO 2-3: favorite-activity best sellers")

# %% Step 4: item-based collaborative filtering, for comparison
# TODO 4: item_sim = cosine_similarity(R.T) with a zero diagonal; results["item-kNN"] = R @ item_sim.
print("TODO 4: item-kNN")

# %% Step 5: the scoreboard: accuracy and variety
top20 = np.argsort(-popularity)[:20]
# TODO 5: for every entry of `results`, print HR@10, the number of different products on anyone's
#         list (len(np.unique(top_k(scores)))) and the share of list slots taken by the top 20.
print("TODO 5: the scoreboard")

# %% Stretch: compare the activity list and item-kNN for customers whose favorite is "general" and for everyone else.

print("\nExpected when done: home-store best sellers HR@10 0.209 (36 products shown, 98% top-20 slots);")
print("favorites camping 811, general 611, hiking 492, climbing 441, running 344, paddling 301;")
print("favorite-activity best sellers 0.265 (183 products, 31%); item-kNN 0.308 (161 products, 51%).")
print("Stretch: 'general' fans 0.229 vs 0.280 for item-kNN; everyone else 0.274 vs 0.315.")
