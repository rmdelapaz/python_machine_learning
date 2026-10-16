# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Debug 8.3: Lesson 8.3's product families, rebuilt in scikit-learn from an older tutorial (deliberately broken).

Three bugs stop the script with an error; one does not, and makes the cross-check against SciPy look
like a failure. Fix them one at a time. When all four are fixed it prints
  family sizes: [31, 36, 36, 38, 59]
  agreement with SciPy's tree (ARI): 1.000
"""
# %%
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

interactions = pd.read_csv(DATA / "interactions.csv")
bought = pd.crosstab(interactions["product_id"], interactions["customer_id"]).clip(upper=1).to_numpy()
D = squareform(pdist(bought, metric="correlation"))        # 200 x 200 correlation distances

# %% Families: everything merged below a correlation distance of 0.97
model = AgglomerativeClustering(n_clusters=5, affinity="precomputed", linkage="ward", distance_threshold=0.97)
families = model.fit_predict(D)
print("family sizes:", sorted(np.bincount(families).tolist()))

# %% Cross-check with SciPy's tree, same distance, same linkage, same cut
scipy_families = fcluster(linkage(bought, method="single", metric="correlation"), t=0.97, criterion="distance")
print(f"agreement with SciPy's tree (ARI): {adjusted_rand_score(families, scipy_families):.3f}")
