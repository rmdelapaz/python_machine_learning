# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA, OFFLINE, OUTPUT, image_root

"""Lab 9.3 A (SOLUTION): a map of the TRAINING images, checked with numbers.

Lesson 9.3 mapped 1,000 Fashion-MNIST test images. Here: 1,500 training images, a UMAP map and a
t-SNE map, a neighbor check for each, which garments the map mixes up, and 500 new images placed on
the UMAP map with transform. Labels are only used to color and to check, never to build the maps.
"""
# %% Step 1: 1,500 training images as rows of 784 pixels, then 50 PCA components
import numpy as np
import pandas as pd
from torchvision import datasets
from sklearn.decomposition import PCA

train = datasets.FashionMNIST(image_root("FashionMNIST"), train=True, download=not OFFLINE)
images = train.data.numpy().reshape(len(train), -1) / 255
labels = train.targets.numpy()
names = train.classes
rng = np.random.default_rng(42)
pick = rng.choice(len(images), size=1500, replace=False)
img, lab = images[pick], labels[pick]
pca50 = PCA(n_components=50, random_state=42).fit(img)
img50 = pca50.transform(img)
print(img.shape, "->", img50.shape, f"variance kept {pca50.explained_variance_ratio_.sum():.3f}")

# %% Step 2: two maps
import warnings
from sklearn.manifold import TSNE
import umap

# UMAP warns that a seed forces one core (Lesson 9.3, Common Mistake 3); expected, so silence it here
warnings.filterwarnings("ignore", message="n_jobs value")

emb_tsne = TSNE(perplexity=30, random_state=42).fit_transform(img50)
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
emb_umap = reducer.fit_transform(img50)

# %% Step 3: check them with numbers
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import trustworthiness


def neighbors_kept(original, embedded, k=10):
    nn_o = NearestNeighbors(n_neighbors=k + 1).fit(original).kneighbors(original, return_distance=False)[:, 1:]
    nn_e = NearestNeighbors(n_neighbors=k + 1).fit(embedded).kneighbors(embedded, return_distance=False)[:, 1:]
    return np.mean([len(set(a) & set(b)) / k for a, b in zip(nn_o, nn_e)])


for name, emb in [("PCA (first 2)", img50[:, :2]), ("t-SNE", emb_tsne), ("UMAP", emb_umap)]:
    print(f"{name:14s} neighbors kept {neighbors_kept(img50, emb):.3f}   "
          f"trustworthiness {trustworthiness(img50, emb, n_neighbors=10):.3f}")

# %% Step 4: which garments does the UMAP map mix up?
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict

guess = cross_val_predict(KNeighborsClassifier(n_neighbors=10), emb_umap, lab, cv=5)
per_class = pd.Series(guess == lab).groupby(pd.Series(lab).map(dict(enumerate(names)))).mean()
print(per_class.round(2).sort_values().to_string())

# %% Step 5: place 500 new training images on the existing UMAP map
rest = np.setdiff1d(np.arange(len(images)), pick)
new = rng.choice(rest, size=500, replace=False)
new_xy = reducer.transform(pca50.transform(images[new]))
knn = KNeighborsClassifier(n_neighbors=10).fit(emb_umap, lab)
print(f"\n500 new images placed {new_xy.shape}; landing among their own class: {knn.score(new_xy, labels[new]):.3f}")

# %% Step 6: save the picture (to OUTPUT, never next to the data)
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
for k, name in enumerate(names):
    ax.scatter(emb_umap[lab == k, 0], emb_umap[lab == k, 1], s=4, color=plt.cm.tab10(k), label=name)
ax.set_xticks([])
ax.set_yticks([])
ax.legend(markerscale=3, fontsize=7)
fig.savefig(OUTPUT / "fashion_umap.png", dpi=80)
print("saved fashion_umap.png")
