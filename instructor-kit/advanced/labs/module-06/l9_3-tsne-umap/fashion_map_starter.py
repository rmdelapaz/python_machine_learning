# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA, OFFLINE, OUTPUT, image_root

"""Lab 9.3 A: a map of the TRAINING images, checked with numbers.

Lesson 9.3 mapped 1,000 Fashion-MNIST test images. Here: 1,500 training images, a UMAP map and a
t-SNE map, a neighbor check for each, which garments the map mixes up, and 500 new images placed on
the UMAP map with transform. Labels only color and check; they never build the maps.
Work through TODO 1 to 5. Fashion-MNIST must be downloaded already (see the session's bring list).
"""
# %% Step 1: 1,500 training images, then 50 PCA components (done for you)
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

warnings.filterwarnings("ignore", message="n_jobs value")      # UMAP's expected seed warning (Lesson 9.3)
# TODO 1: emb_tsne = TSNE(perplexity=30, random_state=42).fit_transform(img50)
#         reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42); emb_umap = reducer.fit_transform(img50)
emb_tsne = emb_umap = reducer = None
if reducer is None:
    print("TODO 1: a t-SNE map and a UMAP map (about 20 seconds)")

# %% Step 3: check with numbers
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.manifold import trustworthiness


def neighbors_kept(original, embedded, k=10):
    nn_o = NearestNeighbors(n_neighbors=k + 1).fit(original).kneighbors(original, return_distance=False)[:, 1:]
    nn_e = NearestNeighbors(n_neighbors=k + 1).fit(embedded).kneighbors(embedded, return_distance=False)[:, 1:]
    return np.mean([len(set(a) & set(b)) / k for a, b in zip(nn_o, nn_e)])

# TODO 2: neighbors kept and trustworthiness (n_neighbors=10) for img50[:, :2] (a PCA view), t-SNE and UMAP.
print("TODO 2: neighbor checks for three maps")

# %% Step 4: which garments get mixed up?
from sklearn.model_selection import cross_val_predict
# TODO 3: guess = cross_val_predict(KNeighborsClassifier(n_neighbors=10), emb_umap, lab, cv=5); print the
#         share of correct guesses per class name, sorted. Which three garments are hardest, and why?
print("TODO 3: per-class agreement on the UMAP map")

# %% Step 5: new images on the old map
# TODO 4: 500 new training images (rng.choice from the rows not in pick), through pca50.transform and then
#         reducer.transform; score a KNeighborsClassifier(n_neighbors=10) fitted on (emb_umap, lab) on them.
# TODO 5: save a scatter of emb_umap colored by class to OUTPUT / "fashion_umap.png".
print("TODO 4-5: place 500 new images; save the picture")

print("\nExpected when done: 50 components keep 0.869 of the pixel variance; neighbors kept 0.195 (PCA view),")
print("0.573 (t-SNE), 0.495 (UMAP), trustworthiness 0.922 / 0.991 / 0.986; hardest: Shirt 0.32, Coat 0.51,")
print("Pullover 0.62, easiest Trouser 0.90; 0.730 of the 500 new images land among their own class.")
