# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import OFFLINE, image_root, use_cached_weights, fixtures

"""Lab 10.5 A (SOLUTION): pretrained features are just a table.

The trip-planning team will not label every tile, and a wrong land-cover label on a trail map is
embarrassing. Using frozen ResNet-18 features of 3,000 NEW EuroSAT tiles (not the lesson's), treat the
512 numbers per tile like any scikit-learn table: cross-validate a head, try k-nearest neighbors, find
the weakest classes, and build a "send to a human" queue for the tiles the model is unsure about.
"""
# %% Step 1: load the cached features (computed once by prepare_eurosat_features.py)
import time

import numpy as np
import pandas as pd

path = fixtures(__file__) / "eurosat_resnet18_feats.npz"
if path.is_file():
    data = np.load(path)
    X_train, y_train = data["X_train"].astype("float32"), data["y_train"].astype(int)
    X_test, y_test = data["X_test"].astype("float32"), data["y_test"].astype(int)
    classes = data["classes"].tolist()
else:  # fallback: compute the same features now (about 15 to 30 seconds)
    print("NOTE: fixture missing, computing the features now (run prepare_eurosat_features.py once to cache them)")
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, models, transforms
    weights = models.ResNet18_Weights.DEFAULT
    tf = transforms.Compose([transforms.Resize(128), transforms.ToTensor(),
                             transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std)])
    eurosat = datasets.EuroSAT(image_root("eurosat"), download=not OFFLINE, transform=tf)
    order = torch.randperm(len(eurosat), generator=torch.Generator().manual_seed(42)).tolist()
    use_cached_weights()
    backbone = models.resnet18(weights=weights)
    backbone.fc = nn.Identity()
    backbone.eval()

    def extract(idx):
        f, l = [], []
        with torch.inference_mode():
            for xb, yb in DataLoader(Subset(eurosat, idx), batch_size=100):
                f.append(backbone(xb))
                l.append(yb)
        return torch.cat(f).numpy().astype("float16").astype("float32"), torch.cat(l).numpy()

    X_train, y_train = extract(order[3500:5500])
    X_test, y_test = extract(order[5500:6500])
    classes = eurosat.classes

print(X_train.shape, X_test.shape)
majority = np.bincount(y_train).argmax()
print(f"majority-class baseline ({classes[majority]}): test accuracy {(y_test == majority).mean():.3f}")

# %% Step 2: a cross-validated head (Module 1 habits on deep-learning features)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
heads = {"logistic regression": LogisticRegression(max_iter=2000),
         "scaled logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))}
for name, head in heads.items():
    scores = cross_val_score(head, X_train, y_train, cv=cv)
    print(f"{name:27s} CV accuracy {scores.mean():.3f} ± {scores.std():.3f}")

head = LogisticRegression(max_iter=2000).fit(X_train, y_train)
print(f"test accuracy (logistic regression, used once): {head.score(X_test, y_test):.3f}")

# %% Step 3: no training at all: k-nearest neighbors on the features
from sklearn.neighbors import KNeighborsClassifier

for k in [1, 5, 15]:
    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine").fit(X_train, y_train)
    print(f"{k:2d}-nearest neighbors (cosine): test accuracy {knn.score(X_test, y_test):.3f}")

# %% Step 4: which land types are hard?
pred = head.predict(X_test)
per_class = pd.Series({c: (pred[y_test == i] == i).mean() for i, c in enumerate(classes)}).sort_values()
print("\nweakest classes:", per_class.head(3).round(3).to_dict())
pairs = pd.crosstab(pd.Series(np.array(classes)[y_test], name="true"), pd.Series(np.array(classes)[pred], name="predicted"))
off = pairs.where(~np.eye(len(pairs), dtype=bool)).stack().sort_values(ascending=False)
print("most frequent mix-ups:", {f"{t} -> {p}": int(n) for (t, p), n in off.head(3).items()})

# %% Step 5: a "send to a human" queue for unsure tiles
proba = head.predict_proba(X_test)
confidence = proba.max(axis=1)
correct = pred == y_test
for cut in [0.6, 0.8]:
    queue = confidence < cut
    print(f"confidence < {cut}: {queue.mean():.1%} of tiles to a human; accuracy in the queue "
          f"{correct[queue].mean():.3f}, on the rest {correct[~queue].mean():.3f}")

# %% Stretch: how many of the 512 numbers does the head need?
from sklearn.decomposition import PCA

for n in [10, 50, 200]:
    small = make_pipeline(PCA(n_components=n, random_state=42), LogisticRegression(max_iter=2000)).fit(X_train, y_train)
    print(f"PCA to {n:3d} components: test accuracy {small.score(X_test, y_test):.3f}")
