# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import OFFLINE, image_root, use_cached_weights, fixtures

"""Session 9 (one-off): cache frozen ResNet-18 features for 3,000 EuroSAT tiles.

Writes labs/module-09/_fixtures/eurosat_resnet18_feats.npz (about 2 MB), which feature_table_*.py
loads in a second. Run it once (about 15 to 30 seconds on a laptop CPU) whenever the fixture is missing:

    python labs/module-09/l10_5-transfer-learning/prepare_eurosat_features.py

The tiles are NOT the lesson's: the lesson used positions 0 to 3,499 of the seeded permutation,
this file uses positions 3,500 to 6,499 (2,000 for training, 1,000 for testing), same seed, same
128-pixel ImageNet-normalized recipe as Lesson 10.5. Features are stored as float16.
"""
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

torch.manual_seed(42)
weights = models.ResNet18_Weights.DEFAULT
prep = weights.transforms()
tf = transforms.Compose([transforms.Resize(128), transforms.ToTensor(),
                         transforms.Normalize(mean=prep.mean, std=prep.std)])
eurosat = datasets.EuroSAT(image_root("eurosat"), download=not OFFLINE, transform=tf)
order = torch.randperm(len(eurosat), generator=torch.Generator().manual_seed(42)).tolist()
train_idx, test_idx = order[3500:5500], order[5500:6500]

use_cached_weights()
backbone = models.resnet18(weights=weights)
backbone.fc = nn.Identity()
backbone.eval()


def extract(idx):
    feats, labels = [], []
    with torch.inference_mode():
        for xb, yb in DataLoader(Subset(eurosat, idx), batch_size=100):
            feats.append(backbone(xb))
            labels.append(yb)
    return torch.cat(feats).numpy(), torch.cat(labels).numpy()


start = time.perf_counter()
X_train, y_train = extract(train_idx)
X_test, y_test = extract(test_idx)
print(f"features for {len(X_train) + len(X_test):,} tiles in {time.perf_counter() - start:.0f} s")

out = fixtures(__file__) / "eurosat_resnet18_feats.npz"
np.savez_compressed(out, X_train=X_train.astype("float16"), y_train=y_train.astype("int8"),
                    X_test=X_test.astype("float16"), y_test=y_test.astype("int8"),
                    train_idx=np.array(train_idx, dtype="int32"), test_idx=np.array(test_idx, dtype="int32"),
                    classes=np.array(eurosat.classes))
print(f"saved {out.name}: {out.stat().st_size / 1e6:.2f} MB")
