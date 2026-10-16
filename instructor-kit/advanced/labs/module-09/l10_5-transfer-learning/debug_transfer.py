# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import OFFLINE, image_root, use_cached_weights

"""Debug 10.5: a transfer-learning script with FOUR bugs (deliberately broken handout).

A colleague wrote a quick frozen-feature classifier on 400 EuroSAT tiles. Two bugs stop the script
with an error; two others let it run and quietly cost accuracy. Find and fix all four. Read every
error from the last line up, and do not trust a number until all four are fixed.
When everything is fixed, it prints: test accuracy of the linear head 0.895 (yours may vary slightly).
"""
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, models, transforms

use_cached_weights()
weights = models.ResNet18_Weights.DEFAULT
tf = transforms.Compose([transforms.Resize(128), transforms.ToTensor()])
eurosat = datasets.EuroSAT(image_root("eurosat"), download=not OFFLINE, transform=tf)
order = torch.randperm(len(eurosat), generator=torch.Generator().manual_seed(42)).tolist()
train_idx, test_idx = order[6500:6900], order[6900:7100]

backbone = models.resnet18(weights=weights)
backbone.classifier = nn.Identity()


def extract(idx):
    feats, labels = [], []
    with torch.no_grad():
        for xb, yb in DataLoader(Subset(eurosat, idx), batch_size=50):
            feats.append(backbone(xb))
            labels.append(yb)
    return torch.cat(feats), torch.cat(labels)


F_train, y_train = extract(train_idx)
F_test, y_test = extract(test_idx)
print("features:", tuple(F_train.shape), tuple(F_test.shape))

torch.manual_seed(42)
head = nn.Linear(512, 10)
for p in list(backbone.parameters()) + list(head.parameters()):
    p.requires_grad = False
optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
for epoch in range(30):
    for fb, yb in DataLoader(TensorDataset(F_train, y_train), batch_size=64, shuffle=True):
        optimizer.zero_grad()
        nn.functional.cross_entropy(head(fb), yb).backward()
        optimizer.step()

with torch.no_grad():
    acc = (head(F_test).argmax(dim=1) == y_test).float().mean().item()
print(f"test accuracy of the linear head: {acc:.3f}")
