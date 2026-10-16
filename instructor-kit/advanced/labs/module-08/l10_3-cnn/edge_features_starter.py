# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import OFFLINE, image_root

"""Lab 10.3 A: hand-made filters against learned ones.

Lesson 10.3 slid two hand-made edge filters over a boot, then let a CNN learn sixteen filters of its
own. Which is better depends on something the lesson did not vary: how many images you have. Build a
classic computer-vision pipeline (two edge filters, pooled into 48 numbers per image, then logistic
regression) and race it against the lesson's SmallCNN on 1,000, 5,000 and 20,000 training images.
Work through TODO 1 to 3; the file runs as it is, and unfinished steps print a reminder. The CNN
code is given: your job is the hand-made pipeline and the race.
"""
# %% Step 1: Fashion-MNIST as plain tensors (the cached copy; nothing downloads under the checker)
import time

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

train_full = datasets.FashionMNIST(image_root("FashionMNIST"), train=True, download=not OFFLINE)
test_set = datasets.FashionMNIST(image_root("FashionMNIST"), train=False, download=not OFFLINE)
X_all = train_full.data.float().unsqueeze(1) / 255          # (60000, 1, 28, 28), values 0 to 1
y_all = train_full.targets
X_test = test_set.data.float().unsqueeze(1) / 255
y_test = test_set.targets
print(tuple(X_all.shape), tuple(X_test.shape))

# %% Step 2: two hand-made filters (the lesson's vertical edge and its transpose)
vertical_edge = torch.tensor([[-1.0, 0.0, 1.0]] * 3)
filters = torch.stack([vertical_edge, vertical_edge.T]).unsqueeze(1)     # (2 filters, 1 channel, 3, 3)
edges = F.conv2d(X_test[:4], filters, padding=1)
print("edge maps:", tuple(edges.shape), " filter shape:", tuple(filters.shape))


# %% Step 3: 48 numbers per image
# TODO 1: write hand_features(X): (a) F.conv2d(X, filters, padding=1).abs() is the edge strength,
#         (n, 2, 28, 28); (b) F.adaptive_avg_pool2d(..., 4).flatten(1) averages it over a 4 x 4 grid of
#         regions, giving 32 numbers; (c) do the same to X itself for 16 brightness numbers; (d) torch.cat
#         both along dim=1 and return .numpy(). Print how many features one image gets (predict it first).
def hand_features(X):
    return None


if hand_features(X_test[:2]) is None:
    print("TODO 1: hand_features")


# %% Step 4: the lesson's SmallCNN, trained for three epochs
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(32 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 10))

    def forward(self, x):
        return self.classifier(self.features(x))


def cnn_accuracy(X, y, epochs=3):
    torch.manual_seed(42)
    model = SmallCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    torch.manual_seed(42)
    loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        return (model(X_test).argmax(dim=1) == y_test).float().mean().item()


# %% Step 5: the race
# TODO 2: for n in [1000, 5000, 20000]: take the first n training images; fit
#         make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)) on hand_features of them and
#         score it on hand_features(X_test); then call cnn_accuracy(X, y). Print both per row.
#         Predict before you run: which wins at 1,000 images? At 20,000?
print("TODO 2: the race at three training-set sizes")

# TODO 3: in two sentences: why do two filters chosen by a person win with little data, and what does
#         the CNN need to overtake them?
print("TODO 3: your explanation")

print("\nExpected when done: 48 features per image; hand-made + logistic regression 0.795 / 0.819 / 0.827")
print("against SmallCNN 0.676 / 0.782 / 0.863 at 1,000 / 5,000 / 20,000 images: the person's filters win with")
print("little data, the learned filters overtake them at 20,000. (CNN numbers may vary slightly.)")
