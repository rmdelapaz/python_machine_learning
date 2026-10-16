# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import OFFLINE, image_root

"""Debug 10.3: a clothing CNN with four bugs (deliberately broken).

Two bugs stop the script with a shape error; two do not, and the script still prints accuracies.
Fix one at a time and rerun. When everything is fixed it prints test accuracy 0.747 on both lines.
"""
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

train_full = datasets.FashionMNIST(image_root("FashionMNIST"), train=True, download=not OFFLINE)
test_set = datasets.FashionMNIST(image_root("FashionMNIST"), train=False, download=not OFFLINE)
X_train = train_full.data[:5000].float() / 255
y_train = train_full.targets[:5000]
X_test = test_set.data.float().unsqueeze(1) / 255
y_test = test_set.targets

torch.manual_seed(42)
model = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(), nn.Linear(32 * 28 * 28, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 10), nn.Softmax(dim=1),
)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
torch.manual_seed(42)
loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
for epoch in range(2):
    model.train()
    for xb, yb in loader:
        optimizer.zero_grad()
        loss_fn(model(xb), yb).backward()
        optimizer.step()


def test_accuracy():
    with torch.no_grad():
        return (model(X_test).argmax(dim=1) == y_test).float().mean().item()


print(f"test accuracy {test_accuracy():.3f}")
print(f"again         {test_accuracy():.3f}")
