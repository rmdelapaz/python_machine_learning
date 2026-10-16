# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import OFFLINE, fixtures, image_root

"""prepare_fashion_checkpoints.py: train Lesson 10.3's two models once and save their weights as fixtures.

Writes labs/module-08/_fixtures/fashion_dense.pt and fashion_small_cnn.pt (state_dicts, float32, about
0.4 MB each): the dense baseline and the SmallCNN trained EXACTLY as on the lesson page (first 20,000
training images, 5 epochs, Adam 1e-3, batch 64, torch.manual_seed(42)), so they reproduce the page's test
accuracies (0.840 and 0.864). shift_test_*.py loads them in a second; the facilitator guide uses them as
the fallback checkpoint when a laptop is too slow to train live. The checker never runs this file.

Rebuild (about 30 to 60 s on 4 threads), from the kit folder:
    OMP_NUM_THREADS=4 python labs/module-08/l10_3-cnn/prepare_fashion_checkpoints.py
"""
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

torch.set_num_threads(4)
train_full = datasets.FashionMNIST(image_root("FashionMNIST"), train=True, download=not OFFLINE,
                                   transform=transforms.ToTensor())
test_set = datasets.FashionMNIST(image_root("FashionMNIST"), train=False, download=not OFFLINE,
                                 transform=transforms.ToTensor())
train_set = Subset(train_full, range(20_000))
val_loader = DataLoader(Subset(train_full, range(55_000, 60_000)), batch_size=1000)
test_loader = DataLoader(test_set, batch_size=1000)


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def accuracy(model, loader):
    model.eval()
    correct, n = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            correct += (model(xb).argmax(dim=1) == yb).sum().item()
            n += len(yb)
    return correct / n


def fit(model, epochs=5, lr=1e-3):
    torch.manual_seed(42)
    loader = DataLoader(train_set, batch_size=64, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss_fn(model(xb), yb).backward()
            optimizer.step()
        accuracy(model, val_loader)        # as on the lesson page; iterating a loader also advances the seed


torch.manual_seed(42)
mlp = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 10))
fit(mlp)
torch.manual_seed(42)
cnn = SmallCNN()
fit(cnn)
out = fixtures(pathlib.Path(__file__).resolve().parent.parent / "_fixtures" / "x")
torch.save(mlp.state_dict(), out / "fashion_dense.pt")
torch.save(cnn.state_dict(), out / "fashion_small_cnn.pt")
print(f"dense test accuracy {accuracy(mlp, test_loader):.3f}, CNN test accuracy {accuracy(cnn, test_loader):.3f}")
print("saved", out / "fashion_dense.pt", "and", out / "fashion_small_cnn.pt")
