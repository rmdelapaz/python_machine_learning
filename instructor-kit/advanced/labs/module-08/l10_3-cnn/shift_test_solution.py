# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import OFFLINE, fixtures, image_root

"""Lab 10.3 B (SOLUTION): does the CNN really not care where the sneaker is?

Lesson 10.3 argued that a CNN finds a pattern anywhere, while a dense network must learn "sneaker"
separately at every position. Test the claim. Load the lesson's two trained models from saved weights
(no training: the files in _fixtures/ hold exactly the page's models), then slide every test image a
few pixels sideways and down and watch both accuracies.
"""
# %% Step 1: the test images and the two saved models
import torch
from torch import nn
from torchvision import datasets

test_set = datasets.FashionMNIST(image_root("FashionMNIST"), train=False, download=not OFFLINE)
X_test = test_set.data.float().unsqueeze(1) / 255
y_test = test_set.targets


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


dense = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 10))
cnn = SmallCNN()
folder = fixtures(__file__)
dense.load_state_dict(torch.load(folder / "fashion_dense.pt", weights_only=True))
cnn.load_state_dict(torch.load(folder / "fashion_small_cnn.pt", weights_only=True))
dense.eval()
cnn.eval()
print("loaded:", sum(p.numel() for p in dense.parameters()), "and", sum(p.numel() for p in cnn.parameters()), "parameters")


def accuracy(model, X):
    with torch.no_grad():
        return (model(X).argmax(dim=1) == y_test).float().mean().item()


print(f"unshifted test accuracy  dense {accuracy(dense, X_test):.3f}   CNN {accuracy(cnn, X_test):.3f}")


# %% Step 2: shift images without wrapping around (the empty strip is filled with black, 0)
def shift_right(X, k):
    out = torch.zeros_like(X)
    out[..., :, k:] = X[..., :, :28 - k]
    return out


def shift_down(X, k):
    out = torch.zeros_like(X)
    out[..., k:, :] = X[..., :28 - k, :]
    return out


one = shift_right(X_test[:1], 3)
print("a boot moved 3 pixels right: first 3 columns are now empty:", bool(one[0, 0, :, :3].sum() == 0))

# %% Step 3: accuracy as the clothes move
print("\nshift (pixels)   right: dense   CNN    down: dense   CNN")
for k in [0, 1, 2, 3, 4]:
    r = [accuracy(m, shift_right(X_test, k) if k else X_test) for m in (dense, cnn)]
    d = [accuracy(m, shift_down(X_test, k) if k else X_test) for m in (dense, cnn)]
    print(f"{k:14d}   {r[0]:11.3f}   {r[1]:.3f}   {d[0]:10.3f}   {d[1]:.3f}")

# %% Step 4: which classes suffer most from a 2-pixel shift right (CNN)?
with torch.no_grad():
    before = cnn(X_test).argmax(dim=1) == y_test
    after = cnn(shift_right(X_test, 2)).argmax(dim=1) == y_test
drops = {name: (before[y_test == c].float().mean() - after[y_test == c].float().mean()).item()
         for c, name in enumerate(test_set.classes)}
print("\nCNN accuracy lost with a 2-pixel shift right, worst three classes:")
for name, drop in sorted(drops.items(), key=lambda item: -item[1])[:3]:
    print(f"  {name:12s} -{drop:.3f}")

print("\nThe CNN holds up better at every shift, but it is NOT shift-proof: its convolutions find a")
print("pattern anywhere, yet the Flatten + Linear head still reads WHERE each feature map lit up.")
print("Pooling buys tolerance of a pixel or two; training on shifted copies (augmentation with random")
print("crops, the lesson's Exercise 4) is the standard way to teach a network more.")
