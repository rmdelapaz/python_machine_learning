# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import OFFLINE, fixtures, image_root

"""Lab 10.3 B: does the CNN really not care where the sneaker is?

Lesson 10.3 argued that a CNN finds a pattern anywhere, while a dense network must learn "sneaker"
separately at every position. Test the claim. Load the lesson's two trained models from saved weights
(no training: the files in _fixtures/ hold exactly the page's models), then slide every test image a
few pixels sideways and down and watch both accuracies. Pairs: one types, one predicts each
number before it prints. Work through TODO 1 to 4; the file runs as it is.
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


# %% Step 2: shift images without wrapping around
# TODO 1: shift_right(X, k) returns a copy of X moved k pixels right with the empty strip filled with 0:
#         out = torch.zeros_like(X); out[..., :, k:] = X[..., :, :28 - k]. Write shift_down the same way
#         along the height. (Why not torch.roll? What would wrap around?)
def shift_right(X, k):
    return None


def shift_down(X, k):
    return None


if shift_right(X_test[:1], 3) is None:
    print("TODO 1: shift_right and shift_down")

# %% Step 3: accuracy as the clothes move
# TODO 2: for k in [0, 1, 2, 3, 4], print the dense and CNN accuracy on images shifted right by k and on
#         images shifted down by k (k = 0 means the original images). Predict first: which model loses
#         less at 2 pixels? Does either keep its accuracy at 4?
print("TODO 2: the shift table")

# %% Step 4: which classes suffer most?
# TODO 3: for the CNN, compare per-class accuracy before and after a 2-pixel shift right; print the three
#         classes that lose the most.
print("TODO 3: the classes that suffer most")

# TODO 4: one sentence: the convolutions find a pattern anywhere, so why does the CNN still lose accuracy?
#         (Hint: look at what comes after self.features.)
print("TODO 4: your explanation")

print("\nExpected when done: unshifted 0.840 / 0.864; shifted right by 1 / 2 / 3 / 4 pixels the dense network")
print("falls to 0.756 / 0.552 / 0.355 / 0.252 and the CNN to 0.838 / 0.743 / 0.471 / 0.301 (down: dense 0.778 /")
print("0.708 / 0.661 / 0.553, CNN 0.821 / 0.770 / 0.726 / 0.592); worst CNN classes at 2 pixels: Dress -0.334,")
print("T-shirt/top -0.229, Coat -0.217. Better than dense at every shift, but not shift-proof.")
