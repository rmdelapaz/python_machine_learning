# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import OFFLINE, image_root, use_cached_weights

"""Lab 10.5 B (SOLUTION): audit a fine-tuning setup before you spend CPU time on it.

Fine-tuning fails quietly: a head that is frozen by accident, an optimizer that never received the
head, batch-norm statistics that drift while you thought everything was frozen. Before a long
training run, a careful practitioner checks the setup on ONE batch. This lab does exactly that with
ResNet-18 and 16 EuroSAT tiles (seconds, not minutes).
"""
# %% Step 1: freeze first, then attach the head (and the opposite order)
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

use_cached_weights()
weights = models.ResNet18_Weights.DEFAULT


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


torch.manual_seed(42)
model = models.resnet18(weights=weights)
for p in model.parameters():
    p.requires_grad = False
model.fc = nn.Linear(512, 10)
print(f"freeze, then new head : {count_trainable(model):,} trainable")

wrong_order = models.resnet18(weights=weights)
wrong_order.fc = nn.Linear(512, 10)
for p in wrong_order.parameters():
    p.requires_grad = False
print(f"new head, then freeze : {count_trainable(wrong_order):,} trainable  <- nothing would ever learn")

# %% Step 2: what each strategy trains
total = sum(p.numel() for p in model.parameters())
strategies = {"head only": [], "head + layer4": ["layer4"], "head + layer3 + layer4": ["layer3", "layer4"]}
for name, blocks in strategies.items():
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True
    for block in blocks:
        for p in getattr(model, block).parameters():
            p.requires_grad = True
    n = count_trainable(model)
    print(f"{name:24s} {n:>10,} trainable ({n / total:.2%} of {total:,})")

# %% Step 3: parameter groups: is every trainable tensor in exactly one group?
for p in model.layer3.parameters():          # back to the lesson's stage 2: head + layer4
    p.requires_grad = False
groups_ok = [{"params": model.layer4.parameters(), "lr": 1e-4}, {"params": model.fc.parameters(), "lr": 1e-3}]
groups_bad = [{"params": model.layer4.parameters(), "lr": 1e-4}]          # the head was forgotten


def audit(groups):
    opt = torch.optim.Adam(groups)
    in_opt = {id(p) for g in opt.param_groups for p in g["params"]}
    missing = [n for n, p in model.named_parameters() if p.requires_grad and id(p) not in in_opt]
    return [(g["lr"], sum(p.numel() for p in g["params"])) for g in opt.param_groups], missing


for label, groups in [("both groups", groups_ok), ("layer4 only", groups_bad)]:
    sizes, missing = audit(groups)
    print(f"{label:12s} groups (lr, parameters): {sizes}  trainable but NOT in the optimizer: {missing}")

# %% Step 4: one training step, then: what changed?
tf = transforms.Compose([transforms.Resize(128), transforms.ToTensor(),
                         transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std)])
eurosat = datasets.EuroSAT(image_root("eurosat"), download=not OFFLINE, transform=tf)
order = torch.randperm(len(eurosat), generator=torch.Generator().manual_seed(42)).tolist()
xb, yb = next(iter(DataLoader(Subset(eurosat, order[6500:6516]), batch_size=16)))
print("\nbatch:", tuple(xb.shape), "labels:", yb.tolist())

watch = {"conv1.weight (frozen)": lambda: model.conv1.weight,
         "layer4 conv (trainable)": lambda: model.layer4[1].conv2.weight,
         "fc.weight (trainable)": lambda: model.fc.weight,
         "bn1.running_mean (frozen stem)": lambda: model.bn1.running_mean}


def one_step(mode):
    before = {k: f().detach().clone() for k, f in watch.items()}
    optimizer = torch.optim.Adam([{"params": model.layer4.parameters(), "lr": 1e-4},
                                  {"params": model.fc.parameters(), "lr": 1e-3}])
    model.train() if mode == "train" else model.eval()
    optimizer.zero_grad()
    loss = nn.functional.cross_entropy(model(xb), yb)
    loss.backward()
    optimizer.step()
    changed = {k: not torch.equal(before[k], f()) for k, f in watch.items()}
    print(f"{mode}() mode, loss {loss.item():.3f}: " + ", ".join(f"{k} {'CHANGED' if v else 'same'}" for k, v in changed.items()))


torch.manual_seed(42)
one_step("eval")
one_step("train")
print("(gradients flow in eval() mode too: eval() changes layer behavior, not gradient tracking)")

# %% Stretch: what does a frozen block save? Count the tensors that need a gradient
with_grad = sum(1 for p in model.parameters() if p.requires_grad)
print(f"\ntensors that get a gradient: {with_grad} of {len(list(model.parameters()))}")
