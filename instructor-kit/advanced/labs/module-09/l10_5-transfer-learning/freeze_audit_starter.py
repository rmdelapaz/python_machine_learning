# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import OFFLINE, image_root, use_cached_weights

"""Lab 10.5 B: audit a fine-tuning setup before you spend CPU time on it.

Fine-tuning fails quietly: a head that is frozen by accident, an optimizer that never received the
head, batch-norm statistics that drift while you thought everything was frozen. Check the setup on ONE
batch of 16 EuroSAT tiles (seconds, not minutes). Work through TODO 1 to 5; run the file after each one.
It runs as it is: unfinished steps print a reminder.
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

# TODO 1: build `wrong_order`: a second ResNet-18 whose head is attached FIRST and frozen AFTER.
#         Predict its trainable count before you print it: ______
print("TODO 1: the wrong order")

# %% Step 2: what each strategy trains
total = sum(p.numel() for p in model.parameters())
# TODO 2: for "head only", "head + layer4" and "head + layer3 + layer4": freeze everything, unfreeze
#         model.fc and the named blocks (getattr(model, "layer4")), then print the trainable count and
#         its share of `total`. Leave the model in "head + layer4" at the end (re-freeze layer3).
print("TODO 2: three strategies")

# %% Step 3: parameter groups: is every trainable tensor in exactly one group?
# TODO 3: write audit(groups): build torch.optim.Adam(groups), collect id(p) of every parameter in
#         opt.param_groups, and return the names (model.named_parameters()) of trainable tensors that
#         are NOT in the optimizer. Audit the lesson's two groups (layer4 1e-4, fc 1e-3), then a version
#         with only the layer4 group. Which tensors are missing?
print("TODO 3: audit two optimizers")

# %% Step 4: one training step, then: what changed?
tf = transforms.Compose([transforms.Resize(128), transforms.ToTensor(),
                         transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std)])
eurosat = datasets.EuroSAT(image_root("eurosat"), download=not OFFLINE, transform=tf)
order = torch.randperm(len(eurosat), generator=torch.Generator().manual_seed(42)).tolist()
xb, yb = next(iter(DataLoader(Subset(eurosat, order[6500:6516]), batch_size=16)))
print("\nbatch:", tuple(xb.shape), "labels:", yb.tolist())

# TODO 4: snapshot (.detach().clone()) model.conv1.weight, model.layer4[1].conv2.weight, model.fc.weight
#         and model.bn1.running_mean. Do ONE optimizer step (Adam, the two groups) in model.eval() mode,
#         and print which of the four changed (torch.equal). Then torch.manual_seed(42) before the first
#         step, and repeat the step in model.train() mode. Which frozen thing changes anyway, and why?
# TODO 5: in one sentence (a comment): why do gradients still reach the head in eval() mode?
print("TODO 4-5: one step in eval() mode, one in train() mode")

print("\nExpected when done: 5,130 trainable, and 0 for the wrong order; head only 5,130 (0.05% of")
print("11,181,642), head + layer4 8,398,858 (75.11%), head + layer3 + layer4 10,498,570 (93.89%); the")
print("layer4-only optimizer misses ['fc.weight', 'fc.bias']; eval() step (loss 2.341): layer4 and fc CHANGED,")
print("conv1 and bn1.running_mean same; train() step (loss 1.820): bn1.running_mean CHANGED as well.")
