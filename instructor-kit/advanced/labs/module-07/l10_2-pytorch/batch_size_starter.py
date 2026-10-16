# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA, OUTPUT

"""Lab 10.2 B: batch size, steps and epochs, then a checkpoint that rebuilds itself.

Lesson 10.2 trained with batches of 64 and never asked why. Here you write the training step from
memory, train the same small network with batch sizes 16, 64, 512 and 3,600 (the whole fitting set)
for ten epochs each, and count the optimizer steps. Then save a checkpoint that carries its own
settings and rebuild the model from it. Pairs: one types, one predicts; swap at TODO 3.
"""
# %% Step 1: the lesson's churn tensors (done for you)
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

customers = pd.read_csv(DATA / "customers_churn.csv")
features = ["tenure_months", "orders_12m", "spend_12m", "avg_order_value", "days_since_last_order",
            "returns_12m", "return_rate", "support_tickets_12m", "discount_share", "app_user"]
X_train_df, X_test_df, y_train_s, y_test_s = train_test_split(
    customers[features], customers["churned"], test_size=0.25, stratify=customers["churned"], random_state=42)
X_fit_df, X_val_df, y_fit_s, y_val_s = train_test_split(
    X_train_df, y_train_s, test_size=0.2, stratify=y_train_s, random_state=42)
scaler = StandardScaler().fit(X_fit_df)


def to_tensors(X_df, y_s):
    return (torch.tensor(scaler.transform(X_df), dtype=torch.float32),
            torch.tensor(y_s.to_numpy(), dtype=torch.float32).unsqueeze(1))


X_fit, y_fit = to_tensors(X_fit_df, y_fit_s)
X_val, y_val = to_tensors(X_val_df, y_val_s)
loss_fn = nn.BCEWithLogitsLoss()
print(X_fit.shape, X_val.shape)


def make_model(n_hidden=32):
    return nn.Sequential(nn.Linear(10, n_hidden), nn.ReLU(), nn.Linear(n_hidden, 1))


def val_scores(model):
    model.eval()
    with torch.no_grad():
        logits = model(X_val)
    return loss_fn(logits, y_val).item(), roc_auc_score(y_val_s, logits.squeeze(1).numpy())


# %% Step 2: one training step, from memory
# TODO 1: without looking at the lesson, write the four numbered lines of the batch loop below
#         (clear gradients, forward and loss, backward, update) and count the steps.
def train_one_epoch(model, loader, optimizer):
    model.train()
    steps = 0
    for xb, yb in loader:
        pass                               # TODO 1: replace with the four lines
        steps += 1
    return steps


# %% Step 3: the batch-size sweep
# TODO 2: for batch_size in [16, 64, 512, 3600]: torch.manual_seed(42), make_model(), Adam with lr=0.003,
#         a shuffled DataLoader with that batch size; train ten epochs, adding up the steps. After epochs 1
#         and 10 print batch size, epoch, steps so far, and val_scores(model). Predict first: which batch
#         size has the best validation AUC after ten epochs, and how many steps did it take?
print("TODO 2: the batch-size sweep")

# %% Step 4: is it the batch size or the number of steps?
# TODO 3: train make_model() with batch size 3600 for 570 epochs (570 steps, the number batch 64 made in
#         ten epochs). Seed 42, Adam lr=0.003. Print val_scores. What does the result tell you?
print("TODO 3: full batch with 570 steps")

# %% Step 5: a checkpoint that carries its own settings
# TODO 4: settings = {"n_hidden": 32}; train make_model(**settings) for ten epochs at batch 64 (seed 42).
#         torch.save a dictionary with "settings", "state_dict", and the scaler's mean_ and scale_ as plain
#         lists (.tolist()) to OUTPUT / "churn_small.pt". Load it with weights_only=True, rebuild the model
#         from checkpoint["settings"], load the weights, and print the rebuilt and original val AUC.
print("TODO 4: save and rebuild")

print("\nExpected when done: after ten epochs val AUC 0.871 (batch 16, 2,250 steps), 0.866 (64, 570 steps),")
print("0.819 (512, 80 steps), 0.518 (3,600, 10 steps); the full batch with 570 steps reaches val loss 0.301,")
print("val AUC 0.873; the checkpoint keys are scaler_mean, scaler_scale, settings, state_dict, and the")
print("rebuilt model scores 0.866 like the original.")
