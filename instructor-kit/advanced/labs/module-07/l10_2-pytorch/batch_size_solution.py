# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA, OUTPUT

"""Lab 10.2 B (SOLUTION): batch size, steps and epochs, then a checkpoint that rebuilds itself.

Lesson 10.2 trained with batches of 64 and never asked why. Here you write the training step from
memory, train the same small network with batch sizes 16, 64, 512 and 3,600 (the whole fitting set)
for ten epochs each, and count the optimizer steps. Then save a checkpoint that carries its own
settings and rebuild the model from it.
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


# %% Step 2: one training step, written from memory, and one epoch
def train_one_epoch(model, loader, optimizer):
    model.train()
    steps = 0
    for xb, yb in loader:
        optimizer.zero_grad()              # 1. clear old gradients
        loss = loss_fn(model(xb), yb)      # 2. forward and loss
        loss.backward()                    # 3. backward
        optimizer.step()                   # 4. update
        steps += 1
    return steps


# %% Step 3: the batch-size sweep (same model, same seed, same learning rate, ten epochs each)
print("\nbatch   epoch  steps so far  val loss  val AUC")
for batch_size in [16, 64, 512, 3600]:
    torch.manual_seed(42)
    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    loader = DataLoader(TensorDataset(X_fit, y_fit), batch_size=batch_size, shuffle=True)
    steps = 0
    for epoch in range(1, 11):
        steps += train_one_epoch(model, loader, optimizer)
        if epoch in (1, 10):
            v_loss, v_auc = val_scores(model)
            print(f"{batch_size:5d}  {epoch:5d}  {steps:12d}  {v_loss:8.3f}  {v_auc:7.3f}")
print("Ten epochs is the same data, but not the same number of steps: 2,250 steps at batch 16, ten at 3,600.")

# %% Step 4: is it the batch size or the number of steps?
torch.manual_seed(42)
full = make_model()
optimizer = torch.optim.Adam(full.parameters(), lr=0.003)
full_loader = DataLoader(TensorDataset(X_fit, y_fit), batch_size=3600, shuffle=True)
for epoch in range(570):                         # 570 full-batch steps = the steps batch 64 made in 10 epochs
    train_one_epoch(full, full_loader, optimizer)
v_loss, v_auc = val_scores(full)
print(f"\nfull batch, 570 steps (570 epochs): val loss {v_loss:.3f}  val AUC {v_auc:.3f}")
print("Given the same number of steps, the full batch catches up: steps drive progress; small batches")
print("simply give you many more of them per pass over the data.")

# %% Step 5: a checkpoint that carries its own settings
settings = {"n_hidden": 32}
torch.manual_seed(42)
keeper = make_model(**settings)
optimizer = torch.optim.Adam(keeper.parameters(), lr=0.003)
loader = DataLoader(TensorDataset(X_fit, y_fit), batch_size=64, shuffle=True)
for epoch in range(10):
    train_one_epoch(keeper, loader, optimizer)
torch.save({"settings": settings, "state_dict": keeper.state_dict(),
            "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist()}, OUTPUT / "churn_small.pt")

checkpoint = torch.load(OUTPUT / "churn_small.pt", weights_only=True)
rebuilt = make_model(**checkpoint["settings"])
rebuilt.load_state_dict(checkpoint["state_dict"])
print(f"\ncheckpoint keys: {sorted(checkpoint)}")
print(f"rebuilt model, val AUC {val_scores(rebuilt)[1]:.3f} (original {val_scores(keeper)[1]:.3f})")
