# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Debug 10.2: a churn training script with five bugs (deliberately broken).

Two bugs stop the script with an error. Three do not, and the script still prints numbers: those are
the dangerous ones. Fix one at a time and rerun. When everything is fixed the script prints
validation AUC 0.839, and the same AUC twice in a row for the two "repeat" lines.
"""
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
X_fit_df, X_val_df, y_fit_s, y_val_s = train_test_split(
    customers[features], customers["churned"], test_size=0.2, stratify=customers["churned"], random_state=42)
scaler = StandardScaler().fit(X_fit_df)

X_fit = torch.from_numpy(scaler.transform(X_fit_df))
X_val = torch.tensor(scaler.transform(X_val_df), dtype=torch.float32)
y_fit = torch.tensor(y_fit_s.to_numpy(), dtype=torch.float32)

torch.manual_seed(42)
model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1), nn.Sigmoid())
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
loader = DataLoader(TensorDataset(X_fit, y_fit), batch_size=64, shuffle=True)

for epoch in range(10):
    model.train()
    for xb, yb in loader:
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()


def val_auc():
    with torch.no_grad():
        scores = model(X_val).squeeze(1).numpy()
    return roc_auc_score(y_val_s, scores)


print(f"validation AUC {val_auc():.3f}")
print(f"repeat         {val_auc():.3f}")
print(f"repeat         {val_auc():.3f}")
