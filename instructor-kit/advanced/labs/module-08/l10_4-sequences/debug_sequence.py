# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Debug 10.4: a Boise sales forecaster with four bugs (deliberately broken).

One bug stops the script with an error. Three do not, and they are the dangerous ones: the script prints
a confident, flattering error that nobody should believe. Fix them all. When everything is fixed the
script prints: 365 test windows, and an MAE in dollars of 1,039.2 (seasonal naive 1,355.6).
"""
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error

sales = pd.read_csv(DATA / "daily_store_sales.csv", parse_dates=["date"])
boise = sales[sales["store"] == "Boise"].set_index("date")["sales"]
is_test_day = boise.index >= "2025-01-01"
log_s = np.log1p(boise.to_numpy())
mu, sd = log_s[~is_test_day].mean(), log_s[~is_test_day].std()
z = (log_s - mu) / sd
weekday = np.eye(7)[boise.index.dayofweek]

X, y, day = [], [], []
for t in range(28, len(z)):
    X.append(np.column_stack([z[t - 28:t], weekday[t - 27:t + 1]]))
    y.append(z[t])
    day.append(t)
X = torch.tensor(np.array(X))
y = torch.tensor(np.array(y), dtype=torch.float32)
rng = np.random.default_rng(42)
test = np.zeros(len(day), dtype=bool)
test[rng.choice(len(day), size=int(is_test_day.sum()), replace=False)] = True
train = ~test
print(f"{test.sum()} test windows")


class Forecaster(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(8, 32)
        self.head = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1]).squeeze(-1)


torch.manual_seed(42)
model = Forecaster()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
torch.manual_seed(42)
loader = DataLoader(TensorDataset(X[train], y[train]), batch_size=32, shuffle=True)
for epoch in range(15):
    model.train()
    for xb, yb in loader:
        optimizer.zero_grad()
        nn.functional.mse_loss(model(xb), yb).backward()
        optimizer.step()

model.eval()
with torch.inference_mode():
    pred = model(X[test]).numpy()
mae = mean_absolute_error(y[test].numpy(), pred)
naive = mean_absolute_error(boise[is_test_day], boise.shift(7)[is_test_day])
print(f"LSTM MAE {mae:,.1f}   (seasonal naive {naive:,.1f})")
