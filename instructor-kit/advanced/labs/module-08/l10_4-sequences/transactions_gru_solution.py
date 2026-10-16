# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 10.4 A (SOLUTION): forecast tomorrow's number of transactions with a GRU.

Lesson 10.4 forecast sales in dollars. Staffing the tills needs something else: how many customers
will pay tomorrow. Same data file, a different target (the transactions column), a GRU instead of an
LSTM. Measure the naive yardsticks in transactions, split by time, build windows, train with and
without the known-in-advance calendar, and report the error in transactions per store-day.
"""
# %% Step 1: the target, wide, and the yardsticks (test = every day of 2025)
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error

sales = pd.read_csv(DATA / "daily_store_sales.csv", parse_dates=["date"])
wide = sales.pivot(index="date", columns="store", values="transactions")
is_test_day = wide.index >= "2025-01-01"
actual_2025 = wide[is_test_day].to_numpy().ravel()
print("mean transactions per day in 2025:", wide[is_test_day].mean().round(1).to_dict())

maes = {}
for name, lag in [("naive (yesterday)", 1), ("seasonal naive (last week)", 7)]:
    maes[name] = mean_absolute_error(actual_2025, wide.shift(lag)[is_test_day].to_numpy().ravel())

# %% Step 2: scale with the training years only (log first: counts grow multiplicatively)
log_tx = np.log1p(wide)
mu = log_tx[~is_test_day].to_numpy().mean()
sd = log_tx[~is_test_day].to_numpy().std()
z = ((log_tx - mu) / sd).to_numpy()


def to_count(z_values):
    return np.expm1(np.asarray(z_values) * sd + mu)


events = sales.groupby("date")[["promo", "event"]].first()
weekday = np.eye(7)[wide.index.dayofweek]
calendar = np.column_stack([weekday, events["promo"].to_numpy(), (events["event"] == "Closed").to_numpy().astype(float)])
print(f"mu {mu:.3f}  sd {sd:.3f}  z {z.shape}  calendar {calendar.shape}")


# %% Step 3: windows of 28 days, each step = (scaled transactions, NEXT day's calendar)
def build_windows(z, cal, window=28):
    X, y, day = [], [], []
    for s in range(z.shape[1]):
        for t in range(window, len(z)):
            X.append(np.column_stack([z[t - window:t, s], cal[t - window + 1:t + 1]]))
            y.append(z[t, s])
            day.append(t)
    return (torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32),
            np.array(day))


# %% Step 4: a GRU forecaster and a training function
class TxGRU(nn.Module):
    def __init__(self, n_features, hidden=32):
        super().__init__()
        self.rnn = nn.GRU(n_features, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):                              # x: (batch, 28, n_features)
        out, _ = self.rnn(x)
        return self.head(out[:, -1]).squeeze(-1)


def train_and_score(cal):
    X, y, day = build_windows(z, cal)
    test = is_test_day[day]
    torch.manual_seed(42)
    model = TxGRU(X.shape[2])
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    torch.manual_seed(42)
    loader = DataLoader(TensorDataset(X[~test], y[~test]), batch_size=64, shuffle=True)
    for epoch in range(15):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            nn.functional.mse_loss(model(xb), yb).backward()
            optimizer.step()
    model.eval()
    with torch.inference_mode():
        pred = to_count(model(X[test]).numpy())
    print(f"windows {tuple(X.shape)}  training {(~test).sum()}  test {test.sum()}  parameters "
          f"{sum(p.numel() for p in model.parameters()):,}")
    return mean_absolute_error(to_count(y[test].numpy()), pred)


# %% Step 5: weekday only, then the full calendar
maes["GRU, weekday only"] = train_and_score(calendar[:, :7])
maes["GRU, weekday + promo + closed"] = train_and_score(calendar)

print("\nnext-day transactions, test year 2025 (MAE in transactions per store-day)")
for name, mae in maes.items():
    print(f"{name:32s} {mae:6.2f}")
print("Weekday flags alone cut the seasonal naive error by about 40%; adding the promotion and closed")
print("flags, known in advance, brings it to half.")
