import torch
import torch.nn as nn
import math
import time

torch.manual_seed(42)

# ── Data ──────────────────────────────────────────────────
# 200 cells, 24 hours, 10 KPIs
X = torch.randn(200, 24, 10)
trend = (X[:, 23, 0] - X[:, 0, 0]).unsqueeze(1)
y = trend

# Train/test split — first 160 train, last 40 test
X_train, X_test = X[:160], X[160:]
y_train, y_test = y[:160], y[160:]

X_flat_train = X_train.mean(dim=1)   # (160, 10) for feedforward
X_flat_test  = X_test.mean(dim=1)    # (40, 10)

# ── Models ────────────────────────────────────────────────
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x): return self.net(x)

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(10, 64, num_layers=2,
                            batch_first=True, dropout=0.2)
        self.scorer = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.scorer(h_n[-1])

class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(10, 64)
        self.pos  = nn.Embedding(100, 64)
        enc = nn.TransformerEncoderLayer(64, nhead=4,
              dim_feedforward=128, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=2)
        self.scorer = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        B, T, _ = x.shape
        x = self.proj(x) + self.pos(torch.arange(T, device=x.device))
        x = self.transformer(x)
        return self.scorer(x.mean(dim=1))

# ── Training function ──────────────────────────────────────
def train(model, X_tr, y_tr, epochs=300):
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    t0 = time.time()
    for epoch in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(X_tr), y_tr)
        loss.backward()
        opt.step()
    return time.time() - t0

# ── Evaluate function ──────────────────────────────────────
def evaluate(model, X_te, y_te):
    model.eval()
    with torch.no_grad():
        preds = model(X_te)
        mse = nn.MSELoss()(preds, y_te).item()
        mae = (preds - y_te).abs().mean().item()
    model.train()
    return mse, mae

# ── Run benchmark ─────────────────────────────────────────
results = {}

print("Training FeedForward...")
ff = FeedForward()
t = train(ff, X_flat_train, y_train)
mse, mae = evaluate(ff, X_flat_test, y_test)
results["FeedForward"] = {"mse": mse, "mae": mae, "time": t,
                           "params": sum(p.numel() for p in ff.parameters())}

print("Training LSTM...")
lstm = LSTMModel()
t = train(lstm, X_train, y_train)
mse, mae = evaluate(lstm, X_test, y_test)
results["LSTM"] = {"mse": mse, "mae": mae, "time": t,
                    "params": sum(p.numel() for p in lstm.parameters())}

print("Training Transformer...")
tfm = TransformerModel()
t = train(tfm, X_train, y_train)
mse, mae = evaluate(tfm, X_test, y_test)
results["Transformer"] = {"mse": mse, "mae": mae, "time": t,
                            "params": sum(p.numel() for p in tfm.parameters())}

# ── Print results ─────────────────────────────────────────
print("\n" + "="*55)
print(f"{'Model':<14} {'MSE':>8} {'MAE':>8} {'Params':>10} {'Time':>8}")
print("="*55)
for name, r in results.items():
    print(f"{name:<14} {r['mse']:>8.4f} {r['mae']:>8.4f} "
          f"{r['params']:>10,} {r['time']:>7.1f}s")
print("="*55)
