#!/usr/bin/env python3
import torch, time
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.serialization import safe_globals
from sklearn.metrics import f1_score, roc_auc_score
from torch import nn
import numpy as np

DATA_PATH = "/app/juliet_cepg_ast.pt"

# --- Load dataset ---
with safe_globals([Data]):
    graphs = torch.load(DATA_PATH, weights_only=False)

x = torch.stack([g.x.mean(dim=0) for g in graphs])  # graph-level average
y = torch.tensor([int(g.y) for g in graphs])

# semgrep = 768 + 16 → 785th dim
semgrep_idx = 768 + 16
semgrep_feat = x[:, semgrep_idx:semgrep_idx+1]

# split
n = len(y)
perm = torch.randperm(n)
train, val, test = perm[:int(0.7*n)], perm[int(0.7*n):int(0.85*n)], perm[int(0.85*n):]

def subset(idx): return semgrep_feat[idx], y[idx]
Xtr, Ytr = subset(train); Xv, Yv = subset(val); Xt, Yt = subset(test)

# --- Model ---
model = nn.Sequential(nn.Linear(1, 8), nn.ReLU(), nn.Linear(8, 1))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCEWithLogitsLoss()

for epoch in range(20):
    model.train()
    out = model(Xtr)
    loss = loss_fn(out.squeeze(), Ytr.float())
    opt.zero_grad(); loss.backward(); opt.step()

model.eval()
with torch.no_grad():
    probs = torch.sigmoid(model(Xt).squeeze()).numpy()
preds = (probs >= 0.5).astype(int)
f1 = f1_score(Yt, preds)
auc = roc_auc_score(Yt, probs)
print(f"📊 Semgrep Baseline → F1={f1:.3f} | AUC={auc:.3f}")
