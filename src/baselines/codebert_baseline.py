#!/usr/bin/env python3
import torch, time
from torch_geometric.data import Data
from torch.serialization import safe_globals
from sklearn.metrics import f1_score, roc_auc_score
from torch import nn
import numpy as np

DATA_PATH = "/app/juliet_cepg_ast.pt"
with safe_globals([Data]):
    graphs = torch.load(DATA_PATH, weights_only=False)

x = torch.stack([g.x.mean(dim=0) for g in graphs])
y = torch.tensor([int(g.y) for g in graphs])

codebert_idx = slice(0, 768)
codebert_feat = x[:, codebert_idx]

n = len(y)
perm = torch.randperm(n)
train, val, test = perm[:int(0.7*n)], perm[int(0.7*n):int(0.85*n)], perm[int(0.85*n):]

def subset(idx): return codebert_feat[idx], y[idx]
Xtr, Ytr = subset(train); Xv, Yv = subset(val); Xt, Yt = subset(test)

model = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, 1))
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.BCEWithLogitsLoss()

for epoch in range(5):
    model.train()
    out = model(Xtr)
    loss = loss_fn(out.squeeze(), Ytr.float())
    opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        val_probs = torch.sigmoid(model(Xv)).squeeze().numpy()
    f1 = f1_score(Yv, (val_probs >= 0.5).astype(int))
    auc = roc_auc_score(Yv, val_probs)
    print(f"Epoch {epoch:02d} | Loss={loss.item():.3f} | Val F1={f1:.3f} | AUC={auc:.3f}")

start = time.time()
with torch.no_grad():
    probs = torch.sigmoid(model(Xt)).squeeze().numpy()
latency = (time.time()-start)/len(Yt)*1000
f1 = f1_score(Yt, (probs>=0.5).astype(int))
auc = roc_auc_score(Yt, probs)
print(f"📊 CodeBERT-only → F1={f1:.3f} | AUC={auc:.3f} | Latency={latency:.2f} ms/sample")
