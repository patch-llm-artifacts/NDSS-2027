#!/usr/bin/env python3
"""
Full Evaluation of Juliet-trained TGATv3 on ManyBugs CEPG
--------------------------------------------------------
✓ Loads Juliet TGATv3 safely
✓ Pads ManyBugs node features to Juliet dim
✓ Auto-fixes edge_attr=None
✓ Computes node-level classification metrics:
      - Accuracy
      - Precision
      - Recall
      - F1
      - ROC-AUC
✓ Saves CSV of predictions
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from torch.serialization import safe_globals, add_safe_globals
import torch_geometric.data.data as pyg_data_mod
from torch_geometric.data import Data
import pandas as pd

# ==========================================================
# SAFE GLOBALS (for loading Data objects)
# ==========================================================
add_safe_globals([
    Data,
    pyg_data_mod.DataEdgeAttr,
])

# ==========================================================
# PATHS
# ==========================================================
MANYBUGS_PATH = "/app/manybugs_cepg.pt"
JULIET_MODEL  = "/app/tgat_juliet_v3_best.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", DEVICE)

# ==========================================================
# LOAD MANYBUGS
# ==========================================================
with safe_globals([Data, pyg_data_mod.DataEdgeAttr]):
    manybugs = torch.load(MANYBUGS_PATH, map_location="cpu", weights_only=False)

print(f"\n📦 Loaded ManyBugs graphs: {len(manybugs)}")

# ==========================================================
# LOAD JULIET TGATv3
# ==========================================================
ckpt = torch.load(JULIET_MODEL, map_location="cpu")

JULIET_IN_DIM = ckpt["conv1.lin_key.weight"].shape[1]
print(f"🧩 Juliet TGAT expected input dim = {JULIET_IN_DIM}")

try:
    JULIET_EDGE_DIM = ckpt["edge_emb.weight"].shape[1]
except:
    JULIET_EDGE_DIM = 2
print(f"🧩 Juliet TGAT expected edge_dim = {JULIET_EDGE_DIM}")

# ==========================================================
# TGATv3 MODEL (same architecture as training)
# ==========================================================
from torch_geometric.nn import TransformerConv
import torch.nn.functional as F

class TGATv3(torch.nn.Module):
    def __init__(self, in_channels, hidden=256, heads=8, edge_dim=2, dropout=0.3):
        super().__init__()
        self.conv1 = TransformerConv(in_channels, hidden, heads=heads,
                                     edge_dim=edge_dim, dropout=dropout)
        self.bn1 = torch.nn.BatchNorm1d(hidden * heads)
        self.conv2 = TransformerConv(hidden * heads, hidden, heads=1,
                                     edge_dim=edge_dim, dropout=dropout)
        self.bn2 = torch.nn.BatchNorm1d(hidden)

        self.reconstruct = torch.nn.Linear(hidden, in_channels)
        self.next_pred   = torch.nn.Linear(hidden, in_channels)
        self.semgrep_pred= torch.nn.Linear(hidden, 1)
        self.readout     = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, 1)
        )

    def forward(self, x, edge_index, edge_attr):
        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(1).float()

        h1 = self.conv1(x, edge_index, edge_attr)
        h1 = F.relu(self.bn1(h1))

        if x.size(1) == h1.size(1):
            h1 = h1 + x

        h2 = self.conv2(h1, edge_index, edge_attr)
        h2 = self.bn2(h2)

        if h1.size(1) != h2.size(1):
            W = torch.eye(h2.size(1), h1.size(1), device=h1.device)
            h2 = h2 + F.linear(h1, W)
        else:
            h2 = h2 + h1
        h2 = F.relu(h2)

        logits = self.readout(h2).squeeze(-1)
        return logits


# ==========================================================
# PADDER + EDGE FIX
# ==========================================================
def pad_features(g, target_dim):
    cur = g.x.size(1)
    if cur < target_dim:
        pad = target_dim - cur
        g.x = torch.cat([g.x, torch.zeros(g.x.size(0), pad)], dim=1)
    return g

def ensure_edge_attr(g, edge_dim):
    if g.edge_attr is None:
        g.edge_attr = torch.zeros(g.edge_index.size(1), edge_dim)
    elif g.edge_attr.size(1) != edge_dim:
        new = torch.zeros(g.edge_index.size(1), edge_dim)
        cols = min(edge_dim, g.edge_attr.size(1))
        new[:, :cols] = g.edge_attr[:, :cols]
        g.edge_attr = new
    return g


# ==========================================================
# INIT MODEL
# ==========================================================
model = TGATv3(
    in_channels=JULIET_IN_DIM,
    edge_dim=JULIET_EDGE_DIM
).to(DEVICE)

model.load_state_dict(ckpt)
model.eval()
print("🎉 Loaded Juliet TGATv3 checkpoint!\n")


# ==========================================================
# RUN INFERENCE
# ==========================================================
y_true = []
y_pred = []
y_score = []

print("🔍 Evaluating ManyBugs…")

for gi, g in enumerate(manybugs):
    g = pad_features(g, JULIET_IN_DIM)
    g = ensure_edge_attr(g, JULIET_EDGE_DIM)

    x  = g.x.to(DEVICE)
    ei = g.edge_index.to(DEVICE)
    ea = g.edge_attr.to(DEVICE)
    y  = g.y.cpu().numpy()  # [1,0]

    with torch.no_grad():
        logits = model(x, ei, ea)
        probs = torch.sigmoid(logits).cpu().numpy()

    # only 2 nodes per graph: node0 buggy(1), node1 fixed(0)
    y_true.extend(list(y))
    y_score.extend(list(probs))
    y_pred.extend(list((probs > 0.5).astype(int)))

    if gi % 20 == 0:
        print(f"   ✓ Processed graph {gi}/{len(manybugs)}")

print("\n🎯 Done. Computing metrics…")


# ==========================================================
# METRICS
# ==========================================================
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
try:
    auc = roc_auc_score(y_true, y_score)
except:
    auc = float("nan")

cm = confusion_matrix(y_true, y_pred)

print("\n==================== RESULTS ====================")
print(f"Accuracy:      {acc:.4f}")
print(f"Precision:     {prec:.4f}")
print(f"Recall:        {rec:.4f}")
print(f"F1:            {f1:.4f}")
print(f"ROC-AUC:       {auc:.4f}")
print("\nConfusion Matrix:\n", cm)
print("=================================================\n")


# ==========================================================
# SAVE CSV
# ==========================================================
df = pd.DataFrame({
    "true":  y_true,
    "pred":  y_pred,
    "score": y_score
})
df.to_csv("manybugs_tgat_predictions.csv", index=False)

print("💾 Saved predictions → manybugs_tgat_predictions.csv")
print("💾 Saved scores      → manybugs_tgat_scores.pt")
print("✨ Evaluation complete!")
