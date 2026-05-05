#!/usr/bin/env python3
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv, SAGEConv
from torch.serialization import safe_globals
import random

# =========================================================
# CONFIG
# =========================================================
DATA_PATH = "/app/juliet_cepg_full.pt"

EPOCHS_TGAT = 30   # was 20
EPOCHS_BASE = 8
LR = 5e-5
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# =========================================================
# LOAD DATA
# =========================================================
with safe_globals([Data]):
    data: Data = torch.load(DATA_PATH, map_location="cpu", weights_only=False)

data = data.to(device)
N = data.num_nodes
in_dim = data.x.size(1)
edge_dim = data.edge_attr.size(1)

print(f"Loaded Juliet → Nodes={N}, Edges={data.edge_index.size(1)}")

# =========================================================
# NORMALIZE EDGE ATTR (CRITICAL)
# =========================================================
ea = data.edge_attr
data.edge_attr = (ea - ea.mean(0)) / (ea.std(0) + 1e-6)

# =========================================================
# SPLIT (fixed for all variants)
# =========================================================
perm = torch.randperm(N)
train_idx = perm[:int(0.8 * N)]
test_idx  = perm[int(0.9 * N):]

# =========================================================
# LOSS (TGAT)
# =========================================================
def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    probs = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = probs * targets + (1 - probs) * (1 - targets)
    return (alpha * (1 - pt) ** gamma * ce).mean()

def find_best_threshold(probs, labels):
    best_f1, best_t = 0.0, 0.5
    for t in np.linspace(0.1, 0.9, 81):
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t

# =========================================================
# MODELS
# =========================================================
class TGAT(torch.nn.Module):
    def __init__(self, use_edge_attr=True, static_temporal=False):
        super().__init__()
        self.use_edge_attr = use_edge_attr
        self.static_temporal = static_temporal

        self.conv1 = TransformerConv(
            in_dim, 256, heads=8,
            edge_dim=edge_dim if use_edge_attr else None
        )
        self.conv2 = TransformerConv(
            256 * 8, 256, heads=1,
            edge_dim=edge_dim if use_edge_attr else None
        )
        self.readout = torch.nn.Linear(256, 1)

    def forward(self, data):
        edge_attr = None
        if self.use_edge_attr:
            edge_attr = data.edge_attr.clone()
            if self.static_temporal:
                edge_attr[:, 0] = 0.0   # remove Δt only

        h = F.relu(self.conv1(data.x, data.edge_index, edge_attr))
        h = F.relu(self.conv2(h, data.edge_index, edge_attr))
        return self.readout(h).squeeze(-1)

class GraphSAGEModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, 256)
        self.conv2 = SAGEConv(256, 256)
        self.readout = torch.nn.Linear(256, 1)

    def forward(self, data):
        h = F.relu(self.conv1(data.x, data.edge_index))
        h = F.relu(self.conv2(h, data.edge_index))
        return self.readout(h).squeeze(-1)

class SeqBaseline(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )

    def forward(self, data):
        return self.net(data.x).squeeze(-1)

# =========================================================
# TRAIN + EVAL
# =========================================================
def train_and_eval(model, epochs, use_focal):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(data)
        if use_focal:
            loss = focal_loss(logits[train_idx], data.y[train_idx].float())
        else:
            loss = F.binary_cross_entropy_with_logits(
                logits[train_idx], data.y[train_idx].float()
            )
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(data)
        probs = torch.sigmoid(logits).cpu().numpy()
        labels = data.y.cpu().numpy()

    # ---- validation-based threshold tuning ----
    val_idx = perm[int(0.8 * N):int(0.9 * N)]
    best_t = find_best_threshold(probs[val_idx], labels[val_idx])

    # ---- test evaluation ----
    test_probs = probs[test_idx]
    test_labels = labels[test_idx]
    test_preds = (test_probs >= best_t).astype(int)

    auc = roc_auc_score(test_labels, test_probs)
    f1 = f1_score(test_labels, test_preds)

    return round(auc, 4), round(f1, 4)

# =========================================================
# RUN ABLATIONS
# =========================================================
results = []

experiments = [
    ("TGAT (full)", TGAT(True, False), EPOCHS_TGAT, True),
    ("TGAT (no Δt or diff edge attr)", TGAT(False, False), EPOCHS_TGAT, True),
    ("TGAT (static edges)", TGAT(True, True), EPOCHS_TGAT, True),
    ("GraphSAGE", GraphSAGEModel(), EPOCHS_BASE, False),
    ("Seq Transformer (flat tokens)", SeqBaseline(), EPOCHS_BASE, False),
]

for name, model, epochs, use_focal in experiments:
    print(f"Running {name} …")
    auc, f1 = train_and_eval(model, epochs, use_focal)
    results.append({
        "Architecture": name,
        "AUC↑": auc,
        "F1↑": f1
    })

# =========================================================
# FINAL TABLE
# =========================================================
df = pd.DataFrame(results)
print("\n📈 Ablation Study 1: Temporal vs Sequential vs Static Structure\n")
print(df.to_markdown(index=False))
