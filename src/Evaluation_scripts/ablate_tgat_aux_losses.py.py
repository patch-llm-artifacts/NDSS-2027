#!/usr/bin/env python3
"""
Ablation Study 2: Semantic + Auxiliary Supervision (Juliet CEPG)

Toggles:
 - Semgrep head loss (node-level regression)
 - Next-node prediction loss (edge-based)
 - Reconstruction loss
 - AST features (zeroed vs kept in x)

Outputs:
 - AUC, F1 (threshold tuned on validation split)
 - Markdown table for paper

Notes:
 - Uses your Juliet graph: /app/juliet_cepg_full.pt
 - Uses TransformerConv-based TGAT core (close to fine_tune_juliet.py)
"""

import os, random
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv
from torch.serialization import safe_globals

# =========================================================
# CONFIG
# =========================================================
DATA_PATH = "/app/juliet_cepg_full.pt"
SEED = 42
LR = 5e-5
WEIGHT_DECAY = 1e-5
EPOCHS = 20

# loss weights (match your style; adjust if needed)
ALPHA = 0.4  # recon
BETA  = 0.2  # next-node
GAMMA = 0.2  # semgrep
DELTA = 0.8  # cls

# focal + smoothing from your code
SMOOTHING = 0.1
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# =========================================================
# HELPERS
# =========================================================
def focal_loss(logits, targets, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA):
    probs = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = probs * targets + (1 - probs) * (1 - targets)
    return (alpha * (1 - p_t) ** gamma * ce).mean()

def edge_next_loss(nexthat, x, edge_index):
    if edge_index.numel() == 0:
        return torch.tensor(0.0, device=x.device)
    src, dst = edge_index
    src = src.clamp(0, x.size(0) - 1)
    dst = dst.clamp(0, x.size(0) - 1)
    return F.mse_loss(nexthat[src], x[dst])

def find_best_threshold(probs, labels):
    best_f1, best_t = 0.0, 0.5
    for t in np.linspace(0.1, 0.9, 81):
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t

@torch.no_grad()
def eval_auc_f1(model, data, val_idx, test_idx, temperature=1.0):
    model.eval()
    logits, *_ = model(data.x, data.edge_index, data.edge_attr)
    logits = torch.clamp(logits, -50, 50)

    probs = torch.sigmoid((logits / temperature)).detach().cpu().numpy()
    labels = data.y.detach().cpu().numpy().astype(int)

    # tune threshold on val only
    t = find_best_threshold(probs[val_idx], labels[val_idx])

    test_probs = probs[test_idx]
    test_labels = labels[test_idx]
    test_preds = (test_probs >= t).astype(int)

    auc = roc_auc_score(test_labels, test_probs)
    f1  = f1_score(test_labels, test_preds)
    return float(auc), float(f1), float(t)

# =========================================================
# LOAD + NORMALIZE
# =========================================================
with safe_globals([Data]):
    data: Data = torch.load(DATA_PATH, map_location="cpu", weights_only=False)

print(f"✅ Loaded Juliet → Nodes={data.num_nodes}, Edges={data.num_edges}, FeatDim={data.x.size(1)}")

data = data.to(device)
in_dim = data.x.size(1)
edge_dim = data.edge_attr.size(1) if hasattr(data, "edge_attr") else 0

# normalize edge_attr (critical)
if hasattr(data, "edge_attr") and data.edge_attr is not None and data.edge_attr.numel() > 0:
    ea = data.edge_attr.clone().float()
    ea = (ea - ea.mean(0)) / (ea.std(0) + 1e-6)
    data.edge_attr = ea

# feature slices (from your pipeline)
# codebert: [0:768], time: [768:784], semgrep: [784:785], ast: [785:835] by your build_cepg layout
CODEBERT_DIM = 768
TIME_DIM = 16
SEMGREP_DIM = 1
AST_DIM = in_dim - (CODEBERT_DIM + TIME_DIM + SEMGREP_DIM)

SEM_S0 = CODEBERT_DIM + TIME_DIM
SEM_S1 = SEM_S0 + 1
AST_S0 = SEM_S1
AST_S1 = AST_S0 + AST_DIM

print(f"🔎 Derived slices: semgrep=({SEM_S0},{SEM_S1}), ast=({AST_S0},{AST_S1}), AST_DIM={AST_DIM}")

# =========================================================
# FIXED SPLIT (same for all ablations)
# =========================================================
N = data.num_nodes
perm = torch.randperm(N, device=device)

train_idx = perm[:int(0.8 * N)]
val_idx   = perm[int(0.8 * N):int(0.9 * N)]
test_idx  = perm[int(0.9 * N):]

# =========================================================
# MODEL (TGATv3-like, with toggleable heads)
# =========================================================
class TGATv3Abl(torch.nn.Module):
    def __init__(self, in_channels, hidden=256, heads=8, edge_dim=2,
                 use_semgrep_head=True):
        super().__init__()
        self.use_semgrep_head = use_semgrep_head

        self.conv1 = TransformerConv(in_channels, hidden, heads=heads,
                                     edge_dim=edge_dim, dropout=0.3)
        self.bn1 = torch.nn.BatchNorm1d(hidden * heads)

        self.conv2 = TransformerConv(hidden * heads, hidden, heads=1,
                                     edge_dim=edge_dim, dropout=0.3)
        self.bn2 = torch.nn.BatchNorm1d(hidden)

        self.reconstruct = torch.nn.Linear(hidden, in_channels)
        self.next_pred   = torch.nn.Linear(hidden, in_channels)

        if use_semgrep_head:
            self.semgrep_pred = torch.nn.Linear(hidden, 1)

        self.readout = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden, 1)
        )

    def forward(self, x, edge_index, edge_attr):
        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(1).float()

        h1 = self.conv1(x, edge_index, edge_attr)
        h1 = F.relu(self.bn1(h1))

        h2 = self.conv2(h1, edge_index, edge_attr)
        h2 = self.bn2(h2)

        # residual projection like your v3 (cheap stable projection)
        if h1.size(1) != h2.size(1):
            W = torch.eye(h2.size(1), h1.size(1), device=h1.device)
            h2 = h2 + F.linear(h1, W)
        else:
            h2 = h2 + h1

        h2 = F.relu(h2)

        x_hat = self.reconstruct(h2)
        next_hat = self.next_pred(h2)
        semg = self.semgrep_pred(h2) if self.use_semgrep_head else None
        logits = self.readout(h2).squeeze(-1)
        return logits, x_hat, next_hat, semg, h2

# =========================================================
# TRAIN ONE SETTING
# =========================================================
def run_setting(setting_name: str,
                use_semgrep_loss: bool,
                use_ast_features: bool,
                use_next_loss: bool,
                use_recon_loss: bool):
    # clone data.x so each run is isolated
    x_orig = data.x

    # AST ablation: zero AST block (keep dim same)
    if not use_ast_features and AST_DIM > 0:
        x_mod = x_orig.clone()
        x_mod[:, AST_S0:AST_S1] = 0.0
    else:
        x_mod = x_orig

    # semgrep head on/off: if loss disabled, head can be disabled too (cleaner)
    model = TGATv3Abl(in_channels=in_dim, edge_dim=edge_dim,
                      use_semgrep_head=use_semgrep_loss).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    for ep in range(1, EPOCHS + 1):
        model.train()
        opt.zero_grad()

        logits, x_hat, next_hat, semg, _ = model(x_mod, data.edge_index, data.edge_attr)
        logits = torch.clamp(logits, -50, 50)

        y = data.y.float()
        y_smooth = (1 - SMOOTHING) * y + 0.5 * SMOOTHING

        loss_cls = focal_loss(logits[train_idx], y_smooth[train_idx])

        loss_recon = torch.tensor(0.0, device=device)
        if use_recon_loss:
            loss_recon = F.mse_loss(x_hat, x_mod)

        loss_next = torch.tensor(0.0, device=device)
        if use_next_loss:
            loss_next = edge_next_loss(next_hat, x_mod, data.edge_index)

        loss_semg = torch.tensor(0.0, device=device)
        if use_semgrep_loss:
            sem_true = x_mod[:, SEM_S0:SEM_S1]   # [N,1]
            loss_semg = F.mse_loss(semg, sem_true)

        loss = (ALPHA * loss_recon) + (BETA * loss_next) + (GAMMA * loss_semg) + (DELTA * loss_cls)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()

    # evaluate
    # temporarily swap in x_mod for eval
    x_backup = data.x
    data.x = x_mod
    auc, f1, best_t = eval_auc_f1(model, data, val_idx.detach().cpu().numpy(),
                                  test_idx.detach().cpu().numpy(),
                                  temperature=1.0)
    data.x = x_backup

    return {
        "Setting": setting_name,
        "Semgrep Loss": "✅" if use_semgrep_loss else "❌",
        "AST Features": "✅" if use_ast_features else "❌",
        "Next-node Loss": "✅" if use_next_loss else "❌",
        "Recon Loss": "✅" if use_recon_loss else "❌",
        "AUC↑": round(auc, 4),
        "F1↑": round(f1, 4),
        "Val-threshold": round(best_t, 3),
    }

# =========================================================
# ABLATION GRID (Study 2)
# =========================================================
settings = [
    ("All aux losses", True,  True,  True,  True),
    ("w/o Semgrep Loss", False, True,  True,  True),
    ("w/o AST Features", True,  False, True,  True),
    ("w/o Next-node Loss", True, True,  False, True),
    ("w/o Recon Loss", True,  True,  True,  False),
    ("No auxiliary losses", False, False, False, False),
]

rows = []
for s in settings:
    name, semg, ast, nxt, rec = s
    print(f"\nRunning: {name} | semgrep={semg} ast={ast} next={nxt} recon={rec}")
    rows.append(run_setting(name, semg, ast, nxt, rec))

df = pd.DataFrame(rows)
print("\n🧠 Ablation Study 2: Semantic + Auxiliary Supervision (Juliet)\n")
print(df.to_markdown(index=False))

# Save to CSV for paper/plotting
out_csv = "ablation2_aux_supervision_juliet.csv"
df.to_csv(out_csv, index=False)
print(f"\n💾 Saved → {out_csv}")
