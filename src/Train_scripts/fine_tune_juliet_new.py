#!/usr/bin/env python3
"""
fine_tune_juliet_with_pretrained.py
------------------------------------------------------------
✓ Loads pretrained TGATMultiHead model (CEPG-trained)
✓ Fine-tunes it on Juliet CEPG graph using Focal Loss
✓ Uses BatchNorm, residuals, cosine LR scheduler
✓ Saves best model and node embeddings
------------------------------------------------------------
"""

import os, torch, torch.nn.functional as F, numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv
from sklearn.metrics import roc_auc_score, f1_score
from collections import Counter
import pandas as pd

# ---------- Config ----------
PRETRAINED_CKPT = "/app/cepg_tgat.pt"
JULIET_DATA     = "/app/juliet_cepg_full.pt"
JULIET_BEST     = "/app/tgat_juliet_finetuned.pt"
JULIET_EMB      = "/app/juliet_tgat_embeddings.pt"
JULIET_CSV      = "/app/juliet_tgat_predictions.csv"

EPOCHS = 15
LR = 5e-5
WEIGHT_DECAY = 1e-5
TEMPERATURE = 2.0
SMOOTHING = 0.1
ALPHA, BETA, GAMMA, DELTA = 0.4, 0.2, 0.2, 0.8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Load Juliet CEPG ----------
from torch.serialization import safe_globals, add_safe_globals
import torch_geometric.data.data as pyg_data_mod

add_safe_globals([pyg_data_mod.Data, pyg_data_mod.DataEdgeAttr])
with safe_globals([pyg_data_mod.Data, pyg_data_mod.DataEdgeAttr]):
    data: Data = torch.load(JULIET_DATA, map_location="cpu", weights_only=False)


print(f"✅ Loaded Juliet → Nodes={data.num_nodes}, Edges={data.num_edges}, FeatDim={data.x.size(1)}")
print("📊 Label dist:", Counter(data.y.tolist()))

data = data.to(device)
in_dim    = data.x.size(1)
data.edge_attr = data.edge_attr[:, :1]  # Keep only first feature (Δt)
edge_dim = 1
data.edge_attr = (data.edge_attr - data.edge_attr.mean(0)) / (data.edge_attr.std(0) + 1e-6)

# ---------- Define TGATModel ----------
class TGATMultiHead(torch.nn.Module):
    def __init__(self, in_channels, hidden=128, heads=4, edge_dim=1, use_semgrep_head=True, use_cls_head=True):
        super().__init__()
        self.use_semgrep_head = use_semgrep_head
        self.use_cls_head     = use_cls_head
        self.conv1 = TransformerConv(in_channels, hidden, heads=heads, edge_dim=edge_dim, dropout=0.1)
        self.conv2 = TransformerConv(hidden * heads, hidden, heads=1, edge_dim=edge_dim, dropout=0.1)
        self.reconstruct = torch.nn.Linear(hidden, in_channels)
        self.next_pred   = torch.nn.Linear(hidden, in_channels)
        if use_semgrep_head:
            self.semgrep_pred = torch.nn.Linear(hidden, 1)
        if use_cls_head:
            self.readout = torch.nn.Linear(hidden, 1)

    def forward(self, x, edge_index, edge_attr):
        h = F.relu(self.conv1(x, edge_index, edge_attr))
        h = F.relu(self.conv2(h, edge_index, edge_attr))
        x_hat   = self.reconstruct(h)
        nexthat = self.next_pred(h)
        semg    = self.semgrep_pred(h) if self.use_semgrep_head else None
        logits  = self.readout(h).squeeze(-1) if self.use_cls_head else None
        return logits, x_hat, nexthat, semg, h

# ---------- Initialize & Load Pretrained ----------
model = TGATMultiHead(
    in_channels=in_dim, hidden=128, heads=4, edge_dim=edge_dim,
    use_semgrep_head=True, use_cls_head=True
).to(device)

model.load_state_dict(torch.load(PRETRAINED_CKPT, map_location=device))
print("📥 Loaded pretrained weights from:", PRETRAINED_CKPT)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ---------- Split ----------
perm = torch.randperm(data.num_nodes)
train_idx = perm[:int(0.8 * len(perm))]
val_idx   = perm[int(0.8 * len(perm)):int(0.9 * len(perm))]
test_idx  = perm[int(0.9 * len(perm)):]

# ---------- Losses ----------
def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    probs = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = probs * targets + (1 - probs) * (1 - targets)
    return (alpha * (1 - p_t) ** gamma * ce).mean()

def edge_next_loss(nexthat, x, edge_index):
    if edge_index.numel() == 0: return torch.tensor(0.0, device=x.device)
    src, dst = edge_index
    src, dst = src.clamp(0, x.size(0)-1), dst.clamp(0, x.size(0)-1)
    return F.mse_loss(nexthat[src], x[dst])

@torch.no_grad()
def evaluate(model, data, idx):
    model.eval()
    logits, *_ = model(data.x, data.edge_index, data.edge_attr)
    logits = torch.clamp(logits, -50, 50)
    probs = torch.sigmoid(logits[idx] / TEMPERATURE).cpu().numpy()
    labels = data.y[idx].cpu().numpy()
    auc = roc_auc_score(labels, probs)
    preds = (probs >= 0.5).astype(int)
    f1 = f1_score(labels, preds)
    return auc, f1

# ---------- Training Loop ----------
best_auc = 0.0
for epoch in range(1, EPOCHS+1):
    model.train()
    optimizer.zero_grad()

    logits, x_hat, nexthat, semg, _ = model(data.x, data.edge_index, data.edge_attr)
    logits = torch.clamp(logits, -50, 50)

    y = data.y.float()
    y_smooth = (1 - SMOOTHING) * y + 0.5 * SMOOTHING

    loss_recon = F.mse_loss(x_hat, data.x)
    loss_next  = edge_next_loss(nexthat, data.x, data.edge_index)
    semgrep_true = data.x[:, 768 + 16 : 768 + 17]
    loss_semg = F.mse_loss(semg, semgrep_true)
    loss_cls  = focal_loss(logits[train_idx], y_smooth[train_idx])

    loss = ALPHA*loss_recon + BETA*loss_next + GAMMA*loss_semg + DELTA*loss_cls
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    scheduler.step()

    val_auc, val_f1 = evaluate(model, data, val_idx)
    print(f"Epoch {epoch:02d} | Loss={loss.item():.4f} | Val AUC={val_auc:.4f} | Val F1={val_f1:.4f}")

    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), JULIET_BEST)
        print("💾 Saved best fine-tuned model!")

# ---------- Final Test ----------
model.load_state_dict(torch.load(JULIET_BEST, map_location=device))
test_auc, test_f1 = evaluate(model, data, test_idx)
print(f"🎯 Test AUC={test_auc:.4f} | F1={test_f1:.4f}")

# ---------- Export ----------
@torch.no_grad()
def export_predictions_csv(model, data, idx, out_csv, temperature=TEMPERATURE):
    model.eval()
    logits, *_ = model(data.x, data.edge_index, data.edge_attr)
    logits = torch.clamp(logits, -50, 50)
    probs = torch.sigmoid(logits[idx] / temperature).cpu().numpy()
    labels = data.y[idx].cpu().numpy().astype(int)
    preds = (probs >= 0.5).astype(int)
    df = pd.DataFrame({"y_true": labels, "y_pred": preds, "y_score": probs})
    df.to_csv(out_csv, index=False)
    print(f"📤 Exported predictions → {out_csv}")

export_predictions_csv(model, data, test_idx, JULIET_CSV)

with torch.no_grad():
    _, _, _, _, H = model(data.x, data.edge_index, data.edge_attr)
torch.save(H.cpu(), JULIET_EMB)
print(f"🧠 Saved node embeddings → {JULIET_EMB}")
