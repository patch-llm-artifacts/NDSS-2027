#!/usr/bin/env python3
"""
rebalance_labels_semgrep_auto.py — ensure both classes appear
even when many semgrep_mean == 0.
"""
import torch, numpy as np
from collections import Counter

IN_PATH  = "/app/checkpoints_cepg_ast/partial_875.pt"
OUT_PATH = "/app/codenet_cepg_ast_balanced.pt"

print(f"📦 Loading graphs from {IN_PATH} ...")
graphs = torch.load(IN_PATH, weights_only=False)
print(f"✅ Loaded {len(graphs)} graphs")

means = np.array([getattr(g, "semgrep_mean", 0.0) for g in graphs])
print(f"📊 semgrep_mean stats → min={means.min():.4f}, max={means.max():.4f}, mean={means.mean():.4f}")

# --- choose a percentile that actually separates 0s and >0s ---
thr = np.percentile(means, 25)   # start at 25th percentile
if thr == 0:
    # climb upward until we hit a non-zero split
    for p in [50, 60, 70, 80, 90]:
        thr = np.percentile(means, p)
        if thr > 0:
            print(f"🔧 Adjusted threshold to {p}th percentile = {thr:.4f}")
            break

# --- assign labels ---
for g in graphs:
    m = getattr(g, "semgrep_mean", 0.0)
    y_semgrep = 1 if m < thr else 0   # 1 → cleaner / fewer rule hits
    y_status  = int(getattr(g, "has_accept", False))
    y_combined = 1 if (y_status == 1 and y_semgrep == 1) else 0
    g.y_semgrep  = torch.tensor(y_semgrep, dtype=torch.long)
    g.y_status   = torch.tensor(y_status, dtype=torch.long)
    g.y_combined = torch.tensor(y_combined, dtype=torch.long)
    g.y = g.y_combined

labels = Counter(int(g.y) for g in graphs)
print(f"📊 New label distribution: {labels}")
torch.save(graphs, OUT_PATH)
print(f"✅ Saved balanced dataset to {OUT_PATH}")
