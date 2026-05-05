#!/usr/bin/env python3
"""
----------------------------------------------------------
This script:

1. Scans the entire /app/manybugs tree
2. Detects real C-modifying patches automatically
3. Extracts buggy + fixed C file pairs
4. Builds CEPG graphs using:
       - CodeBERT
       - AST features
       - Time positional encoding
       - Diff stats
5. Saves:
       /app/manybugs_cepg.pt
       /app/manybugs_cepg_stats.json
       /app/manybugs_cepg_debug.json
"""

import os, re, json, torch, numpy as np
from tqdm import tqdm
from torch_geometric.data import Data
from transformers import RobertaTokenizer, RobertaModel
from features import extract_ast_features_cached, compute_diff_features

ROOT = "/app/manybugs"

OUT_DATASET = "/app/manybugs_cepg.pt"
OUT_STATS   = "/app/manybugs_cepg_stats.json"
OUT_DEBUG   = "/app/manybugs_cepg_debug.json"

# ==========================================================
# Init CodeBERT
# ==========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model     = RobertaModel.from_pretrained("microsoft/codebert-base").to(device).eval()

CODEBERT_DIM = 768
AST_DIM = 50
TIME_DIM = 16

# ==========================================================
# Helpers
# ==========================================================
def embed(code):
    """CodeBERT CLS embedding."""
    try:
        toks = tokenizer(code, return_tensors="pt", truncation=True,
                         padding=True, max_length=256)
        toks = {k: v.to(device) for k, v in toks.items()}
        with torch.no_grad():
            out = model(**toks)
        return out.last_hidden_state[:, 0, :].squeeze(0).cpu()
    except:
        return torch.zeros(CODEBERT_DIM)

def time_enc(t):
    pe = np.zeros(TIME_DIM)
    div = np.exp(np.arange(0, TIME_DIM, 2) * -(np.log(10000)/TIME_DIM))
    for i in range(0, TIME_DIM, 2):
        pe[i] = np.sin(t * div[i//2])
        pe[i+1] = np.cos(t * div[i//2])
    return torch.tensor(pe, dtype=torch.float)

def is_patch_file(path):
    """Return True if file is a patch-like file containing a unified diff."""
    if not (path.endswith(".diff") or path.endswith(".patch") or path.endswith(".txt")):
        return False
    try:
        with open(path, "r", errors="ignore") as f:
            txt = f.read()
        return "--- " in txt and "+++" in txt
    except:
        return False

def extract_c_touches(patch_path):
    """Extract C files modified inside a diff."""
    try:
        lines = open(patch_path, "r", errors="ignore").read().splitlines()
    except:
        return []

    touched = []
    for ln in lines:
        if ln.startswith("---") or ln.startswith("+++"):
            parts = ln.split()
            if len(parts) >= 2 and parts[1].endswith(".c"):
                touched.append(parts[1])
    return list(set([t.strip() for t in touched]))

def resolve_paths(project_root, touched_list):
    """Resolve relative paths to actual disk paths."""
    out = []
    for t in touched_list:
        t_clean = t.replace("a/", "").replace("b/", "").replace("orig/", "").lstrip("./")
        for root, dirs, files in os.walk(project_root):
            for f in files:
                if f.endswith(".c") and f == os.path.basename(t_clean):
                    out.append(os.path.join(root, f))
    return list(set(out))

def detect_bug_fix_pair(paths):
    """Detect buggy and fixed file from list."""
    buggy = [p for p in paths if "orig" in p.lower() or "old" in p.lower() or "bug" in p.lower()]
    fixed = [p for p in paths if "new" in p.lower() or "fix" in p.lower() or "patched" in p.lower()]

    if buggy and fixed:
        return buggy[0], fixed[0]
    if len(paths) >= 2:
        return sorted(paths)[0], sorted(paths)[1]
    return None, None

# ==========================================================
# BUILD GRAPH (FIXED, WITH CORRECT LABELS)
# ==========================================================
def build_graph(project, buggy_path, fixed_path, t):
    """Build final 2-node CEPG graph with correct labels."""
    try:
        buggy = open(buggy_path, "r", errors="ignore").read()
        fixed = open(fixed_path, "r", errors="ignore").read()
    except:
        return None

    xb = torch.cat([
        embed(buggy),
        time_enc(t),
        extract_ast_features_cached(buggy, "c")
    ]).unsqueeze(0)

    xf = torch.cat([
        embed(fixed),
        time_enc(t + 1),
        extract_ast_features_cached(fixed, "c")
    ]).unsqueeze(0)

    x = torch.cat([xb, xf], dim=0)

    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_time  = torch.tensor([1.0, 1.0], dtype=torch.float)

    diff_stats, _ = compute_diff_features(buggy, fixed)

    # --- FIXED LABELS ---
    # Node 0 = buggy (label 1)
    # Node 1 = fixed (label 0)
    y = torch.tensor([1, 0], dtype=torch.long)

    g = Data(
        x=x,
        edge_index=edge_index,
        edge_time=edge_time,
        y=y
    )

    # node_meta, also with labels
    g.node_meta = [
        {
            "label": 1,
            "type": "buggy",
            "path": buggy_path,
            "full_source": buggy[:50000],
            "diff": diff_stats
        },
        {
            "label": 0,
            "type": "fixed",
            "path": fixed_path,
            "full_source": fixed[:50000],
            "diff": diff_stats
        }
    ]

    return g

# ==========================================================
# MAIN
# ==========================================================
def main():

    print("\nScanning for real C patches...")
    real_pairs = []
    debug_info = []

    projects = sorted(os.listdir(ROOT))
    for proj in tqdm(projects):
        proj_root = os.path.join(ROOT, proj)
        if not os.path.isdir(proj_root):
            continue

        patch_files = []
        for root, dirs, files in os.walk(proj_root):
            for f in files:
                path = os.path.join(root, f)
                if is_patch_file(path):
                    patch_files.append(path)

        if not patch_files:
            continue

        for pfile in patch_files:
            touched_rel = extract_c_touches(pfile)
            if not touched_rel:
                continue

            resolved = resolve_paths(proj_root, touched_rel)
            if len(resolved) < 2:
                continue

            buggy, fixed = detect_bug_fix_pair(resolved)
            if buggy and fixed:
                real_pairs.append((proj, buggy, fixed))
                debug_info.append({
                    "project": proj,
                    "patch": pfile,
                    "buggy": buggy,
                    "fixed": fixed
                })

    print(f"\nFOUND {len(real_pairs)} real C modification pairs.\n")

    graphs = []
    t = 0
    print("Building graphs...")
    for proj, buggy, fixed in tqdm(real_pairs):
        g = build_graph(proj, buggy, fixed, t)
        if g:
            graphs.append(g)
            t += 2

    torch.save(graphs, OUT_DATASET)

    stats = {
        "projects_total": len(projects),
        "c_pairs": len(real_pairs),
        "graphs": len(graphs),
        "avg_nodes": float(np.mean([g.x.size(0) for g in graphs])),
        "avg_dim": float(np.mean([g.x.size(1) for g in graphs]))
    }

    json.dump(stats, open(OUT_STATS, "w"), indent=2)
    json.dump(debug_info, open(OUT_DEBUG, "w"), indent=2)

    print("\n==== DONE ====")
    print("Saved dataset:", OUT_DATASET)
    print("Stats:", OUT_STATS)
    print("Debug:", OUT_DEBUG)

if __name__ == "__main__":
    main()
