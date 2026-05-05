#!/usr/bin/env python3
"""
build_juliet_temporal_cepg.py
------------------------------------------------------------
Temporal DAG builder for Juliet Test Suite (C/C++).
Each node = function version (bug or fix)
Each edge = temporal evolution (bug→fix, or consecutive versions)
Node features = CodeBERT + time + Semgrep + AST summary (+diff)
------------------------------------------------------------
Output:
  /app/juliet_cepg_full.pt
------------------------------------------------------------
"""

import os, torch, json, traceback, numpy as np
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
from transformers import RobertaTokenizer, RobertaModel
from torch_geometric.data import Data
from torch.serialization import safe_globals
from features import extract_ast_features_cached, hash_code, compute_diff_features
from build_cepg import time_encoding, SLICE

# --------------------------------------------------------------------
# Config
# --------------------------------------------------------------------
OUT_PATH = "/app/juliet_cepg_full.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("microsoft/codebert-base").to(device)
model.eval()

# Overwrite protection (so we don't overwrite a valid file)
if os.path.exists(OUT_PATH) and os.path.getsize(OUT_PATH) > 10_000:
    print(f"⚠️  File already exists ({os.path.getsize(OUT_PATH)/1e6:.1f} MB) — aborting overwrite.")
    import sys; sys.exit(0)

def dbg(msg): print(f"[DBG] {msg}", flush=True)

# --------------------------------------------------------------------
# Feature builders
# --------------------------------------------------------------------
def embed_code(code: str):
    try:
        toks = tokenizer(code, return_tensors="pt", truncation=True,
                         max_length=512, padding=True)
        toks = {k: v.to(device) for k, v in toks.items()}
        with torch.no_grad():
            out = model(**toks)
        return out.last_hidden_state[:, 0, :].squeeze(0).cpu()
    except Exception as e:
        dbg(f"⚠️ Embedding failed: {e}")
        return torch.zeros(768)

def build_function_node(code: str, label: int, cwe_id: str,
                        pseudo_time: int, prev_code=None):
    """Return node-level feature vector + metadata for one function."""
    emb = embed_code(code)
    t_enc = time_encoding(pseudo_time, dim=16)
    semgrep_tensor = torch.tensor([0.0], dtype=torch.float)
    ast_feat = extract_ast_features_cached(code, "c")
    if ast_feat.ndim > 1:
        ast_feat = ast_feat.mean(dim=0)
    feat = torch.cat([emb, t_enc, semgrep_tensor, ast_feat], dim=0)

    diff_stats, diff_hash = compute_diff_features(prev_code, code) if prev_code else ({"added":0,"removed":0}, None)
    hsh = hash_code(code)
   
    meta = {
        "label": int(label),
        "cwe_id": cwe_id,
        "hash": hsh,
        "diff_hash": diff_hash,
        "diff_stats": diff_stats,
        "pseudo_time": pseudo_time,
        "code_snippet": code[:400],     # keep for previews
        "full_source": code,            # <-- 🔥 full code here
        "source_type": "bad" if label == 1 else "good"

    }
    return feat.unsqueeze(0), meta

# --------------------------------------------------------------------
# Build temporal DAG across all CWEs
# --------------------------------------------------------------------
if __name__ == "__main__":
    ds_train = load_dataset("LorenzH/juliet_test_suite_c_1_3", split="train")
    ds_test  = load_dataset("LorenzH/juliet_test_suite_c_1_3", split="test")
    full_ds  = concatenate_datasets([ds_train, ds_test])
    print(f"📚 Combined Juliet samples: {len(full_ds)}")

    all_feats, all_meta = [], []
    edge_list, edge_attr = [], []
    time_counter = 0

    # -- Build nodes
    for row in tqdm(full_ds, desc="Encoding functions"):
        bad, good = row["bad"].strip(), row["good"].strip()
        cwe = str(row["class"])
        prev = None
        if bad:
            time_counter += 1
            f, m = build_function_node(bad, 1, cwe, time_counter)
            all_feats.append(f); all_meta.append(m)
            prev = bad
        if good:
            time_counter += 1
            f, m = build_function_node(good, 0, cwe, time_counter, prev)
            all_feats.append(f); all_meta.append(m)

    X = torch.cat(all_feats, dim=0)
    print(f"✅ Built {len(all_meta)} nodes → X shape {X.shape}")

    # -- Temporal DAG: chain all functions per CWE chronologically
    cwe_groups = {}
    for idx, m in enumerate(all_meta):
        cwe_groups.setdefault(m["cwe_id"], []).append((idx, m))

    for cwe, entries in cwe_groups.items():
        entries = sorted(entries, key=lambda x: x[1]["pseudo_time"])
        for i in range(1, len(entries)):
            src, dst = entries[i-1][0], entries[i][0]
            m_src, m_dst = entries[i-1][1], entries[i][1]
            Δt = m_dst["pseudo_time"] - m_src["pseudo_time"]
            diff_sz = m_dst["diff_stats"]["added"] + m_dst["diff_stats"]["removed"]
            edge_list.append([src, dst])
            edge_attr.append([Δt, diff_sz])
            # If explicit bug→fix, duplicate for stronger attention
            if m_src["label"] == 1 and m_dst["label"] == 0:
                edge_list.append([src, dst])
                edge_attr.append([Δt, diff_sz])

    edge_index = torch.tensor(edge_list, dtype=torch.long).T
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    if edge_index.numel() == 0:
        edge_index = torch.zeros((2,0), dtype=torch.long)
        edge_attr = torch.zeros((0,2), dtype=torch.float)
    else:
        # Bidirectional for TGAT stability
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_attr = torch.cat([edge_attr, edge_attr], dim=0)

    print(f"🔗 Temporal DAG edges: {edge_index.size(1)}")

    # -- Labels
    Y = torch.tensor([m["label"] for m in all_meta], dtype=torch.long)

    # -- Pack into Data object
    data = Data(
        x=X,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=Y
    )
    data.node_meta = all_meta
    data.feature_slices = SLICE
    data.num_nodes = X.size(0)
    data.num_edges = edge_index.size(1)

    # ----------------------------------------------------------------
    # ✅ Safe save (PyTorch ≥2.6 fix)
    # ----------------------------------------------------------------

    try:
        # Newer PyTorch (2.6+)
        from torch.serialization import safe_globals
        with safe_globals([]):
            torch.save(data, OUT_PATH, _use_new_zipfile_serialization=True, weights_only=False)
    except TypeError:
        # Older PyTorch (<2.6)
        print("ℹ️ Detected PyTorch <2.6 — using legacy save() format")
        with safe_globals([]):
            torch.save(data, OUT_PATH, _use_new_zipfile_serialization=True)


    print(f"✅ Saved temporal Juliet graph → {OUT_PATH}")
    print(f"💾 File size: {os.path.getsize(OUT_PATH)/1e6:.1f} MB")
