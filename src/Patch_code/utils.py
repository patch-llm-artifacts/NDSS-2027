# utils.py
import torch
import os
import json
from torch.serialization import safe_globals
from torch_geometric.data import Data

# 📍 Update this if needed
CEPG_PATH = "/app/juliet_cepg_full.pt"

# ✅ Safe loader for full CEPG dataset
def load_cepg_dataset():
    if not os.path.exists(CEPG_PATH):
        raise FileNotFoundError(f"Missing: {CEPG_PATH}")
    
    print("📦 Loading CEPG dataset safely...")
    with safe_globals([Data]):
        cepg_data = torch.load(CEPG_PATH, map_location="cpu", weights_only=False)
    
    return cepg_data

# 🔎 Get subgraph for a given node ID like "node_2069"
def load_cepg_dataset_by_id(node_id: str) -> Data:
    data = load_cepg_dataset()
    lookup = {n["id"]: i for i, n in enumerate(data.node_meta)}
    
    if node_id not in lookup:
        raise ValueError(f"Node ID {node_id} not found in dataset.")
    
    idx = lookup[node_id]
    g = data.subgraph([idx])  # single-node subgraph
    g.id = node_id
    g.meta = data.node_meta[idx]
    return g

# 🔎 Get subgraph by node index (int)
def load_vuln_node_graph_by_index(node_idx: int) -> Data:
    data = load_cepg_dataset()
    if node_idx < 0 or node_idx >= data.x.size(0):
        raise IndexError(f"Node index {node_idx} out of bounds.")
    
    g = data.subgraph([node_idx])
    g.id = data.node_meta[node_idx]["id"]
    g.meta = data.node_meta[node_idx]
    return g
