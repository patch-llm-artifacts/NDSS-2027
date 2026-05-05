# === build_cepg.py ===
import os, torch, pandas as pd, numpy as np, hashlib, json
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel
from torch_geometric.data import Data
import sys
sys.path.insert(0, os.path.dirname(__file__))
from features import compute_diff_features, hash_code, extract_ast_features_cached
from semgrep_runner import run_semgrep_parallel   # ⚡️ upgraded parallel version

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")

# === Feature layout (keep consistent everywhere) ===
CODEBERT_DIM = 768
TIME_DIM     = 16
SEMGREP_DIM  = 1
AST_DIM      = 50

# Convenience: start indices for each block in x
SLICE = {
    "codebert": (0, CODEBERT_DIM),
    "time":     (CODEBERT_DIM, CODEBERT_DIM + TIME_DIM),
    "semgrep":  (CODEBERT_DIM + TIME_DIM, CODEBERT_DIM + TIME_DIM + SEMGREP_DIM),
    "ast":      (CODEBERT_DIM + TIME_DIM + SEMGREP_DIM, CODEBERT_DIM + TIME_DIM + SEMGREP_DIM + AST_DIM),
}
TOTAL_DIM = CODEBERT_DIM + TIME_DIM + SEMGREP_DIM + AST_DIM

# === Paths ===
CSV_PATH = "/app/codenet_subset.csv"                # Input CSV
OUT_PATH = "/app/codenet_cepg_ast.pt"               # Output graphs with AST
CHECKPOINT_DIR = "/app/checkpoints_cepg_ast"        # Partial checkpoints
SEMGR_CACHE_PATH = "/app/semgrep_cache.json"        # Semgrep cache
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# === CodeBERT setup ===
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("microsoft/codebert-base").to(device)
model.eval()

# === Load Semgrep cache ===
semgrep_cache = {}
if os.path.exists(SEMGR_CACHE_PATH):
    with open(SEMGR_CACHE_PATH, "r") as f:
        try:
            semgrep_cache = json.load(f)
        except json.JSONDecodeError:
            semgrep_cache = {}

# === Helper: Embed Code ===
def embed_code(code, max_len=256):
    try:
        tokens = tokenizer(code, return_tensors="pt", truncation=True,
                           max_length=max_len, padding=True)
        tokens = {k: v.to(device) for k, v in tokens.items()}
        with torch.no_grad():
            out = model(**tokens)
        return out.last_hidden_state[:, 0, :].squeeze(0).cpu()
    except Exception as e:
        print(f"Embedding error: {e}")
        return None

# === Helper: Time Encoding ===
def time_encoding(t, dim=16):
    pe = np.zeros(dim)
    div = np.exp(np.arange(0, dim, 2) * -(np.log(10000.0) / dim))
    for i in range(0, dim, 2):
        pe[i] = np.sin(t * div[i//2])
        pe[i+1] = np.cos(t * div[i//2])
    return torch.tensor(pe, dtype=torch.float)

# === Helper: Semgrep with policy metadata ===
# === Helper: Semgrep with policy metadata ===
def get_semgrep_hits(path: str):
    """Return (#hits, rule_metadata) using cached JSON for fast reuse."""
    if not os.path.exists(path):
        return 0, []

    # use cache if present
    if path in semgrep_cache:
        value = semgrep_cache[path]
        # backward compatibility: handle old cache entries storing int only
        if isinstance(value, int):
            return value, []
        return value

    try:
        # use the new parallel runner if available
        from semgrep_runner import run_semgrep_parallel
        result = run_semgrep_parallel(path)
        if isinstance(result, dict):  # parallel mode for batch
            hits, rules = list(result.values())[0]
        else:
            hits, rules = result
    except Exception as e:
        print(f"⚠️ Semgrep failed on {path}: {e}")
        hits, rules = 0, []

    semgrep_cache[path] = (hits, rules)
    return hits, rules


# === Helper: Infer language from file path ===
def infer_language_from_path(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".py"]: return "python"
    if ext in [".c"]: return "c"
    if ext in [".cpp", ".cc", ".cxx", ".hpp", ".h"]: return "cpp"
    if ext in [".java"]: return "java"
    return "c"  # default fallback

# === Main CEPG builder ===
def build_problem_cepg(problem_df, pid, idx, time_dim=TIME_DIM, ast_dim=AST_DIM):
    node_feats, edges, edge_times = [], [], []
    node_meta, edge_meta = [], []

    # ROOT node
    root_feat = torch.zeros(TOTAL_DIM)
    node_feats.append(root_feat.unsqueeze(0))
    node_meta.append({
        "submission_id": "root", "user_id": None,
        "problem_id": pid, "hash": None,
        "diff_stats": None, "semgrep_hits": 0, "ast_dim": 0
    })
    node_idx, root_idx = 1, 0
    accepted = False

    for user_id, user_df in problem_df.groupby("user_id"):
        print(f"🔄 User {user_id}, {len(user_df)} submissions for {pid}")
        user_df = user_df.sort_values("date")
        prev_code, prev_idx, prev_time = None, None, None

        for _, row in user_df.iterrows():
            if not os.path.exists(row["code_path"]):
                print(f"🚫 File not found: {row['code_path']}")
                continue

            try:
                with open(row["code_path"], "r", encoding="utf-8", errors="ignore") as f:
                    code = f.read()
            except Exception as e:
                print(f"❌ Error reading file {row['code_path']}: {e}")
                continue

            emb = embed_code(code)
            if emb is None or emb.numel() == 0:
                print(f"⚠️ Skipping embedding for {row['submission_id']}")
                continue

            # Timestamp and encodings
            tstamp = pd.to_datetime(row["date"], errors="coerce", utc=True)
            t_val = tstamp.value / 1e12 if pd.notna(tstamp) else 0.0
            t_enc = time_encoding(t_val, dim=time_dim)

            # Semgrep (hits + rule metadata)
            semgrep_hits, semgrep_rules = get_semgrep_hits(row["code_path"])
            semgrep_tensor = torch.tensor([float(semgrep_hits)], dtype=torch.float)

            # Language inference + AST
            language = infer_language_from_path(row["code_path"])
            ast_feat = extract_ast_features_cached(code, language)

            feat = torch.cat([emb, t_enc, semgrep_tensor, ast_feat])
            node_feats.append(feat.unsqueeze(0))

            diff_stats, diff_hash = compute_diff_features(prev_code, code) if prev_code else ({"added": 0, "removed": 0}, None)
            code_hash = hash_code(code)

            node_meta.append({
                "submission_id": row["submission_id"],
                "user_id": user_id,
                "problem_id": row["problem_id"],
                "ts": t_val,
                "hash": code_hash,
                "diff_stats": diff_stats,
                "semgrep_hits": semgrep_hits,
                "semgrep_rules": semgrep_rules,   
                "ast_dim": ast_feat.shape[0],
                "language": language
            })

            # Edges
            if prev_idx is not None:
                dt = t_val - prev_time if prev_time else 0.0
                edges.append([prev_idx, node_idx])
                edge_times.append(dt)
                edge_meta.append({
                    "src": prev_idx, "dst": node_idx, "dt": dt,
                    "diff_hash": diff_hash,
                    "diff_size": diff_stats["added"] + diff_stats["removed"],
                    "edge_type": "edit"
                })
            else:
                edges.append([root_idx, node_idx])
                edge_times.append(0.0)
                edge_meta.append({
                    "src": root_idx, "dst": node_idx, "dt": 0.0,
                    "diff_hash": None, "diff_size": 0, "edge_type": "edit"
                })

            prev_code, prev_idx, prev_time = code, node_idx, t_val
            node_idx += 1
            if row["status"] == "Accepted":
                accepted = True

    if len(node_feats) <= 2:
        print(f"❌ Skipping {pid} — only {len(node_feats)} nodes")
        return None

    x = torch.cat(node_feats, dim=0)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_time = torch.tensor(edge_times, dtype=torch.float)
    y = torch.tensor(1 if accepted else 0)

    data = Data(x=x, edge_index=edge_index, edge_time=edge_time, y=y)
    data.node_meta, data.edge_meta = node_meta, edge_meta
    data.problem_id = pid

    # === Attach feature layout + Semgrep stats ===
    data.feature_slices = SLICE
    node_hits = [m["semgrep_hits"] for m in node_meta[1:] if m.get("semgrep_hits") is not None]
    data.semgrep_mean = float(np.mean(node_hits)) if node_hits else 0.0
    data.semgrep_max  = float(np.max(node_hits)) if node_hits else 0.0
    data.has_accept   = bool(accepted)

    return data

# === Run Full Dataset ===
def main():
    df = pd.read_csv(CSV_PATH).fillna("")
    print(f"📄 Loaded {len(df)} rows")

    # Quick diagnostic
    from collections import Counter
    exts = [os.path.splitext(p)[1].lower() for p in df["code_path"]]
    print("🧩 Language distribution:", Counter(exts))

    graphs = []
    for idx, (pid, gdf) in enumerate(tqdm(df.groupby("problem_id"))):
        g = build_problem_cepg(gdf, pid, idx)
        if g:
            graphs.append(g)
        if idx > 0 and idx % 25 == 0:
            partial_path = os.path.join(CHECKPOINT_DIR, f"partial_{idx}.pt")
            torch.save(graphs, partial_path)
            with open(SEMGR_CACHE_PATH, "w") as f:
                json.dump(semgrep_cache, f)
            print(f"📂 Saved checkpoint + cache: {partial_path}")

    # === Add weak supervision labels (one-time, no rerun) ===
    means = [g.semgrep_mean for g in graphs]
    thr = float(np.percentile(means, 75)) if means else 1e9

    for g in graphs:
        y_status = 1 if g.has_accept else 0
        y_semgrep = 1 if g.semgrep_mean < thr else 0
        y_combined = 1 if (y_status == 1 or y_semgrep == 1) else 0

        g.y_status   = torch.tensor(y_status, dtype=torch.long)
        g.y_semgrep  = torch.tensor(y_semgrep, dtype=torch.long)
        g.y_combined = torch.tensor(y_combined, dtype=torch.long)
        g.y          = torch.tensor(y_combined, dtype=torch.long)

    torch.save(graphs, OUT_PATH)
    with open(SEMGR_CACHE_PATH, "w") as f:
        json.dump(semgrep_cache, f)
    print(f"✅ Final saved {len(graphs)} graphs to {OUT_PATH}")
    print(f"🧠 Semgrep cache size: {len(semgrep_cache)} entries")

if __name__ == "__main__":
    main()
