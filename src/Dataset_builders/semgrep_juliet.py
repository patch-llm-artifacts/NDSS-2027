#!/usr/bin/env python3
"""
semgrep_enrich_juliet_seq_safe.py
====================================================================
🧠 Safe sequential Semgrep enrichment for Juliet graphs
✅ No multiprocessing (no Bus errors, no zombies)
✅ Automatic checkpoint resume
✅ Periodic memory + heartbeat logs
✅ Safe Semgrep timeout + recovery
✅ Coverage statistics and sample node dump
====================================================================
"""

import os, json, torch, tempfile, subprocess, gc, re, hashlib, time, psutil
from tqdm import tqdm
from torch_geometric.data import Data
from torch.serialization import safe_globals
from collections import Counter, defaultdict

# ============================================================
# CONFIG
# ============================================================
DATA_PATH   = "/app/juliet_cepg_full.pt"
SPLIT_PATH  = "/app/juliet_cepg_split.pt"
OUT_PATH    = "/app/juliet_cepg_semgrep.pt"
TMP_DIR     = "/app/tmp_semgrep"
STATS_PATH  = "/app/semgrep_rule_stats.json"
CUSTOM_RULE = "/app/rules/juliet_rules.yaml"

os.makedirs(TMP_DIR, exist_ok=True)
TIMEOUT = 45
CHUNK_SIZE = 20
MAX_MEM_GB = 10.0

RULESETS = [
    "p/c", "p/cpp", "p/security-audit", "p/memory-leaks",
    "p/owasp-top-ten", "p/cwe-top25", "p/cwe-top100", CUSTOM_RULE
]

# ============================================================
# UTILS
# ============================================================
def get_mem_usage_gb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)

def short_hash(code: str) -> str:
    return hashlib.sha1(code.encode("utf-8", "ignore")).hexdigest()[:16]

def clean_code(src: str) -> str:
    if not isinstance(src, str): return ""
    s = src.replace("\r", "\n")
    s = re.sub(r"^/\*.*?\*/", "", s, flags=re.DOTALL)
    s = re.sub(r"//.*", "", s)
    return s.strip()

def extract_function(src: str) -> str:
    s = clean_code(src)
    func_re = re.compile(r"\b[A-Za-z_][A-Za-z0-9_\*\s]*\([^)]*\)\s*\{", re.M)
    m = func_re.search(s)
    if not m: return s[:2000]
    start = m.start()
    brace = 0
    for i, ch in enumerate(s[start:], start=start):
        if ch == '{': brace += 1
        elif ch == '}': brace -= 1
        if brace == 0 and i > start:
            return s[start:i+1]
    return s[start:]

# ============================================================
# SEMGREP WRAPPER
# ============================================================
def run_semgrep_once(ruleset, path):
    try:
        res = subprocess.run(
            ["semgrep", "--config", ruleset, "--json", path],
            capture_output=True, text=True, timeout=TIMEOUT
        )
        if res.returncode not in (0, 1):  # 1 = findings
            return 0, []
        data = json.loads(res.stdout or "{}")
        results = data.get("results", [])
        return len(results), [r.get("check_id", "") for r in results]
    except subprocess.TimeoutExpired:
        print(f"⚠️ Timeout in {ruleset} for {path}")
        return 0, []
    except Exception as e:
        print(f"⚠️ Semgrep failure: {e}")
        return 0, []

def run_semgrep_on_code(code: str):
    """Run Semgrep on single function snippet."""
    if not code.strip():
        return 0, []
    func_code = extract_function(code)
    if len(func_code) < 40:
        return 0, []

    ext = ".cpp" if "class" in func_code or "std::" in func_code else ".c"
    tmp_path = None
    total_hits, all_rules = 0, []

    try:
        with tempfile.NamedTemporaryFile("w", suffix=ext, dir=TMP_DIR, delete=False) as f:
            f.write(func_code)
            tmp_path = f.name

        for r in RULESETS:
            h, rules = run_semgrep_once(r, tmp_path)
            total_hits += h
            all_rules.extend(rules)

        uniq_rules = sorted(set(all_rules))
        return total_hits, uniq_rules
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except: pass

# ============================================================
# GRAPH LOGIC
# ============================================================
def enrich_node(node_meta_entry):
    code = node_meta_entry.get("full_source") or node_meta_entry.get("code_snippet") or ""
    hits, rules = run_semgrep_on_code(code)
    node_meta_entry["semgrep_hits"] = hits
    node_meta_entry["semgrep_rules"] = rules
    return hits

def process_graph(args):
    idx, g = args
    total_hits = 0
    try:
        for nmeta in g.node_meta:
            total_hits += enrich_node(nmeta)
        g.semgrep_total_hits = total_hits
        return idx, g, total_hits
    except Exception as e:
        print(f"⚠️ Graph {idx} failed: {e}")
        return idx, g, 0

# ============================================================
# SPLIT GIANT GRAPH
# ============================================================
def split_big_graph_if_needed():
    if os.path.exists(SPLIT_PATH):
        print(f"ℹ️ Found existing split dataset → {SPLIT_PATH}")
        with safe_globals([Data]):
            graphs = torch.load(SPLIT_PATH, weights_only=False)
        return graphs

    with safe_globals([Data]):
        big = torch.load(DATA_PATH, weights_only=False)
    if isinstance(big, list):
        return big

    print(f"⚙️ Splitting single graph → {big.num_nodes} nodes across CWEs…")
    groups = defaultdict(list)
    for i, m in enumerate(big.node_meta):
        groups[m.get("cwe_id", f"UNK_{i}")].append(i)

    mini_graphs = []
    for cwe, idxs in tqdm(groups.items(), desc="Splitting by CWE"):
        sub_x = big.x[idxs]
        mask = torch.isin(big.edge_index[0], torch.tensor(idxs)) & torch.isin(big.edge_index[1], torch.tensor(idxs))
        sub_e = big.edge_index[:, mask]
        sub_a = big.edge_attr[mask]
        sub_y = big.y[idxs]
        g = Data(x=sub_x, edge_index=sub_e, edge_attr=sub_a, y=sub_y)
        g.node_meta = [big.node_meta[i] for i in idxs]
        g.cwe_id = cwe
        mini_graphs.append(g)

    torch.save(mini_graphs, SPLIT_PATH)
    print(f"✅ Saved {len(mini_graphs)} mini-graphs → {SPLIT_PATH}")
    return mini_graphs

# ============================================================
# CHECKPOINT + STATS
# ============================================================
def load_checkpoint():
    if os.path.exists(OUT_PATH) and os.path.getsize(OUT_PATH) > 10_000:
        try:
            print("🔄 Found checkpoint → loading partial progress…")
            with safe_globals([Data]):
                enriched = torch.load(OUT_PATH, weights_only=False)
            print(f"✅ Resumed from {len(enriched)} graphs.")
            return enriched
        except Exception as e:
            print(f"⚠️ Failed to load checkpoint: {e}")
    return []

def compute_stats(graphs):
    rc, cc = Counter(), Counter()
    hit_graphs = 0
    for g in graphs:
        rules = []
        for n in getattr(g, "node_meta", []):
            rules.extend(n.get("semgrep_rules", []))
        if rules:
            hit_graphs += 1
        rc.update(rules)
        for r in rules:
            m = re.search(r"(CWE[-_]?\d+)", r, re.I)
            if m:
                cc[m.group(1).upper()] += 1
    stats = {
        "total_graphs": len(graphs),
        "graphs_with_hits": hit_graphs,
        "coverage_percent": round(100 * hit_graphs / max(1, len(graphs)), 2),
        "distinct_rules": len(rc),
        "top_rules": rc.most_common(15),
        "top_cwes": cc.most_common(15)
    }
    with open(STATS_PATH, "w") as f: json.dump(stats, f, indent=2)
    print(f"\n📊 Coverage: {hit_graphs}/{len(graphs)} "
          f"({stats['coverage_percent']}%) | {len(rc)} unique rules")
    for rule, n in rc.most_common(10):
        print(f"   - {rule}: {n}")
    return stats

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    graphs = split_big_graph_if_needed()
    print(f"📦 Loaded {len(graphs)} sub-graphs for enrichment")

    enriched = load_checkpoint()
    done = len(enriched)
    remaining = graphs[done:]
    total_hits = sum(getattr(g, "semgrep_total_hits", 0) for g in enriched)
    print(f"🎯 {len(remaining)} remaining | starting total_hits={total_hits}")

    for ci in range(0, len(remaining), CHUNK_SIZE):
        chunk = remaining[ci:ci + CHUNK_SIZE]
        if not chunk: break

        print(f"\n🧩 Chunk {ci // CHUNK_SIZE + 1}/{(len(remaining) // CHUNK_SIZE) + 1} "
              f"({len(chunk)} graphs, sequential)")
        start_time = time.time()
        results = []

        for idx, g in enumerate(chunk, start=done):
            _, enriched_g, hits = process_graph((idx, g))
            results.append((idx, enriched_g, hits))
            if idx % 5 == 0:
                print(f"💬 Progress {idx - done}/{len(chunk)} | hits={hits} | mem={get_mem_usage_gb():.2f} GB", flush=True)

            if get_mem_usage_gb() > MAX_MEM_GB:
                print(f"⚠️ Memory high ({get_mem_usage_gb():.1f} GB) → checkpointing early")
                break

        hits_chunk = sum(h for _, _, h in results)
        total_hits += hits_chunk
        enriched.extend(g for _, g, _ in results)
        done += len(chunk)

        torch.save(enriched, OUT_PATH)
        gc.collect()

        print(f"💾 Checkpoint → {OUT_PATH} | Δhits={hits_chunk} | total={total_hits} "
              f"| mem={get_mem_usage_gb():.1f} GB | time={(time.time()-start_time):.1f}s")

    compute_stats(enriched)
    if enriched:
        ex = enriched[min(3, len(enriched) - 1)].node_meta[0]
        print("\n🔍 Example node_meta:")
        print(json.dumps(ex, indent=2)[:800])

    print(f"\n✅ Finished sequential Semgrep enrichment.\nResults → {OUT_PATH}")
