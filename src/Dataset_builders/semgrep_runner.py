# === semgrep_runner.py ===
"""
Parallelized Semgrep wrapper with detailed rule metadata and caching.
⚙️ Requirements: pip install semgrep
"""

import subprocess, json, os
from concurrent.futures import ThreadPoolExecutor, as_completed

# ----------------------------------------------------------------------
# 🧩 Internal: scan one file
# ----------------------------------------------------------------------
def _scan_file(path: str, config: str = "auto", timeout: int = 20):
    """
    Run Semgrep on a single file and return (path, hits, rules).
    """
    if not os.path.exists(path):
        return (path, 0, [])

    try:
        result = subprocess.run(
            ["semgrep", "--quiet", "--json", "--config", config, path],
            capture_output=True, text=True, timeout=timeout
        )

        if result.returncode not in (0, 1):  # 0=clean, 1=matches found
            return (path, 0, [])

        data = json.loads(result.stdout or "{}")

        # Extract detailed rule metadata
        rules = [
            {
                "id": r.get("check_id"),
                "message": r.get("extra", {}).get("message", ""),
                "severity": r.get("extra", {}).get("severity", ""),
                "start_line": r.get("start", {}).get("line"),
                "end_line": r.get("end", {}).get("line"),
            }
            for r in data.get("results", [])
        ]
        return (path, len(rules), rules)

    except subprocess.TimeoutExpired:
        print(f"⏱️ Semgrep timeout on {path}")
        return (path, 0, [])
    except json.JSONDecodeError:
        print(f"⚠️ JSON decode failed for {path}")
        return (path, 0, [])
    except Exception as e:
        print(f"⚠️ Semgrep error on {path}: {e}")
        return (path, 0, [])


# ----------------------------------------------------------------------
# 🚀 Parallel interface for fast scans
# ----------------------------------------------------------------------
def run_semgrep_parallel(paths, workers: int = 4, config: str = "auto", timeout: int = 20):
    """
    Run Semgrep in parallel for one or multiple paths.
    Returns:
      - (hits, rules) for single path
      - {path: (hits, rules)} dict for multiple paths
    """
    if isinstance(paths, str):
        # Single path mode
        _, hits, rules = _scan_file(paths, config=config, timeout=timeout)
        return hits, rules

    results = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_scan_file, p, config, timeout): p for p in paths}
        for f in as_completed(futs):
            p, hits, rules = f.result()
            results[p] = (hits, rules)
    return results


# ----------------------------------------------------------------------
# 🧠 Caching-friendly wrapper for builder (drop-in replacement)
# ----------------------------------------------------------------------
def get_semgrep_hits(path: str, semgrep_cache: dict = None):
    """
    Cached interface compatible with builder.
    Returns (hits, rules) and updates cache if provided.
    """
    if not os.path.exists(path):
        return 0, []

    if semgrep_cache is not None and path in semgrep_cache:
        val = semgrep_cache[path]
        if isinstance(val, int):
            return val, []
        return val

    try:
        hits, rules = run_semgrep_parallel(path, workers=4)
    except Exception as e:
        print(f"⚠️ Semgrep failed on {path}: {e}")
        hits, rules = 0, []

    if semgrep_cache is not None:
        semgrep_cache[path] = (hits, rules)
    return hits, rules
