#!/usr/bin/env python3
"""
Generate eval_20cases.csv over CWEs {415, 762, 120, 121, 78}
× {regex/template baseline, CodeT5, hybrid}.

Columns (exact spec):
  compile, tests, guardrail, Δvuln, reward, seed, commit, method, original_code, patch

- compile: 1 if gcc compilation succeeds, 0 otherwise
- guardrail: 1 if SimpleVulnDetector/guardrail passes, 0 otherwise
- Δvuln: (#vulns_before - #vulns_after) from SimpleVulnDetector
- reward: PatchEnv reward (same function as RL training)
- seed: integer seed for reproducibility
"""
import csv
import random
import subprocess
import os
import time
from collections import defaultdict, Counter

import numpy as np
import torch

from patch_env import PatchEnv, extract_variables, extract_identifiers
from stable_baselines3 import PPO

# ============================================================
# CONFIG
# ============================================================
CWE_LIST = [415, 762, 120, 121, 78]
METHODS = ["regex", "codet5", "hybrid"]
N_CASES = 60  # more samples: 5 CWEs × 3 methods × 4 each
OUTPUT_CSV = "eval_20cases.csv"


def git_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def load_cwe_snippets_from_file(path="proper_vulnerable_code.txt", max_per_cwe=20):
    snippets_by_cwe = defaultdict(list)
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) < 10 or ";" not in line:
                continue
            if "strcpy" in line or "gets" in line:
                snippets_by_cwe[120].append(line)
                snippets_by_cwe[121].append(line)
            elif "free" in line:
                snippets_by_cwe[415].append(line)
                snippets_by_cwe[762].append(line)
            elif "system" in line or "shell_command" in line:
                snippets_by_cwe[78].append(line)
            elif "memcpy" in line or "memset" in line:
                snippets_by_cwe[120].append(line)
                snippets_by_cwe[121].append(line)
            elif "printf(" in line and '"' not in line:
                snippets_by_cwe[78].append(line)
    return {k: v[:max_per_cwe] for k, v in snippets_by_cwe.items()}


CWE_SNIPPETS = load_cwe_snippets_from_file()


def prepare_env_for_code(env: PatchEnv, code: str):
    _obs, _info = env.reset()
    env.current_code = code
    env.var_list = extract_variables(code) or ["buf", "src"]
    ids = extract_identifiers(code)
    env.required_vars = [v for v in ids if v in env.var_list] or ids
    env.allowed_vars = set(env.var_list) | set(env.required_vars)
    env._patch_history = []
    env._current_node_embedding = None
    env.steps = 0
    return env._get_state()


def apply_regex_template(env, code):
    vuln_type = env._detect_vulnerability_type(code)
    return env._apply_improved_template_fix(code, vuln_type)


def apply_codet5(env, code):
    vuln_type = env._detect_vulnerability_type(code)
    try:
        patch, _ = env.codet5.fallback_patch(vuln_type, code)
    except Exception:
        patch = None
    return patch or env._apply_improved_template_fix(code, vuln_type)


def apply_hybrid_rl(env, model, code, max_steps=3):
    obs = prepare_env_for_code(env, code)
    ep_reward = 0.0
    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)
        ep_reward += reward
        if done or truncated:
            break
    return env.current_code, ep_reward


def evaluate_patch(env, original, patch, override_reward=None):
    prepare_env_for_code(env, original)
    compile_ok, _ = env._full_compile(patch)
    orig_vulns = env.vuln_detector.detect_vulnerabilities(original)
    patch_vulns = env.vuln_detector.detect_vulnerabilities(patch)
    delta = sum(v['count'] for v in orig_vulns) - sum(v['count'] for v in patch_vulns)
    guard_ok = int(env.vuln_detector.is_safe(patch))
    reward = override_reward if override_reward is not None else env._reward(patch, original)[0]
    return {
        "compile": int(compile_ok),
        "tests": 0,
        "guardrail": guard_ok,
        "Δvuln": delta,
        "reward": float(reward),
    }


def print_summary(rows):
    by_method = defaultdict(list)
    for row in rows:
        by_method[row['method']].append(row)

    print("\n\n==== 📊 Summary Table by Method ====")
    print(f"{'Method':<10} | {'Compile%':<8} | {'Guard%':<8} | {'Δvuln>0%':<10} | {'Avg Reward':<10}")
    print("-" * 60)
    for method, rows in by_method.items():
        n = len(rows)
        compile_rate = 100 * sum(r['compile'] for r in rows) / n
        guard_rate = 100 * sum(r['guardrail'] for r in rows) / n
        vuln_rate = 100 * sum(1 for r in rows if r['Δvuln'] > 0) / n
        avg_reward = sum(r['reward'] for r in rows) / n
        print(f"{method:<10} | {compile_rate:>7.1f}% | {guard_rate:>7.1f}% | {vuln_rate:>9.1f}% | {avg_reward:>9.2f}")


def main():
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    env = PatchEnv(use_proper_samples=True, training_mode=False)
    model = PPO.load("ppo_patch_best", device="cpu")
    commit = git_commit_hash()

    results = []
    seed = 0
    for cwe in CWE_LIST:
        for method in METHODS:
            candidates = CWE_SNIPPETS.get(cwe, [])
            for snippet in candidates[:4]:  # 4 per method per CWE
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)

                original = snippet
                if method == "regex":
                    patch = apply_regex_template(env, original)
                    metrics = evaluate_patch(env, original, patch)
                elif method == "codet5":
                    patch = apply_codet5(env, original)
                    metrics = evaluate_patch(env, original, patch)
                elif method == "hybrid":
                    patch, reward = apply_hybrid_rl(env, model, original)
                    metrics = evaluate_patch(env, original, patch, reward)

                row = {
                    **metrics,
                    "seed": seed,
                    "commit": commit,
                    "method": method,
                    "original_code": original,
                    "patch": patch,
                }
                results.append(row)
                seed += 1

    with open(OUTPUT_CSV, "w", newline="") as f:
        fieldnames = ["compile", "tests", "guardrail", "Δvuln", "reward", "seed", "commit", "method", "original_code", "patch"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Wrote {len(results)} rows to {OUTPUT_CSV}")
    print_summary(results)


if __name__ == "__main__":
    main()
