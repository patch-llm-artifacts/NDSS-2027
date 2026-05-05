import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from patch_env import PatchEnv
from collections import defaultdict
import random
import os

# ============================================================
# CONFIG
# ============================================================
MODEL_PATH = "ppo_patch_best.zip"
NUM_EPISODES = 30
MAX_STEPS = 3
OUT_DIR = "ablation_results"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# ABLATION WRAPPER (NO PatchEnv changes)
# ============================================================
class AblationWrapper(gym.Wrapper):
    def __init__(self, env, mode="FULL", model=None):
        super().__init__(env)
        self.mode = mode
        self.model = model

    def step(self, action):
        # ---------------- NO RL ----------------
        if self.mode == "NO_RL":
            action = self.action_space.sample()

        obs, reward, terminated, truncated, info = self.env.step(action)

        # ---------------- NO TGAT ----------------
        if self.mode == "NO_TGAT":
            if "tgat_score" in info:
                reward -= info.get("tgat_score", 0.0)

        # ---------------- NO GUARDRAILS ----------------
        if self.mode == "NO_GUARDRAILS":
            # Ignore compile / semgrep failures post‑hoc
            if reward < 0:
                reward = max(reward, -1.0)

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

# ============================================================
# EVALUATION LOOP
# ============================================================
def evaluate_mode(mode):
    print(f"\n🔧 Evaluating mode: {mode}")

    env = PatchEnv(training_mode=False)
    model = PPO.load(MODEL_PATH)

    env = AblationWrapper(env, mode=mode, model=model)

    stats = defaultdict(int)
    rewards = []

    for ep in range(NUM_EPISODES):
        obs, _ = env.reset()
        ep_reward = 0
        done = False
        steps = 0

        while not done and steps < MAX_STEPS:
            if mode == "NO_RL":
                action = env.action_space.sample()
            else:
                action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            steps += 1

        rewards.append(ep_reward)

        if ep_reward > 20:
            stats["success"] += 1
        if ep_reward < 0:
            stats["failed"] += 1
        if info.get("is_nonsense", False):
            stats["nonsense"] += 1

    return {
        "mode": mode,
        "success_rate": stats["success"] / NUM_EPISODES,
        "failure_rate": stats["failed"] / NUM_EPISODES,
        "avg_reward": np.mean(rewards),
        "nonsense_rate": stats["nonsense"] / NUM_EPISODES
    }

# ============================================================
# RUN ALL MODES
# ============================================================
MODES = ["FULL", "NO_RL", "NO_TGAT", "NO_GUARDRAILS"]

results = []
for m in MODES:
    results.append(evaluate_mode(m))

df = pd.DataFrame(results)
df.to_csv(f"{OUT_DIR}/ablation_metrics.csv", index=False)
print("\n📊 Saved ablation_metrics.csv")

# ============================================================
# PLOTTING (Paper‑ready)
# ============================================================
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14
})

fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# --- Success Rate ---
axs[0].bar(df["mode"], df["success_rate"])
axs[0].set_ylabel("Patch Success Rate")
axs[0].set_ylim(0, 1)
axs[0].set_title("Patch Success Across Ablations")

# --- Avg Reward ---
axs[1].bar(df["mode"], df["avg_reward"])
axs[1].set_ylabel("Average Episode Reward")
axs[1].set_title("Reward Degradation Without Components")

# --- Nonsense Rate ---
axs[2].bar(df["mode"], df["nonsense_rate"])
axs[2].set_ylabel("Nonsense Patch Rate")
axs[2].set_ylim(0, 1)
axs[2].set_title("Guardrail Impact")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/ablation_summary.png", dpi=300)
plt.show()

print("✅ Ablation plots saved to ablation_results/")
