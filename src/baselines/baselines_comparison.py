import os
import sys
import json
import time
import numpy as np
import torch
from stable_baselines3 import PPO

# Add original source directories to path
sys.path.insert(0, os.getcwd())
root = os.path.dirname(os.getcwd())
sys.path.append(os.path.join(root, 'src', 'Patch_code'))

from patch_env import PatchEnv
from baselines.chatrepair import ChatRepairAgent
from baselines.san2patch import SAN2PATCHAgent

# CONFIGURATION
# Load test data to get size
with open('temporal_study/cvefixes_test.json', 'r') as f:
    test_data = json.load(f)
NUM_TRIALS = len(test_data)


def evaluate_ccs():
    print("="*60)
    print(" ACM CCS COMPARATIVE EVALUATION: RL vs SOTA BASELINES ")
    print("="*60)
    
    # 1. Setup Environment
    print("\n[1/3] Initializing Environment with Clean Test Set...")
    env = PatchEnv(dataset_path='temporal_study/cvefixes_test.json', training_mode=False)
    
    # 2. Load Agents
    print("\n[1/3] Loading RL Agent (PPO)...")
    rl_model = PPO.load("temporal_study/ppo_cvefixes_final")
    
    print("[2/3] Loading ChatRepair Agent (Conversational)...")
    chat_agent = ChatRepairAgent(model_dir="temporal_study/models/finetuned_cve_codet5", max_turns=5)
    
    print("[3/3] Loading SAN2PATCH Agent (Tree-of-Thought)...")
    san_agent = SAN2PATCHAgent(model_dir="temporal_study/models/finetuned_cve_codet5")
    
    results = {
        "RL_Agent": {"success": 0, "avg_steps": 0, "history": []},
        "ChatRepair": {"success": 0, "avg_steps": 0, "history": []},
        "SAN2PATCH": {"success": 0, "avg_steps": 0, "history": []}
    }
    
    # 3. Execution Loop
    for i in range(NUM_TRIALS):
        obs, info = env.reset() # Single reset per trial for fairness
        buggy_code = env.current_code 
        vuln_type = info['vuln_type']
        project = "Linux" if "linux" in info['repo_url'].lower() else "FFmpeg" if "ffmpeg" in info['repo_url'].lower() else "ImageMagick"
        
        print(f"\nTrial {i+1}/{NUM_TRIALS} | Project: {project} | Type: {vuln_type}")
        
        # --- Evaluate RL Agent ---
        obs_rl = obs 
        done = False
        steps = 0
        total_rew = 0
        while not done and steps < 3:
            action, _ = rl_model.predict(obs_rl, deterministic=True)
            obs_rl, reward, terminated, truncated, info_rl = env.step(action)
            total_rew += reward
            done = terminated or truncated
            steps += 1
        
        # SUCCESS: Use the SAME compile_with_feedback test as baselines
        # This ensures an apples-to-apples comparison
        final_patch = env.current_code
        rl_success_compile, _ = env.compile_with_feedback(final_patch)
        rl_success = rl_success_compile
        results["RL_Agent"]["success"] += 1 if rl_success else 0
        results["RL_Agent"]["avg_steps"] += steps
        results["RL_Agent"]["history"].append({"project": project, "vuln": vuln_type, "success": rl_success, "steps": steps, "reward": total_rew})
        
        # --- RESET ENV FOR BASELINES ---
        env.current_code = buggy_code
        env.steps = 0 
        
        # --- Evaluate ChatRepair ---
        patch, turns, chat_history = chat_agent.repair(env, buggy_code, vuln_type)
        success = patch is not None
        results["ChatRepair"]["success"] += 1 if success else 0
        results["ChatRepair"]["avg_steps"] += turns
        results["ChatRepair"]["history"].append({"project": project, "vuln": vuln_type, "success": success, "turns": turns, "reasoning": chat_history})
        
        # --- Evaluate SAN2PATCH ---
        patch, stages, success = san_agent.repair(env, buggy_code, vuln_type)
        results["SAN2PATCH"]["success"] += 1 if success else 0
        results["SAN2PATCH"]["avg_steps"] += stages
        results["SAN2PATCH"]["history"].append({"project": project, "vuln": vuln_type, "success": success, "stages": stages})

    # 4. Final Aggregation
    summary = {}
    for agent in results:
        results[agent]["avg_steps"] /= NUM_TRIALS
        accuracy = (results[agent]["success"] / NUM_TRIALS) * 100
        summary[agent] = {
            "Accuracy": f"{accuracy:.1f}%",
            "Avg Steps/Stages": f"{results[agent]['avg_steps']:.2f}"
        }
    
    # 5. Export Results
    with open(OUTF_JSON, "w") as f:
        json.dump(results, f, indent=2)
        
    # Generate Markdown Report
    report = "# CCS Evaluation Summary\n\n"
    report += "| Agent | Accuracy | Avg Steps/Stages |\n"
    report += "| :--- | :--- | :--- |\n"
    for agent, stats in summary.items():
        report += f"| **{agent}** | {stats['Accuracy']} | {stats['Avg Steps/Stages']} |\n"
    
    with open(OUTF_REPORT, "w") as f:
        f.write(report)
        
    print("\n" + "="*40)
    print(" EVALUATION COMPLETE ")
    print("="*40)
    for agent, stats in summary.items():
        print(f"{agent:12s}: {stats['Accuracy']} (Steps: {stats['Avg Steps/Stages']})")
    print(f"\nDetailed artifacts saved to {OUTF_JSON}")


if __name__ == "__main__":
    evaluate_ccs()
