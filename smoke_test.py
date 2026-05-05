import sys
import os
import io

# ------------------------------------------------------------
# FORCE UTF-8 FOR STDOUT/STDERR TO PREVENT ENCODING ERRORS
# ------------------------------------------------------------
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from unittest.mock import MagicMock
import joblib
import numpy as np

# ------------------------------------------------------------
# REVIEWER SMOKE TEST MOCK INJECTION
# ------------------------------------------------------------
# Mock tree_sitter_languages to prevent build/compilation issues on Windows/Reviewer machines
mock_tsl = MagicMock()
mock_parser = MagicMock()
mock_tsl.get_parser.return_value = mock_parser
mock_parser.parse.return_value.root_node = MagicMock()
sys.modules["tree_sitter_languages"] = mock_tsl

# Create a dummy ast_policy_model.joblib if it doesn't exist
if not os.path.exists("ast_policy_model.joblib"):
    try:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier()
        model.fit(np.zeros((1, 50)), np.array([0]))
        label_map = {"DELETE_CALL": 0}
        rev_map = {0: "DELETE_CALL"}
        joblib.dump((model, label_map, rev_map), "ast_policy_model.joblib")
        print("Created a dummy ast_policy_model.joblib for testing.")
    except Exception as e:
        print(f"Skipped creating dummy model: {e}")

def run_smoke_test():
    print("=" * 60)
    print("RUNNING SMOKE TEST FOR REVIEWERS")
    print("=" * 60)

    print("\n[Step 1/4] Checking Core Imports...")
    try:
        import torch
        import gymnasium as gym
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        import torch_geometric
        import transformers
        print("Core libraries imported successfully!")
    except ImportError as e:
        print(f"Import failed: {e}")
        print("Please check if the required dependencies are installed.")
        return False

    # Check if we can import the local PatchEnv
    print("\n[Step 2/4] Initializing Patch Environment...")
    try:
        # Add src/Patch_code and src/Dataset_builders to sys.path
        sys.path.append(os.path.join(os.getcwd(), "src", "Patch_code"))
        sys.path.append(os.path.join(os.getcwd(), "src", "Dataset_builders"))
        
        from patch_env import PatchEnv
        
        # Initialize
        env = PatchEnv(use_proper_samples=True, training_mode=True)
        print("Environment initialized successfully!")
    except Exception as e:
        print(f"Environment initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n[Step 3/4] Testing Environment Reset and Step...")
    try:
        obs, info = env.reset()
        print(f"Reset successful. Obs shape: {obs.shape}, Info: {info}")

        # Step through all actions
        for action in range(7):
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step with action {action} successful. Reward: {reward:.2f}")
    except Exception as e:
        print(f"Environment step failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n[Step 4/4] Running Quick Training Trial...")
    try:
        from stable_baselines3.common.monitor import Monitor
        
        # Custom short training
        def make_env():
            def _init():
                e = PatchEnv(use_proper_samples=True, training_mode=True)
                e = Monitor(e)
                return e
            return _init

        vec_env = DummyVecEnv([make_env()])
        
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            n_steps=64,        # Extremely small for rapid testing
            batch_size=64,       # Equal to n_steps
            n_epochs=1,          # Minimal epochs
            device="cpu"
        )
        
        print("Starting a 64-step test training...")
        model.learn(total_timesteps=64)
        print("Training completed without errors!")
        
        # Testing inference
        print("\nTesting inference...")
        obs = vec_env.reset()
        action, _ = model.predict(obs, deterministic=True)
        print(f"Prediction successful. Selected Action ID: {action[0]}")
    except Exception as e:
        print(f"Quick training trial failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("ALL SMOKE TESTS PASSED!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
