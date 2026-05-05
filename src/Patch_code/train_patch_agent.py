# train_patch_agent_fixed.py
import numpy as np
import torch
import time
import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from patch_env import PatchEnv


DEVICE = torch.device("cpu")
TOTAL_STEPS = 200_000
N_ENVS = 1

print("\n🚀 PPO Patch Training with Progress Tracking")
print("Device:", DEVICE)
print(f"Total Steps: {TOTAL_STEPS:,}")

# ============================================================
# FIXED CALLBACK - SIMPLIFIED
# ============================================================
class TrainingProgressCallback(BaseCallback):
    def __init__(self, check_freq=1000, verbose=0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        self.best_mean_reward = -np.inf
        self.start_time = time.time()
        self.action_counts = {}  # Track actions during training
        self.episode_count = 0
        
    def _on_step(self) -> bool:
        # Track actions from info dict (if available)
        if 'info' in self.locals:
            info = self.locals['info']
            if isinstance(info, dict) and 'actual_action' in info:
                action = info['actual_action']
                self.action_counts[action] = self.action_counts.get(action, 0) + 1
        
        # Track episode rewards
        if len(self.model.ep_info_buffer) > 0:
            for ep_info in self.model.ep_info_buffer:
                if 'r' in ep_info:
                    self.episode_rewards.append(ep_info['r'])
                    self.episode_count += 1
        
        # Periodic reporting
        if self.n_calls % self.check_freq == 0:
            elapsed = time.time() - self.start_time
            steps_per_sec = self.num_timesteps / elapsed if elapsed > 0 else 0
            
            if len(self.episode_rewards) > 0:
                mean_reward = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards)
                
                print(f"\n📊 Training Progress - Step {self.num_timesteps:,}")
                print(f"   Steps/sec: {steps_per_sec:.1f}")
                print(f"   Mean reward (last {min(100, len(self.episode_rewards))} episodes): {mean_reward:.2f}")
                print(f"   Episodes completed: {self.episode_count}")
                
                # Print action statistics
                print("\n📈 Action Distribution (last {self.check_freq} steps):")
                total_actions = sum(self.action_counts.values())
                if total_actions > 0:
                    # Define action names (from your PatchEnv)
                    ACTION_NAMES = {
                        0: "TEMPLATE_FIX",
                        1: "AST_DELETE_CALL", 
                        2: "SAFE_NOP",
                        3: "CODET5_LOCAL",
                        4: "NULL_GUARD",
                        5: "STRCPY_TO_STRNCPY",
                        6: "PREVENT_DOUBLE_FREE"
                    }
                    
                    for action_id in sorted(self.action_counts.keys()):
                        count = self.action_counts[action_id]
                        percentage = (count / total_actions) * 100
                        action_name = ACTION_NAMES.get(action_id, f"ACTION_{action_id}")
                        print(f"   {action_name:20s}: {count:4d} ({percentage:5.1f}%)")
                
                # Reset action counts for next period
                self.action_counts = {}
                
                # Save best model
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.model.save("ppo_patch_best")
                    print(f"   🎉 New best model saved! (reward: {mean_reward:.2f})")
                
                # Reset rewards for next period
                self.episode_rewards = []
        
        return True

# ============================================================
# FIXED ENVIRONMENT CREATION
# ============================================================
def make_env():
    def _init():
        env = PatchEnv(use_proper_samples=True, training_mode=True)
        # IMPORTANT: Add action mask to info for PPO
        env = Monitor(env)
        return env
    return _init

# ============================================================
# CUSTOM ENVIRONMENT WRAPPER FOR ACTION MASKS
# ============================================================
class ActionMaskEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # PPO needs action mask in observation space
        # We'll add it to the info dict instead
        
    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        # Add action mask to info for PPO's action masking
        if hasattr(self.env, '_get_action_mask'):
            info['action_mask'] = self.env._get_action_mask()
        return obs, reward, done, truncated, info
    
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        if hasattr(self.env, '_get_action_mask'):
            info['action_mask'] = self.env._get_action_mask()
        return obs, info

# ============================================================
# TRAINING FUNCTION - FIXED
# ============================================================
def train_with_progress():
    print("\n🚀 Initializing environment...")
    
    # Create environment with wrapper
    def make_wrapped_env():
        def _init():
            env = PatchEnv(use_proper_samples=True, training_mode=True)
            env = ActionMaskEnvWrapper(env)
            env = Monitor(env)
            return env
        return _init
    
    env = DummyVecEnv([make_wrapped_env()])
    
    print("✅ Setting up PPO model...")
    
    # Create callback
    progress_callback = TrainingProgressCallback(check_freq=2000)
    
    # IMPORTANT: Use smaller network for faster training
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,  # Increased for faster learning
        n_steps=2048,        # Reduced for more frequent updates
        batch_size=64,       # Reduced for faster training
        n_epochs=10,         # Reduced for faster training
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,       # Lower entropy to encourage exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        device="cpu",
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 64], vf=[128, 64]),  # Smaller network
            activation_fn=torch.nn.ReLU,
            ortho_init=True,
            log_std_init=-0.5,  # Start with more exploration
        ),
    )
    
    print(f"\n🎯 Starting Training with Progress Tracking")
    print("=" * 60)
    
    start_time = time.time()
    
    # Train with callback
    model.learn(
        total_timesteps=TOTAL_STEPS,
        callback=progress_callback,
        log_interval=1,      # Log every update for better monitoring
        progress_bar=True,
        tb_log_name="ppo_patch_training"
    )
    
    training_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("💾 Saving final model: ppo_patch_final")
    model.save("ppo_patch_final")
    
    print(f"\n⏱️  Total training time: {training_time:.1f} seconds")
    print(f"   Speed: {TOTAL_STEPS/training_time:.1f} steps/second")
    print(f"   Best mean reward: {progress_callback.best_mean_reward:.2f}")
    
    return model, progress_callback

# ============================================================
# FIXED TEST FUNCTION
# ============================================================
def test_model(model, num_tests=20):
    """Test the trained model on new samples"""
    import re
    
    def is_nonsense_patch(patch):
        """Detect nonsense patches."""
        if not patch or not isinstance(patch, str):
            return True
        
        nonsense_patterns = [
            r'if\s*\(\s*NULL\s*!=\s*NULL\s*\)',
            r'if\s*\(\s*\d+\s*!=\s*NULL\s*\)',
            r'if\s*\(\s*[A-Z_][A-Z0-9_]*\s*!=\s*NULL\s*\)',
            r'if\s*\(\s*\w+\s*!=\s*NULL\)\s*{\s*/\*\s*safe\s*\*/\s*}\s*$',
        ]
        
        for pattern in nonsense_patterns:
            if re.search(pattern, patch, re.IGNORECASE):
                return True
        
        return False
    
    print(f"\n🧪 Testing model on {num_tests} new samples...")
    
    test_env = PatchEnv(use_proper_samples=True, training_mode=False)
    
    results = {
        'total_reward': 0,
        'successful_fixes': 0,
        'failed_fixes': 0,
        'nonsense_patches': 0,
        'test_cases': []
    }
    
    for i in range(num_tests):
        try:
            obs, _ = test_env.reset()
            episode_reward = 0
            done = False
            step = 0
            
            original_code = test_env.current_code
            actions_taken = []
            
            while not done and step < 3:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = test_env.step(action)
                episode_reward += reward
                done = terminated or truncated
                step += 1
                
                # Track action
                if 'actual_action' in info:
                    actions_taken.append(info['actual_action'])
            
            results['total_reward'] += episode_reward
            
            # Check for nonsense patches
            final_code = test_env.current_code
            is_nonsense = is_nonsense_patch(final_code)
            
            if is_nonsense:
                results['nonsense_patches'] += 1
            
            # Track success
            fixed = episode_reward >= 20  # <<< THIS IS THE KEY LINE

            if fixed:
                results['successful_fixes'] += 1
            elif episode_reward < 0:
                results['failed_fixes'] += 1

            # Store test case details (ADD Fix Status)
            results['test_cases'].append({
                'test_num': i + 1,
                'original_code': original_code[:100],
                'final_code': final_code[:100],
                'reward': episode_reward,
                'actions': actions_taken,
                'is_nonsense': is_nonsense,
                'Fix Status': 'Fixed' if fixed else 'Unfixed'
            })

            
            print(f"\nTest {i+1}/{num_tests}:")
            print(f"   Original: {original_code[:60]}...")
            print(f"   Actions: {actions_taken}")
            print(f"   Final reward: {episode_reward:.2f}")
            print(f"   Nonsense: {is_nonsense}")
        
        except Exception as e:
            print(f"   Error in test {i+1}: {e}")
    
    # Calculate averages
    avg_reward = results['total_reward'] / num_tests if num_tests > 0 else 0
    nonsense_rate = results['nonsense_patches'] / num_tests * 100 if num_tests > 0 else 0
    success_rate = results['successful_fixes'] / num_tests * 100 if num_tests > 0 else 0
    
    print(f"\n📊 Test Results Summary:")
    print(f"   Average reward: {avg_reward:.2f}")
    print(f"   Success rate: {success_rate:.1f}% ({results['successful_fixes']}/{num_tests})")
    print(f"   Nonsense patch rate: {nonsense_rate:.1f}% ({results['nonsense_patches']}/{num_tests})")
    print(f"   Failed fixes: {results['failed_fixes']}/{num_tests}")
    
    return results

# ============================================================
# MAIN EXECUTION - FIXED
# ============================================================
if __name__ == "__main__":
    import gymnasium as gym
    
    print("=" * 60)
    print("🚀 STARTING TRAINING SESSION")
    print("=" * 60)
    
    # Train the model
    model, callback = train_with_progress()
    
    # Test the model
    print("\n" + "=" * 60)
    print("🧪 MODEL EVALUATION")
    print("=" * 60)
    
    test_results = test_model(model, num_tests=15)
    
    # Save training summary
    with open("training_summary.txt", "w") as f:
        f.write("Training Summary\n")
        f.write("================\n")
        f.write(f"Total steps: {TOTAL_STEPS}\n")
        f.write(f"Best mean reward: {callback.best_mean_reward:.2f}\n")
        f.write(f"Test average reward: {test_results['total_reward']/15:.2f}\n")
        f.write(f"Success rate: {test_results['successful_fixes']/15*100:.1f}%\n")
        f.write(f"Nonsense patch rate: {test_results['nonsense_patches']/15*100:.1f}%\n")
    
    print("\n✅ Training completed and evaluated!")
    print("   Check 'ppo_patch_best.zip' for the best model")
    print("   Check 'training_summary.txt' for detailed results")