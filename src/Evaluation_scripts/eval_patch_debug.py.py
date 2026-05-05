#!/usr/bin/env python3
"""
Model Evaluation Script for Patch Environment
Evaluates success rate, patch quality, and vulnerability-specific performance
WITH ACTION TRACKING
"""

import torch
import numpy as np
import json
import pandas as pd
from tqdm import tqdm
import os
import sys
from collections import Counter

# Import ACTION_NAMES from your patch environment
from patch_env import ACTION_NAMES

# Import your environment
from patch_env import PatchEnv  
class ModelEvaluator:
    def __init__(self, model_path, test_samples=139, max_steps=3):
        """
        Initialize the evaluator.
        
        Args:
            model_path: Path to trained PPO model
            test_samples: Number of samples to test
            max_steps: Maximum steps per episode
        """
        self.model_path = model_path
        self.test_samples = min(test_samples, 139)  # Max 139 proper samples
        self.max_steps = max_steps
        
        # Load the model
        self.model = self._load_model(model_path)
        
        # Create evaluation environment
        self.env = PatchEnv(
            graph_paths=None,
            use_proper_samples=True,
            training_mode=False  # EVALUATION mode
        )
        
        # Statistics
        self.results = []
        self.vuln_type_stats = {}
        
        # ACTION TRACKING ATTRIBUTES
        self.global_action_counts = {name: 0 for name in ACTION_NAMES.values()}
        self.vuln_action_counts = {}  # Track actions per vulnerability type
        self.episode_action_sequences = []  # Track sequences of actions
        self.action_effectiveness = {}  # Track reward per action
        
    def _load_model(self, model_path):
        """Load the trained PPO model."""
        from stable_baselines3 import PPO
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        print(f"📦 Loading model from {model_path}")
        model = PPO.load(model_path)
        print("✅ Model loaded successfully")
        return model
    
    def _categorize_fix(self, final_reward, final_code, original_code):
        """Categorize the fix based on reward and code analysis."""
        # Get vulnerability types
        from patch_env import SimpleVulnDetector
        detector = SimpleVulnDetector()
        original_type = detector._detect_vulnerability_type(original_code)
        
        # Remove comments for comparison
        def clean_code(code):
            import re
            code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
            code = re.sub(r'//.*', '', code)
            return code.strip()
        
        clean_original = clean_code(original_code)
        clean_final = clean_code(final_code)
        
        # Categorization
        if final_reward > 25:
            return "EXCELLENT"  # Complete fix with good practices
        elif final_reward > 10:
            if clean_original != clean_final:
                return "GOOD"  # Meaningful change but incomplete
            else:
                return "MINIMAL"  # Mostly comments
        elif final_reward > 0:
            return "WEAK"  # Minor improvement
        else:
            return "FAILED"  # No fix or negative reward
    
    def _analyze_patch_quality(self, original_code, patch_code):
        """Analyze patch quality metrics."""
        import re
        
        metrics = {
            'original_length': len(original_code),
            'patch_length': len(patch_code),
            'added_comments': 0,
            'added_validation': 0,
            'vulnerability_fixed': False
        }
        
        # Count comments added
        original_comments = len(re.findall(r'/\*|\*/|//', original_code))
        patch_comments = len(re.findall(r'/\*|\*/|//', patch_code))
        metrics['added_comments'] = patch_comments - original_comments
        
        # Check for validation added
        validation_patterns = [
            r'if\s*\(.*!=.*NULL.*\)',
            r'sizeof.*-.*1',
            r'strncpy.*sizeof',
            r'snprintf.*sizeof',
            r'fgets.*sizeof'
        ]
        
        for pattern in validation_patterns:
            if re.search(pattern, patch_code) and not re.search(pattern, original_code):
                metrics['added_validation'] += 1
        
        # Check if vulnerability appears to be fixed
        vuln_functions = ['strcpy', 'gets', 'sprintf', 'scanf', 'system']
        fixed = True
        for func in vuln_functions:
            if func in original_code.lower():
                # Check if still present in patch (not fixed)
                if func in patch_code.lower():
                    # But check if it's now safe (e.g., strcpy -> strncpy)
                    if func == 'strcpy' and 'strncpy' in patch_code:
                        fixed = True  # strcpy was replaced
                    elif func == 'gets' and 'fgets' in patch_code:
                        fixed = True  # gets was replaced
                    elif func == 'sprintf' and 'snprintf' in patch_code:
                        fixed = True  # sprintf was replaced
                    else:
                        fixed = False  # Still present and not fixed
        
        metrics['vulnerability_fixed'] = fixed
        
        return metrics
    
    def _run_semgrep_test(self, original_code, patch_code):
        """Run REAL semgrep security testing with specific C vulnerability rules."""
        import tempfile
        import subprocess
        import json
        import os
        
        semgrep_results = {
            'original_vulns': [],
            'patch_vulns': [],
            'vulns_fixed': 0,
            'vulns_introduced': 0,
            'semgrep_passed': False,
            'original_has_vulns': False,
            'patch_has_vulns': False,
            'error': None
        }
        
        try:
            # Create proper C files with context for semgrep
            def create_test_file(code, filename):
                """Create a proper C file for semgrep testing."""
                # Simple C function that includes common headers
                full_code = f"""#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void test_vulnerable_code() {{
    char buffer[100];
    char* ptr = (char*)malloc(100);
    char* user_input = "test";
    int size = 100;
    
    // The vulnerable code to test
    {code}
    
    // Cleanup to avoid memory leaks in test
    if (ptr) free(ptr);
}}
"""
                with open(filename, 'w') as f:
                    f.write(full_code)
                return filename
            
            # Create temp files
            import tempfile
            orig_file = tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False).name
            patch_file = tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False).name
            
            # Create files with context
            create_test_file(original_code, orig_file)
            create_test_file(patch_code, patch_file)
            
            try:
                # Use SPECIFIC C vulnerability rules, not 'auto'
                # Try multiple rule sets to find vulnerabilities
                rule_sets = [
                    "p/security-audit",  # Primary security rules
                    "p/cwe-top-25",      # CWE Top 25 vulnerabilities
                    "p/best-practices",  # Best practices
                ]
                
                all_original_vulns = []
                all_patch_vulns = []
                
                for rule_set in rule_sets:
                    # Test original code
                    cmd = ['semgrep', '--config', rule_set, '--json', orig_file, '--quiet']
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
                    
                    if result.returncode in [0, 1]:  # 0=success, 1=findings
                        try:
                            data = json.loads(result.stdout)
                            if 'results' in data:
                                for r in data['results']:
                                    vuln_info = {
                                        'check_id': r.get('check_id', ''),
                                        'severity': r.get('extra', {}).get('severity', ''),
                                        'message': r.get('extra', {}).get('message', ''),
                                        'line': r.get('start', {}).get('line', 0),
                                        'rule_set': rule_set
                                    }
                                    # Only add if not already found
                                    if vuln_info not in all_original_vulns:
                                        all_original_vulns.append(vuln_info)
                        except json.JSONDecodeError:
                            pass
                    
                    # Test patched code
                    cmd = ['semgrep', '--config', rule_set, '--json', patch_file, '--quiet']
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
                    
                    if result.returncode in [0, 1]:
                        try:
                            data = json.loads(result.stdout)
                            if 'results' in data:
                                for r in data['results']:
                                    vuln_info = {
                                        'check_id': r.get('check_id', ''),
                                        'severity': r.get('extra', {}).get('severity', ''),
                                        'message': r.get('extra', {}).get('message', ''),
                                        'line': r.get('start', {}).get('line', 0),
                                        'rule_set': rule_set
                                    }
                                    if vuln_info not in all_patch_vulns:
                                        all_patch_vulns.append(vuln_info)
                        except json.JSONDecodeError:
                            pass
                
                semgrep_results['original_vulns'] = all_original_vulns
                semgrep_results['patch_vulns'] = all_patch_vulns
                
                # Calculate metrics using unique check_id + line
                def get_vuln_key(vuln):
                    return f"{vuln['check_id']}:{vuln['line']}"
                
                original_keys = {get_vuln_key(v) for v in all_original_vulns}
                patch_keys = {get_vuln_key(v) for v in all_patch_vulns}
                
                semgrep_results['vulns_fixed'] = len(original_keys - patch_keys)
                semgrep_results['vulns_introduced'] = len(patch_keys - original_keys)
                semgrep_results['original_has_vulns'] = len(original_keys) > 0
                semgrep_results['patch_has_vulns'] = len(patch_keys) > 0
                
                # REALISTIC PASS CRITERIA:
                # 1. If original had vulns: must fix at least one AND not introduce new ones
                # 2. If no original vulns: just don't introduce new ones
                
                if semgrep_results['original_has_vulns']:
                    semgrep_results['semgrep_passed'] = (
                        semgrep_results['vulns_fixed'] > 0 and
                        semgrep_results['vulns_introduced'] == 0
                    )
                else:
                    semgrep_results['semgrep_passed'] = (
                        semgrep_results['vulns_introduced'] == 0
                    )
                
                # Print brief summary for first few tests
                if len(all_original_vulns) > 0 or len(all_patch_vulns) > 0:
                    print(f"\n🔍 Semgrep: {len(all_original_vulns)}→{len(all_patch_vulns)} vulns")
                
            finally:
                # Cleanup
                if os.path.exists(orig_file):
                    os.unlink(orig_file)
                if os.path.exists(patch_file):
                    os.unlink(patch_file)
                    
        except FileNotFoundError:
            semgrep_results['error'] = 'semgrep not installed. Run: pip install semgrep'
        except Exception as e:
            semgrep_results['error'] = f"Semgrep error: {str(e)}"
        
        return semgrep_results
    
    def run_evaluation(self, num_tests=None):
        """Run comprehensive evaluation."""
        print(f"🧪 Running evaluation on {num_tests} samples...")
        print("=" * 60)
        
        # Reset environment to get samples
        self.env.reset()
        
        # Reset action tracking for this evaluation run
        self.global_action_counts = {name: 0 for name in ACTION_NAMES.values()}
        self.vuln_action_counts = {}
        self.episode_action_sequences = []
        self.action_effectiveness = {}
        self.attempted_action_counts = {name: 0 for name in ACTION_NAMES.values()}  # NEW: Track attempted actions
        
        # Track per-vulnerability performance
        vuln_performance = {}
        
        for test_num in tqdm(range(num_tests), desc="Testing"):
            # Reset environment for new episode
            obs, _ = self.env.reset()
            
            # Store episode data
            episode_data = {
                'test_num': test_num + 1,
                'original_code': self.env.current_code,
                'original_vuln': self.env._detect_vulnerability_type(self.env.current_code),
                'actions': [],
                'action_names': [],
                'attempted_actions': [],  # NEW: Track what model tried to do
                'actual_actions': [],     # NEW: Track what actually happened after masking
                'rewards': [],
                'patches': [],
                'total_reward': 0,
                'steps': 0
            }
            
            done = False
            truncated = False
            step_count = 0
            
            # Run episode
            while not done and not truncated and step_count < self.max_steps:
                # Predict action using model
                action, _states = self.model.predict(obs, deterministic=True)
                
                # ============================================================
                # LET THE ENVIRONMENT HANDLE ACTION MASKING (like in training)
                # ============================================================
                obs, reward, done, truncated, info = self.env.step(action)
                
                # Get actual action that was applied (after masking)
                actual_action_int = info.get('actual_action', action)
                action_name = ACTION_NAMES.get(int(actual_action_int), f"ACTION_{actual_action_int}")
                
                # Track attempted vs actual
                attempted_action_name = ACTION_NAMES.get(int(action), f"ACTION_{int(action)}")
                episode_data['attempted_actions'].append(attempted_action_name)
                episode_data['actual_actions'].append(action_name)
                
                # Record step data
                episode_data['actions'].append(int(actual_action_int))
                episode_data['action_names'].append(action_name)
                episode_data['rewards'].append(float(reward))
                episode_data['patches'].append(info.get('patch', ''))
                episode_data['total_reward'] += reward
                episode_data['steps'] += 1
                
                # ============================================================
                # ACTION TRACKING
                # ============================================================
                # Global action counts (track ACTUAL actions used)
                self.global_action_counts[action_name] = self.global_action_counts.get(action_name, 0) + 1
                
                # Track attempted actions
                self.attempted_action_counts[attempted_action_name] = \
                    self.attempted_action_counts.get(attempted_action_name, 0) + 1
                
                # Track by vulnerability type
                vuln_type = episode_data['original_vuln']
                if vuln_type not in self.vuln_action_counts:
                    self.vuln_action_counts[vuln_type] = {}
                
                self.vuln_action_counts[vuln_type][action_name] = \
                    self.vuln_action_counts[vuln_type].get(action_name, 0) + 1
                
                # Track action effectiveness (reward per action)
                if action_name not in self.action_effectiveness:
                    self.action_effectiveness[action_name] = {
                        'total_reward': 0,
                        'count': 0,
                        'success_count': 0  # Reward > 10
                    }
                
                self.action_effectiveness[action_name]['total_reward'] += reward
                self.action_effectiveness[action_name]['count'] += 1
                if reward > 10:
                    self.action_effectiveness[action_name]['success_count'] += 1
                # ============================================================
                
                step_count += 1
            
            # Store the action sequence for this episode
            if episode_data['action_names']:
                self.episode_action_sequences.append({
                    'test_num': test_num + 1,
                    'vuln_type': episode_data['original_vuln'],
                    'attempted_actions': episode_data['attempted_actions'].copy(),
                    'actual_actions': episode_data['actual_actions'].copy(),
                    'final_reward': episode_data['total_reward'],
                    'fix_category': None  # Will be set below
                })
            
            # Final analysis
            final_code = self.env.current_code if episode_data['patches'] else episode_data['original_code']
            episode_data['final_code'] = final_code
            episode_data['final_vuln'] = self.env._detect_vulnerability_type(final_code)
            episode_data['fix_category'] = self._categorize_fix(
                episode_data['total_reward'], 
                final_code, 
                episode_data['original_code']
            )
            
            # Update action sequence with fix category
            if self.episode_action_sequences and test_num < len(self.episode_action_sequences):
                self.episode_action_sequences[test_num]['fix_category'] = episode_data['fix_category']
            
            # Analyze patch quality
            episode_data['patch_metrics'] = self._analyze_patch_quality(
                episode_data['original_code'], 
                final_code
            )
            
            # Run semgrep test (but only print details for first 5 tests)
            episode_data['semgrep_results'] = self._run_semgrep_test(
                episode_data['original_code'], 
                final_code
            )
            
            # Update vulnerability statistics
            vuln_type = episode_data['original_vuln']
            if vuln_type not in vuln_performance:
                vuln_performance[vuln_type] = {
                    'count': 0,
                    'total_reward': 0,
                    'excellent': 0,
                    'good': 0,
                    'weak': 0,
                    'failed': 0
                }
            
            vuln_performance[vuln_type]['count'] += 1
            vuln_performance[vuln_type]['total_reward'] += episode_data['total_reward']
            
            if episode_data['fix_category'] == 'EXCELLENT':
                vuln_performance[vuln_type]['excellent'] += 1
            elif episode_data['fix_category'] == 'GOOD':
                vuln_performance[vuln_type]['good'] += 1
            elif episode_data['fix_category'] == 'WEAK':
                vuln_performance[vuln_type]['weak'] += 1
            else:
                vuln_performance[vuln_type]['failed'] += 1
            
            self.results.append(episode_data)
            
            # Print detailed result for first 3 tests
            if test_num < 3:
                print(f"\nTest {test_num + 1}:")
                print(f"  Original: {episode_data['original_code'][:60]}...")
                print(f"  Vuln type: {episode_data['original_vuln']}")
                print(f"  Attempted: {episode_data['attempted_actions']}")
                print(f"  Actual: {episode_data['actual_actions']}")
                print(f"  Category: {episode_data['fix_category']}")
                print(f"  Reward: {episode_data['total_reward']:.2f}")
                
                sr = episode_data.get('semgrep_results', {})
                if 'error' not in sr:
                    print(f"  Semgrep: {sr.get('vulns_fixed', 0)} fixed, {sr.get('vulns_introduced', 0)} introduced")
        
        # Print action statistics
        self._print_action_statistics()
        
        # Calculate overall statistics
        self._calculate_statistics(vuln_performance)
        
        return self.results
    
    def _print_action_statistics(self):
        """Print detailed action usage statistics."""
        print("\n" + "=" * 60)
        print("🎯 ACTION USAGE STATISTICS")
        print("=" * 60)
        
        total_attempted = sum(self.attempted_action_counts.values())
        total_actual = sum(self.global_action_counts.values())
        
        if total_attempted == 0:
            print("No actions taken during evaluation.")
            return
        
        print(f"Total attempted actions: {total_attempted}")
        print(f"Total actual actions: {total_actual}")
        
        # Calculate blocking rate
        if total_attempted > 0:
            blocking_rate = (total_attempted - total_actual) / total_attempted * 100
            print(f"Action blocking rate: {blocking_rate:.1f}%")
        
        print("\nAttempted vs Actual Distribution:")
        print("  Attempted Actions (what model chose):")
        for action_name, count in sorted(self.attempted_action_counts.items(), 
                                        key=lambda x: x[1], reverse=True):
            if count > 0:
                percentage = (count / total_attempted) * 100
                print(f"    {action_name:20s}: {count:4d} ({percentage:5.1f}%)")
        
        print("\n  Actual Actions (after masking):")
        for action_name, count in sorted(self.global_action_counts.items(), 
                                        key=lambda x: x[1], reverse=True):
            if count > 0:
                percentage = (count / total_actual) * 100
                print(f"    {action_name:20s}: {count:4d} ({percentage:5.1f}%)")
        
        # Most common action sequences
        print(f"\nMost Common Action Sequences:")
        sequence_counts = {}
        for episode_seq in self.episode_action_sequences:
            # Use 'actual_actions' which contains the action names after masking
            if episode_seq.get('actual_actions'):
                seq_str = " → ".join(episode_seq['actual_actions'])
                sequence_counts[seq_str] = sequence_counts.get(seq_str, 0) + 1
        
        if sequence_counts:
            # Show top 5 sequences
            for seq, count in sorted(sequence_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  '{seq}': {count} times")
        else:
            print("  No action sequences recorded.")
        
        print("=" * 60)
        
    def _calculate_statistics(self, vuln_performance):
        """Calculate comprehensive statistics."""
        print("\n" + "=" * 60)
        print("📊 EVALUATION RESULTS SUMMARY")
        print("=" * 60)
        
        # Overall statistics
        total_tests = len(self.results)
        total_reward = sum(r['total_reward'] for r in self.results)
        avg_reward = total_reward / total_tests if total_tests > 0 else 0
        
        # Count by category
        categories = {}
        for r in self.results:
            cat = r['fix_category']
            categories[cat] = categories.get(cat, 0) + 1
        
        # Success rate (EXCELLENT + GOOD)
        success_count = categories.get('EXCELLENT', 0) + categories.get('GOOD', 0)
        success_rate = (success_count / total_tests) * 100
        
        # Compilation success rate
        compile_success = sum(1 for r in self.results if r['total_reward'] > 0)
        compile_rate = (compile_success / total_tests) * 100
        
        print(f"\n📈 OVERALL STATISTICS:")
        print(f"   Total tests: {total_tests}")
        print(f"   Average reward: {avg_reward:.2f}")
        print(f"   Success rate (EXCELLENT+GOOD): {success_rate:.1f}% ({success_count}/{total_tests})")
        print(f"   Compilation success rate: {compile_rate:.1f}% ({compile_success}/{total_tests})")
        
        print(f"\n📊 FIX CATEGORIES:")
        for cat in ['EXCELLENT', 'GOOD', 'WEAK', 'MINIMAL', 'FAILED']:
            count = categories.get(cat, 0)
            percentage = (count / total_tests) * 100 if total_tests > 0 else 0
            print(f"   {cat}: {count} ({percentage:.1f}%)")
        
        print(f"\n🔧 VULNERABILITY-SPECIFIC PERFORMANCE:")
        for vuln_type, stats in vuln_performance.items():
            if stats['count'] > 0:
                avg_reward = stats['total_reward'] / stats['count']
                success_rate = ((stats['excellent'] + stats['good']) / stats['count']) * 100
                print(f"   {vuln_type.upper():15s}: {stats['count']:3d} tests | "
                      f"Avg reward: {avg_reward:6.2f} | "
                      f"Success: {success_rate:5.1f}% | "
                      f"[E:{stats['excellent']} G:{stats['good']} W:{stats['weak']} F:{stats['failed']}]")
        
        # Action distribution from Counter (already in action statistics)
        all_actions = []
        for r in self.results:
            all_actions.extend(r['actions'])
        
        if all_actions:
            action_counts = Counter(all_actions)
            print(f"\n🎯 ACTION DISTRIBUTION (Simple):")
            for action_num in range(7):
                count = action_counts.get(action_num, 0)
                percentage = (count / len(all_actions)) * 100 if all_actions else 0
                action_name = ACTION_NAMES.get(action_num, f"ACTION_{action_num}")
                print(f"   {action_name:20s}: {count:3d} ({percentage:.1f}%)")
        
        # REAL SEMGREP RESULTS
        semgrep_results = [r.get('semgrep_results', {}) for r in self.results]
        
        # Separate successful runs from errors
        successful_semgrep = [sr for sr in semgrep_results if sr.get('error') is None]
        failed_semgrep = [sr for sr in semgrep_results if sr.get('error') is not None]
        
        if successful_semgrep:
            total_semgrep_tests = len(successful_semgrep)
            semgrep_passed = sum(1 for sr in successful_semgrep if sr.get('semgrep_passed', False))
            
            # Detailed analysis
            original_had_vulns = sum(1 for sr in successful_semgrep if sr.get('original_has_vulns', False))
            fixes_made = sum(1 for sr in successful_semgrep 
                           if sr.get('original_has_vulns', False) and sr.get('vulns_fixed', 0) > 0)
            
            # Calculate totals
            total_vulns_fixed = sum(sr.get('vulns_fixed', 0) for sr in successful_semgrep)
            total_vulns_introduced = sum(sr.get('vulns_introduced', 0) for sr in successful_semgrep)
            total_original_vulns = sum(len(sr.get('original_vulns', [])) for sr in successful_semgrep)
            
            print(f"\n🔍 SEMGREP SECURITY TESTING:")
            print(f"   Tests completed: {total_semgrep_tests}/{total_tests}")
            print(f"   ✅ Security audit passed: {semgrep_passed}/{total_semgrep_tests} ({semgrep_passed/total_semgrep_tests*100:.1f}%)")
            
            if original_had_vulns > 0:
                print(f"\n   📊 Vulnerability Analysis:")
                print(f"      Original code had vulnerabilities: {original_had_vulns}/{total_semgrep_tests} tests")
                print(f"      Successfully fixed: {fixes_made}/{original_had_vulns} tests")
                print(f"      Total vulnerabilities found: {total_original_vulns}")
                print(f"      Total vulnerabilities fixed: {total_vulns_fixed}")
                print(f"      Total vulnerabilities introduced: {total_vulns_introduced}")
                
                if total_vulns_introduced > 0:
                    print(f"\n   ⚠️  WARNING: Patches introduced {total_vulns_introduced} NEW vulnerabilities!")
                    # Show which checks were introduced
                    introduced_checks = set()
                    for sr in successful_semgrep:
                        if sr.get('vulns_introduced', 0) > 0:
                            for vuln in sr.get('patch_vulns', []):
                                if vuln.get('check_id'):
                                    introduced_checks.add(vuln['check_id'])
                    if introduced_checks:
                        print(f"      New vulnerability types: {', '.join(list(introduced_checks)[:3])}")
                
                if fixes_made < original_had_vulns:
                    print(f"   ⚠️  WARNING: Only fixed {fixes_made}/{original_had_vulns} vulnerable tests")
            
            # Show common issues
            if total_semgrep_tests > 0 and semgrep_passed < total_semgrep_tests:
                print(f"\n   🔧 Common failure reasons:")
                
                introduced_vulns = sum(1 for sr in successful_semgrep 
                                      if not sr.get('semgrep_passed') and sr.get('vulns_introduced', 0) > 0)
                no_fixes = sum(1 for sr in successful_semgrep 
                              if not sr.get('semgrep_passed') and sr.get('original_has_vulns') 
                              and sr.get('vulns_fixed', 0) == 0)
                
                if introduced_vulns > 0:
                    print(f"      - Introduced new vulnerabilities: {introduced_vulns} tests")
                if no_fixes > 0:
                    print(f"      - Didn't fix existing vulnerabilities: {no_fixes} tests")
        
        if failed_semgrep:
            print(f"\n❌ SEMGREP ERRORS ({len(failed_semgrep)} tests):")
            for i, sr in enumerate(failed_semgrep[:2]):  # Show first 2 errors
                print(f"   {i+1}. {sr.get('error', 'Unknown error')}")
            if len(failed_semgrep) > 2:
                print(f"   ... and {len(failed_semgrep) - 2} more")
            print(f"   Fix: pip install semgrep && semgrep --version")
        
        # Patch quality metrics
        print(f"\n🔍 PATCH QUALITY METRICS:")
        avg_length_increase = sum(r['patch_metrics']['patch_length'] - r['patch_metrics']['original_length'] 
                                 for r in self.results) / total_tests
        avg_validation_added = sum(r['patch_metrics']['added_validation'] 
                                  for r in self.results) / total_tests
        vulnerabilities_fixed = sum(1 for r in self.results 
                                   if r['patch_metrics']['vulnerability_fixed'])
        
        print(f"   Avg. length increase: {avg_length_increase:.1f} chars")
        print(f"   Avg. validation checks added: {avg_validation_added:.1f}")
        print(f"   Vulnerabilities fully fixed: {vulnerabilities_fixed}/{total_tests} "
              f"({(vulnerabilities_fixed/total_tests)*100:.1f}%)")
        
        # Action-vulnerability correlation analysis
        print(f"\n🔗 ACTION-VULNERABILITY CORRELATION:")
        for vuln_type in self.vuln_action_counts:
            actions = self.vuln_action_counts.get(vuln_type, {})
            if actions:
                # Find best action for this vulnerability (highest average reward)
                vuln_results = [r for r in self.results if r['original_vuln'] == vuln_type]
                
                if vuln_results:
                    action_rewards = {}
                    action_counts = {}
                    
                    for result in vuln_results:
                        if result['actions']:
                            first_action = result['actions'][0]
                            action_name = ACTION_NAMES.get(first_action, f"ACTION_{first_action}")
                            
                            action_rewards[action_name] = action_rewards.get(action_name, 0) + result['total_reward']
                            action_counts[action_name] = action_counts.get(action_name, 0) + 1
                    
                    # Calculate averages
                    if action_rewards:
                        best_action = None
                        best_avg = -1000
                        
                        for action_name, total_reward in action_rewards.items():
                            count = action_counts[action_name]
                            avg_reward = total_reward / count
                            
                            if avg_reward > best_avg:
                                best_avg = avg_reward
                                best_action = action_name
                        
                        if best_action:
                            print(f"   {vuln_type.upper():15s}: Best action = {best_action} "
                                  f"(avg reward: {best_avg:.2f})")
    
    def save_results(self, output_dir="evaluation_results"):
        """Save evaluation results to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results as JSON
        results_dict = []
        for r in self.results:
            # Convert numpy types to Python types
            clean_result = {}
            for key, value in r.items():
                if isinstance(value, (np.integer, np.int64, np.int32)):
                    clean_result[key] = int(value)
                elif isinstance(value, (np.floating, np.float64, np.float32)):
                    clean_result[key] = float(value)
                elif isinstance(value, np.ndarray):
                    clean_result[key] = value.tolist()
                else:
                    clean_result[key] = value
            results_dict.append(clean_result)
        
        with open(os.path.join(output_dir, "detailed_results.json"), "w") as f:
            json.dump(results_dict, f, indent=2)
        
        # Save summary as CSV
        summary_data = []
        for r in self.results:
            summary_data.append({
                'test_num': r['test_num'],
                'original_vuln': r['original_vuln'],
                'final_vuln': r['final_vuln'],
                'total_reward': r['total_reward'],
                'steps': r['steps'],
                'fix_category': r['fix_category'],
                'vulnerability_fixed': r['patch_metrics']['vulnerability_fixed'],
                'added_validation': r['patch_metrics']['added_validation'],
                'semgrep_passed': r.get('semgrep_results', {}).get('semgrep_passed', False),
                'vulns_fixed': r.get('semgrep_results', {}).get('vulns_fixed', 0),
                'vulns_introduced': r.get('semgrep_results', {}).get('vulns_introduced', 0)
            })
        
        df = pd.DataFrame(summary_data)
        df.to_csv(os.path.join(output_dir, "summary.csv"), index=False)
        
        # Save action statistics
        action_stats = {
            "global_distribution": self.global_action_counts,
            "attempted_distribution": self.attempted_action_counts,
            "vulnerability_distribution": self.vuln_action_counts,
            "action_effectiveness": self.action_effectiveness,
            "total_attempted": sum(self.attempted_action_counts.values()),
            "total_actual": sum(self.global_action_counts.values())
        }
        
        with open(os.path.join(output_dir, "action_statistics.json"), "w") as f:
            json.dump(action_stats, f, indent=2)
        
        # Create CSV for action statistics
        action_data = []
        for vuln_type, actions in self.vuln_action_counts.items():
            for action_name, count in actions.items():
                action_data.append({
                    'vulnerability_type': vuln_type,
                    'action': action_name,
                    'count': count,
                    'percentage_vuln': (count / sum(actions.values())) * 100 if sum(actions.values()) > 0 else 0
                })
        
        action_df = pd.DataFrame(action_data)
        action_df.to_csv(os.path.join(output_dir, "action_distribution.csv"), index=False)
        
        # Save action sequences - FIXED VERSION
        sequences_data = []
        for seq in self.episode_action_sequences:
            # Get actions from available fields
            actions_list = []
            if 'actual_actions' in seq:
                actions_list = seq['actual_actions']
            elif 'actions' in seq:
                actions_list = [str(a) for a in seq['actions']]
            elif 'action_names' in seq:
                actions_list = seq['action_names']
            
            sequences_data.append({
                'test_num': seq['test_num'],
                'vuln_type': seq['vuln_type'],
                'actions': " → ".join(actions_list) if actions_list else "",
                'final_reward': seq.get('final_reward', 0),
                'fix_category': seq.get('fix_category', '')
            })
        
        sequences_df = pd.DataFrame(sequences_data)
        sequences_df.to_csv(os.path.join(output_dir, "action_sequences.csv"), index=False)
        
        print(f"\n💾 Results saved to '{output_dir}/'")
        print(f"   - detailed_results.json: Full evaluation data")
        print(f"   - summary.csv: Summary statistics")
        print(f"   - action_statistics.json: Action usage data")
        print(f"   - action_distribution.csv: Action distribution by vulnerability type")
        print(f"   - action_sequences.csv: Episode action sequences")

def main():
    """Main evaluation function."""
    # Configuration
    MODEL_PATH = "ppo_patch_best.zip"  # Your trained model
    NUM_TESTS = 50  # Number of samples to test (max 139)
    
    # Create evaluator
    print("🚀 Starting Model Evaluation")
    print("=" * 60)
    
    try:
        evaluator = ModelEvaluator(
            model_path=MODEL_PATH,
            test_samples=NUM_TESTS,
            max_steps=3
        )
        
        # Run evaluation
        results = evaluator.run_evaluation(num_tests=NUM_TESTS)
        
        # Save results
        evaluator.save_results()
        
        # Generate improvement recommendations
        print("\n" + "=" * 60)
        print("💡 IMPROVEMENT RECOMMENDATIONS")
        print("=" * 60)
        
        # Analyze common failure patterns
        failed_tests = [r for r in results if r['fix_category'] in ['WEAK', 'FAILED']]
        if failed_tests:
            print(f"\n🔴 {len(failed_tests)} tests need improvement:")
            
            # Group by vulnerability type
            failed_by_vuln = {}
            for r in failed_tests:
                vuln = r['original_vuln']
                failed_by_vuln[vuln] = failed_by_vuln.get(vuln, 0) + 1
            
            for vuln, count in failed_by_vuln.items():
                print(f"   - {vuln}: {count} failures")
        
        # Check action effectiveness
        excellent_tests = [r for r in results if r['fix_category'] == 'EXCELLENT']
        if excellent_tests:
            print(f"\n🟢 {len(excellent_tests)} excellent fixes:")
            
            # Find most successful actions for excellent fixes
            successful_actions = []
            for r in excellent_tests:
                if r['actions']:
                    successful_actions.append(r['actions'][0])  # First action
            
            if successful_actions:
                common_action = Counter(successful_actions).most_common(1)[0]
                action_name = ACTION_NAMES.get(common_action[0], f"ACTION_{common_action[0]}")
                print(f"   Most successful action: {action_name} ({common_action[1]} times)")
        
        # Action-specific recommendations
        if evaluator.global_action_counts:
            print(f"\n🎯 ACTION-SPECIFIC RECOMMENDATIONS:")
            
            # Check for underused actions
            total_actions = sum(evaluator.global_action_counts.values())
            avg_usage = total_actions / len(evaluator.global_action_counts)
            
            underused_actions = []
            for action_name, count in evaluator.global_action_counts.items():
                if count < avg_usage * 0.3:  # Less than 30% of average
                    underused_actions.append((action_name, count))
            
            if underused_actions:
                print(f"   Underused actions (consider training more on these):")
                for action_name, count in underused_actions:
                    print(f"   - {action_name}: {count} uses")
            
            # Check for overused ineffective actions
            for action_name, stats in evaluator.action_effectiveness.items():
                usage = evaluator.global_action_counts.get(action_name, 0)
                usage_percentage = (usage / total_actions) * 100 if total_actions > 0 else 0
                
                avg_reward = stats['total_reward'] / stats['count'] if stats['count'] > 0 else 0
                
                if usage_percentage > 20 and avg_reward < 5:  # High usage, low reward
                    print(f"   ⚠️  {action_name}: High usage ({usage_percentage:.1f}%) but "
                          f"low effectiveness (avg reward: {avg_reward:.2f})")
        
        # Semgrep effectiveness
        semgrep_tests = [r for r in results if 'semgrep_results' in r and 'error' not in r['semgrep_results']]
        if semgrep_tests:
            passed = sum(1 for r in semgrep_tests if r['semgrep_results'].get('semgrep_passed', False))
            print(f"\n🔍 Semgrep Security Audit:")
            print(f"   Passed: {passed}/{len(semgrep_tests)} ({passed/len(semgrep_tests)*100:.1f}%)")
            if passed < len(semgrep_tests):
                print(f"   🔧 Consider: Improve patches to pass security audit")
        
        # Learning progression (if enough tests)
        if len(results) > 20:
            print(f"\n📈 LEARNING PROGRESSION (first vs last 10 tests):")
            
            first_10 = results[:10]
            last_10 = results[-10:] if len(results) >= 20 else results[-10:]
            
            first_reward = sum(r['total_reward'] for r in first_10) / len(first_10)
            last_reward = sum(r['total_reward'] for r in last_10) / len(last_10)
            
            first_success = sum(1 for r in first_10 if r['fix_category'] in ['EXCELLENT', 'GOOD'])
            last_success = sum(1 for r in last_10 if r['fix_category'] in ['EXCELLENT', 'GOOD'])
            
            print(f"   Average reward: {first_reward:.1f} → {last_reward:.1f} "
                  f"(Δ: {last_reward - first_reward:+.1f})")
            print(f"   Success rate: {first_success/len(first_10)*100:.1f}% → "
                  f"{last_success/len(last_10)*100:.1f}%")
    
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please ensure your model file exists at the specified path.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
# Add this to your main() function
    def diagnostic_check():
        print("\n🔍 DIAGNOSTIC: Why is model only using 2 actions?")
        print("=" * 60)
        
        # Check if action masking is too aggressive
        print("\n1. Action Masking Analysis:")
        print("   - CODET5_LOCAL blocked for: strcat, malloc, free, memcpy, memset, wmemset, fscanf")
        print("   - STRCPY_TO_STRNCPY blocked for everything except strcpy")
        print("   - Model defaults to CODET5_LOCAL when blocked")
        
        print("\n2. Reward Analysis:")
        print("   - CODET5_LOCAL gets high rewards: 30.74 avg")
        print("   - Model doesn't need to learn other actions")
        
        print("\n3. Recommendations:")
        print("   - Reduce rewards for CODET5_LOCAL on simple vulnerabilities")
        print("   - Increase rewards for correct specialized actions")
        print("   - Remove or reduce action blocking during evaluation")
        print("   - Retrain with action diversity bonus")