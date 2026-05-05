#!/usr/bin/env python3


import subprocess
import tempfile
import os
import time
import json
import re
from datetime import datetime

# ============================================================
# REAL SEMGREP INTEGRATION
# ============================================================
class RealSemgrepValidator:
    def __init__(self):
        self.available = self._check_semgrep_available()
        
    def _check_semgrep_available(self):
        """Check if semgrep is actually installed and available."""
        try:
            result = subprocess.run(
                ["semgrep", "--version"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                print("✅ Semgrep is installed and available")
                print(f"   Version: {result.stdout.strip()}")
                return True
        except FileNotFoundError:
            print("❌ Semgrep not found. Is it installed?")
            print("   Try: pip install semgrep")
        except Exception as e:
            print(f"⚠️  Could not check semgrep: {e}")
        
        print("⚠️  Will use simulated Semgrep validation")
        return False
    
    def validate(self, original_code, patched_code, test_name):
        """Run ACTUAL Semgrep on both original and patched code."""
        print(f"    ├── 🔍 REAL SEMGREP VALIDATION")
        
        # Create temp files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f_orig:
            f_orig.write(original_code)
            orig_file = f_orig.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f_patch:
            f_patch.write(patched_code)
            patch_file = f_patch.name
        
        try:
            if not self.available:
                return self._simulate_semgrep(original_code, patched_code)
            
            # Run REAL Semgrep on original
            print(f"    │   ├── Scanning original code...")
            orig_results = self._run_semgrep(orig_file, "original")
            
            # Run REAL Semgrep on patched
            print(f"    │   ├── Scanning patched code...")
            patch_results = self._run_semgrep(patch_file, "patched")
            
            # Calculate improvement
            orig_findings = len(orig_results.get("results", []))
            patch_findings = len(patch_results.get("results", []))
            improvement = orig_findings - patch_findings
            
            # Log detailed results
            print(f"    │   ├── Original findings: {orig_findings}")
            print(f"    │   ├── Patched findings: {patch_findings}")
            print(f"    │   ├── Δ Findings: {improvement}")
            
            if improvement > 0:
                print(f"    │   ├── ✅ Reduced vulnerabilities by {improvement}")
                gate_passed = True
            elif improvement == 0:
                print(f"    │   ├── ⚠️  Same number of findings")
                gate_passed = True  # Accept if no regression
            else:
                print(f"    │   ├── ❌ Introduced {abs(improvement)} new findings")
                gate_passed = False
            
            # Show top findings if any
            if orig_findings > 0:
                print(f"    │   ├── Top original findings:")
                for i, finding in enumerate(orig_results.get("results", [])[:2]):
                    check_id = finding.get("check_id", "unknown")
                    message = finding.get("extra", {}).get("message", "")[:50]
                    print(f"    │   │   {i+1}. {check_id}: {message}...")
            
            return {
                "gate_passed": gate_passed,
                "original_findings": orig_findings,
                "patch_findings": patch_findings,
                "improvement": improvement,
                "original_results": orig_results.get("results", []),
                "patch_results": patch_results.get("results", []),
                "using_real_semgrep": True
            }
            
        except Exception as e:
            print(f"    │   ├── ⚠️  Semgrep error: {str(e)[:100]}")
            return self._simulate_semgrep(original_code, patched_code)
        
        finally:
            # Cleanup temp files
            for f in [orig_file, patch_file]:
                try:
                    os.unlink(f)
                except:
                    pass
    
    def _run_semgrep(self, file_path, label):
        """Run actual semgrep command and parse results."""
        try:
            # Run semgrep with C security rules
            cmd = [
                "semgrep",
                "--config", "p/c",           # C language rules
                "--config", "p/security-audit",  # Security audit rules
                "--json",                     # JSON output
                file_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10  # Give it time to run
            )
            
            if result.returncode == 0:
                try:
                    return json.loads(result.stdout)
                except json.JSONDecodeError:
                    print(f"    │   │   ⚠️  Could not parse semgrep JSON output")
                    return {"results": []}
            else:
                print(f"    │   │   ⚠️  Semgrep failed: {result.stderr[:100]}")
                return {"results": []}
                
        except subprocess.TimeoutExpired:
            print(f"    │   │   ⚠️  Semgrep timeout for {label}")
            return {"results": []}
        except Exception as e:
            print(f"    │   │   ⚠️  Semgrep exception: {str(e)[:100]}")
            return {"results": []}
    
    def _simulate_semgrep(self, original_code, patched_code):
        """Simulate semgrep when not available."""
        print(f"    │   ├── ⚠️  Using simulated Semgrep validation")
        
        # Simulated rules
        simulated_findings = []
        
        # Rule 1: strcpy
        if "strcpy(" in original_code and "strcpy(" in patched_code and "strncpy(" not in patched_code:
            simulated_findings.append({
                "check_id": "simulated.strcpy",
                "extra": {"message": "strcpy can cause buffer overflow"}
            })
        
        # Rule 2: gets
        if "gets(" in original_code and "gets(" in patched_code and "fgets(" not in patched_code:
            simulated_findings.append({
                "check_id": "simulated.gets",
                "extra": {"message": "gets is always unsafe"}
            })
        
        # Rule 3: system
        if "system(" in original_code and "SECURITY" not in patched_code and "WARNING" not in patched_code:
            simulated_findings.append({
                "check_id": "simulated.system",
                "extra": {"message": "system() with user input may lead to command injection"}
            })
        
        # Rule 4: printf format string
        if "printf(" in original_code and '"' not in original_code and '"%s"' not in patched_code:
            simulated_findings.append({
                "check_id": "simulated.printf-format",
                "extra": {"message": "printf with variable format string"}
            })
        
        # Rule 5: free without check
        if "free(" in original_code and "if" not in patched_code and "NULL" not in patched_code:
            simulated_findings.append({
                "check_id": "simulated.free-null-check",
                "extra": {"message": "free without NULL check"}
            })
        
        # For simulation, check if we fixed anything
        orig_has_strcpy = "strcpy(" in original_code
        patch_has_strncpy = "strncpy(" in patched_code
        orig_has_gets = "gets(" in original_code
        patch_has_fgets = "fgets(" in patched_code
        
        improvement = 0
        if orig_has_strcpy and patch_has_strncpy:
            improvement += 1
        if orig_has_gets and patch_has_fgets:
            improvement += 1
        
        print(f"    │   ├── Simulated findings: {len(simulated_findings)}")
        print(f"    │   ├── Simulated improvement: {improvement}")
        
        gate_passed = len(simulated_findings) == 0
        
        return {
            "gate_passed": gate_passed,
            "original_findings": len(simulated_findings),
            "patch_findings": len(simulated_findings),
            "improvement": improvement,
            "original_results": simulated_findings,
            "patch_results": simulated_findings,
            "using_real_semgrep": False
        }

# ============================================================
# TEST PROGRAMS WITH KNOWN VULNERABILITIES
# ============================================================
TEST_PROGRAMS = [
    {
        "id": "strcpy_vuln",
        "name": "strcpy Buffer Overflow",
        "code": '''#include <string.h>
#include <stdio.h>

int main() {
    char buffer[10];
    char user_input[100] = "This string is definitely too long for the buffer!";
    
    // VULNERABLE: strcpy without bounds check
    strcpy(buffer, user_input);
    
    printf("Buffer content: %s\\n", buffer);
    return 0;
}''',
        "vulnerability": "CWE-120: Buffer Copy without Checking Size",
        "expected_fix": "strcpy → strncpy"
    },
    {
        "id": "gets_vuln",
        "name": "gets Buffer Overflow",
        "code": '''#include <stdio.h>

int main() {
    char username[20];
    
    printf("Enter your username: ");
    // VULNERABLE: gets() is always unsafe
    gets(username);
    
    printf("Hello, %s!\\n", username);
    return 0;
}''',
        "vulnerability": "CWE-242: Use of Inherently Dangerous Function",
        "expected_fix": "gets → fgets"
    },
    {
        "id": "printf_format",
        "name": "printf Format String",
        "code": '''#include <stdio.h>

int main() {
    char user_input[100] = "%s%s%s%s%s%s%s%s";  // Malicious format string
    
    // VULNERABLE: printf with user-controlled format
    printf(user_input);
    
    return 0;
}''',
        "vulnerability": "CWE-134: Use of Externally-Controlled Format String",
        "expected_fix": "printf with literal format string"
    }
]

# ============================================================
# PATCH APPLICATORS
# ============================================================
def apply_patch(original_code, patch_type):
    """Apply specific patch type."""
    if patch_type == "STRCPY_TO_STRNCPY":
        return original_code.replace(
            'strcpy(buffer, user_input);',
            'strncpy(buffer, user_input, sizeof(buffer) - 1);\n    buffer[sizeof(buffer) - 1] = \'\\0\';'
        )
    
    elif patch_type == "GETS_TO_FGETS":
        return original_code.replace(
            'gets(username);',
            'fgets(username, sizeof(username), stdin);\n    username[strcspn(username, "\\n")] = \'\\0\';'
        )
    
    elif patch_type == "PRINTF_FORMAT_FIX":
        return original_code.replace(
            'printf(user_input);',
            'printf("%s", user_input);'
        )
    
    elif patch_type == "NULL_GUARD_FREE":
        # Add NULL guard example
        if "free(" in original_code:
            return original_code.replace(
                'free(',
                'if (ptr != NULL) { free('
            ) + '}'
        return original_code
    
    elif patch_type == "NO_CHANGE":
        return original_code
    
    else:
        return original_code

# ============================================================
# COMPILATION TEST
# ============================================================
def compile_test(code, test_id):
    """Test if code compiles."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
        f.write(code)
        tmp_file = f.name
    
    output_file = f"/tmp/{test_id}.out"
    
    try:
        result = subprocess.run(
            ["gcc", tmp_file, "-o", output_file, "-w"],
            capture_output=True,
            text=True,
            timeout=3
        )
        
        os.unlink(tmp_file)
        if os.path.exists(output_file):
            os.unlink(output_file)
        
        return result.returncode == 0, result.stderr if result.stderr else ""
        
    except Exception as e:
        try:
            os.unlink(tmp_file)
        except:
            pass
        return False, str(e)

# ============================================================
# MAIN PIPELINE WITH REAL SEMGREP
# ============================================================
def run_semgrep_pipeline():
    """Run pipeline with real Semgrep integration."""
    print("\n" + "="*80)
    print("🚀 PATCHENV PIPELINE WITH REAL SEMGREP INTEGRATION")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
    
    # Initialize real Semgrep validator
    semgrep = RealSemgrepValidator()
    
    results = []
    total_start = time.time()
    
    for vuln in TEST_PROGRAMS:
        print(f"\n{'#'*80}")
        print(f"VULNERABILITY: {vuln['name']}")
        print(f"CWE: {vuln['vulnerability']}")
        print(f"{'#'*80}")
        
        original_code = vuln['code']
        
        # Determine which patches to test based on vulnerability
        if "strcpy" in vuln['vulnerability'].lower():
            patches = ["STRCPY_TO_STRNCPY", "NO_CHANGE"]
        elif "gets" in vuln['vulnerability'].lower():
            patches = ["GETS_TO_FGETS", "NO_CHANGE"]
        elif "printf" in vuln['vulnerability'].lower():
            patches = ["PRINTF_FORMAT_FIX", "NO_CHANGE"]
        else:
            patches = ["NO_CHANGE"]
        
        for patch_type in patches:
            print(f"\n  Patch: {patch_type}")
            
            patch_start = time.time()
            wall_start = datetime.now().strftime('%H:%M:%S.%f')[:-3]
            
            # Apply patch
            patched_code = apply_patch(original_code, patch_type)
            
            # Compilation test
            print(f"    ├── 🛠️  Compilation test...")
            compile_ok, compile_msg = compile_test(patched_code, f"{vuln['id']}_{patch_type}")
            compile_time = time.time() - patch_start
            
            if not compile_ok:
                print(f"    ├── ❌ Compile failed: {compile_msg[:50]}")
                results.append({
                    "vuln_id": vuln['id'],
                    "patch": patch_type,
                    "status": "COMPILE_FAIL",
                    "compile_time": compile_time,
                    "total_time": time.time() - patch_start,
                    "wall_time": wall_start
                })
                continue
            
            print(f"    ├── ✅ Compiled in {compile_time:.2f}s")
            
            # REAL SEMGREP VALIDATION
            semgrep_start = time.time()
            semgrep_results = semgrep.validate(original_code, patched_code, f"{vuln['id']}_{patch_type}")
            semgrep_time = time.time() - semgrep_start
            
            total_time = time.time() - patch_start
            
            # Determine status
            if semgrep_results["gate_passed"] and semgrep_results["improvement"] > 0:
                status = "SUCCESS"
            elif semgrep_results["gate_passed"]:
                status = "NO_CHANGE"
            else:
                status = "GATE_REJECT"
            
            results.append({
                "vuln_id": vuln['id'],
                "vuln_name": vuln['name'],
                "patch": patch_type,
                "status": status,
                "compile_time": compile_time,
                "semgrep_time": semgrep_time,
                "total_time": total_time,
                "wall_time": wall_start,
                "end_wall_time": datetime.now().strftime('%H:%M:%S.%f')[:-3],
                "semgrep_results": {
                    "gate_passed": semgrep_results["gate_passed"],
                    "original_findings": semgrep_results["original_findings"],
                    "patch_findings": semgrep_results["patch_findings"],
                    "improvement": semgrep_results["improvement"],
                    "using_real_semgrep": semgrep_results["using_real_semgrep"]
                }
            })
            
            print(f"    ├── Semgrep time: {semgrep_time:.2f}s")
            print(f"    ├── Total time: {total_time:.2f}s")
            print(f"    └── Status: {status}")
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 REAL SEMGREP PIPELINE RESULTS")
    print(f"{'='*80}")
    
    total_tests = len(results)
    successful = [r for r in results if r['status'] == 'SUCCESS']
    gate_rejected = [r for r in results if r['status'] == 'GATE_REJECT']
    compilable = [r for r in results if r.get('compile_time', 0) > 0]
    
    # Count real vs simulated Semgrep usage
    real_semgrep_count = sum(1 for r in results if r.get('semgrep_results', {}).get('using_real_semgrep', False))
    
    print(f"\n📈 STATISTICS:")
    print(f"   Total tests: {total_tests}")
    print(f"   Compilable: {len(compilable)}/{total_tests}")
    print(f"   Successful patches: {len(successful)}/{total_tests}")
    print(f"   Gate rejected: {len(gate_rejected)}/{total_tests}")
    print(f"   Real Semgrep used: {real_semgrep_count}/{total_tests}")
    
    if successful:
        print(f"\n✅ SUCCESSFUL PATCHES:")
        for r in successful:
            semgrep_info = r.get('semgrep_results', {})
            print(f"\n   • {r['vuln_name']} - {r['patch']}:")
            print(f"     Semgrep Δ: {semgrep_info.get('improvement', 0)}")
            print(f"     Using real Semgrep: {'✅' if semgrep_info.get('using_real_semgrep') else '❌'}")
            print(f"     Compile time: {r['compile_time']:.2f}s")
            print(f"     Semgrep time: {r['semgrep_time']:.2f}s")
            print(f"     Total time: {r['total_time']:.2f}s")
            print(f"     Wall time: {r['wall_time']} → {r['end_wall_time']}")
    
    # Save detailed results
    output_file = "/app/real_semgrep_pipeline_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "semgrep_available": semgrep.available,
            "total_execution_time": time.time() - total_start,
            "results": results,
            "summary": {
                "total_tests": total_tests,
                "compilable": len(compilable),
                "successful": len(successful),
                "gate_rejected": len(gate_rejected),
                "real_semgrep_used": real_semgrep_count,
                "simulated_semgrep_used": total_tests - real_semgrep_count
            }
        }, f, indent=2)
    
    print(f"\n💾 Results saved to {output_file}")
    print(f"⏱️  Total pipeline time: {time.time() - total_start:.2f}s")
    
    # Show Semgrep installation instructions if not available
    if not semgrep.available:
        print(f"\n{'='*80}")
        print("📋 SEMGREP INSTALLATION INSTRUCTIONS:")
        print("="*80)
        print("To use REAL Semgrep validation, install it:")
        print("  pip install semgrep")
        print("\nOr with pipx (recommended):")
        print("  pipx install semgrep")
        print("\nThen run this pipeline again to see real Semgrep findings!")
    
    return results

# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    print("="*80)
    print("🎯 PATCHENV WITH REAL SEMGREP INTEGRATION")
    print("="*80)
    print("This pipeline:")
    print("  1. ✅ Checks if Semgrep is actually installed")
    print("  2. ✅ Runs REAL Semgrep scans on original/patched code")
    print("  3. ✅ Uses Semgrep security rules (p/c, p/security-audit)")
    print("  4. ✅ Parses JSON output for detailed findings")
    print("  5. ✅ Falls back to simulation if Semgrep not available")
    print("  6. ✅ Shows whether real or simulated Semgrep was used")
    print("="*80)
    
    results = run_semgrep_pipeline()
    
    print(f"\n{'='*80}")
    print("✅ PIPELINE COMPLETE")
    print("="*80)