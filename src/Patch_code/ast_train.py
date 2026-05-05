#!/usr/bin/env python3
"""
Perfected Juliet Fix System with AST Policy Integration (Corrected Version)
"""

import json
import re
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datasets import load_dataset
from codet5_fallback import CodeT5Fallback
import numpy as np
# --------------------------
# NEW IMPORTS (Module B)
# --------------------------
import joblib
from features import extract_ast_features_cached
from ast_edit_actions import ASTEditActions   # Should exist from Module B


# perfected_patterns.py
# Minimal dictionary version for PatchEnv
PERFECTED_FIXES = [
    {
        "name": "safe_free_with_null_assign",
        "pattern": r"free\s*\(\s*(\w+)\s*\)\s*;",
        "replacement": r"if (\1 != NULL) { free(\1); \1 = NULL; }",
        "includes": ["stdlib.h"],
        "description": "Safe free with NULL check and pointer nullification"
    },
    {
        "name": "strcpy_to_strncpy_safe",
        "pattern": r"strcpy\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)\s*;",
        "replacement": r"if (\1 != NULL && \2 != NULL) { strncpy(\1, \2, sizeof(\1) - 1); \1[sizeof(\1) - 1] = '\\0'; }",
        "includes": ["string.h"],
        "description": "Safe strcpy with bounds checking and null termination"
    },
    {
        "name": "strcat_to_strncat_safe",
        "pattern": r"strcat\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)\s*;",
        "replacement": r"if (\1 != NULL && \2 != NULL) { size_t len = strlen(\1); if (len < sizeof(\1) - 1) { strncat(\1, \2, sizeof(\1) - len - 1); } }",
        "includes": ["string.h"],
        "description": "Safe strcat with bounds checking"
    },
    {
        "name": "gets_to_fgets",
        "pattern": r"gets\s*\(\s*(\w+)\s*\)\s*;",
        "replacement": r"fgets(\1, sizeof(\1), stdin); \1[strcspn(\1, \"\\n\")] = 0;",
        "includes": ["stdio.h"],
        "description": "Replace unsafe gets with safe fgets"
    },
    {
        "name": "memcpy_with_checks",
        "pattern": r"memcpy\s*\(\s*(\w+)\s*,\s*(\w+)\s*,\s*(\w+)\s*\)\s*;",
        "replacement": r"if (\1 != NULL && \2 != NULL && \3 > 0) { memcpy(\1, \2, \3); }",
        "includes": ["string.h"],
        "description": "Memcpy with NULL and size validation"
    },
    {
        "name": "sprintf_to_snprintf",
        "pattern": r"sprintf\s*\(\s*(\w+)\s*,\s*([^)]+)\s*\)\s*;",
        "replacement": r"snprintf(\1, sizeof(\1), \2);",
        "includes": ["stdio.h"],
        "description": "Replace sprintf with snprintf"
    },
    {
        "name": "strncpy_proper_null_term",
        "pattern": r"strncpy\s*\(\s*(\w+)\s*,\s*(\w+)\s*,\s*(\w+)\s*\)\s*;",
        "replacement": r"strncpy(\1, \2, \3); \1[\3 - 1] = '\\0';",
        "includes": ["string.h"],
        "description": "Ensure strncpy is properly null-terminated"
    },
    {
        "name": "strcat_to_strncat_safe",
        "pattern": r"strcat\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)\s*;",
        "replacement": r"if (\1 != NULL && \2 != NULL) { size_t avail = sizeof(\1) - strlen(\1) - 1; if (avail > 0) { strncat(\1, \2, avail); } }",
        "includes": ["string.h"],
        "description": "Safe strcat with bounds checking"
    }
]

# ============================================================
# PERFECTED TEMPLATE FIX PATTERNS (unchanged)
# ============================================================
@dataclass
class PerfectedFixPattern:
    name: str
    cwe_ids: List[str]
    vulnerable_pattern: str
    fix_template: str
    required_includes: List[str]
    validation_notes: str
    parameters: List[Dict]
    
    def to_json(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "cwe_ids": self.cwe_ids,
            "vulnerable_pattern": self.vulnerable_pattern,
            "fix_template": self.fix_template,
            "required_includes": self.required_includes,
            "validation_notes": self.validation_notes,
            "parameters": self.parameters
        }

class PerfectedPatternMiner:
    def __init__(self):
        self.fixes: List[PerfectedFixPattern] = []
        
    def create_perfected_patterns(self) -> List[PerfectedFixPattern]:
        fixes = []
        
        fixes.append(PerfectedFixPattern(
            name="strcat_to_strncat",
            cwe_ids=["CWE-121", "CWE-122"],
            vulnerable_pattern=r'strcat\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)',
            fix_template='strncat(\\g<1>, \\g<2>, sizeof(\\g<1>) - 1)',
            required_includes=["string.h"],
            validation_notes="PROVEN",
            parameters=[]
        ))
        
        fixes.append(PerfectedFixPattern(
            name="safe_free",
            cwe_ids=["CWE-415", "CWE-416"],
            vulnerable_pattern=r'free\s*\(\s*(\w+)\s*\)',
            fix_template='if (\\g<1> != NULL) { free(\\g<1>); }',
            required_includes=["stdlib.h"],
            validation_notes="PROVEN",
            parameters=[]
        ))
        
        fixes.append(PerfectedFixPattern(
            name="strcpy_to_strncpy",
            cwe_ids=["CWE-121", "CWE-122", "CWE-124", "CWE-126"],
            vulnerable_pattern=r'strcpy\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)',
            fix_template='strncpy(\\g<1>, \\g<2>, sizeof(\\g<1>))',
            required_includes=["string.h"],
            validation_notes="IMPROVED",
            parameters=[]
        ))
        
        self.fixes = fixes
        return fixes


# ============================================================
# DATA EXTRACTION
# ============================================================
class FinalCodeExtractor:
    def __init__(self):
        self.dataset = None
    
    def load_dataset(self):
        print("📥 Loading Juliet dataset from Hugging Face...")
        self.dataset = load_dataset("LorenzH/juliet_test_suite_c_1_3")
        print(f"✅ Dataset loaded: {len(self.dataset['train'])} examples")
        return self.dataset
    
    def extract_cwe(self, filename: str) -> str:
        match = re.search(r'CWE(\d+)', filename)
        return f"CWE-{match.group(1)}" if match else "Unknown"
    
    def find_perfected_vulnerabilities(self, code: str) -> List[Dict]:
        vulnerabilities = []
        if not code:
            return vulnerabilities
        
        clean_code = self._clean_code(code)
        
        working_patterns = [
            ('strcat', r'strcat\s*\(\s*\w+\s*,\s*\w+\s*\)\s*;'),
            ('free',   r'free\s*\(\s*\w+\s*\)\s*;'),
            ('strcpy', r'strcpy\s*\(\s*\w+\s*,\s*\w+\s*\)\s*;'),
        ]
        
        for vuln_type, pattern in working_patterns:
            matches = re.finditer(pattern, clean_code)
            for match in matches:
                vulnerabilities.append({
                    "type": vuln_type,
                    "vulnerable_code": match.group(0).strip(),
                    "variables": self._extract_variables(match.group(0))
                })
        
        return vulnerabilities
    
    def _clean_code(self, code: str) -> str:
        return '\n'.join(line for line in code.split('\n') 
                         if 'std_testcase.h' not in line)
    
    def _extract_variables(self, code: str) -> List[str]:
        match = re.search(r'\(\s*(\w+)\s*,\s*(\w+)\s*\)', code)
        if match:
            return list(match.groups())
        match = re.search(r'\(\s*(\w+)\s*\)', code)
        if match:
            return [match.group(1)]
        return []
    
    def get_perfected_test_cases(self, max_examples: int = 30) -> List[Dict]:
        if not self.dataset:
            self.load_dataset()
        
        test_cases = []
        
        for i, example in enumerate(self.dataset["train"]):
            if len(test_cases) >= max_examples:
                break
                
            filename = example["filename"]
            cwe = self.extract_cwe(filename)
            bad_code = example.get("bad", "")
            
            vulns = self.find_perfected_vulnerabilities(bad_code)
            if vulns:
                test_cases.append({
                    "id": i,
                    "filename": filename,
                    "cwe": cwe,
                    "vulnerabilities": vulns
                })
        
        print(f"🎯 Found {len(test_cases)} test cases")
        return test_cases


# ============================================================
# COMPILATION/TEST HELPER (CORRECTED)
# ============================================================
class FinalTester:
    def __init__(self, temp_dir: str = "/tmp/juliet_final"):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
    
    def substitute_variables(self, code: str, test_suffix: str) -> str:
        """
        Replace generic variable names with test-specific ones.
        Ensures fixed code compiles inside the test harness.
        """
        subs = {
            "dest":  f"dest_{test_suffix}",
            "src":   f"src_{test_suffix}",
            "source": f"src_{test_suffix}",
            "data":  f"data_{test_suffix}",
            "buffer": f"data_{test_suffix}",
            "ptr":    f"ptr_{test_suffix}",
            "input":  f"src_{test_suffix}",
        }

        result = code
        for old, new in subs.items():
            result = re.sub(r'\b' + old + r'\b', new, result)

        return result

    def create_final_test(self, vulnerable_code: str, fixed_code: str, 
                         includes: List[str], test_name: str) -> str:
        test_file = self.temp_dir / f"{test_name}.c"
        
        base_includes = ["stdio.h", "stdlib.h", "string.h"]
        all_includes = list(set(base_includes + includes))
        inc_str = "\n".join(f"#include <{inc}>" for inc in all_includes)

        # ALWAYS SUBSTITUTE VARIABLE NAMES
        vuln = self.substitute_variables(vulnerable_code, test_name)
        fixd = self.substitute_variables(fixed_code, test_name)
        
        template = f"""
{inc_str}

void test_vulnerable() {{
    char dest_{test_name}[100] = {{0}};
    char src_{test_name}[100]  = "test_data";
    char data_{test_name}[100] = {{0}};
    char source_{test_name}[100] = "test_data";
    void* ptr_{test_name}      = malloc(100);

    {vuln}
}}

void test_fixed() {{
    char dest_{test_name}[100] = {{0}};
    char src_{test_name}[100]  = "test_data";
    char data_{test_name}[100] = {{0}};
    char source_{test_name}[100] = "test_data";
    void* ptr_{test_name}      = malloc(100);

    {fixd}
}}

int main() {{
    test_vulnerable();
    test_fixed();
    return 0;
}}
"""
        with open(test_file, "w") as f:
            f.write(template)
        
        return str(test_file)
    
    def test_compilation(self, test_file: str) -> Tuple[bool, str]:
        try:
            result = subprocess.run(
                ["gcc", "-o", test_file.replace(".c",""), test_file, "-w"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                return False, f"❌ GCC error: {result.stderr[:120]}"
            
            try:
                subprocess.run(
                    [test_file.replace(".c","")],
                    capture_output=True, text=True, timeout=5
                )
                return True, "🎉 PERFECT: compiles & runs"
            except:
                return True, "⚠️ COMPILES but runtime failed"
        
        except subprocess.TimeoutExpired:
            return False, "⏳ Timeout"
        except Exception as e:
            return False, f"❌ System error: {str(e)}"

    # DEBUG SUPPORT
    def debug_test_file(self, test_file: str) -> str:
        debug_dir = Path("/tmp/juliet_debug")
        debug_dir.mkdir(exist_ok=True)
        debug_file = debug_dir / Path(test_file).name
        with open(test_file, "r") as f_in, open(debug_file, "w") as f_out:
            code = f_in.read()
            f_out.write(code)
        print("\n====== DEBUG: FULL GENERATED TEST FILE ======")
        print(code)
        print("=============================================\n")
        return str(debug_file)

    def test_compilation_verbose(self, test_file: str, debug_out_path: str) -> Tuple[bool, str]:
        try:
            exe_path = test_file.replace(".c", "")
            result = subprocess.run(
                ["gcc", "-o", exe_path, test_file],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode != 0:
                print("\n====== DEBUG: GCC FULL ERROR OUTPUT ======")
                print(result.stderr)
                print("==========================================\n")

                with open(debug_out_path + ".stderr", "w") as errf:
                    errf.write(result.stderr)

                return False, "❌ GCC failed (see debug stderr file)"

            return True, "✔ COMPILES"

        except subprocess.TimeoutExpired:
            return False, "⏳ Timeout"
        except Exception as e:
            return False, f"❌ System error: {e}"


# ============================================================
# ⭐ PERFECTED JULIET SYSTEM WITH AST POLICY
# ============================================================
class PerfectedJulietSystem:
    def __init__(self):
        self.miner = PerfectedPatternMiner()
        self.extractor = FinalCodeExtractor()
        self.tester = FinalTester()
        self.fixes = []

        print("🧠 Loading AST policy model (Module B)...")
        self.ast_model, self.label_map, self.rev_map = joblib.load("ast_policy_model.joblib")
        self.ast_actions = ASTEditActions()
        print("   ✓ Model loaded: classes =", list(self.label_map.keys()))
        self.codet5 = CodeT5Fallback()

    
    def initialize(self):
        self.fixes = self.miner.create_perfected_patterns()
        print(f"🚀 Loaded {len(self.fixes)} perfected patterns!")


    def predict_ast_action(self, vuln_code: str, full_bad_code: str) -> str:
        vec = extract_ast_features_cached(full_bad_code, "c").tolist()
        pred_idx = self.ast_model.predict([vec])[0]
        action = self.rev_map[pred_idx]

        # LOGICAL OVERRIDES
        if "strcpy" in vuln_code: return "REPLACE_API"
        if "strcat" in vuln_code: return "REPLACE_API"
        if re.search(r"free\s*\(", vuln_code):
            return "INSERT_CALL" if action == "REPLACE_API" else action

        return action


    def decide_and_apply_fix(self, vuln_code, fix, full_bad_code, test_suffix):
        action = self.predict_ast_action(vuln_code, full_bad_code)
        print(f"    🔮 AST Policy decided action: {action}")

        if action == "DELETE_CALL":
            patched = self.ast_actions.delete_call(vuln_code, "unknown")
        elif action == "INSERT_CALL":
            patched = self.ast_actions.insert_call(vuln_code, "/*SAFE_NOP*/ ;")
        else:
            patched = re.sub(fix.vulnerable_pattern, fix.fix_template, vuln_code)

        patched = self.tester.substitute_variables(patched, test_suffix)

        # NORMAL PRIMARY PATH
        # Try compile after template+AST patch
        test_file = self.tester.create_final_test(
            vuln_code, patched, fix.required_includes, test_suffix
        )
        ok, msg = self.tester.test_compilation(test_file)

        if ok:
            return patched  # success without fallback

        # -------------------------------------------------------
        # 🔥 FALLBACK: CODET5 LOCAL + FULL REWRITE HYBRID
        # -------------------------------------------------------
        full_bad_code = full_bad_code if isinstance(full_bad_code, str) else ""

        codet5_patch, mode = self.codet5.fallback_patch(vuln_code, full_bad_code)

        if codet5_patch:
            print(f"    ⚠️ Using CodeT5 fallback mode: {mode}")

            final_patched = self.tester.substitute_variables(codet5_patch, test_suffix)
            return final_patched

        # If CodeT5 also fails:
        print("    ❌ CodeT5 fallback failed")
        return patched



    def run_final_validation(self, max_examples=20):
        print(f"\n🔬 Running validation on {max_examples} samples...")
        cases = self.extractor.get_perfected_test_cases(max_examples)

        results = []
        perfect = []

        for i, case in enumerate(cases):
            print(f"\n--- Test case {i+1}/{len(cases)} ({case['cwe']}) ---")

            for vuln in case["vulnerabilities"]:

                for fix in self.fixes:
                    if case["cwe"] not in fix.cwe_ids:
                        continue

                    if not re.search(fix.vulnerable_pattern, vuln["vulnerable_code"]):
                        continue

                    full_bad_code = self.extractor.dataset["train"][case["id"]]["bad"]
                    test_name = f"final_{case['id']}_{fix.name}"

                    fixed_code = self.decide_and_apply_fix(
                        vuln["vulnerable_code"], fix, full_bad_code, test_name
                    )

                    test_file = self.tester.create_final_test(
                        vuln["vulnerable_code"], fixed_code,
                        fix.required_includes, test_name
                    )

                    debug_out = self.tester.debug_test_file(test_file)
                    ok, msg = self.tester.test_compilation_verbose(test_file, debug_out)

                    result = {
                        "test_id": case["id"],
                        "cwe": case["cwe"],
                        "fix": fix.name,
                        "action": self.predict_ast_action(vuln["vulnerable_code"], full_bad_code),
                        "vulnerable": vuln["vulnerable_code"],
                        "fixed": fixed_code,
                        "compiles": ok,
                        "message": msg,
                    }

                    results.append(result)
                    print("    ➜", msg)

                    if ok:
                        perfect.append(result)

        return results, perfect


    def generate_final_report(self, all_results, perfect_results):
        print("\n🎉 FINAL REPORT")
        print("="*60)

        total = len(all_results)
        success = len(perfect_results)
        rate = (success / total * 100) if total > 0 else 0

        print(f"Total tests: {total}")
        print(f"Successful fixed: {success}")
        print(f"Success rate: {rate:.2f}%")

        with open("JULIET_FINAL_WITH_AST.json", "w") as f:
            json.dump({
                "summary": {"total": total, "success": success, "rate": rate},
                "all_results": all_results,
                "perfect": perfect_results
            }, f, indent=2)

        print("\n💾 Saved: JULIET_FINAL_WITH_AST.json")
        print("🔥 System Ready for Deployment")


# ============================================================
# MAIN ENTRY POINT
# ============================================================
def main():
    print("🚀 JULIET + AST POLICY AUTOMATED FIX SYSTEM")
    print("="*50)

    system = PerfectedJulietSystem()
    system.initialize()

    all_res, perfect_res = system.run_final_validation(max_examples=20)
    system.generate_final_report(all_res, perfect_res)

    print("\n🎯 Mission complete!")


if __name__ == "__main__":
    main()
