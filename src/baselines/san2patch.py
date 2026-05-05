import torch
import re
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class SAN2PATCHAgent:
    """
    Implements a SAN2PATCH-style baseline (USENIX Security 2025).
    Adapts the 'Tree-of-Thought' (ToT) reasoning paradigm for vulnerability repair.
    
    Original stages: Comprehend Log -> Localize Fault -> Plan Strategy -> Generate Patch.
    Our adaptation: Comprehend Vuln -> Localize Fault -> Plan Strategy -> Generate Patch.
    """
    def __init__(self, model_dir="models/finetuned_cve_codet5"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  [SAN2PATCH] Loading model from {model_dir}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
            self.model = self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"  [SAN2PATCH] Error: {e}")
            self.model = None

    def repair(self, env, buggy_code, vuln_type):
        """
        Executes the 4-stage Tree-of-Thought reasoning loop.
        """
        print(f"  [SAN2PATCH] Starting Tree-of-Thought for: {vuln_type}")
        
        # --- STAGE 1: Vulnerability Comprehension ---
        # Instead of sanitizer logs, we use the vulnerability type and code context
        comp_prompt = f"Comprehend: {buggy_code}\nVuln: {vuln_type}\nExplanation:"
        comprehension = self._generate(comp_prompt, max_len=64)
        print(f"    1. Comprehension: {comprehension[:50]}...")

        # --- STAGE 2: Fault Localization ---
        loc_prompt = f"Code: {buggy_code}\nVuln: {vuln_type}\nReason: {comprehension}\nLocalize bug line:"
        location = self._generate(loc_prompt, max_len=32)
        print(f"    2. Localization: {location[:50]}...")

        # --- STAGE 3: Fix Strategy Formulation ---
        strat_prompt = f"Vuln: {vuln_type}\nLocation: {location}\nPlan fix strategy:"
        strategy = self._generate(strat_prompt, max_len=64)
        print(f"    3. Strategy: {strategy[:50]}...")

        # --- STAGE 4: Patch Generation ---
        # Final generation conditioned on the chain of thought
        patch_prompt = (
            f"Fix: {buggy_code}\n"
            f"Reasoning: {comprehension}\n"
            f"Strategy: {strategy}\n"
            f"Safe:"
        )
        patch = self._generate(patch_prompt, max_len=128)
        
        if patch:
            success, feedback = env.compile_with_feedback(patch)
            print(f"    4. Patch Result -> {feedback}")
            return patch, 4, success # 4 stages = 4 'turns' in terms of effort
        
        return None, 4, False

    def _generate(self, prompt, max_len=128):
        if not self.model: return ""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_len,
                    num_beams=1,
                    temperature=0.7,
                    do_sample=True
                )
            raw = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract portion after prompt if needed
            if "Safe:" in raw:
                raw = raw.split("Safe:", 1)[1].strip()
            elif "Explanation:" in raw:
                raw = raw.split("Explanation:", 1)[1].strip()
            elif "strategy:" in raw:
                raw = raw.split("strategy:", 1)[1].strip()
                
            return re.sub(r'\s+', ' ', raw).strip()
        except:
            return ""
