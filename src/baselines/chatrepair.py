import torch
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class ChatRepairAgent:
    """
    Implements a ChatRepair-style baseline (Xia et al.)
    using conversational feedback loops.
    """
    def __init__(self, model_dir="models/finetuned_cve_codet5", max_turns=5):
        self.max_turns = max_turns
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  [ChatRepair] Loading model from {model_dir} on {self.device}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
            self.model = self.model.to(self.device)
            self.model.eval()
            print("  [ChatRepair] Model loaded successfully.")
        except Exception as e:
            print(f"  [ChatRepair] Error loading model: {e}")
            self.model = None

    def repair(self, env, buggy_code, vuln_type):
        """
        Runs the conversational repair loop for a single bug.
        """
        history = []
        current_code = buggy_code
        
        print(f"\n  [ChatRepair] Starting repair for: {vuln_type}")
        
        for turn in range(1, self.max_turns + 1):
            # 1. Construct conversational prompt
            prompt = self._build_prompt(buggy_code, vuln_type, history)
            
            # 2. Generate patch
            # Note: Using same params as CodeT5 baseline for fairness
            patch = self._generate(prompt)
            if not patch:
                print(f"    Turn {turn}: Model failed to generate patch.")
                break
                
            # 3. Validate with Oracle Feedback
            # Using the new method we added to patch_env
            success, feedback = env.compile_with_feedback(patch)
            
            # Record history for next turn
            history.append({
                "patch": patch,
                "feedback": feedback
            })
            
            print(f"    Turn {turn}: Patch generated -> {feedback}")
            
            if success:
                print(f"    [ChatRepair] Success found in turn {turn}!")
                return patch, turn, history
                
        print(f"    [ChatRepair] Failed to find success within {self.max_turns} turns.")
        return None, self.max_turns, history

    def _build_prompt(self, code, vuln_type, history):
        """
        Constructs the conversational history prompt.
        """
        if not history:
            # Turn 1: Initial fix request
            return f"Fix: {code}\nType: {vuln_type}\nSafe:"
        
        # Turn N: Interactive feedback
        # Format: History of failures -> Feedback -> Request for new fix
        prompt = f"Fix: {code}\nType: {vuln_type}\n"
        for i, entry in enumerate(history):
            prompt += f"Attempt {i+1}: {entry['patch']}\n"
            prompt += f"Feedback: {entry['feedback']}\n"
        
        prompt += "Safe:"
        return prompt

    def _generate(self, prompt):
        if not self.model: return None
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=128,
                    num_beams=1,
                    temperature=0.7,
                    do_sample=True
                )
            
            raw = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract Safe portion if present
            if "Safe:" in raw:
                raw = raw.split("Safe:", 1)[1].strip()
            
            # Basic cleanup (matching original CodeT5 baseline)
            raw = re.sub(r'\s+', ' ', raw).strip()
            return raw
        except Exception as e:
            print(f"    [ChatRepair Error] Generation failed: {e}")
            return None

if __name__ == "__main__":
    # Small smoke test if run directly
    import os
    import sys
    # Add parent dir to path to import PatchEnv if needed
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # We won't run a full loop here without the Env, but we test the prompt building
    agent = ChatRepairAgent()
    p = agent._build_prompt("strcpy(d,s);", "strcpy", [{"patch": "strncpy(d,s,10);", "feedback": "Error: Missing semicolon"}])
    print(f"Prompt Test:\n{p}")
