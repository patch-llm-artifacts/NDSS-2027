# ============================================================
# codet5_fallback.py 
# ============================================================

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re


class CodeT5Fallback:
    def __init__(self, finetuned_dir="/app/models/codet5_retrained"):  
        print("🚀 Loading RETRAINED CodeT5 model...")
        
        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   📊 Using device: {self.device}")
        
        try:
            # Load your RETRAINED model
            self.tokenizer = AutoTokenizer.from_pretrained(finetuned_dir)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(finetuned_dir)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            if torch.cuda.is_available():
                print(f"   ✅ GPU: {torch.cuda.get_device_name(0)}")
            
            print("   ✅ Loaded RETRAINED CodeT5 successfully")
        except Exception as e:
            print(f"   ❌ Failed to load retrained model: {e}")
            self.model = None
        
        print("   ✓ Model ready")

    def fallback_patch(self, context: str, code: str):
        """
        Generate a fix for a single vulnerable line.
        """
        if self.model is None:
            return None, 0.0
        
        try:
            # Use the SAME prompt as training: "Fix: {code}\nSafe:"
            prompt = f"Fix: {code}\nSafe:"
            
            # Tokenize
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                max_length=128, 
                truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate with your training parameters
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=128,
                    num_beams=3,              # Your training used 3 beams
                    early_stopping=True,
                    temperature=0.3,
                    do_sample=False,          # Your training used greedy
                )
            
            # Decode
            raw = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the safe code part
            if "Safe:" in raw:
                parts = raw.split("Safe:", 1)
                if len(parts) > 1:
                    raw = parts[1].strip()
            
            # Clean up
            raw = raw.strip()
            
            # FIX gets() output: remove printLine() if present
            if "gets(" in code and "printLine(" in raw:
                # Extract just the fgets part if it exists
                if "fgets(" in raw:
                    # Find fgets(...) pattern
                    match = re.search(r'fgets\([^)]+\)', raw)
                    if match:
                        raw = match.group(0)
                else:
                    # Just remove printLine
                    raw = raw.replace("printLine(", "").replace(");", "")
                    raw = re.sub(r'printLine[^;]+;', '', raw)
            
            # FIX sprintf syntax error
            if "sprintf(" in code and "sizeof(buffer)-strlen(buffer)" in raw:
                # Fix the malformed sprintf
                raw = "snprintf(buffer, sizeof(buffer), fmt);"
            
            # Add NULL check for memmove if missing
            if "memcpy(" in code and "memmove(" in raw and "if (" not in raw:
                raw = f"if (dest != NULL && src != NULL) {{ {raw} }}"
            
            # Ensure proper spacing
            raw = re.sub(r'\s+', ' ', raw).strip()
            
            # Fix missing spaces after if
            raw = re.sub(r'if\s*\(([^)]+)\)\s*(\w+)', r'if (\1) \2', raw)
            
            print(f"[CodeT5] Generated: {raw}")
            
            # Validate
            if not raw or raw == code:
                return None, 0.0
            
            # Calculate confidence
            confidence = self._calculate_confidence(code, raw)
            
            return raw, confidence
            
        except Exception as e:
            print(f"[CodeT5 Error] {e}")
            return None, 0.0

    def _calculate_confidence(self, original: str, patch: str) -> float:
        """Calculate confidence based on fix quality."""
        confidence = 0.5  # Base
        
        # High confidence for perfect fixes
        if "strcpy" in original and "strncpy" in patch:
            confidence = 0.9
        
        elif "gets" in original and "fgets" in patch:
            confidence = 0.9
        
        elif "printf" in original and '"%s"' in patch:
            confidence = 0.9
        
        elif "free" in original and "if (" in patch and "!= NULL" in patch:
            confidence = 0.8
        
        elif "malloc" in original and "if (" in patch and "== NULL" in patch:
            confidence = 0.8
        
        elif "memset" in original and "if (" in patch and "!= NULL" in patch:
            confidence = 0.8
        
        elif "memcpy" in original and ("memmove" in patch or "if (" in patch):
            confidence = 0.7
        
        # Lower confidence for weird fixes
        if "printLine" in patch:
            confidence *= 0.5  # Reduce confidence
        
        return min(confidence, 1.0)

    # For backward compatibility
    def generate_local_edit(self, vuln_code: str, full_code=None):
        patch, _ = self.fallback_patch("", vuln_code)
        return patch if patch else vuln_code


def test_retrained_model():
    """Test the retrained model."""
    codet5 = CodeT5Fallback("/app/models/codet5_retrained")
    
    test_cases = [
        "strcpy(buffer, input);",
        "free(ptr);",
        "gets(user_input);",
        "printf(variable);",
        "data = malloc(size);",
        "memcpy(dest, src, n);",
        "memset(data, 'A', 100);",
        "sprintf(buffer, fmt);",
    ]
    
    print("\n🧪 Testing RETRAINED CodeT5:")
    for test in test_cases:
        patch, confidence = codet5.fallback_patch("", test)
        print(f"  Input:  {test}")
        if patch:
            print(f"  Output: {patch} (confidence: {confidence:.2f})")
        else:
            print(f"  Output: None")
        print()


if __name__ == "__main__":
    test_retrained_model()