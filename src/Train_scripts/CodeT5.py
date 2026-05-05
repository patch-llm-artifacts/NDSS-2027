#!/usr/bin/env python3
import json
import torch
import re
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

PAIRS_FILE = "juliet_pairs_merged.jsonl"   
MODEL_NAME = "Salesforce/codet5-base"
OUT_DIR = "/app/models/codet5_juliet_ft"  

# -------------------------------------------------------
# Extract single vulnerable lines from your dataset
# -------------------------------------------------------
def extract_single_line_pairs():
    """Extract single vulnerable lines and their fixes from full files."""
    buggy_lines = []
    fixed_lines = []
    cwe_labels = []
    
    print("📥 Loading dataset and extracting single-line pairs...")
    
    with open(PAIRS_FILE, "r") as f:
        for line_num, line in enumerate(f):
            try:
                ex = json.loads(line.strip())
                buggy_code = ex["buggy"]
                fixed_code = ex["fixed"]
                cwe = ex["cwe"]
                
                # Split into lines
                buggy_lines_list = buggy_code.split('\n')
                fixed_lines_list = fixed_code.split('\n')
                
                # Match lines - assume same number of lines
                min_lines = min(len(buggy_lines_list), len(fixed_lines_list))
                
                for i in range(min_lines):
                    bug_line = buggy_lines_list[i].strip()
                    fix_line = fixed_lines_list[i].strip()
                    
                    # Skip empty lines, comments, preprocessor directives
                    if not bug_line or len(bug_line) < 3:
                        continue
                    
                    if bug_line.startswith(('#', '//', '/*', '*/', '*')):
                        continue
                    
                    # Check if this line contains a vulnerable function
                    vulnerable_patterns = [
                        r'\bstrcpy\s*\(', r'\bfree\s*\(', r'\bmalloc\s*\(',
                        r'\bgets\s*\(', r'\bmemcpy\s*\(', r'\bsystem\s*\(',
                        r'\bprintf\s*\([^"]', r'\bsprintf\s*\(', r'\bstrcat\s*\(',
                        r'\bscanf\s*\(', r'\bfscanf\s*\(', r'\bwmemset\s*\(',
                        r'\bmemset\s*\(', r'\bstrncpy\s*\(', r'\bmemmove\s*\(',
                        r'\bcalloc\s*\(', r'\brealloc\s*\('
                    ]
                    
                    is_vulnerable = False
                    for pattern in vulnerable_patterns:
                        if re.search(pattern, bug_line, re.IGNORECASE):
                            is_vulnerable = True
                            break
                    
                    if is_vulnerable:
                        # Make sure the fix is actually different
                        if bug_line != fix_line and fix_line:
                            buggy_lines.append(bug_line)
                            fixed_lines.append(fix_line)
                            cwe_labels.append(cwe)
                
            except Exception as e:
                print(f"⚠️ Error processing line {line_num}: {e}")
                continue
    
    print(f"✅ Extracted {len(buggy_lines)} single-line vulnerable/fix pairs")
    
    # Add synthetic examples for common patterns if we don't have enough
    if len(buggy_lines) < 100:
        print(f"⚠️ Only {len(buggy_lines)} pairs found, adding synthetic examples...")
        buggy_lines, fixed_lines, cwe_labels = add_synthetic_examples(buggy_lines, fixed_lines, cwe_labels)
        print(f"✅ Now have {len(buggy_lines)} total pairs")
    
    return Dataset.from_dict({
        "buggy": buggy_lines,
        "fixed": fixed_lines,
        "cwe": cwe_labels,
    })

# -------------------------------------------------------
# Add synthetic examples for better coverage
# -------------------------------------------------------
def add_synthetic_examples(buggy_lines, fixed_lines, cwe_labels):
    """Add synthetic examples for common vulnerabilities."""
    synthetic_pairs = [
        # strcpy -> strncpy
        ("strcpy(dest, src);", "strncpy(dest, src, sizeof(dest)-1); dest[sizeof(dest)-1] = '\\0';", "CWE-120"),
        ("strcpy(buffer, input);", "if (buffer != NULL) { strncpy(buffer, input, sizeof(buffer)-1); buffer[sizeof(buffer)-1] = '\\0'; }", "CWE-120"),
        
        # free -> NULL check
        ("free(ptr);", "if (ptr != NULL) { free(ptr); ptr = NULL; }", "CWE-416"),
        ("free(data);", "if (data != NULL) { free(data); }", "CWE-416"),
        
        # malloc -> NULL check
        ("data = malloc(size);", "data = malloc(size); if (data == NULL) { /* handle error */ }", "CWE-789"),
        ("ptr = (char*)malloc(100);", "ptr = (char*)malloc(100); if (ptr == NULL) { exit(EXIT_FAILURE); }", "CWE-789"),
        
        # gets -> fgets
        ("gets(buffer);", "if (buffer != NULL) { fgets(buffer, sizeof(buffer), stdin); }", "CWE-242"),
        ("gets(input);", "fgets(input, sizeof(input), stdin);", "CWE-242"),
        
        # printf format string
        ("printf(variable);", 'printf("%s", variable);', "CWE-134"),
        ("printf(user_input);", 'printf("%s", user_input);', "CWE-134"),
        
        # sprintf -> snprintf (FIXED: escaped quotes)
        ('sprintf(buffer, fmt);', 'snprintf(buffer, sizeof(buffer), fmt);', "CWE-120"),
        ('sprintf(dest, src);', 'snprintf(dest, sizeof(dest), "%s", src);', "CWE-120"),
        
        # memcpy bounds check
        ("memcpy(dest, src, n);", "if (dest != NULL && src != NULL) { memcpy(dest, src, n); }", "CWE-120"),
        ("memcpy(a, b, size);", "if (a != NULL && b != NULL && size > 0) { memcpy(a, b, size); }", "CWE-120"),
        
        # memset bounds check
        ("memset(data, 'A', 100);", "if (data != NULL) { memset(data, 'A', 100); }", "CWE-120"),
        ("wmemset(buffer, L'A', count);", "if (buffer != NULL && count > 0) { wmemset(buffer, L'A', count); }", "CWE-120"),
        
        # scanf validation (FIXED: escaped quotes)
        ('scanf("%s", buffer);', 'if (buffer != NULL) { scanf("%99s", buffer); }', "CWE-120"),
        ('fscanf(stdin, "%d", &data);', 'if (&data != NULL) { fscanf(stdin, "%d", &data); }', "CWE-120"),
    ]
    
    for buggy, fixed, cwe in synthetic_pairs:
        buggy_lines.append(buggy)
        fixed_lines.append(fixed)
        cwe_labels.append(cwe)
    
    return buggy_lines, fixed_lines, cwe_labels

# -------------------------------------------------------
# Simple prompt for single-line fixes
# -------------------------------------------------------
def make_single_line_prompt(buggy_line, cwe=None):
    """Create prompt for single-line fixes."""
    if cwe:
        return f"Fix CWE-{cwe} vulnerability: {buggy_line}\nFixed:"
    else:
        return f"Fix this vulnerable C code: {buggy_line}\nSafe version:"

def preprocess_batch(batch, tokenizer):
    """Tokenize batch for training."""
    prompts = []
    
    for buggy, fixed, cwe in zip(batch["buggy"], batch["fixed"], batch["cwe"]):
        prompts.append(make_single_line_prompt(buggy, cwe))
    
    # Tokenize inputs (prompts)
    inputs = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=128,  # Short for single lines
    )
    
    # Tokenize targets (fixed lines)
    labels = tokenizer(
        batch["fixed"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )
    
    inputs["labels"] = labels["input_ids"]
    return inputs

# -------------------------------------------------------
# Fine-tune CodeT5 on single-line fixes
# -------------------------------------------------------
def finetune():
    print("=" * 60)
    print("🚀 FINE-TUNING CODET5 FOR SINGLE-LINE FIXES")
    print("=" * 60)
    
    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📊 Device: {device}")
    if torch.cuda.is_available():
        print(f"📊 GPU: {torch.cuda.get_device_name(0)}")
        print(f"📊 CUDA Version: {torch.version.cuda}")
        print(f"📊 Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Extract single-line pairs
    ds = extract_single_line_pairs()
    
    if len(ds) == 0:
        print("❌ No training data found!")
        return
    
    # Show samples
    print("\n📊 Dataset samples:")
    for i in range(min(3, len(ds))):
        print(f"  {i+1}. {ds[i]['buggy']} → {ds[i]['fixed']}")
    
    # Load model and tokenizer
    print("\n🔧 Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    
    # Move model to GPU if available
    model = model.to(device)
    print(f"✅ Model moved to: {next(model.parameters()).device}")
    
    # Tokenize dataset
    print("⚙️ Tokenizing dataset...")
    ds_tokenized = ds.map(
        lambda b: preprocess_batch(b, tokenizer),
        batched=True,
        remove_columns=ds.column_names,
    )
    
    # Data collator
    collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    # Training arguments
    fp16 = torch.cuda.is_available()
    print(f"📊 Using FP16: {fp16}")
    
    args = Seq2SeqTrainingArguments(
        output_dir=OUT_DIR,
        per_device_train_batch_size=16 if fp16 else 8,  # Larger if GPU available
        gradient_accumulation_steps=1,
        num_train_epochs=15,
        learning_rate=4e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        save_steps=200,
        save_total_limit=2,
        logging_steps=50,
        eval_strategy="no",
        predict_with_generate=True,
        fp16=fp16,
        bf16=False,  # Use bf16 if you have Ampere GPU (A100, etc.)
        tf32=False,  # Enable TF32 if available (A100+)
        
        # GPU optimizations
        dataloader_pin_memory=True if torch.cuda.is_available() else False,
        dataloader_num_workers=4 if torch.cuda.is_available() else 0,
        gradient_checkpointing=False,  # Enable if OOM, but slower
        
        # Optimizer
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        
        # Logging
        report_to="none",
        push_to_hub=False,
        load_best_model_at_end=False,
        metric_for_best_model=None,
        greater_is_better=None,
        
        # Additional GPU optimizations
        remove_unused_columns=True,
        label_smoothing_factor=0.0,
    )
    
    # Enable TF32 if available (for A100, etc.)
    if hasattr(torch, 'backends') and hasattr(torch.backends, 'cuda'):
        if hasattr(torch.backends.cuda, 'matmul'):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cuda, 'allow_tf32'):
            torch.backends.cuda.allow_tf32 = True
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=collator,
        args=args,
        train_dataset=ds_tokenized,
    )
    
    print("\n🎯 Starting training...")
    print(f"   Model: {MODEL_NAME}")
    print(f"   Output: {OUT_DIR}")
    print(f"   Examples: {len(ds)}")
    print(f"   Batch size: {args.per_device_train_batch_size}")
    print(f"   Epochs: {args.num_train_epochs}")
    print(f"   Learning rate: {args.learning_rate}")
    print("=" * 60)
    
    # Train
    train_result = trainer.train()
    
    # Save
    print("\n💾 Saving model...")
    trainer.save_model(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    
    print("✅ Training complete!")
    
    # Print training metrics
    print(f"\n📈 Training metrics:")
    print(f"   Loss: {train_result.training_loss:.4f}")
    print(f"   Global step: {train_result.global_step}")
    print(f"   Flops: {train_result.metrics.get('train_runtime_flops_per_second', 'N/A')}")
    
    # Test the model
    print("\n🧪 Testing model...")
    test_model_after_training(model, tokenizer, device)

def test_model_after_training(model, tokenizer, device):
    """Quick test of the trained model."""
    test_cases = [
        "strcpy(buffer, input);",
        "free(ptr);",
        "gets(user_input);",
        "printf(variable);",
        "data = malloc(size);",
        "memcpy(dest, src, n);",
    ]
    
    print("\n🧪 Model test outputs:")
    model.eval()  # Set to evaluation mode
    
    for test in test_cases:
        prompt = f"Fix this vulnerable C code: {test}\nSafe version:"
        
        inputs = tokenizer(prompt, return_tensors="pt", max_length=128, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=128,
                num_beams=3,
                early_stopping=True,
                temperature=0.7,
                do_sample=True,
            )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  Input:  {test}")
        print(f"  Output: {result}")
        print()

# -------------------------------------------------------
# Update CodeT5Fallback class to work with single-line model
# -------------------------------------------------------
def update_codet5_fallback():
    """Code to update your existing CodeT5Fallback class."""
    update_code = '''
# UPDATE YOUR CodeT5Fallback class in codet5_fallback.py:
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class CodeT5Fallback:
    def __init__(self, finetuned_dir="/app/models/codet5_juliet_ft"):
        print(f"🚀 Loading CodeT5 model from {finetuned_dir}...")
        
        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   📊 Using device: {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(finetuned_dir)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(finetuned_dir)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            if torch.cuda.is_available():
                print(f"   ✅ GPU: {torch.cuda.get_device_name(0)}")
                print(f"   ✅ Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
            
            print("   ✅ CodeT5 model loaded successfully")
        except Exception as e:
            print(f"   ❌ Failed to load CodeT5: {e}")
            self.model = None
    
    def fallback_patch(self, context, code):
        """Generate a fix for a single vulnerable line."""
        if self.model is None:
            return None, 0.0
        
        try:
            # Create a simple prompt for single-line fixes
            prompt = f"Fix this vulnerable C code: {code}\\nSafe version:"
            
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=128, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=128,
                    num_beams=5,
                    early_stopping=True,
                    temperature=0.8,
                    do_sample=True,
                )
            
            # Decode
            raw = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the code part
            if "Safe version:" in raw:
                raw = raw.split("Safe version:", 1)[1].strip()
            elif "Fixed:" in raw:
                raw = raw.split("Fixed:", 1)[1].strip()
            
            # Clean up
            raw = raw.strip()
            
            # Remove quotes if present
            if raw.startswith('"') and raw.endswith('"'):
                raw = raw[1:-1]
            
            print(f"[CodeT5] Generated: {raw}")
            
            if raw and raw != code:
                return raw, 1.0
            else:
                return None, 0.0
                
        except Exception as e:
            print(f"[CodeT5 Error] {e}")
            return None, 0.0
'''
    print("\n📝 Update your CodeT5Fallback class with this code:")
    print(update_code)

# -------------------------------------------------------
# Main
# -------------------------------------------------------
if __name__ == "__main__":
    # Train the model
    finetune()
    
    # Show how to update the existing CodeT5Fallback
    update_codet5_fallback()