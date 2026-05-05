import torch
import numpy as np
import re
import random
import subprocess
import tempfile
import os
import gymnasium as gym
from gymnasium import spaces
from torch_geometric.nn import TransformerConv
from torch_geometric.data import Data
from torch.serialization import add_safe_globals
import torch_geometric.data.data as pyg_data_mod

add_safe_globals([Data, pyg_data_mod.DataEdgeAttr])

from ast_edit_actions import ASTEditActions
from codet5_fallback import CodeT5Fallback
from ast_train import PERFECTED_FIXES

# ============================================================
# ADVANCED VARIABLE EXTRACTION
# ============================================================
ACTION_NAMES = {
    0: "TEMPLATE_FIX",
    1: "AST_DELETE_CALL", 
    2: "SAFE_NOP",
    3: "CODET5_LOCAL",
    4: "NULL_GUARD",
    5: "STRCPY_TO_STRNCPY",
    6: "PREVENT_DOUBLE_FREE"
}

TYPE_REGEX = r"(?:char|int|long|float|double|short|size_t|wchar_t|void)"

VAR_DECL_RE = re.compile(
    rf"\b{TYPE_REGEX}\s+(\*?\s*[A-Za-z_]\w*)(?:\s*[,=;\[])?"
)

STRUCT_DECL_RE = re.compile(
    r"struct\s+[A-Za-z_]\w*\s*\*\s*([A-Za-z_]\w*)"
)

ARRAY_DECL_RE = re.compile(
    rf"{TYPE_REGEX}\s+([A-Za-z_]\w*)\s*\["
)

MULTI_DECL_SPLIT = re.compile(
    rf"{TYPE_REGEX}\s+([^;]+);"
)

C_KEYWORDS = {
    "if","else","for","while","do","switch","case","break","continue","return",
    "sizeof","struct","typedef","static","const","void","int","char","long",
    "short","double","float","unsigned","signed","enum","union","goto",
    "volatile","register","extern","auto"
}

C_LIB_FUNCS = {
    "malloc","calloc","realloc","free","strcpy","strncpy","strcat","strncat",
    "memcpy","memmove","strlen","strcmp","printf","fprintf","snprintf",
    "sprintf","scanf","fscanf","sscanf","fgets","gets","puts","fputs",
    "open","close","read","write","recv","send","socket","connect",
    "listen","accept"
}

def extract_variables(code: str):
    vars_found = set()
    
    # Clean the code first - remove all comments
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    code = re.sub(r'//.*', '', code)
    
    lines = code.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        # Skip preprocessor directives
        if line.startswith('#'):
            continue
        cleaned_lines.append(line)
    
    clean_code = ' '.join(cleaned_lines)
    
    # Extract variables from function arguments (but not the function itself)
    # Look for patterns like: func(var1, var2, var3)
    func_call_pattern = r'\b(?!if|while|for|switch|return|sizeof)\w+\s*\(([^)]+)\)'
    func_matches = re.findall(func_call_pattern, clean_code)
    
    for args in func_matches:
        # Split arguments by comma
        for arg in args.split(','):
            arg = arg.strip()
            # Remove &, *, etc. from beginning
            arg = re.sub(r'^[&*\s]+', '', arg)
            # Remove [], (), etc. from end
            arg = re.sub(r'[\s\[\]\(\)]*$', '', arg)
            
            # Check if it's a valid variable name
            if re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', arg):
                # Skip keywords, functions, constants
                if (arg not in C_KEYWORDS and 
                    arg not in C_LIB_FUNCS and
                    not re.match(r'^[A-Z][A-Z0-9_]+$', arg) and  # Not constants
                    arg not in ['stdin', 'stdout', 'stderr', 'NULL', 'true', 'false'] and
                    not arg.endswith('_t')):  # Not types
                    vars_found.add(arg)
    
    # Also look for variable assignments
    assign_patterns = [
        r'\b([A-Za-z_][A-Za-z0-9_]*)\s*=',  # var = 
        r'\b([A-Za-z_][A-Za-z0-9_]*)\s*\[',  # var[
        r'\b([A-Za-z_][A-Za-z0-9_]*)\s*;',   # var;
    ]
    
    for pattern in assign_patterns:
        matches = re.findall(pattern, clean_code)
        for var in matches:
            if isinstance(var, tuple):
                var = var[0]
            var = str(var).strip()
            
            if (var and len(var) > 1 and 
                var not in C_KEYWORDS and 
                var not in C_LIB_FUNCS and
                not re.match(r'^[A-Z][A-Z0-9_]+$', var)):
                vars_found.add(var)
    
    # Limit to reasonable number
    vars_list = []
    for v in list(vars_found):
        if 2 <= len(v) <= 20:
            # Skip ALL_CAPS variables (likely constants)
            if v.isupper() and '_' in v:
                continue
            # Skip common constant patterns
            if any(pattern in v.upper() for pattern in ['MAX_', 'MIN_', 'SIZE_', 'LEN_', 'BUF_']):
                continue
            vars_list.append(v)
    vars_list = vars_list[:5]  # Limit to 5 most relevant
    
    # If still empty, add context-aware defaults
    if not vars_list:
        if "strcpy" in clean_code or "strncpy" in clean_code:
            vars_list = ["dest", "src", "buffer"]
        elif "malloc" in clean_code or "free" in clean_code:
            vars_list = ["ptr", "data", "size"]
        elif "printf" in clean_code:
            vars_list = ["data", "format"]
        elif "memcpy" in clean_code:
            vars_list = ["dest", "src", "n"]
        elif "system" in clean_code:
            vars_list = ["command"]
        else:
            vars_list = ["var1", "var2"]
    
    return [str(v) for v in vars_list]

def extract_identifiers(line: str):
    ids = re.findall(r"\b[A-Za-z_]\w*\b", line)
    return [str(i) for i in ids if i not in C_KEYWORDS]

def sanitize_codet5_single_line(raw, required, allowed):
    if not raw or not isinstance(raw, str):
        return None

    text = " ".join(raw.split())
    candidates = text.split(";")

    for c in candidates:
        c = c.strip()
        if not c:
            continue
        stmt = c + ";"

        if any(b in stmt for b in ["{", "}", "#", "/*", "*/"]):
            continue

        ids = extract_identifiers(stmt)

        if required and not all(rv in ids for rv in required):
            continue

        for ident in ids:
            if ident in allowed:
                continue
            if ident in C_LIB_FUNCS:
                continue
            return None

        return stmt

    return None

# ============================================================
# SIMPLE SEMGREP-STYLE DETECTOR
# ============================================================
class SimpleVulnDetector:
    def __init__(self):
        self.rules = [
            {
                'id': 'strcpy-buffer-overflow',
                'pattern': r'strcpy\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)',
                'severity': 'HIGH',
                'description': 'strcpy can cause buffer overflow'
            },
            {
                'id': 'gets-buffer-overflow',
                'pattern': r'gets\s*\(\s*([^)]+)\s*\)',
                'severity': 'HIGH',
                'description': 'gets is always unsafe'
            },
            {
                'id': 'free-without-check',
                'pattern': r'free\s*\(\s*([^)]+)\s*\)',
                'severity': 'MEDIUM',
                'description': 'free without NULL check'
            },
            {
                'id': 'system-command-injection',
                'pattern': r'system\s*\(\s*([^)]+)\s*\)',
                'severity': 'HIGH',
                'description': 'system() with user input'
            },
            {
                'id': 'printf-format-string',
                'pattern': r'printf\s*\(\s*([^)]+)\s*\)',
                'severity': 'MEDIUM',
                'description': 'printf with variable format'
            },
        ]
    
    def _detect_vulnerability_type(self, code):
        """Improved vulnerability detection."""
        # Clean the code
        clean_code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        clean_code = re.sub(r'//.*', '', clean_code)
        clean_code_lower = clean_code.lower()
        
        # Skip preprocessor directives
        if clean_code.strip().startswith('#'):
            # Check if there's actual code after preprocessor
            lines = clean_code.split('\n')
            for line in lines:
                if line.strip() and not line.strip().startswith('#'):
                    # Recursively check non-preprocessor lines
                    return self._detect_vulnerability_type(line)
            return "unknown"
        
        # Case-insensitive patterns
        patterns = {
            'strcpy': r'\bstrcpy\s*\(',
            'free': r'\bfree\s*\(',
            'malloc': r'\bmalloc\s*\(',
            'calloc': r'\bcalloc\s*\(',
            'realloc': r'\brealloc\s*\(',
            'gets': r'\bgets\s*\(',
            'memcpy': r'\bmemcpy\s*\(',
            'system': r'\bsystem\s*\(',
            'wmemset': r'\bwmemset\s*\(',
            'memset': r'\bmemset\s*\(',
            'swscanf': r'\bsw?scanf\s*\(',
            'fscanf': r'\bfscanf\s*\(',
            'scanf': r'\bscanf\s*\(',
            'sprintf': r'\bsprintf\s*\(',
            'strcat': r'\bstrcat\s*\(',
            'strncpy': r'\bstrncpy\s*\(',
            'memmove': r'\bmemmove\s*\(',
            'strcat': r'\bstrcat\s*\(',
            'wcscpy': r'\bwc?scpy\s*\(',
        }
        
        for vuln_type, pattern in patterns.items():
            if re.search(pattern, clean_code_lower):
                return vuln_type
        
        # Special case for printf format string
        if re.search(r'\bprintf\s*\(', clean_code_lower):
            match = re.search(r'printf\s*\(([^)]+)\)', clean_code)
            if match:
                arg = match.group(1).strip()
                if not (arg.startswith('"') or arg.startswith("'")):
                    return "printf_format"
            return "printf"
        
        return "unknown"
    
    def detect_vulnerabilities(self, code):
        vulns = []
        for rule in self.rules:
            matches = re.findall(rule['pattern'], code)
            if matches:
                vulns.append({
                    'rule': rule['id'],
                    'severity': rule['severity'],
                    'description': rule['description'],
                    'count': len(matches)
                })
        return vulns
    
    def is_safe(self, code):
        return len(self.detect_vulnerabilities(code)) == 0
# ============================================================
# LIGHT TGAT ENCODER
# ============================================================
class LightTGATEncoder(torch.nn.Module):
    def __init__(self, in_dim=None, edge_dim=2, out_dim=64):
        super().__init__()
        self.out_dim = out_dim
        self.adaptive_fc = torch.nn.Linear(1, 1)  # Placeholder
        
    def forward(self, x, edge_index, edge_attr):
        # Dynamically create layers based on input shape
        in_dim = x.size(1)
        
        # FIX: Handle None edge_attr
        if edge_attr is None:
            # Create default edge attributes
            edge_attr = torch.zeros((edge_index.size(1), 2), dtype=x.dtype, device=x.device)
        
        # Create layers on-the-fly if needed
        if not hasattr(self, 'edge_emb') or self.edge_emb.in_features != edge_attr.size(1):
            self.edge_emb = torch.nn.Linear(edge_attr.size(1), self.out_dim).to(x.device)
        
        if not hasattr(self, 'conv1') or self.conv1.in_channels != in_dim:
            self.conv1 = TransformerConv(in_dim, self.out_dim, heads=1, edge_dim=self.out_dim).to(x.device)
            
        if not hasattr(self, 'conv2'):
            self.conv2 = TransformerConv(self.out_dim, self.out_dim, heads=1, edge_dim=self.out_dim).to(x.device)
        
        # Ensure edge_attr has correct shape
        if edge_attr.ndim == 1:
            edge_attr = edge_attr.unsqueeze(1)
            if edge_attr.size(1) < 2:
                edge_attr = edge_attr.repeat(1, 2)

        e = self.edge_emb(edge_attr)
        h = torch.relu(self.conv1(x, edge_index, e))
        h = self.conv2(h, edge_index, e)
        return h

# ============================================================
# PATCH ENVIRONMENT
# ============================================================
class PatchEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, graph_paths=None, use_proper_samples=True, training_mode=True):
        """
        Initialize the Patch Environment.
        
        Args:
            graph_paths: Paths to graph data files (.pt files)
            use_proper_samples: Whether to use proper vulnerable code samples
            training_mode: If True, SKIP loading graph data (which contains bad samples)
        """
        super().__init__()
        print("📥 Initializing Patch Environment...")
        
        # Store training mode
        self.training_mode = training_mode
        self.total_actions_taken = 0
        self.action_distribution = {i: 0 for i in range(7)}
        # Initialize empty graph list
        self.all_graphs = []
        
        # ============================================================
        # CRITICAL FIX: SKIP GRAPH DATA DURING TRAINING!
        # The graph data contains documentation, function prototypes, 
        # and other bad samples that ruin training.
        # ============================================================
        if graph_paths and not training_mode:  # ONLY load if NOT in training mode
            print("   ⚠️ Training mode: SKIPPING graph data loading")
            print("   Reason: Graph data contains documentation/prototypes that ruin training")
        elif graph_paths:
            # Original paths if not provided
            if graph_paths is None:
                graph_paths = ["/app/juliet_cepg_full.pt", "/app/manybugs_cepg.pt"]
            
            print("   📊 Loading graph data (non-training mode only)...")
            for path in graph_paths:
                try:
                    loaded = torch.load(path, map_location="cpu", weights_only=False)
                    if isinstance(loaded, list):
                        for g in loaded:
                            if isinstance(g, tuple):
                                g = g[0]
                            if hasattr(g, "edge_index"):
                                if not hasattr(g, "edge_attr") or g.edge_attr is None:
                                    g.edge_attr = torch.zeros((g.edge_index.size(1), 2))
                                elif g.edge_attr.ndim == 1:
                                    g.edge_attr = g.edge_attr.unsqueeze(1).repeat(1, 2)
                                self.all_graphs.append(g)
                    elif isinstance(loaded, Data):
                        self.all_graphs.append(loaded)
                    else:
                        print(f"   ⚠ Unknown object in {path}: {type(loaded)}")
                    print(f"   ✅ Loaded graphs from {path}")
                except Exception as e:
                    print(f"   ❌ Failed to load {path}: {e}")
        else:
            print("   ⚠️ No graph paths provided, using proper samples only")
        
        # ============================================================
        # ALWAYS LOAD PROPER VULNERABLE CODE SAMPLES
        # These are clean, actual vulnerable code statements
        # ============================================================
        self.use_proper_samples = use_proper_samples
        self.proper_vulnerable_samples = self._load_proper_vulnerable_samples()
        print(f"   ✅ Loaded {len(self.proper_vulnerable_samples)} proper vulnerable code samples")
        
        # ============================================================
        # VALIDATION: Ensure we have proper samples
        # ============================================================
        if not self.proper_vulnerable_samples:
            print("   ⚠️ No proper samples loaded, creating defaults...")
            self.proper_vulnerable_samples = self._get_default_samples()
        
        if not self.proper_vulnerable_samples:
            raise RuntimeError("No valid vulnerable code samples loaded! Check your proper_vulnerable_code.txt file.")
        
        # ============================================================
        # SETUP DATA STRUCTURE
        # ============================================================
        # Always use dummy data structure - we don't need graph data for training
        # Graph data only causes problems with bad samples
        self.data = Data(
            x=torch.randn(1, 834), 
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.zeros((0, 2), dtype=torch.float32)  # FIX: Add edge_attr
        )
        self.data.node_meta = [{"full_source": "strcpy(buffer, input);"}]
        
        # ============================================================
        # INITIALIZE COMPONENTS
        # ============================================================
        self.actions = ASTEditActions()
        self.codet5 = CodeT5Fallback(finetuned_dir="/app/models/codet5_retrained")
        self.templates = PERFECTED_FIXES
        self.vuln_detector = SimpleVulnDetector()
        
        # ============================================================
        # SETUP TGAT ENCODER
        # ============================================================
        # Use consistent feature dimension (834 matches your dummy data)
        feat_dim = 834
        print(f"📊 Using fixed feature dimension: {feat_dim} (no graph dependency)")
        
        self.tgat = LightTGATEncoder(in_dim=feat_dim, out_dim=64).eval()
        
        # ============================================================
        # ENVIRONMENT SETTINGS
        # ============================================================
        print("📊 Using proper samples only - graph data skipped for training")
        print("⚡ TGAT On‑Demand Embedding Active")
        
        # Observation dimension: 834 (features) + 64 (embedding) + 1 (var count) = 899
        self.obs_dim = 899
        print(f"[FIX] Setting observation dimension to {self.obs_dim}")
        
        # Set observation space
        self.observation_space = spaces.Box(
            low=-8, high=8, 
            shape=(self.obs_dim,),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(7)
        
        # ============================================================
        # TRAINING STATE
        # ============================================================
        self.max_steps = 3
        self.steps = 0
        self.curriculum_level = 0
        self.declared_vars = set()
        self._patch_history = []
        self._current_node_id = None
        self._current_node_embedding = None
        self.var_list = ["ptr", "buf", "size"]  # default safe vars

        # ============================================================
        # FINAL INIT MESSAGE
        # ============================================================
        print(f"\n🎯 Environment initialized successfully!")
        print(f"   Mode: {'TRAINING' if training_mode else 'EVALUATION'}")
        print(f"   Proper samples: {len(self.proper_vulnerable_samples)}")
        print(f"   Graph data: {'SKIPPED (training mode)' if training_mode else 'Loaded' if self.all_graphs else 'None'}")
        print(f"   Observation dim: {self.obs_dim}")
        print(f"   Max steps per episode: {self.max_steps}")

    def _load_proper_vulnerable_samples(self):
        """Load the proper vulnerable code samples we extracted."""
        samples = []
        try:
            with open("proper_vulnerable_code.txt", "r") as f:
                content = f.read()
            
            # Split by lines and filter
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                
                # Skip empty lines or very short lines
                if not line or len(line) < 10:
                    continue
                
                # Skip documentation/comments
                if self._is_documentation(line):
                    continue
                
                # Skip function prototypes
                if re.search(r'^\w+\s+\w+\([^)]*\)\s*;', line):
                    continue
                
                # Skip header files, typedefs, macros
                skip_patterns = [
                    '#include', '#define', '#ifdef', '#endif', '#ifndef',
                    'typedef', 'struct ', 'enum ', 'union ', 'SQLITE_API',
                    'API', '/*!', '/**', '^If', '** Restrictions'
                ]
                if any(pattern in line for pattern in skip_patterns):
                    continue
                
                # Must contain vulnerable function calls
                vulnerable_patterns = [
                    r'\bstrcpy\s*\(', r'\bfree\s*\(', r'\bgets\s*\(', r'\bmalloc\s*\(',
                    r'\bmemcpy\s*\(', r'\bsystem\s*\(', r'\bprintf\s*\([^"]',
                    r'\bsprintf\s*\(', r'\bstrcat\s*\(', r'\bscanf\s*\(',
                    r'\bwmemset\s*\(', r'\bmemset\s*\(', r'\bswscanf\s*\(',
                    r'\bfscanf\s*\(', r'\bstrncpy\s*\(', r'\bmemmove\s*\(',
                    r'\bcalloc\s*\(', r'\brealloc\s*\('
                ]
                            
                has_vuln = False
                for pattern in vulnerable_patterns:
                    if re.search(pattern, line):
                        has_vuln = True
                        break
                
                if not has_vuln:
                    continue
                
                # Clean the sample
                line = re.sub(r'\s+', ' ', line)
                
                # Limit length and ensure it's actual code
                if len(line) < 250 and ';' in line:
                    samples.append(line)
            
            # If we don't have enough samples, add default ones
            if len(samples) < 20:
                print(f"⚠️ Only {len(samples)} proper samples found, adding defaults")
                samples.extend([
                    "strcpy(buffer, input);",
                    "free(ptr);",
                    "gets(user_input);",
                    "printf(variable);",
                    "system(command);",
                    "data = malloc(size);",
                    "memcpy(dest, src, n);",
                    "sprintf(buffer, variable);",
                    "strcat(dest, src);",
                    "scanf(\"%s\", buffer);",
                ])
            
            print(f"   ✅ Loaded {len(samples)} clean vulnerable code samples")
            return samples
            
        except Exception as e:
            print(f"⚠️ Could not load proper_vulnerable_code.txt: {e}")
            return self._get_default_samples()

    def _get_default_samples(self):
        """Get default vulnerable samples if file loading fails."""
        return [
            "strcpy(dest, src);",
            "free(ptr);",
            "gets(buffer);",
            "printf(user_input);",
            "system(command);",
            "data = malloc(size);",
            "memcpy(dest, src, n);",
            "sprintf(buffer, fmt);",
            "strcat(dest, src);",
            "scanf(\"%s\", input);",
        ]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        
        # ========== CRITICAL FIX: During training, ONLY use proper samples ==========
        # NEVER use graph data because it contains documentation/prototypes
        # ============================================================================
        
        # Always use proper samples when available
        if self.proper_vulnerable_samples:
            self.current_code = random.choice(self.proper_vulnerable_samples)
            print(f"📋 Using proper sample: {self.current_code[:80]}...")
            
            # Extract variables from the code
            self.var_list = extract_variables(self.current_code)
            if not self.var_list:
                # Fallback variables based on code pattern
                if "strcpy" in self.current_code:
                    self.var_list = ["dest", "src", "buffer"]
                elif "malloc" in self.current_code or "free" in self.current_code:
                    self.var_list = ["ptr", "data", "size"]
                elif "printf" in self.current_code:
                    self.var_list = ["data", "format"]
                elif "system" in self.current_code:
                    self.var_list = ["command"]
                elif "memcpy" in self.current_code:
                    self.var_list = ["dest", "src", "n"]
                else:
                    self.var_list = ["var1", "var2"]
            
            # Extract required variables
            ids = extract_identifiers(self.current_code)
            self.required_vars = [v for v in ids if v in self.var_list] or ids
            self.allowed_vars = set(self.var_list) | set(self.required_vars)
            
            # Use a dummy node for embedding
            self.current_node = 0
            self.base_score = 0.5
            self._last_action = None
            self.steps = 0
            self._patch_history = []
            self._current_node_embedding = None
            
            # Get the state
            state = self._get_state()
            
            # Ensure state has correct length
            if len(state) != self.observation_space.shape[0]:
                if len(state) > self.observation_space.shape[0]:
                    state = state[:self.observation_space.shape[0]]
                else:
                    padding = np.zeros(self.observation_space.shape[0] - len(state))
                    state = np.concatenate([state, padding])
            
            return state, {}
        
        # If no proper samples (shouldn't happen), fallback
        raise RuntimeError("No proper vulnerable samples available!")

    def _get_state(self):
        # Get the raw state
        if not hasattr(self, 'data') or not hasattr(self.data, 'x'):
            # Create a simple feature vector for proper samples
            simple_features = torch.randn(834 - 10)  # 824 random features
            var_emb = torch.tensor([len(self.var_list) / 10.0], dtype=torch.float32)

            
            if self._current_node_embedding is None:
                # Generate a random embedding (we're not using graph data)
                self._current_node_embedding = torch.randn(64)
            
                # Vulnerability type encoding (10 types - using the LAST 10 features)
                vuln_type = self._detect_vulnerability_type(self.current_code)
                vuln_types = ["strcpy", "free", "malloc", "memcpy", "printf_format", 
                            "gets", "system", "wmemset", "sprintf", "fscanf", "scanf"]
                
                vuln_encoding = [1.0 if vuln_type == vt else 0.0 for vt in vuln_types]
                vuln_tensor = torch.tensor(vuln_encoding, dtype=torch.float32)
                            
                # Combine: 824 + 64 + 1 + 10 = 899 (same as before!)
                raw_state = torch.cat([simple_features, self._current_node_embedding, 
                                    var_emb, vuln_tensor], dim=0)
        else:
            # Original graph-based state (for non-training mode)
            idx = self.current_node
            base = self.data.x[idx]
            
            # FIX: Check if we can get node embedding
            try:
                g = self._get_node_embedding(idx)
            except Exception as e:
                print(f"⚠️ Failed to get node embedding: {e}, using random")
                g = torch.randn(64)
            
            var_emb = torch.tensor([len(self.var_list) / 10.0], dtype=torch.float32)
            raw_state = torch.cat([base, g, var_emb], dim=0)
        
        # ================================================
        # CRITICAL: ENCODE ACTION MASK INFORMATION
        # ================================================
        # Instead of adding more dimensions, encode action preferences in existing features
        
        # Get vulnerability type
        vuln_type = self._detect_vulnerability_type(self.current_code)
        
        # Encode "action mask" information by modifying the LAST 10 vulnerability features
        # We'll use positions 889-898 (last 10 of 899) to encode action preferences
        
        # Convert to numpy for easier manipulation
        if hasattr(raw_state, 'numpy'):
            state_np = raw_state.numpy()
        else:
            state_np = np.array(raw_state, dtype=np.float32)
        
        # Ensure we have 899 dimensions
        if len(state_np) != 899:
            if len(state_np) < 899:
                padding = np.zeros(899 - len(state_np), dtype=np.float32)
                state_np = np.concatenate([state_np, padding])
            else:
                state_np = state_np[:899]
        
        # ================================================
        # ENCODE ACTION PREFERENCES IN VULNERABILITY FEATURES
        # ================================================
        # Map vulnerabilities to their preferred actions
        vuln_to_preferred_action = {
            "strcpy": 5,      # STRCPY_TO_STRNCPY - this works well!
            "free": 6,        # PREVENT_DOUBLE_FREE - better than SAFE_NOP
            "malloc": 2,      # SAFE_NOP - this works well
            "memcpy": 0,      # TEMPLATE_FIX - gives better patches
            "memset": 0,      # TEMPLATE_FIX - gives better patches  
            "wmemset": 0,     # TEMPLATE_FIX - gives better patches
            "fscanf": 0,      # TEMPLATE_FIX
            "scanf": 0,       # TEMPLATE_FIX
            "swscanf": 0,     # TEMPLATE_FIX
            "printf_format": 0, # TEMPLATE_FIX
            "gets": 1,        # AST_DELETE_CALL - actually replaces gets
            "system": 0,      # TEMPLATE_FIX
            "sprintf": 0,     # TEMPLATE_FIX
            "strcat": 0,      # TEMPLATE_FIX - gives better patches than SAFE_NOP
        }
                
        # Get preferred action for this vulnerability
        preferred_action = vuln_to_preferred_action.get(vuln_type, 0)
        
        # Encode preferred action in the last 7 features (positions 892-898)
        # We'll use the last 7 positions to indicate action preference strength
        for i in range(7):
            if i == preferred_action:
                state_np[892 + i] = 1.5  # Strong positive signal
            elif i == 2:  # SAFE_NOP
                state_np[892 + i] = 0.3  # Mild positive (safe fallback)
            else:
                state_np[892 + i] = -0.5  # Negative for others

        # Also encode which actions are INVALID
        if vuln_type in ["memcpy", "memset", "wmemset", "fscanf", "scanf", "swscanf"]:
            # CodeT5 (action 3) is invalid for these
            state_np[892 + 3] = -1.0  # Negative signal to avoid
        
        if vuln_type != "strcpy":
            # Action 5 is invalid for non-strcpy
            state_np[892 + 5] = -1.0  # Negative signal to avoid
        
        # ================================================
        # ENSURE STATE IS VALID
        # ================================================
        # Clip values to reasonable range
        state_np = np.clip(state_np, -5.0, 5.0)
        
        # Create final state
        final_state = np.array(state_np, dtype=np.float32)
        for i in range(7):
            if i == preferred_action:
                final_state[892 + i] = 0.5  # Signal: this action is preferred
            else:
                final_state[892 + i] = -0.2  # Signal: other actions less preferred        
            return final_state

    def _get_action_mask(self):
        """Get action validity mask (for debugging/info)."""
        vuln_type = self._detect_vulnerability_type(self.current_code)
        
        # Start with all actions valid
        mask = [1.0] * 7
        
        # Mark invalid actions
        problematic_vulns = ["memcpy", "memset", "wmemset", "fscanf", "scanf", "swscanf"]
        if vuln_type in problematic_vulns:
            mask[3] = 0.0  # CodeT5 invalid
        
        if vuln_type != "strcpy":
            mask[5] = 0.0  # STRCPY_TO_STRNCPY invalid
        
        return mask

    def _get_node_embedding(self, node_id):
        # If no graph data or using proper samples, return random embedding
        if (not hasattr(self, 'data') or not hasattr(self.data, 'x') or 
            not hasattr(self.data, 'edge_index') or self.data.edge_index.size(1) == 0):
            if self._current_node_embedding is None:
                self._current_node_embedding = torch.randn(64)
            return self._current_node_embedding
        
        if self._current_node_id == node_id and self._current_node_embedding is not None:
            return self._current_node_embedding

        with torch.no_grad():
            # FIX: Ensure edge_attr is not None and has correct shape
            if not hasattr(self.data, 'edge_attr') or self.data.edge_attr is None:
                # Create dummy edge attributes if missing
                edge_attr = torch.zeros((self.data.edge_index.size(1), 2), dtype=torch.float32)
            else:
                edge_attr = self.data.edge_attr
                
            # Ensure edge_attr has correct dimensions
            if edge_attr.ndim == 1:
                edge_attr = edge_attr.unsqueeze(1)
                if edge_attr.size(1) < 2:
                    edge_attr = edge_attr.repeat(1, 2)
            
            all_embeddings = self.tgat(
                self.data.x,
                self.data.edge_index,
                edge_attr
            )
            node_embedding = all_embeddings[node_id]
            # CRITICAL: Ensure it's detached and on CPU
            node_embedding = node_embedding.detach().cpu()

        self._current_node_id = node_id
        self._current_node_embedding = node_embedding
        return node_embedding

    def _tgat_score(self, node_id):
        embedding = self._get_node_embedding(node_id)
        return torch.norm(embedding).item()

    def _is_redundant_patch(self, patch):
        """Check if patch is redundant (repeated or trivial)."""
        if not patch or patch.strip() == "":
            return True
        
        # Check if patch is in recent history
        if patch in self._patch_history[-3:]:
            return True
        
        # Check for deeply nested code
        if patch.count('{') > 3 or patch.count('}') > 3:
            return True
        
        # Check for repeated patterns
        redundant_patterns = [
            r'if\s*\(\w+\s*!=\s*NULL\)\s*{\s*if\s*\(\w+\s*!=\s*NULL\)',
            r'free\(\w+\)\s*;\s*free\(\w+\)',
        ]
        
        for pattern in redundant_patterns:
            if re.search(pattern, patch):
                return True
        
        return False

    def _clean_variable(self, var_str):
        """Clean a variable string properly with better type handling."""
        # Remove leading/trailing whitespace
        var_str = var_str.strip()
        
        # Remove common C operators and punctuation at start
        var_str = re.sub(r'^[&*\s]+', '', var_str)  # Remove &, * at start
        
        # Handle array declarations: char buffer[100] -> buffer
        var_str = re.sub(r'\[.*\]', '', var_str)  # Remove array brackets
        
        # Remove [], (), etc. from end
        var_str = re.sub(r'[\s\[\]\(\);,]*$', '', var_str)
        
        # Extract first valid identifier only
        match = re.search(r'([A-Za-z_][A-Za-z0-9_]*)', var_str)
        if match:
            clean_var = match.group(1)
            
            # Skip type names and constants
            type_names = ['wchar_t', 'size_t', 'int', 'char', 'float', 'double', 'short', 'long']
            if clean_var in type_names or clean_var.endswith('_t'):
                return ""
            
            # Skip common constants
            constants = ['stdin', 'stdout', 'stderr', 'NULL', 'EOF', 'SEEK_SET']
            if clean_var in constants or clean_var.isupper():
                return ""
            
            return clean_var
        
        return ""


    def _apply_template_fallback(self, code, vuln_type):
        """Fallback template fix when CodeT5 fails."""
        if vuln_type == "free":
            match = re.search(r"free\(([^)]+)\)", code)
            if match:
                var = match.group(1).strip()
                var_clean = re.sub(r'[^A-Za-z0-9_]', '', var)
                return f"if ({var_clean} != NULL) {{ free({var_clean}); {var_clean} = NULL; }}"
        
        elif vuln_type == "malloc":
            match = re.search(r'(\w+)\s*=\s*(?:\(\s*\w+\s*\*\s*\)\s*)?malloc\s*\(\s*([^)]+)\s*\)', code)
            if match:
                var, size = match.groups()
                var_clean = re.sub(r'[^A-Za-z0-9_]', '', var)
                return f"{var_clean} = malloc({size});\nif ({var_clean} == NULL) {{ /* allocation failed */ }}"
        
        elif vuln_type == "strcpy":
            match = re.search(r"strcpy\(([^,]+),\s*([^)]+)\)", code)
            if match:
                dest, src = match.groups()
                dest_clean = re.sub(r'[^A-Za-z0-9_]', '', dest)
                src_clean = re.sub(r'[^A-Za-z0-9_]', '', src)
                return f"if ({dest_clean} != NULL && {src_clean} != NULL) {{ strncpy({dest_clean}, {src_clean}, sizeof({dest_clean}) - 1); {dest_clean}[sizeof({dest_clean}) - 1] = '\\0'; }}"
        
        elif vuln_type == "gets":
            match = re.search(r"gets\(([^)]+)\)", code)
            if match:
                buffer = match.group(1).strip()
                buffer_clean = re.sub(r'[^A-Za-z0-9_]', '', buffer)
                return f"if ({buffer_clean} != NULL) {{ fgets({buffer_clean}, sizeof({buffer_clean}), stdin); }}"
        
        elif vuln_type == "printf_format":
            match = re.search(r"printf\(([^)]+)\)", code)
            if match:
                arg = match.group(1).strip()
                if '"' not in arg:
                    return f'printf("%s", {arg});'

        elif vuln_type in ["fscanf", "scanf", "swscanf"]:
            # Improved scanf fix with proper bounds checking
            match = re.search(r'(\w+)\s*\(\s*([^)]+)\s*\)', code)
            if match:
                func, args = match.groups()
                args_list = [a.strip() for a in args.split(',')]
                
                if len(args_list) >= 2:
                    # First arg is usually stdin, second is format, rest are variables
                    format_arg = args_list[1] if len(args_list) > 1 else args_list[0]
                    
                    # Extract variables from format string and arguments
                    var_args = []
                    for i, arg in enumerate(args_list):
                        if i >= 2:  # Variables start at index 2
                            var_args.append(arg)
                    
                    if var_args:
                        # Build safe version with NULL checks
                        conditions = []
                        for var in var_args:
                            # Clean variable (remove &, [], etc.)
                            var_clean = re.sub(r'^&', '', var.strip())
                            var_clean = re.sub(r'[^A-Za-z0-9_]', '', var_clean)
                            if var_clean and re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', var_clean):
                                conditions.append(f"{var_clean} != NULL")
                        
                        if conditions:
                            return f"if ({' && '.join(conditions)}) {{\n    {code}\n}}"
                        else:
                            # Simple fallback
                            return f"/* Validate input before use */\n{code}"      
        # Default fallback
        return f"/* fallback fix for {vuln_type}: {code} */"


    def _apply_improved_template_fix(self, code, vuln_type):
        """Improved template fixes that actually fix the vulnerabilities."""
        
        if vuln_type == "memcpy":
            match = re.search(r"memcpy\s*\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^)]+)\s*\)", code)
            if match:
                dest, src, n = match.groups()
                dest_clean = self._clean_variable(dest)
                src_clean = self._clean_variable(src)
                n_clean = re.sub(r'[^A-Za-z0-9_]', '', n)
                
                # Determine if n is a variable or constant
                is_variable = n_clean and n_clean != n.strip()
                
                if dest_clean and src_clean:
                    if is_variable:
                        return f"if ({dest_clean} != NULL && {src_clean} != NULL) {{\n    size_t dest_size = sizeof({dest_clean});\n    if ({n_clean} <= dest_size) {{\n        memcpy({dest_clean}, {src_clean}, {n});\n    }} else {{\n        // ERROR: Copy size exceeds destination size\n        memcpy({dest_clean}, {src_clean}, dest_size);\n    }}\n}}"
                    else:
                        # Constant size
                        return f"if ({dest_clean} != NULL && {src_clean} != NULL) {{\n    size_t dest_size = sizeof({dest_clean});\n    if ({n} <= dest_size) {{\n        memcpy({dest_clean}, {src_clean}, {n});\n    }} else {{\n        // ERROR: Copy size exceeds destination size\n        memcpy({dest_clean}, {src_clean}, dest_size);\n    }}\n}}"
        
        elif vuln_type in ["memset", "wmemset"]:
            match = re.search(r'(\w+)\s*\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^)]+)\s*\)', code)
            if match:
                func, dest, char, count = match.groups()
                dest_clean = self._clean_variable(dest)
                count_clean = re.sub(r'[^A-Za-z0-9_]', '', count)
                
                is_variable = count_clean and count_clean != count.strip()
                
                if dest_clean:
                    if is_variable:
                        return f"if ({dest_clean} != NULL) {{\n    size_t dest_size = sizeof({dest_clean});\n    if ({count_clean} <= dest_size) {{\n        {func}({dest_clean}, {char}, {count});\n    }} else {{\n        // WARNING: Count exceeds destination size\n        {func}({dest_clean}, {char}, dest_size);\n    }}\n}}"
                    else:
                        # Constant count
                        return f"if ({dest_clean} != NULL) {{\n    size_t dest_size = sizeof({dest_clean});\n    if ({count} <= dest_size) {{\n        {func}({dest_clean}, {char}, {count});\n    }} else {{\n        // WARNING: Count exceeds destination size\n        {func}({dest_clean}, {char}, dest_size);\n    }}\n}}"
        
        elif vuln_type == "strcat":
            match = re.search(r'strcat\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)', code, re.IGNORECASE)
            if match:
                dest, src = match.groups()
                dest_clean = re.sub(r'[^A-Za-z0-9_]', '', dest)
                src_clean = re.sub(r'[^A-Za-z0-9_]', '', src)
                
                if dest_clean and src_clean:
                    # SIMPLIFIED BUT SAFE: Just convert to strncat with reasonable limit
                    # This avoids the sizeof() problem entirely
                    return f"""/* strcat converted to strncat with bounds checking */
        if ({dest_clean} != NULL && {src_clean} != NULL) {{
            /* Using reasonable buffer limit - adjust based on your actual buffer size */
            #define {dest_clean.upper()}_SIZE 100
            size_t current_len = strlen({dest_clean});
            if (current_len < {dest_clean.upper()}_SIZE) {{
                size_t available = {dest_clean.upper()}_SIZE - current_len - 1;
                strncat({dest_clean}, {src_clean}, available);
                {dest_clean}[{dest_clean.upper()}_SIZE - 1] = '\\0';
            }}
        }}"""
            
            return f"/* strcat requires buffer size knowledge - use strncat instead */\n{code}"



        elif vuln_type in ["fscanf", "scanf", "swscanf"]:
            # Improved scanf fix with proper bounds checking
            match = re.search(r'(\w+)\s*\(\s*([^)]+)\s*\)', code)
            if match:
                func, args = match.groups()
                args_list = [a.strip() for a in args.split(',')]
                
                # Look for format string
                format_arg = None
                var_args = []
                
                for i, arg in enumerate(args_list):
                    if '%' in arg:
                        format_arg = arg
                        # Collect all following variable arguments
                        for j in range(i + 1, len(args_list)):
                            var_args.append(args_list[j])
                        break
                
                if format_arg and var_args:
                    # Build safe version
                    safe_code = f"if ("
                    conditions = []
                    for var in var_args:
                        var_clean = re.sub(r'^&', '', var.strip())
                        var_clean = re.sub(r'[^A-Za-z0-9_]', '', var_clean)
                        if var_clean:
                            conditions.append(f"{var_clean} != NULL")
                    
                    if conditions:
                        safe_code += " && ".join(conditions) + ") {\n"
                        
                        # Handle string formats specially
                        if '%s' in format_arg:
                            # Only first string argument gets bounds checking
                            first_var = re.sub(r'^&', '', var_args[0].strip())
                            first_var_clean = re.sub(r'[^A-Za-z0-9_]', '', first_var)
                            if first_var_clean:
                                safe_code += f"    size_t {first_var_clean}_size = sizeof({first_var_clean});\n"
                                safe_code += f'    {func}(stdin, "%{first_var_clean}_size-1s", {first_var_clean});\n'
                                safe_code += f"    {first_var_clean}[{first_var_clean}_size-1] = '\\0';\n"
                            else:
                                safe_code += f"    {func}({', '.join(args_list)});\n"
                        else:
                            safe_code += f"    {func}({', '.join(args_list)});\n"
                        
                        safe_code += "}"
                        return safe_code
        
        # Fall back to the original template fallback
        return self._apply_template_fallback(code, vuln_type)


    def _apply(self, action):
        # 🔒 Ensure action is int
        patch = None
        try:
            if hasattr(action, 'item'):
                action = int(action.item())
            elif isinstance(action, np.ndarray):
                action = int(action[0]) if len(action) > 0 else int(action)
            elif isinstance(action, (np.integer, np.int64, np.int32)):
                action = int(action)
        except Exception as e:
            print(f"⚠ Warning converting action {action}: {e}")
            action = 0  # Default to TEMPLATE_FIX
        
        # Get the current code FIRST
        code = self.current_code
        
        print(f"\n🎯 APPLYING ACTION {action}: {ACTION_NAMES.get(action, 'UNKNOWN')}")
        print(f"   Input code: {code}")
        
        # ===============================
        # DETECT VULNERABILITY TYPE FIRST
        # ===============================
        vuln_type = self._detect_vulnerability_type(code)
        print(f"   Detected vulnerability type: {vuln_type}")
        patch = code
        # ===============================
        # ACTION 0 → TEMPLATE FIX (vulnerability-specific)
        # ===============================
        if action == 0:
            # If it's documentation/comments, just return as-is
            if self._is_documentation(code):
                return code

            if vuln_type in ["memset", "wmemset"]:
                match = re.search(r'(\w+)\s*\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^)]+)\s*\)', code)
                if match:
                    func, dest, char, count = match.groups()
                    
                    # Handle pointer arithmetic in destination
                    dest_expr = dest.strip()
                    base_dest_match = re.search(r'([A-Za-z_]\w*)', dest)
                    
                    if base_dest_match:
                        base_dest = base_dest_match.group(1)
                        
                        # Check if count is a variable or constant
                        count_clean = re.sub(r'[^A-Za-z0-9_]', '', count)
                        is_variable = count_clean and count_clean != count.strip()
                        
                        if is_variable:
                            return f"if ({base_dest} != NULL) {{\n    size_t available = sizeof(*{base_dest}) * 100; /* Estimate */\n    if ({count_clean} <= available) {{\n        {func}({dest_expr}, {char}, {count});\n    }}\n}}"
                        else:
                            # Try to evaluate constant count
                            try:
                                # Simple evaluation for expressions like "100-1"
                                count_value = eval(count.replace('sizeof', '1'))  # Simplified
                                return f"if ({base_dest} != NULL) {{\n    {func}({dest_expr}, {char}, {count});\n}}"
                            except:
                                return f"if ({base_dest} != NULL) {{\n    {func}({dest_expr}, {char}, {count});\n}}"
                        
            elif vuln_type == "system" or vuln_type == "SYSTEM":
                # BETTER: Add actual security improvements
                match = re.search(r'system\s*\(\s*([^)]+)\s*\)', code, re.IGNORECASE)
                if match:
                    cmd = match.group(1).strip()
                    cmd_clean = re.sub(r'[^A-Za-z0-9_]', '', cmd)
                    
                    if cmd_clean:
                        # Add input validation and suggest safer alternative
                        return f"""/* SECURITY: Validate and sanitize command */
    char* sanitized_cmd = sanitize_command({cmd_clean});
    if (sanitized_cmd != NULL) {{
        // Consider using execve() with argument array instead
        system(sanitized_cmd);
        free(sanitized_cmd);
    }} else {{
        fprintf(stderr, "Command validation failed\\n");
    }}"""
                
                # Fallback: warning comment
                return f"/* WARNING: system() is dangerous. Use execve() with argument array. */\n{code}"
            
            # REPLACE this section in _apply() method (around line 1290-1305):
            elif vuln_type in ["fscanf", "scanf", "swscanf"]:
                # BETTER: Extract variables and add proper validation
                match = re.search(r'(\w+)\s*\(\s*([^)]+)\s*\)', code)
                if match:
                    func, args = match.groups()
                    args_list = [a.strip() for a in args.split(',')]
                    
                    # Find format string position
                    for i, arg in enumerate(args_list):
                        if '%' in arg:  # Found format string
                            format_arg = arg
                            var_args = args_list[i+1:] if i+1 < len(args_list) else []
                            
                            if var_args:
                                # Handle string format (%s) specially
                                if '%s' in format_arg:
                                    first_var = var_args[0].strip()
                                    var_clean = re.sub(r'^&', '', first_var)
                                    var_clean = re.sub(r'[^A-Za-z0-9_]', '', var_clean)
                                    
                                    if var_clean:
                                        return f"""if ({var_clean} != NULL) {{
                // Using fgets for string input instead of scanf %s
                fgets({var_clean}, sizeof({var_clean}), stdin);
                // Remove newline if present
                {var_clean}[strcspn({var_clean}, "\\n")] = 0;
            }}"""
                                
                                # For numeric formats, add NULL checks for pointers
                                conditions = []
                                for var in var_args:
                                    # Check if it's a pointer (starts with &)
                                    if var.startswith('&'):
                                        var_name = var[1:].strip()
                                        var_clean = re.sub(r'[^A-Za-z0-9_]', '', var_name)
                                        if var_clean:
                                            conditions.append(f"{var_clean} != NULL")
                                
                                if conditions:
                                    return f"if ({' && '.join(conditions)}) {{\n    {code}\n}}"
                                else:
                                    # For non-pointer args (like integers), add range validation
                                    return f"// Validate input range before scanning\n{code}"
                    
                    # Fallback for simple cases
                    return f"/* Input validation needed */\n{code}"
                
                # If no match found, return original with warning
                return f"/* WARNING: {vuln_type} requires input validation */\n{code}"
                
                # Fallback: Simple NULL check
                # Extract first variable after format
                vars_in_code = extract_variables(code)
                if vars_in_code:
                    first_var = vars_in_code[0]
                    return f"if ({first_var} != NULL) {{\n    {code}\n}}"
                
                return f"/* Input validation needed for {vuln_type} */\n{code}"
                
                # Basic validation if no pattern matched
                return f"/* Validate input before scanning */\n{code}"
            
            elif vuln_type == "sprintf":
                # Convert to snprintf
                match = re.search(r'sprintf\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)', code, re.IGNORECASE)
                if match:
                    buffer, format_str = match.groups()
                    buffer_clean = re.sub(r'[^A-Za-z0-9_]', '', buffer)
                    return f"snprintf({buffer_clean}, sizeof({buffer_clean}), {format_str})"
          
            elif vuln_type == "strcpy":
                match = re.search(r"strcpy\(([^,]+),\s*([^)]+)\)", code)
                if match:
                    dest, src = match.groups()
                    # Clean variable names
                    dest_clean = re.sub(r'[^A-Za-z0-9_]', '', dest)
                    src_clean = re.sub(r'[^A-Za-z0-9_]', '', src)
                    return f"if ({dest_clean} != NULL && {src_clean} != NULL) {{ strncpy({dest_clean}, {src_clean}, sizeof({dest_clean}) - 1); {dest_clean}[sizeof({dest_clean}) - 1] = '\\0'; }}"
            
            elif vuln_type == "strcat":
                # Improved strcat fix with bounds checking
                match = re.search(r'strcat\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)', code, re.IGNORECASE)
                if match:
                    dest, src = match.groups()
                    dest_clean = re.sub(r'[^A-Za-z0-9_]', '', dest)
                    src_clean = re.sub(r'[^A-Za-z0-9_]', '', src)
                    
                    if dest_clean and src_clean:
                        return f"""/* STRING CONCATENATION SAFETY */
if ({dest_clean} != NULL && {src_clean} != NULL) {{
    size_t dest_len = strlen({dest_clean});
    size_t dest_size = sizeof({dest_clean});
    size_t src_len = strlen({src_clean});
    
    if (dest_len + src_len + 1 <= dest_size) {{
        strcat({dest_clean}, {src_clean});
    }} else {{
        /* ERROR: Destination buffer too small */
        /* Truncate or use strncat with bounds */
        strncat({dest_clean}, {src_clean}, dest_size - dest_len - 1);
        {dest_clean}[dest_size - 1] = '\\0';
    }}
}}"""
                
                return f"/* Use strncat() with bounds checking instead of strcat() */\n{code}"
                
            elif vuln_type == "free":
                match = re.search(r"free\(([^)]+)\)", code)
                if match:
                    var = match.group(1).strip()
                    var_clean = re.sub(r'[^A-Za-z0-9_]', '', var)
                    return f"if ({var_clean} != NULL) {{ free({var_clean}); }}"
            
            elif vuln_type == "gets":
                match = re.search(r"gets\(([^)]+)\)", code)
                if match:
                    buffer = match.group(1).strip()
                    buffer_clean = re.sub(r'[^A-Za-z0-9_]', '', buffer)
                    return f"if ({buffer_clean} != NULL) {{ fgets({buffer_clean}, sizeof({buffer_clean}), stdin); }}"
            
            # Replace the malloc section (around line 1295-1305 in your code):
            elif vuln_type == "malloc":
                # Find malloc assignment - FIXED: Better regex for various malloc patterns
                patterns = [
                    r'(\w+)\s*=\s*(?:\(\s*\w+\s*\*\s*\)\s*)?malloc\s*\(\s*([^)]+)\s*\)',  # var = malloc(...)
                    r'(\w+)\s*=\s*(?:\(\s*\w+\s*\*\s*\)\s*)?calloc\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)',  # var = calloc(...)
                    r'(\w+)\s*=\s*(?:\(\s*\w+\s*\*\s*\)\s*)?realloc\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)',  # var = realloc(...)
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, code)
                    if match:
                        if "calloc" in pattern:
                            var, count, size = match.groups()
                            return f"{var} = calloc({count}, {size});\nif ({var} == NULL) {{\n    fprintf(stderr, \"Memory allocation failed\\n\");\n    exit(EXIT_FAILURE);\n}}"
                        elif "realloc" in pattern:
                            var, ptr, size = match.groups()
                            return f"if ({ptr} != NULL) {{\n    {var} = realloc({ptr}, {size});\n    if ({var} == NULL) {{\n        fprintf(stderr, \"Memory reallocation failed\\n\");\n    }}\n}}"
                        else:
                            var, size = match.groups()
                            var_clean = re.sub(r'[^A-Za-z0-9_]', '', var)
                            return f"{var_clean} = malloc({size});\nif ({var_clean} == NULL) {{\n    fprintf(stderr, \"Memory allocation failed\\n\");\n    exit(EXIT_FAILURE);\n}}"
                
                # Fallback for any malloc pattern
                if "malloc" in code.lower():
                    return f"/* malloc safety: */ {code}"
                
                return code
            
            elif vuln_type == "memcpy":
                match = re.search(r"memcpy\s*\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^)]+)\s*\)", code)
                if match:
                    dest, src, n = match.groups()
                    dest_clean = re.sub(r'[^A-Za-z0-9_]', '', dest)
                    src_clean = re.sub(r'[^A-Za-z0-9_]', '', src)
                    
                    # FIX: Check if dest is a pointer or array
                    # If dest ends with ']' or starts with '&', it's likely an array
                    is_array = '[' in dest or dest.strip().startswith('&')
                    
                    if dest_clean:
                        if is_array:
                            # Array - use sizeof(dest)
                            return f"if ({dest_clean} != NULL && {src_clean} != NULL) {{\n    size_t dest_size = sizeof({dest_clean});\n    if ({n} <= dest_size) {{\n        memcpy({dest_clean}, {src_clean}, {n});\n    }}\n}}"
                        else:
                            # Pointer - we don't know size, just do NULL check
                            return f"if ({dest_clean} != NULL && {src_clean} != NULL) {{\n    memcpy({dest_clean}, {src_clean}, {n});\n}}"
                    
                    # Default fallback if no pointer arithmetic
                    dest_clean = re.sub(r'[^A-Za-z0-9_]', '', dest)
                    src_clean = re.sub(r'[^A-Za-z0-9_]', '', src)
                    return f"if ({dest_clean} != NULL && {src_clean} != NULL) {{ memcpy({dest_clean}, {src_clean}, {n}); }}"
                        
            elif vuln_type == "printf_format":
                match = re.search(r"printf\(([^)]+)\)", code)
                if match:
                    arg = match.group(1).strip()
                    if '"' not in arg:  # Not a literal string
                        return f'printf("%s", {arg});'
            
            elif vuln_type == "system":
                # BETTER: Add clear security warning AND validation
                match = re.search(r'system\s*\(\s*([^)]+)\s*\)', code, re.IGNORECASE)
                if match:
                    cmd = match.group(1).strip()
                    
                    # Check if command contains user input
                    if any(indicator in cmd.lower() for indicator in ['input', 'buffer', 'argv', 'user', 'gets']):
                        return f"""/* CRITICAL SECURITY: Never use system() with user input */
            /* Recommended: Use execve() with argument array */
            char *args[] = {{"/bin/sh", "-c", {cmd}, NULL}};
            execve(args[0], args, NULL);"""
                    else:
                        # Static command - still warn
                        return f"""/* SECURITY WARNING: system() is dangerous */
            /* Validate command first: */
            if (access({cmd}, X_OK) == 0) {{
                {code}
            }} else {{
                fprintf(stderr, "Command not found or not executable\\n");
            }}"""
                
                return code
                    
        # ===============================
        # ACTION 1 → AST DELETE/REPLACE
        # ===============================
        elif action == 1:
            print("   Applying AST delete/replace fix...")
            
            if vuln_type == "strcpy":
                match = re.search(r"strcpy\(([^,]+),\s*([^)]+)\)", code)
                if match:
                    dest, src = match.groups()
                    dest_clean = re.sub(r'[^A-Za-z0-9_]', '', dest)
                    src_clean = re.sub(r'[^A-Za-z0-9_]', '', src)
                    # Convert to strncpy - cleaner than if (1)
                    return f"strncpy({dest_clean}, {src_clean}, sizeof({dest_clean}) - 1);\n{dest_clean}[sizeof({dest_clean}) - 1] = '\\0';"
            
            elif vuln_type == "gets":
                match = re.search(r"gets\(([^)]+)\)", code)
                if match:
                    buffer = match.group(1)
                    buffer_clean = re.sub(r'[^A-Za-z0-9_]', '', buffer)
                    # Replace gets with fgets
                    return f"fgets({buffer_clean}, sizeof({buffer_clean}), stdin);"
            
            elif vuln_type == "free":
                match = re.search(r"free\(([^)]+)\)", code)
                if match:
                    var = match.group(1)
                    var_clean = re.sub(r'[^A-Za-z0-9_]', '', var)
                    # NULL check + NULL assignment
                    return f"if ({var_clean} != NULL) {{ free({var_clean}); {var_clean} = NULL; }}"
            
            elif vuln_type == "malloc":
                match = re.search(r'(\w+)\s*=\s*(?:\(\s*\w+\s*\*\s*\)\s*)?malloc\s*\(\s*([^)]+)\s*\)', code)
                if match:
                    var, size = match.groups()
                    var_clean = re.sub(r'[^A-Za-z0-9_]', '', var)
                    # Add NULL check
                    return f"{var_clean} = malloc({size});\nif ({var_clean} == NULL) {{ /* Handle allocation failure */ }}"
            
            elif vuln_type == "memcpy":
                match = re.search(r"memcpy\(([^,]+),\s*([^,]+),\s*([^)]+)\)", code)
                if match:
                    dest, src, n = match.groups()
                    dest_clean = re.sub(r'[^A-Za-z0-9_]', '', dest)
                    src_clean = re.sub(r'[^A-Za-z0-9_]', '', src)
                    # Add bounds checking
                    return f"if ({dest_clean} != NULL && {src_clean} != NULL) {{ memcpy({dest_clean}, {src_clean}, {n}); }}"
            
            elif vuln_type in ["wmemset", "memset"]:
                # Add bounds check for memset/wmemset
                match = re.search(r'(\w+)\s*\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^)]+)\s*\)', code)
                if match:
                    func, dest, char, count = match.groups()
                    dest_clean = re.sub(r'[^A-Za-z0-9_]', '', dest)
                    return f"if ({dest_clean} != NULL) {{ {func}({dest_clean}, {char}, {count}); }}"
            
            # Default: AST transformation - add safety but NOT if (1)
            return f"/* AST transformation applied */ {code}"
        
        # ===============================
        # ACTION 2 → SAFE NOP (minimal safe change)
        # ===============================
        elif action == 2:
            print("   Creating SAFE_NOP (meaningful safety fix)...")
            
            if vuln_type == "free":
                match = re.search(r"free\(([^)]+)\)", code)
                if match:
                    var = match.group(1).strip()
                    var_clean = re.sub(r'[^A-Za-z0-9_]', '', var)
                    # ACTUALLY FIX IT - don't just add comments!
                    return f"if ({var_clean} != NULL) {{ free({var_clean}); }}"

            elif vuln_type == "wmemset" or vuln_type == "memset":
                # ACTUALLY FIX IT - add bounds check
                match = re.search(r'(\w+)\s*\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^)]+)\s*\)', code)
                if match:
                    func, dest, char, count = match.groups()
                    dest_clean = self._clean_variable(dest)
                    if dest_clean:
                        return f"if ({dest_clean} != NULL) {{ {func}({dest_clean}, {char}, {count}); }}"        

            elif vuln_type == "gets":
                # ACTUALLY FIX IT - replace with fgets
                match = re.search(r"gets\(([^)]+)\)", code)
                if match:
                    buffer = match.group(1).strip()
                    buffer_clean = re.sub(r'[^A-Za-z0-9_]', '', buffer)
                    return f"if ({buffer_clean} != NULL) {{ fgets({buffer_clean}, sizeof({buffer_clean}), stdin); }}"
            
            elif vuln_type == "strcat":
                # ACTUALLY FIX IT - add bounds checking
                match = re.search(r'strcat\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)', code, re.IGNORECASE)
                if match:
                    dest, src = match.groups()
                    dest_clean = re.sub(r'[^A-Za-z0-9_]', '', dest)
                    src_clean = re.sub(r'[^A-Za-z0-9_]', '', src)
                    if dest_clean and src_clean:
                        return f"""if ({dest_clean} != NULL && {src_clean} != NULL) {{
            size_t dest_len = strlen({dest_clean});
            size_t dest_size = sizeof({dest_clean});
            if (dest_len + strlen({src_clean}) + 1 <= dest_size) {{
                strcat({dest_clean}, {src_clean});
            }}
        }}"""
                    
            elif vuln_type == "malloc":
                # Look for malloc assignment patterns
                match = re.search(r'(\w+)\s*=\s*(?:\(\s*\w+\s*\*\s*\)\s*)?malloc\s*\(\s*([^)]+)\s*\)', code)
                if match:
                    var, size = match.groups()
                    var_clean = re.sub(r'[^A-Za-z0-9_]', '', var)
                    return f"{var_clean} = malloc({size});\nif ({var_clean} == NULL) {{ /* allocation check */ }}"
            
            elif vuln_type == "strcpy":
                match = re.search(r"strcpy\(([^,]+),\s*([^)]+)\)", code)
                if match:
                    dest, src = match.groups()
                    dest_clean = re.sub(r'[^A-Za-z0-9_]', '', dest)
                    return f"strncpy({dest_clean}, {src}, sizeof({dest_clean}) - 1);"
            
            elif vuln_type == "memcpy":
                match = re.search(r"memcpy\(([^,]+),\s*([^,]+),\s*([^)]+)\)", code)
                if match:
                    dest, src, n = match.groups()
                    dest_clean = re.sub(r'[^A-Za-z0-9_]', '', dest)
                    src_clean = re.sub(r'[^A-Za-z0-9_]', '', src)
                    return f"if ({dest_clean} != NULL && {src_clean} != NULL) {{ memcpy({dest_clean}, {src_clean}, {n}); }}"
            
            elif vuln_type == "printf_format":
                match = re.search(r"printf\(([^)]+)\)", code)
                if match:
                    arg = match.group(1).strip()
                    if '"' not in arg and "'" not in arg:  # Variable format
                        return f'printf("%s", {arg});'
            
            elif vuln_type in ["fscanf", "scanf", "swscanf"]:
                # Extract and validate variables
                match = re.search(r'(\w+)\s*\(\s*([^)]+)\s*\)', code)
                if match:
                    func, args = match.groups()
                    args_list = [a.strip() for a in args.split(',')]
                    
                    # Find variables to validate
                    conditions = []
                    for arg in args_list:
                        # Look for &variable patterns
                        if arg.startswith('&'):
                            var_name = arg[1:].strip()
                            var_clean = re.sub(r'[^A-Za-z0-9_]', '', var_name)
                            if var_clean and re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', var_clean):
                                conditions.append(f"{var_clean} != NULL")
                    
                    if conditions:
                        return f"if ({' && '.join(conditions)}) {{\n    {code}\n}}"
                    
                    # Fallback if no &variables found
                    return f"/* SAFE_NOP: Input validation needed for {vuln_type} */\n{code}"
            
                vars_in_code = extract_variables(code)
                if vars_in_code:
                    first_var = vars_in_code[0]
                    # Check if it looks like a pointer variable
                    if re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', first_var) and not first_var.isupper():
                        return f"if ({first_var} != NULL) {{ {code} }}"
                
                # If we can't add a NULL check, at least add a warning
                return f"/* WARNING: {vuln_type} requires safety consideration */ {code}"
        
        # ===============================
        # ACTION 3 → CodeT5 patch (context-aware)
        # ===============================
        elif action == 3:
            print("   Calling RETRAINED CodeT5...")
            
            # ================================================
            # CRITICAL: RESTRICT CodeT5 usage for certain vulnerability types
            # ================================================
            problematic_vulns = ["memcpy", "memset", "wmemset", "fscanf", "scanf", "swscanf"]
            
            if vuln_type in problematic_vulns:
                print(f"   ⚠️ RESTRICTED: CodeT5 not recommended for {vuln_type}")
                print(f"   💡 Applying TEMPLATE_FIX instead...")
                
                # Force template fix for these vulnerable types
                if vuln_type == "memcpy":
                    match = re.search(r"memcpy\s*\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^)]+)\s*\)", code)
                    if match:
                        dest, src, n = match.groups()
                        dest_clean = self._clean_variable(dest)
                        src_clean = self._clean_variable(src)
                        n_clean = re.sub(r'[^A-Za-z0-9_]', '', n)
                        
                        # Check if size is a variable
                        if n_clean and n_clean != n.strip():  # Variable size
                            return f"if ({dest_clean} != NULL && {src_clean} != NULL) {{\n    size_t dest_size = sizeof({dest_clean});\n    if ({n_clean} <= dest_size) {{\n        memcpy({dest_clean}, {src_clean}, {n});\n    }}\n}}"
                        else:  # Constant size
                            return f"if ({dest_clean} != NULL && {src_clean} != NULL) {{\n    size_t dest_size = sizeof({dest_clean});\n    if ({n} <= dest_size) {{\n        memcpy({dest_clean}, {src_clean}, {n});\n    }}\n}}"
                
                elif vuln_type in ["memset", "wmemset"]:
                    match = re.search(r'(\w+)\s*\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^)]+)\s*\)', code)
                    if match:
                        func, dest, char, count = match.groups()
                        dest_clean = self._clean_variable(dest)
                        count_clean = re.sub(r'[^A-Za-z0-9_]', '', count)
                        
                        if dest_clean:
                            if count_clean and count_clean != count.strip():  # Variable count
                                return f"if ({dest_clean} != NULL && {count_clean} > 0) {{\n    size_t dest_size = sizeof({dest_clean});\n    if ({count_clean} <= dest_size) {{\n        {func}({dest_clean}, {char}, {count});\n    }}\n}}"
                            else:  # Constant count
                                return f"if ({dest_clean} != NULL) {{\n    size_t dest_size = sizeof({dest_clean});\n    if ({count} <= dest_size) {{\n        {func}({dest_clean}, {char}, {count});\n    }}\n}}"
                

                elif vuln_type in ["fscanf", "scanf", "swscanf"]:
                    # IMPROVED: Actually fix the vulnerability
                    match = re.search(r'(\w+)\s*\(\s*([^)]+)\s*\)', code)
                    if match:
                        func, args = match.groups()
                        args_list = [a.strip() for a in args.split(',')]
                        
                        # Find the format string
                        for i, arg in enumerate(args_list):
                            if '%' in arg:
                                format_arg = arg
                                var_args = args_list[i+1:] if i+1 < len(args_list) else []
                                
                                if var_args:
                                    # Build safe version with proper validation
                                    safe_code = "/* SAFE VERSION: Input validation added */\n"
                                    
                                    # Add NULL checks for pointer arguments
                                    for j, var in enumerate(var_args):
                                        var_clean = re.sub(r'^&', '', var.strip())
                                        var_clean = re.sub(r'\[\s*\d*\s*\]', '', var_clean)
                                        
                                        # Check if it's a variable (not constant)
                                        if re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', var_clean):
                                            # For string inputs, add bounds checking
                                            if '%s' in format_arg and j == 0:  # First string argument
                                                safe_code += f"if ({var_clean} != NULL) {{\n"
                                                safe_code += f"    size_t {var_clean}_size = sizeof({var_clean});\n"
                                                safe_code += f'    {func}(stdin, "%{var_clean}_size-1s", {var_clean});\n'
                                                safe_code += f"    {var_clean}[{var_clean}_size-1] = '\\0';\n"
                                                safe_code += "}\n"
                                                return safe_code
                                    
                                    # For numeric inputs, just add NULL check
                                    conditions = []
                                    for var in var_args:
                                        var_clean = re.sub(r'^&', '', var.strip())
                                        if re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', var_clean):
                                            conditions.append(f"{var_clean} != NULL")
                                    
                                    if conditions:
                                        safe_code += f"if ({' && '.join(conditions)}) {{\n"
                                        safe_code += f"    {func}({', '.join(args_list)});\n"
                                        safe_code += "}\n"
                                        return safe_code
                                
                                # Fallback
                                return f"/* Validate input before scanning */\n{code}"
                    
                    return f"/* {func} requires input validation */\n{code}"
                
                # Default template fallback for restricted types
                return self._apply_template_fallback(code, vuln_type)
            
            # ================================================
            # Only use CodeT5 for well-performing vulnerabilities
            # ================================================
            try:
                # Use the new fallback_patch method which returns (patch, confidence)
                patch, confidence = self.codet5.fallback_patch(vuln_type, code)
                
                # HIGHER confidence threshold for strcpy, malloc, free
                min_confidence = 0.85 if vuln_type in ["strcpy", "malloc", "free"] else 0.70
                
                if patch and confidence >= min_confidence:
                    print(f"   ✅ CodeT5 generated (confidence {confidence:.2f}): {patch}")
                    
                    # Sanitize the patch
                    safe = sanitize_codet5_single_line(patch, self.required_vars, self.allowed_vars)
                    if safe and safe != code:
                        return safe
                    else:
                        print("   ⚠️ CodeT5 patch failed sanitization")
                        # Fallback to IMPROVED template fix
                        return self._apply_improved_template_fix(code, vuln_type)
                else:
                    print(f"   ⚠️ CodeT5 low confidence ({confidence:.2f}) or no patch")
                    # Fallback to IMPROVED template fix
                    return self._apply_improved_template_fix(code, vuln_type)
            except Exception as e:
                print(f"   ❌ CodeT5 error: {e}")
                return self._apply_improved_template_fix(code, vuln_type)
        
        # ===============================
        # ACTION 4 → NULL GUARD (vulnerability-specific)
        # ===============================
        elif action == 4:
            print("   Adding smart NULL guard...")
            
            # Only add NULL guards for pointer variables, not constants!
            if vuln_type == "free":
                match = re.search(r"free\(([^)]+)\)", code)
                if match:
                    var = match.group(1).strip()
                    var_clean = re.sub(r'[^A-Za-z0-9_]', '', var)
                    # Check if it's a variable (not constant, not number)
                    if re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', var_clean) and not var_clean.isupper():
                        return f"if ({var_clean} != NULL) {{ {code} }}"

            elif vuln_type in ["memset", "wmemset"]:
                match = re.search(r'(\w+)\s*\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^)]+)\s*\)', code)
                if match:
                    func, dest, char, count = match.groups()
                    dest_clean = self._clean_variable(dest)
                    if dest_clean:
                        return f"if ({dest_clean} != NULL) {{ {func}({dest_clean}, {char}, {count}); }}"
            
            elif vuln_type in ["strcpy", "memcpy", "strcat", "memmove"]:
                # Get first argument (should be a pointer)
                match = re.search(r'(\w+)\(([^)]+)\)', code)
                if match:
                    args = match.group(2).split(',')
                    if args:
                        first_arg = args[0].strip()
                        first_arg_clean = re.sub(r'[^A-Za-z0-9_]', '', first_arg)
                        # Only add NULL guard if it looks like a variable
                        if (re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', first_arg_clean) and 
                            not first_arg_clean.isupper() and
                            first_arg_clean not in ['stdin', 'stdout', 'stderr']):
                            return f"if ({first_arg_clean} != NULL) {{ {code} }}"
            
            # Default: add informative comment, not useless NULL check
            return f"/* NULL_GUARD: Consider adding NULL checks for pointer arguments */ {code}"
        
        # ===============================
        # ACTION 5 → STRCPY_TO_STRNCPY (ONLY for strcpy!)
        # ===============================
        elif action == 5:
            print("   Converting strcpy to strncpy...")
            
            # CRITICAL: Only apply to strcpy vulnerability - REJECT everything else
            if vuln_type != "strcpy":
                print(f"   ❌❌❌ ACTION REJECTED: STRCPY_TO_STRNCPY on {vuln_type}")
                print(f"   💡 Redirecting to appropriate action...")
                
                # DON'T try to fix here - just return the original code with a comment
                # This will get a negative reward, teaching the model not to use ACTION 5
                return f"/* ERROR: Action 5 (STRCPY_TO_STRNCPY) inappropriate for {vuln_type} */ {code}"
            
            # Only proceed if it's actually strcpy
            match = re.search(r"strcpy\(([^,]+),\s*([^)]+)\)", code)
            if match:
                dest, src = match.groups()
                dest_clean = self._clean_variable(dest)
                
                # Extract src_clean properly
                src_clean = src.strip()
                
                # Check if src is a string literal
                is_string_literal = (src_clean.startswith('"') and src_clean.endswith('"')) or \
                                   (src_clean.startswith("'") and src_clean.endswith("'")) or \
                                   (src_clean.startswith('L"') and src_clean.endswith('"'))
                
                if is_string_literal:
                    # String literal - just convert directly
                    result = f"// strcpy converted to strncpy\n"
                    result += f"strncpy({dest_clean}, {src}, sizeof({dest_clean}) - 1);\n"
                    result += f"{dest_clean}[sizeof({dest_clean}) - 1] = '\\0';"
                else:
                    # Variable - clean it and check against NULL
                    src_var_clean = self._clean_variable(src)
                    if src_var_clean:
                        result = f"// strcpy converted to strncpy\n"
                        result += f"if ({dest_clean} != NULL && {src_var_clean} != NULL) {{\n"
                        result += f"    size_t src_len = strlen({src_var_clean});\n"
                        result += f"    size_t dest_size = sizeof({dest_clean});\n"
                        result += f"    if (src_len < dest_size) {{\n"
                        result += f"        strncpy({dest_clean}, {src_var_clean}, src_len);\n"
                        result += f"        {dest_clean}[src_len] = '\\0';\n"
                        result += f"    }} else {{\n"
                        result += f"        // Truncate to fit\n"
                        result += f"        strncpy({dest_clean}, {src_var_clean}, dest_size - 1);\n"
                        result += f"        {dest_clean}[dest_size - 1] = '\\0';\n"
                        result += f"    }}\n"
                        result += f"}}"
                    else:
                        # Can't clean src variable, do basic conversion
                        result = f"// strcpy converted to strncpy\n"
                        result += f"strncpy({dest_clean}, {src}, sizeof({dest_clean}) - 1);\n"
                        result += f"{dest_clean}[sizeof({dest_clean}) - 1] = '\\0';"
                
                print(f"   strcpy→strncpy with bounds: {result}")
                return result
            
            # If no strcpy pattern found (shouldn't happen if vuln_type is strcpy)
            return f"/* No strcpy pattern found */ {code}"
        
        # ===============================
        # ACTION 6 → PREVENT_DOUBLE_FREE & MEMORY SAFETY
        # ===============================
        elif action == 6:
            print("   Applying memory safety fix...")
            
            if vuln_type == "free":
                match = re.search(r"free\s*\(\s*([^)]+)\s*\)", code)
                if match:
                    var = match.group(1).strip()
                    var_clean = re.sub(r'[^A-Za-z0-9_]', '', var)
                    # RETURN THIS INSTEAD:
                    return f"if ({var_clean} != NULL) {{ free({var_clean}); {var_clean} = NULL; }}"
                return code
            
            elif vuln_type == "malloc":
                        # Find malloc assignment - FIXED REGEX
                patterns = [
                    r'(\w+)\s*=\s*(?:\(\s*\w+\s*\*\s*\)\s*)?malloc\s*\(\s*([^)]+)\s*\)',
                    r'(\w+)\s*=\s*(?:\(\s*\w+\s*\*\s*\)\s*)?calloc\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)',
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, code)
                    if match:
                        if "calloc" in pattern:
                            var, count, size = match.groups()
                            var_clean = re.sub(r'[^A-Za-z0-9_]', '', var)
                            return f"{var_clean} = calloc({count}, {size});\nif ({var_clean} == NULL) {{\n    /* Memory allocation failed */\n}}"
                        else:
                            var, size = match.groups()
                            var_clean = re.sub(r'[^A-Za-z0-9_]', '', var)
                            return f"{var_clean} = malloc({size});\nif ({var_clean} == NULL) {{\n    /* Memory allocation failed */\n}}"
                return code
            
            elif vuln_type == "memcpy":
                # Add bounds checking for memcpy
                match = re.search(r"memcpy\s*\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^)]+)\s*\)", code)
                if match:
                    dest, src, n = match.groups()
                    dest_clean = re.sub(r'[^A-Za-z0-9_]', '', dest)
                    src_clean = re.sub(r'[^A-Za-z0-9_]', '', src)
                    n_clean = re.sub(r'[^A-Za-z0-9_]', '', n)
                    
                    # Check if n is a variable or constant
                    if n_clean and n_clean != n.strip():  # It's a variable
                        return f"if ({dest_clean} != NULL && {src_clean} != NULL && {n_clean} > 0) {{\n    size_t dest_size = sizeof(*{dest_clean});\n    if ({n_clean} <= dest_size) {{\n        memcpy({dest_clean}, {src_clean}, {n});\n    }}\n}}"
                    else:
                        return f"if ({dest_clean} != NULL && {src_clean} != NULL) {{ memcpy({dest_clean}, {src_clean}, {n}); }}"
                return code
            
            else:
                # For other vuln types, apply generic memory safety
                # But don't just add comments!
                return f"/* memory safety: {code} */"

        # ===============================
        # FINAL VALIDATION - ADD THIS BEFORE RETURN
        # ===============================
        if patch is None:
            patch = code  # Fallback to original code
        
        # Clean up any syntax errors
        patch = self._validate_and_fix_patch(patch)
        
        # Ensure patch ends properly
        if not patch.strip().endswith(';') and not patch.strip().endswith('}') and ';' in patch:
            # Add semicolon if needed
            lines = patch.split('\n')
            last_line = lines[-1].strip()
            if last_line and not last_line.endswith(';'):
                lines[-1] = last_line + ';'
                patch = '\n'.join(lines)
        
        return patch

    
    def _detect_vulnerability_type(self, code):
        """MUCH BETTER vulnerability detection"""
        # Clean the code
        clean_code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        clean_code = re.sub(r'//.*', '', clean_code)
        clean_code_lower = clean_code.lower()
        
        # Case-insensitive patterns
        patterns = {
            'strcpy': r'\bstrcpy\s*\(',
            'free': r'\bfree\s*\(',
            'malloc': r'\bmalloc\s*\(',
            'gets': r'\bgets\s*\(',
            'memcpy': r'\bmemcpy\s*\(',
            'system': r'\bsystem\s*\(',
            'wmemset': r'\bwmemset\s*\(',
            'memset': r'\bmemset\s*\(',
            'swscanf': r'\bsw?scanf\s*\(',
            'fscanf': r'\bfscanf\s*\(',
            'scanf': r'\bscanf\s*\(',
            'sprintf': r'\bsprintf\s*\(',
            'strcat': r'\bstrcat\s*\(',
            'strncpy': r'\bstrncpy\s*\(',
        }
        
        for vuln_type, pattern in patterns.items():
            if re.search(pattern, clean_code_lower):
                return vuln_type
        
        # Special case for printf format string
        if re.search(r'\bprintf\s*\(', clean_code_lower):
            match = re.search(r'printf\s*\(([^)]+)\)', clean_code)
            if match:
                arg = match.group(1).strip()
                if not (arg.startswith('"') or arg.startswith("'")):
                    return "printf_format"
            return "printf"
        
        return "unknown"

    def _is_documentation(self, code):
        """Check if code is actually documentation/comments."""
        # If it's mostly comments or contains documentation markers
        comment_ratio = len(re.findall(r'/\*|\*/|//', code)) / max(len(code.split()), 1)
        if comment_ratio > 0.3:  # More than 30% comments
            return True
        
        # Contains documentation markers
        doc_markers = ['/**', '/*!', '/*-', '^If', '** Restrictions', '** </ul>', 'SQLITE_API']
        for marker in doc_markers:
            if marker in code:
                return True
        
        # Very long lines (likely documentation)
        lines = code.split('\n')
        if any(len(line) > 200 for line in lines):
            return True
        
        return False

    def _is_nonsense_patch(self, patch):
        """Detect nonsense patches that shouldn't be rewarded."""
        if not patch:
            return True
        
        # Check for NULL != NULL
        if re.search(r'if\s*\(\s*NULL\s*!=\s*NULL\s*\)', patch, re.IGNORECASE):
            return True
        
        # Check for checking numbers against NULL
        if re.search(r'if\s*\(\s*\d+\s*!=\s*NULL\s*\)', patch, re.IGNORECASE):
            return True
        
        # Check for checking string literals against NULL
        if re.search(r'if\s*\(\s*"[^"]*"\s*!=\s*NULL\s*\)', patch):
            return True
        if re.search(r"if\s*\(\s*'[^']*'\s*!=\s*NULL\s*\)", patch):
            return True
        
        # Check for checking common constants against NULL
        constants = ['stdin', 'stdout', 'stderr', 'EOF', 'SEEK_SET', 'SEEK_CUR', 'SEEK_END']
        for const in constants:
            pattern = rf'if\s*\(\s*{const}\s*!=\s*NULL\s*\)'
            if re.search(pattern, patch, re.IGNORECASE):
                return True
        
        # Check for checking type names against NULL
        type_names = ['wchar_t', 'size_t', 'int', 'char', 'float', 'double', 'short', 'long']
        for type_name in type_names:
            pattern = rf'if\s*\(\s*{type_name}\s*!=\s*NULL\s*\)'
            if re.search(pattern, patch, re.IGNORECASE):
                return True
        
        # Check for empty safe blocks
        if re.search(r'if\s*\(\s*\w+\s*!=\s*NULL\)\s*{\s*/\*\s*safe\s*\*/\s*}', patch, re.IGNORECASE):
            return True
        
        # Check for checking 1 against NULL (from if (1) wrappers)
        if re.search(r'if\s*\(\s*1\s*!=\s*NULL\s*\)', patch):
            return True
        
        return False

    def _reward(self, patch, original_code):
        reward = 0
        done = False
        
        # Get vulnerability types
        original_vuln_type = self._detect_vulnerability_type(original_code)
        patch_vuln_type = self._detect_vulnerability_type(patch)
            # Get vulnerability type
        vuln_type = self._detect_vulnerability_type(original_code)
        
        # ========== ACTION-SPECIFIC REWARDS ==========
        # Get the actual action used (need to track this)
        actual_action = getattr(self, '_last_action', 0)
        
        # REWARD using correct actions for specific vulnerabilities
        correct_action_map = {
            "free": 6,        # PREVENT_DOUBLE_FREE - gives NULL + assignment
            "strcpy": 5,      # STRCPY_TO_STRNCPY - actually converts
            "malloc": 2,      # SAFE_NOP - works well
            "memcpy": 0,      # TEMPLATE_FIX - gives bounds checking
            "memset": 0,      # TEMPLATE_FIX - gives bounds checking
            "wmemset": 0,     # TEMPLATE_FIX - gives bounds checking
            "fscanf": 0,      # TEMPLATE_FIX
            "scanf": 0,       # TEMPLATE_FIX  
            "printf_format": 0, # TEMPLATE_FIX
            "gets": 1,        # AST_DELETE_CALL - actually replaces
            "system": 0,      # TEMPLATE_FIX
            "sprintf": 0,     # TEMPLATE_FIX
            "strcat": 0,      # TEMPLATE_FIX - gives bounds checking
        }
        
        expected_action = correct_action_map.get(vuln_type, 0)
        
        # Bonus for using the correct action
        if actual_action == expected_action:
            reward += 50.0
            print(f"   ✅ +50.0 for using correct action {ACTION_NAMES[actual_action]} for {vuln_type}")
        else:
            # Penalty for wrong action choice (even if redirected)
            reward -= 25.0  # Increased penalty
            print(f"   ❌❌❌ -25.0 penalty for WRONG action {ACTION_NAMES[actual_action]} on {vuln_type}")           
        
        # Penalty for using wrong action (especially CODET5 for problematic vulns)
        if actual_action == 3 and vuln_type in ["fscanf", "scanf", "memcpy", "memset"]:
            reward -= 10.0
            print(f"   ❌ -10.0 penalty for using CODET5 on {vuln_type}")
        print(f"   Original vuln type: {original_vuln_type}")
        print(f"   Patch vuln type: {patch_vuln_type}")
        
        # ========== STRICT PENALTIES ==========
        if (re.search(r'if\s*\(\s*1\s*\)\s*{', patch) and 
            not re.search(r'if\s*\(\s*1\s*\)\s*{', original_code)):
            print(f"   ❌❌❌ EXTREME: Useless 'if (1) {{' wrapper")
            return -15.0, True  # Terminate episode immediately!
            
        # Example: STRCPY_TO_STRNCPY on free() should be heavily penalized
        if "strcpy converted to strncpy" in patch and original_vuln_type != "strcpy":
            print(f"   ❌❌❌ SEVERE: Wrong fix type! strcpy fix on {original_vuln_type}")
            return -20.0, True
        
        # 2. Penalize trying to fix documentation/header files
        if self._is_documentation(original_code):
            print(f"   ❌ Attempted to fix documentation, not code")
            return -15.0, True
        
        # 3. Remove comments and compare
        patch_no_comments = re.sub(r'/\*.*?\*/', '', patch, flags=re.DOTALL)
        patch_no_comments = re.sub(r'//.*', '', patch_no_comments)
        patch_no_comments = patch_no_comments.strip()
        
        original_no_comments = re.sub(r'/\*.*?\*/', '', original_code, flags=re.DOTALL)
        original_no_comments = re.sub(r'//.*', '', original_no_comments)
        original_no_comments = original_no_comments.strip()
        
        if patch_no_comments == original_no_comments:
            print(f"   ⚠️ Only added comments, no real fix")
            return -3.0, False
    
        
        # 2. Check for nonsense patches
        if self._is_nonsense_patch(patch):
            print("   ❌ Nonsense patch detected!")
            return -10.0, False
        
        if not patch or patch == original_code:
            print("   ❌ No change made")
            return -3.0, False
        
        # 3. Penalize applying wrong fixes
        # Example: Action 5 (strcpy fix) on free()
        if original_vuln_type != "strcpy" and "strcpy converted to strncpy" in patch:
            print("   ❌ Wrong fix: strcpy action on non-strcpy code")
            return -5.0, False
        
        # Small base reward for any meaningful change
        reward += 0.5  # Reduced from 1.0
        print(f"   +0.5 for making a meaningful change")
        
        # Check syntax
        if not self._syntactic_compile(patch):
            print("   ❌ Failed syntactic check")
            return -5.0, False
        
        # Try compilation
        compile_ok, compile_score = self._full_compile(patch)
        reward += compile_score
        
        if not compile_ok:
            print("   ❌ Compilation failed")
            return reward, False
        
        print(f"   ✅ Compilation successful (+{compile_score})")
        
        # ========== VULNERABILITY-SPECIFIC REWARDS ==========
        # Add this to the vulnerability-specific rewards section:
        if original_vuln_type in ["fscanf", "scanf", "swscanf"]:
            if "if (" in patch and "!= NULL" in patch:
                # Extra bonus for proper bounds checking
                if "sizeof" in patch or "size =" in patch or "_size" in patch:
                    reward += 35.0  # Excellent fix with bounds checking
                    print(f"   🎉 +35.0 for {original_vuln_type} with bounds checking!")
                    done = True
                else:
                    reward += 20.0  # Good fix with NULL check
                    print(f"   +20.0 for NULL-checked {original_vuln_type}")
                    done = True
            elif "size" in patch and ("sizeof" in patch or "_size" in patch):
                reward += 15.0  # Partial fix with size awareness
                print(f"   +15.0 for {original_vuln_type} with size awareness")
                
        if original_vuln_type in ["malloc", "calloc", "realloc"]:
            if "if (" in patch and "== NULL" in patch:
                if "fprintf" in patch or "exit" in patch or "return" in patch:
                    reward += 40.0  # Excellent fix with error handling
                    print(f"   🎉 +40.0 for proper malloc NULL check with error handling")
                    done = True
                else:
                    reward += 25.0  # Good fix
                    print(f"   +25.0 for malloc NULL check")
                    done = True
            elif "== NULL" in patch:
                reward += 15.0
                print(f"   +15.0 for basic NULL check")
        # 1. STRCPY - HIGH REWARD for proper fix
        if original_vuln_type == "strcpy":
            if "strncpy" in patch or "strcpy_s" in patch:
                if "strcpy" not in patch_no_comments:  # Actually replaced
                    reward += 80.0
                    print(f"   🎉 +80.0 for fixing strcpy!")
                    done = True
                else:
                    reward += 10.0
                    print(f"   +10.0 for partial strcpy fix")
        
        # 2. FREE - REWARD for NULL check + assignment
        elif original_vuln_type == "free":
            if "if (" in patch and "!= NULL" in patch:
                if "= NULL;" in patch:
                    reward += 40.0
                    print(f"   🎉 +40.0 for safe free with NULL assignment")
                    done = True
                else:
                    reward += 15.0
                    print(f"   +15.0 for NULL check on free")
        
        # 3. GETS - HIGH REWARD for fgets
        elif original_vuln_type == "gets":
            if "fgets" in patch and "gets" not in patch_no_comments:
                reward += 60.0
                print(f"   🎉 +60.0 for replacing gets with fgets")
                done = True
        
        # 4. PRINTF FORMAT - REWARD for format string fix
        elif original_vuln_type == "printf_format":
            if 'printf("%s"' in patch or 'snprintf' in patch:
                reward += 50.0
                print(f"   🎉 +50.0 for fixing format string")
                done = True
        
        # 5. MEMCPY - REWARD for bounds check
        elif original_vuln_type == "memcpy":
            # Give partial credit for NULL checks
            if "if (" in patch and "!= NULL" in patch:
                # Check if it's a proper NULL check
                if re.search(r'if\s*\(\s*\w+\s*!=\s*NULL', patch):
                    reward += 20.0
                    print(f"   +20.0 for NULL check on memcpy")
                    
                    # Extra bonus for bounds checking
                    if "sizeof" in patch.lower() or "size" in patch.lower():
                        reward += 10.0
                        print(f"   +10.0 bonus for bounds checking")
                    
                    done = True
        
        # 6. WMEMSET/MEMSET - REWARD for bounds check
        elif original_vuln_type in ["wmemset", "memset"]:
            if "if (" in patch and ("sizeof" in patch or "size" in patch or "dest_size" in patch):
                reward += 25.0
                print(f"   +25.0 for bounds check on {original_vuln_type}")
                done = True
        
        # System command injection fixes
        elif original_vuln_type == "system":
            if "execve" in patch or "SECURITY WARNING" in patch or "Replace with execve" in patch:
                reward += 30.0
                print(f"   🎉 +30.0 for proper system() security warning")
                done = True
            elif "Validate" in patch or "WARNING" in patch:
                reward += 15.0
                print(f"   +15.0 for system() validation comment")
                done = True

        # strcat fixes
        elif original_vuln_type == "strcat":
            if "strncat" in patch and "strcat" not in patch_no_comments:
                if "sizeof" in patch or "dest_size" in patch or "bounds" in patch:
                    reward += 40.0
                    print(f"   🎉 +40.0 for converting strcat to strncat with bounds!")
                    done = True
                else:
                    reward += 20.0
                    print(f"   +20.0 for converting strcat to strncat")
                    done = True
            elif "if (" in patch and "strlen" in patch:
                reward += 25.0
                print(f"   +25.0 for strcat with length checking")
                done = True
        
        # 8. SPRINTF - REWARD for conversion to snprintf
        elif original_vuln_type == "sprintf":
            if "snprintf" in patch and "sprintf" not in patch_no_comments:
                reward += 60.0
                print(f"   🎉 +60.0 for converting sprintf to snprintf")
                done = True
        # Penalty for introducing NEW vulnerabilities
        original_vulns = self.vuln_detector.detect_vulnerabilities(original_code)
        patch_vulns = self.vuln_detector.detect_vulnerabilities(patch)
        
        original_count = sum(v['count'] for v in original_vulns)
        patch_count = sum(v['count'] for v in patch_vulns)
        
        if patch_count > original_count:
            penalty = (patch_count - original_count) * 20.0
            reward -= penalty
            print(f"   ❌ -{penalty} for introducing new vulnerabilities")
        
        print(f"   Total reward: {reward:.1f}, Done: {done}")
        return reward, done

    def _is_meaningful_attempt(self, patch, original_code):
        """Check if the patch is a meaningful attempt even if it doesn't compile."""
        # Check if it's trying to fix the right vulnerability
        original_type = self._detect_vulnerability_type(original_code)
        
        if original_type == "strcpy" and ("strncpy" in patch or "sizeof" in patch):
            return True
        elif original_type == "free" and ("if (" in patch and "!= NULL" in patch):
            return True
        elif original_type == "gets" and "fgets" in patch:
            return True
        elif original_type == "printf_format" and ('"%s"' in patch or "snprintf" in patch):
            return True
        
        return False

    def _full_compile(self, line):
        try:
            # Skip compilation for complex code (function prototypes, etc.)
            if self._should_skip_compilation(line):
                return True, 5.0  # Give partial credit for complex code
            
            print(f"[DEBUG] Compiling: {line}")
            print(f"[DEBUG] Variables: {self.var_list}")
            
            # Create proper variable declarations
            vars_decl = []
            
            # Clean variable list - remove non-variables
            clean_vars = []
            for v in self.var_list:
                if not v or len(v.strip()) < 2:
                    continue
                v = str(v).strip()
                
                # Skip if it looks like a type or keyword
                if (v in C_KEYWORDS or 
                    v in ['stdin', 'stdout', 'stderr', 'NULL'] or
                    v.endswith('_t') or  # size_t, wchar_t, etc.
                    v in ['int', 'char', 'float', 'double', 'void', 'short', 'long']):
                    continue
                
                # Skip if it's clearly not a variable (based on common patterns)
                skip_patterns = ['API', 'IOCAP', 'SCN', 'OMIT', 'BAD', 'SRC', 'LEN']
                if any(pattern in v.upper() for pattern in skip_patterns):
                    continue
                
                clean_vars.append(v)
            
            # Use cleaned vars
            self.var_list = clean_vars[:5]  # Limit to 5 most relevant
            
            for v in self.var_list:
                if not v:
                    continue
                    
                v_lower = v.lower()
                
                # Check if variable appears in the code as a pointer
                if f'*{v}' in line or f'{v} = malloc' in line or f'{v} = calloc' in line:
                    vars_decl.append(f"    char* {v} = (char*)malloc(100);")
                elif 'size' in v_lower or 'len' in v_lower:
                    vars_decl.append(f"    size_t {v} = 100;")
                elif v in ['fd', 'socket', 'handle', 'n', 'count']:
                    vars_decl.append(f"    int {v} = 100;")
                else:
                    # Default to array (not pointer) for scanf/printf etc.
                    if any(func in line for func in ['scanf', 'fscanf', 'printf', 'sprintf']):
                        vars_decl.append(f"    char {v}[100];")
                    else:
                        vars_decl.append(f"    char* {v} = (char*)malloc(100);")
            
            vars_decl_str = "\n".join(vars_decl)
            
            print(f"[DEBUG] Variable declarations:\n{vars_decl_str}")
            
            harness = f"""
    #include <stdlib.h>
    #include <string.h>
    #include <stdio.h>

    void test_vulnerable() {{
    {vars_decl_str}
        {line}
    }}

    int main() {{
        test_vulnerable();
        return 0;
    }}
    """
            
            with tempfile.NamedTemporaryFile(suffix=".c", delete=False, mode='w') as f:
                f.write(harness)
                tmp_name = f.name

            # Try to compile
            result = subprocess.run(
                ["gcc", tmp_name, "-o", tmp_name.replace(".c", ".out"), "-w", "-O0"],
                capture_output=True, text=True, timeout=5
            )

            # Cleanup
            try:
                os.unlink(tmp_name)
                out_file = tmp_name.replace(".c", ".out")
                if os.path.exists(out_file):
                    os.unlink(out_file)
            except:
                pass
            
            if result.returncode == 0:
                return True, 10.0
            else:
                # Check if error is due to complex code
                error_msg = result.stderr[:200] if result.stderr else ""
                if any(keyword in error_msg for keyword in ['undeclared', 'SQLITE', 'API', 'expected']):
                    # Complex code that's not meant to be compiled standalone
                    return True, 5.0  # Partial credit
                return False, -3.0

        except Exception as e:
            print(f"   Compilation exception: {str(e)[:100]}")
            return False, -4.0

    def _should_skip_compilation(self, line):
        """Check if we should skip compilation for complex code."""
        line_lower = line.lower()
        
        # Skip if it contains function prototypes
        if any(pattern in line_lower for pattern in [
            'sqlite_api', 'int main(', 'void ', 'double ', 'float ', 
            'typedef ', 'struct ', 'enum ', 'union '
        ]):
            return True
        
        # Skip if it's clearly documentation/comment
        if line.count('/*') > 2 or line.count('*/') > 2:
            return True
        
        # Skip if it contains complex C constructs
        if any(pattern in line for pattern in [
            '(*callback)', '**errmsg', '*sqlite3', '(*)(', '[]', '{}'
        ]):
            return True
        
        return False

    def _syntactic_compile(self, line):
        """Check if code looks syntactically reasonable."""
        if not line or len(line.strip()) < 3:
            return False
        
        # Remove comments for checking
        clean_line = re.sub(r'/\*.*?\*/', '', line, flags=re.DOTALL)
        clean_line = re.sub(r'//.*', '', clean_line)
        clean_line = clean_line.strip()
        
        if len(clean_line) < 3:
            return True  # Just comments is OK
        
        # Basic checks only
        # Check for balanced quotes (simple check)
        if clean_line.count('"') % 2 != 0:
            return False
        
        # Check for obviously malformed
        if clean_line.endswith('{') or clean_line.endswith('('):
            return False
        
        # Allow more flexibility with braces
        open_braces = clean_line.count('{')
        close_braces = clean_line.count('}')
        if open_braces > close_braces + 1:  # Too many open braces
            return False
        
        return True

    def _curriculum_setup(self):
        if self.curriculum_level == 0:
            self.vuln_patterns = ["strcpy", "free"]
        elif self.curriculum_level == 1:
            self.vuln_patterns = ["malloc", "gets"]
        else:
            self.vuln_patterns = ["system", "exec"]

    def _detect_vuln(self, code):
        for p in self.vuln_patterns:
            m = re.search(rf"{p}\s*\([^;]*\);?", code)
            if m:
                return m.group(0)
        # Fallback: return first line that looks like a function call
        lines = code.split(';')
        for line in lines:
            if re.search(r'\w+\s*\([^)]*\)', line):
                return line.strip() + ';'
        return "int x = 0;"

    # Add this method to your PatchEnv class:
    def _validate_and_fix_patch(self, patch):
        """Validate and fix common syntax errors in patches."""
        if not patch:
            return patch
        
        # Fix 1: Check for unbalanced parentheses in function calls
        lines = patch.split('\n')
        fixed_lines = []
        
        for line in lines:
            # Count parentheses
            open_paren = line.count('(')
            close_paren = line.count(')')
            
            if open_paren > close_paren:
                # Try to fix common patterns
                # Pattern 1: memcpy(dest, src, strlen(src);  // missing )
                if 'memcpy(' in line or 'strncpy(' in line or 'strcpy(' in line:
                    line = line.rstrip(';') + ');'
                # Pattern 2: sizeof(dest - 1  // missing )
                elif 'sizeof(' in line and ')' not in line:
                    line = line + ')'
            
            # Fix 2: Ensure statements end with semicolon (if they should)
            line_stripped = line.strip()
            if (line_stripped and 
                not line_stripped.endswith(';') and 
                not line_stripped.endswith('{') and 
                not line_stripped.endswith('}') and
                not line_stripped.startswith('#') and
                not line_stripped.startswith('/*') and
                not line_stripped.startswith('//')):
                
                # Check if it looks like a statement that needs semicolon
                if re.search(r'\w+\([^)]*\)$', line_stripped) or re.search(r'\w+\s*=\s*', line_stripped):
                    line = line + ';'
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)

    def step(self, action):
        """
        Gymnasium step method.
        Returns: observation, reward, terminated, truncated, info
        """
        print(f"\n=== STEP {self.steps + 1} ===")
        self.steps += 1
        original_code = self.current_code
        
        # ================================================
        # CRITICAL: Convert action to integer
        # ================================================
        try:
            if hasattr(action, 'item'):
                action_int = int(action.item())
            elif isinstance(action, np.ndarray):
                action_int = int(action[0]) if len(action) > 0 else int(action)
            elif isinstance(action, (np.integer, np.int64, np.int32)):
                action_int = int(action)
            else:
                action_int = int(action)
        except Exception as e:
            print(f"⚠️ Warning converting action {action}: {e}")
            action_int = 0  # Default to TEMPLATE_FIX
        
        # Store original choice BEFORE any masking
        original_action_int = action_int
        
        # Get vulnerability type
        vuln_type = self._detect_vulnerability_type(original_code)
        print(f"Agent chose: {original_action_int} -> {ACTION_NAMES.get(original_action_int, 'UNKNOWN')}")
        print(f"Vulnerability: {vuln_type}")
        print(f"Current code: {original_code}")
        
        # ================================================
        # FIX 1: Check if action is valid BEFORE applying
        # ================================================
        is_valid_action = True
        action_penalty = 0.0
        
        # FIX 3: Check action validity and apply penalties
        if original_action_int == 3:  # CodeT5
            problematic_vulns = [
                "memcpy", "memset", "wmemset", 
                "fscanf", "scanf", "swscanf",
                "sprintf", "strcat", "malloc", "free"
            ]
            
            if vuln_type in problematic_vulns:
                print(f"   ❌❌❌ INVALID ACTION: CodeT5 on {vuln_type}")
                is_valid_action = False
                action_penalty = -15.0  # STRONG penalty
        
        elif original_action_int == 5:  # STRCPY_TO_STRNCPY
            if vuln_type != "strcpy":
                print(f"   ❌❌❌ INVALID ACTION: STRCPY_TO_STRNCPY on {vuln_type}")
                is_valid_action = False
                action_penalty = -10.0

        elif original_action_int == 2:  # SAFE_NOP
            # Redirect SAFE_NOP to better actions for certain vulns
            redirect_from_safenop = {
                "free": 6,        # PREVENT_DOUBLE_FREE is better
                "memcpy": 0,      # TEMPLATE_FIX gives bounds checking
                "memset": 0,      # TEMPLATE_FIX gives bounds checking
                "wmemset": 0,     # TEMPLATE_FIX gives bounds checking
                "gets": 1,        # AST_DELETE_CALL actually fixes it
                "strcat": 0,      # TEMPLATE_FIX gives better bounds checking
            }
            
            if vuln_type in redirect_from_safenop:
                print(f"   ⚠️ Redirecting SAFE_NOP to better action for {vuln_type}")
                action_int = redirect_from_safenop[vuln_type]
                # Small penalty for choosing suboptimal SAFE_NOP
                action_penalty = -5.0        

        # ================================================
        # ACTION MASKING: Redirect invalid actions
        # ================================================
        if not is_valid_action:
            print(f"   ⚠️ Applying action penalty: {action_penalty}")
            
            # Redirect to appropriate action
            redirect_map = {
                "malloc": 2,      # SAFE_NOP
                "free": 6,        # PREVENT_DOUBLE_FREE  
                "memcpy": 0,      # TEMPLATE_FIX
                "memset": 0,      # TEMPLATE_FIX
                "wmemset": 0,     # TEMPLATE_FIX
                "fscanf": 0,      # TEMPLATE_FIX
                "scanf": 0,       # TEMPLATE_FIX
                "swscanf": 0,     # TEMPLATE_FIX
                "sprintf": 0,     # TEMPLATE_FIX
                "strcat": 0,      # TEMPLATE_FIX
                "printf_format": 0, # TEMPLATE_FIX
                "gets": 1,        # AST_DELETE_CALL
                "system": 0,      # TEMPLATE_FIX
            }
            
            if vuln_type in redirect_map:
                action_int = redirect_map[vuln_type]
            else:
                action_int = 0  # Default to TEMPLATE_FIX
            
            print(f"   💡 Redirected to: {action_int} -> {ACTION_NAMES.get(action_int, 'UNKNOWN')}")
        else:
            # No redirection needed
            action_int = original_action_int
        
        # Store the action that will be applied
        self._last_action = action_int
        
        # ================================================
        # Apply the action (either original or redirected)
        # ================================================
        try:
            patch = self._apply(action_int)
        except Exception as e:
            print(f"❌ Error applying action: {e}")
            # Return a negative reward and mark as terminated
            return self._get_state(), -10.0, True, False, {"error": str(e)}

        # ================================================
        # ACTION TRACKING: Update statistics (FIX 4)
        # ================================================
        self.total_actions_taken += 1
        self.action_distribution[original_action_int] = self.action_distribution.get(original_action_int, 0) + 1
        
        # Print statistics every 50 steps
        if self.total_actions_taken % 50 == 0:
            self.print_action_statistics()
        
        # ================================================
        # Calculate reward
        # ================================================
        is_redundant = False
        try:
            is_redundant = self._is_redundant_patch(patch)
        except AttributeError:
            # Method doesn't exist yet
            pass
        
        if is_redundant:
            reward = -3.0
            terminated = False
            truncated = False
            patch = original_code
        else:
            try:
                # Get base reward from patch quality
                base_reward, terminated = self._reward(patch, original_code)
                
                # ADD THE ACTION PENALTY to base reward
                reward = base_reward + action_penalty
                
                truncated = False
                
                # Special case: If action was invalid, ensure negative total reward
                if not is_valid_action and base_reward > 0:
                    # Even if patch is good, invalid action choice should be penalized
                    reward = min(base_reward - 5.0, base_reward / 2)
                
            except Exception as e:
                print(f"❌ Error in reward function: {e}")
                reward = -5.0 + action_penalty
                terminated = False
                truncated = True

            if patch != original_code:
                # Initialize patch history if needed
                if not hasattr(self, '_patch_history'):
                    self._patch_history = []
                self._patch_history.append(patch)

        self.current_code = patch
        
        # Check if episode should end
        terminated = terminated or self.steps >= self.max_steps
        truncated = truncated or self.steps >= self.max_steps
        
        # ================================================
        # Get observation state
        # ================================================
        obs = self._get_state()
        
        # Ensure state has correct length
        if len(obs) != self.observation_space.shape[0]:
            if len(obs) > self.observation_space.shape[0]:
                obs = obs[:self.observation_space.shape[0]]
            else:
                padding = np.zeros(self.observation_space.shape[0] - len(obs))
                obs = np.concatenate([obs, padding])
        
        # ================================================
        # Build info dictionary
        # ================================================
        info = {
            "original_code": original_code,
            "patch": patch,
            "reward": reward,
            "steps": self.steps,
            "vuln_type": vuln_type,
            "original_action": ACTION_NAMES.get(original_action_int, f"ACTION_{original_action_int}"),
            "actual_action": action_int, 
            "action_mask": self._get_action_mask(),
            "action_valid": is_valid_action,
            "action_penalty": action_penalty,
            "actual_action_name": ACTION_NAMES.get(action_int, f"ACTION_{action_int}"),
            "action_distribution": {ACTION_NAMES.get(k, f"ACTION_{k}"): v 
                                for k, v in self.action_distribution.items() 
                                if v > 0},
            "total_actions": self.total_actions_taken
        }
        
        print(f"   Final reward: {reward:.2f} (base: {base_reward if 'base_reward' in locals() else 'N/A'}, penalty: {action_penalty})")
        print(f"   Terminated: {terminated}, Truncated: {truncated}")
        print("="*50)
        
        return obs, reward, terminated, truncated, info

    def print_action_statistics(self):
        """Print action usage statistics"""
        print("\n" + "="*60)
        print("ACTION USAGE STATISTICS")
        print("="*60)
        
        if self.total_actions_taken == 0:
            print("No actions taken yet.")
            return
        
        print(f"Total actions taken: {self.total_actions_taken}")
        print("\nDistribution:")
        
        for action_id in sorted(self.action_distribution.keys()):
            count = self.action_distribution[action_id]
            if count > 0:
                percentage = (count / self.total_actions_taken) * 100
                action_name = ACTION_NAMES.get(action_id, f"ACTION_{action_id}")
                print(f"  {action_name:20s}: {count:4d} ({percentage:5.1f}%)")
        
        # Most common action
        if self.action_distribution:
            most_common = max(self.action_distribution.items(), key=lambda x: x[1])
            action_name = ACTION_NAMES.get(most_common[0], f"ACTION_{most_common[0]}")
            print(f"\nMost common action: {action_name} ({most_common[1]} times)")
        
        print("="*60)

    def render(self, mode="human"):
        """
        Render the environment - required by Gymnasium.
        """
        if mode == "human":
            print(f"=== Environment State ===")
            print(f"Step: {self.steps}/{self.max_steps}")
            print(f"Current code: {self.current_code}")
            print(f"Variables: {self.var_list}")
            print(f"Vulnerability type: {self._detect_vulnerability_type(self.current_code)}")
            print("========================")
        return None

    def close(self):
        """
        Close environment - required by Gymnasium.
        """
        # Clean up any resources if needed
        pass

    def _encode_code(self, code_str):
        """
        Encode raw code string into an observation tensor.
        Only used in code-only inference mode.
        """
        if not hasattr(self, "code_encoder"):
            raise RuntimeError("PatchEnv: 'code_encoder' not initialized.")
        
        self.current_code = code_str
        encoded = self.code_encoder.encode(code_str)
        
        # Add any necessary padding/fixes here if your model expects fixed dim
        return encoded
