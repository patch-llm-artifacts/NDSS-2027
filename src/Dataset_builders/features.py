# === features.py ===
import hashlib, difflib, torch
from tree_sitter_languages import get_language, get_parser

# Parsers ready out of the box
LANG_PARSERS = {
    "python": get_parser("python"),
    "c": get_parser("c"),
    "c++": get_parser("cpp"),
    "cpp": get_parser("cpp"),
    "java": get_parser("java"),
}

AST_CACHE = {}

def hash_code(code: str):
    return hashlib.sha256(code.encode("utf-8")).hexdigest()

def compute_diff_features(old_code: str, new_code: str):
    diff = list(difflib.unified_diff(old_code.splitlines(), new_code.splitlines(), lineterm=""))
    added = sum(1 for l in diff if l.startswith("+") and not l.startswith("+++"))
    removed = sum(1 for l in diff if l.startswith("-") and not l.startswith("---"))
    diff_hash = hashlib.sha256("\n".join(diff).encode("utf-8")).hexdigest()
    return {"added": added, "removed": removed}, diff_hash

def extract_ast_features(code: str, language: str, max_nodes=5000, max_dim=50):
    lang_key = language.strip().lower()
    parser = LANG_PARSERS.get(lang_key, LANG_PARSERS["c"])
    try:
        tree = parser.parse(bytes(code, "utf8"))
        root = tree.root_node
        node_types = {}
        def visit(node):
            node_types[node.type] = node_types.get(node.type, 0) + 1
            for child in node.children:
                visit(child)
        visit(root)
        top = sorted(node_types.items(), key=lambda kv: kv[1], reverse=True)[:max_dim]
        vec = torch.zeros(max_dim)
        for i, (_, count) in enumerate(top):
            vec[i] = count / max_nodes
        return vec
    except Exception as e:
        print(f"⚠️ AST parse error: {e}")
        return torch.zeros(max_dim)

def extract_ast_features_cached(code: str, language: str):
    h = hashlib.md5(code.encode("utf-8")).hexdigest()
    if h not in AST_CACHE:
        AST_CACHE[h] = extract_ast_features(code, language)
    return AST_CACHE[h]
