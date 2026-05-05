# ============================================================
# ast_edit_actions.py
# ============================================================
import re

class ASTEditActions:

    def replace_api(self, code, old_api, new_api):
        return re.sub(rf"\b{old_api}\b", new_api, code)

    def insert_if_guard(self, code, cond="1"):
        return f"if ({cond}) {{\n{code}\n}}"

    def add_null_check(self, code, cond):
        return f"if ({cond}) {{\n{code}\n}}"

    def insert_call(self, code, call):
        # `call` already includes its own semicolon
        return call + "\n" + code

    def delete_call(self, code, api):
        """
        Deletes a function call like  api(...)  while avoiding regex crashes.
        """
        safe_api = re.escape(api)       # <-- FIXED
        pattern = rf"{safe_api}\s*\([^)]*\);"
        try:
            return re.sub(pattern, "", code)
        except re.error:
            # Fallback: don't delete anything, just return original line
            return code

    def update_literal(self, code):
        return code  # placeholder

    def update_identifier(self, code):
        return code  # placeholder

    def apply(self, code, action):
        t = action["action"]

        if t == "REPLACE_API":
            return self.replace_api(code, action["old_api"], action["new_api"])

        if t == "INSERT_CALL":
            return self.insert_call(code, action["api"])

        if t == "DELETE_CALL":
            return self.delete_call(code, action["api"])

        if t == "ADD_IF_GUARD":
            return self.insert_if_guard(code, action["cond"])

        if t == "ADD_NULL_CHECK":
            return self.add_null_check(code, action["cond"])

        return code
