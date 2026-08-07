"""Utility script to filter consolidated_data.csv and prepare python/java if-else datasets."""
import os
import ast
import re
import pandas as pd

# Python helper functions to extract AST spans
def line_col_to_char_offset(code, line, col):
    lines = code.splitlines(keepends=True)
    if line - 1 < len(lines):
        return sum(len(lines[i]) for i in range(line - 1)) + col
    return len(code)

def find_if_else_spans_py(code):
    try:
        tree = ast.parse(code)
    except Exception:
        return []
    
    spans = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            # Then branch (if body)
            if node.body:
                start_node = node.body[0]
                end_node = node.body[-1]
                try:
                    start_char = line_col_to_char_offset(code, start_node.lineno, start_node.col_offset)
                    end_char = line_col_to_char_offset(code, end_node.end_lineno, end_node.end_col_offset)
                    spans.append((start_char, end_char))
                except AttributeError:
                    pass
            
            # Else/Elif branch
            if node.orelse:
                # If orelse contains a nested ast.If (which is an elif), ast.walk will visit it separately.
                # Thus, we only process orelse here if it is a plain 'else' block.
                if not (len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If)):
                    start_node = node.orelse[0]
                    end_node = node.orelse[-1]
                    try:
                        start_char = line_col_to_char_offset(code, start_node.lineno, start_node.col_offset)
                        end_char = line_col_to_char_offset(code, end_node.end_lineno, end_node.end_col_offset)
                        spans.append((start_char, end_char))
                    except AttributeError:
                        pass
    return spans

# Java helper functions to extract spans
def java_find_matching_brace(code, open_idx):
    depth = 0
    for i in range(open_idx, len(code)):
        ch = code[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1

def find_if_else_spans_java(code):
    spans = []
    
    # Match: if (...) {
    if_pat = re.compile(r"\bif\s*\([^)]*\)\s*\{")
    for m in if_pat.finditer(code):
        brace_open = code.find("{", m.start())
        if brace_open != -1:
            brace_close = java_find_matching_brace(code, brace_open)
            if brace_close != -1:
                spans.append((brace_open + 1, brace_close))
                
    # Match: else {
    else_pat = re.compile(r"\belse\s*\{")
    for m in else_pat.finditer(code):
        brace_open = code.find("{", m.start())
        if brace_open != -1:
            brace_close = java_find_matching_brace(code, brace_open)
            if brace_close != -1:
                spans.append((brace_open + 1, brace_close))
                
    return spans

def main():
    csv_path = "./data/consolidated_data.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    print("Reading consolidated_data.csv...")
    df = pd.read_csv(csv_path)

    # 1. Process Python
    print("Processing Python snippets...")
    py_snippets = []
    for code in df["Python"]:
        if pd.notna(code) and isinstance(code, str):
            spans = find_if_else_spans_py(code)
            if len(spans) > 0:
                py_snippets.append(code)

    py_df = pd.DataFrame({"code": py_snippets})
    py_out = "./data/python_ifelse_snippets.csv"
    py_df.to_csv(py_out, index=False)
    print(f"Saved {len(py_snippets)} Python if-else snippets to {py_out}")

    # 2. Process Java
    print("Processing Java snippets...")
    java_snippets = []
    for code in df["Java"]:
        if pd.notna(code) and isinstance(code, str):
            spans = find_if_else_spans_java(code)
            if len(spans) > 0:
                java_snippets.append(code)

    java_df = pd.DataFrame({"code": java_snippets})
    java_out = "./data/java_ifelse_snippets.csv"
    java_df.to_csv(java_out, index=False)
    print(f"Saved {len(java_snippets)} Java if-else snippets to {java_out}")

if __name__ == "__main__":
    main()
