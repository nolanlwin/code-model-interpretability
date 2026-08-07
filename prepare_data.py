"""Utility script to filter consolidated_data.csv and prepare python/java iterator datasets."""
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

def extract_target_names(target):
    names = set()
    if isinstance(target, ast.Name):
        names.add(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            names.update(extract_target_names(elt))
    return names

def node_to_span(code, node):
    start = line_col_to_char_offset(code, node.lineno, node.col_offset)
    end = line_col_to_char_offset(code, node.end_lineno, node.end_col_offset)
    return (start, end)

def find_iterator_and_body_use_spans_py(code):
    try:
        tree = ast.parse(code)
    except Exception:
        return []
    
    spans = []

    def visit_for_node(for_node):
        iterator_names = extract_target_names(for_node.target)

        def add_target_spans(target):
            if isinstance(target, ast.Name):
                spans.append(node_to_span(code, target))
            elif isinstance(target, (ast.Tuple, ast.List)):
                for elt in target.elts:
                    add_target_spans(elt)

        add_target_spans(for_node.target)

        def collect_uses(node, blocked_names=None):
            if blocked_names is None:
                blocked_names = set()

            if isinstance(node, ast.For):
                inner_bound = extract_target_names(node.target)
                new_blocked = blocked_names | (inner_bound & iterator_names)
                for child in node.body:
                    collect_uses(child, new_blocked)
                for child in node.orelse:
                    collect_uses(child, new_blocked)
                return

            if isinstance(node, ast.Name):
                if node.id in iterator_names and node.id not in blocked_names and isinstance(node.ctx, ast.Load):
                    spans.append(node_to_span(code, node))

            for child in ast.iter_child_nodes(node):
                collect_uses(child, blocked_names)

        for stmt in for_node.body:
            collect_uses(stmt)
        for stmt in for_node.orelse:
            collect_uses(stmt)

    for node in ast.walk(tree):
        if isinstance(node, ast.For):
            visit_for_node(node)

    return spans

# Java helper functions to extract spans
JAVA_FOR_BINDER_RE = re.compile(r"for\s*\(\s*int\s+([A-Za-z_][A-Za-z0-9_]*)\s*=")

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

def java_find_loop_blocks(code):
    loops = []
    for m in JAVA_FOR_BINDER_RE.finditer(code):
        var = m.group(1)
        binder_span = (m.start(1), m.end(1))

        header_end = code.find(")", m.end())
        if header_end == -1:
            continue

        brace_open = code.find("{", header_end)
        if brace_open == -1:
            continue

        brace_close = java_find_matching_brace(code, brace_open)
        if brace_close == -1:
            continue

        loops.append({
            "var": var,
            "binder_span": binder_span,
            "loop_start": m.start(),
            "body_start": brace_open + 1,
            "body_end": brace_close,
        })

    return loops

def _java_in_ranges(pos, ranges):
    return any(s <= pos < e for s, e in ranges)

def find_iterator_and_body_use_spans_java(code):
    try:
        loops = java_find_loop_blocks(code)
    except Exception:
        return []
    spans = []

    for lp in loops:
        var = lp["var"]
        spans.append(lp["binder_span"])

        shadow_ranges = []
        for child in loops:
            if child is lp:
                continue
            if (
                child["loop_start"] >= lp["body_start"]
                and child["loop_start"] < lp["body_end"]
                and child["var"] == var
            ):
                shadow_ranges.append((child["loop_start"], child["body_end"]))

        pat = re.compile(rf"\b{re.escape(var)}\b")
        body_text = code[lp["body_start"]:lp["body_end"]]

        for m in pat.finditer(body_text):
            abs_s = lp["body_start"] + m.start()
            abs_e = lp["body_start"] + m.end()
            if not _java_in_ranges(abs_s, shadow_ranges):
                spans.append((abs_s, abs_e))

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
            spans = find_iterator_and_body_use_spans_py(code)
            if len(spans) > 0:
                py_snippets.append(code)

    py_df = pd.DataFrame({"code": py_snippets})
    py_out = "./data/python_iterator_snippets.csv"
    py_df.to_csv(py_out, index=False)
    print(f"Saved {len(py_snippets)} Python iterator snippets to {py_out}")

    # 2. Process Java
    print("Processing Java snippets...")
    java_snippets = []
    for code in df["Java"]:
        if pd.notna(code) and isinstance(code, str):
            spans = find_iterator_and_body_use_spans_java(code)
            if len(spans) > 0:
                java_snippets.append(code)

    java_df = pd.DataFrame({"code": java_snippets})
    java_out = "./data/java_iterator_snippets.csv"
    java_df.to_csv(java_out, index=False)
    print(f"Saved {len(java_snippets)} Java iterator snippets to {java_out}")

if __name__ == "__main__":
    main()
