"""Utility script to filter consolidated_data.csv and prepare C++ and C# class/struct datasets."""
import os
import re
import pandas as pd

def find_matching_brace(code, open_idx):
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

def find_class_struct_spans(code):
    spans = []
    pattern = re.compile(r'\b(class|struct)\s+([A-Za-z_][A-Za-z0-9_]*)\b')
    for m in pattern.finditer(code):
        start_pos = m.start()
        search_start = m.end()
        brace_pos = -1
        invalid = False
        for idx in range(search_start, len(code)):
            char = code[idx]
            if char == '{':
                brace_pos = idx
                break
            if char in (';', '(', ')', '=', ','):
                invalid = True
                break
        
        if invalid or brace_pos == -1:
            continue
            
        matching_brace = find_matching_brace(code, brace_pos)
        if matching_brace == -1:
            continue
            
        end_pos = matching_brace + 1
        for idx in range(matching_brace + 1, len(code)):
            if code[idx].isspace():
                continue
            if code[idx] == ';':
                end_pos = idx + 1
                break
            break
            
        spans.append((start_pos, end_pos))
    return spans

def is_valid_snippet(code, max_lines=None):
    if not isinstance(code, str) or len(code.strip()) == 0:
        return False
    if code.count("{") != code.count("}"):
        return False
    if max_lines is not None and len(code.splitlines()) > max_lines:
        return False
    return True

def main():
    csv_path = "./data/consolidated_data.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    print("Reading consolidated_data.csv...")
    df = pd.read_csv(csv_path)

    # 1. Process C++
    print("Processing C++ snippets...")
    cpp_snippets = []
    for code in df["C++"]:
        if is_valid_snippet(code, max_lines=None):
            spans = find_class_struct_spans(code)
            if len(spans) > 0:
                cpp_snippets.append(code)

    cpp_df = pd.DataFrame({"code": cpp_snippets})
    cpp_out = "./data/cpp_class_struct_snippets.csv"
    cpp_df.to_csv(cpp_out, index=False)
    print(f"Saved {len(cpp_snippets)} C++ class/struct snippets to {cpp_out}")

    # 2. Process C#
    print("Processing C# snippets...")
    csharp_snippets = []
    for code in df["C#"]:
        if is_valid_snippet(code, max_lines=None):
            spans = find_class_struct_spans(code)
            if len(spans) > 0:
                csharp_snippets.append(code)

    csharp_df = pd.DataFrame({"code": csharp_snippets})
    csharp_out = "./data/csharp_class_struct_snippets.csv"
    csharp_df.to_csv(csharp_out, index=False)
    print(f"Saved {len(csharp_snippets)} C# class/struct snippets to {csharp_out}")

if __name__ == "__main__":
    main()
