"""Utility script to convert python_ifelse_probe.py and java_ifelse_probe.py to Jupyter Notebook (.ipynb) files."""
from pathlib import Path
import json

def convert_py_to_notebook(py_path: Path, nb_path: Path):
    if not py_path.exists():
        print(f"Error: {py_path} not found.")
        return
        
    content = py_path.read_text(encoding="utf-8")
    lines = content.splitlines()
    
    cells = []
    current_cell_type = None
    current_cell_lines = []
    
    for line in lines:
        # Check if it starts with "# " or is exactly "#"
        is_md_line = line.startswith("# ") or line == "#"
        
        if is_md_line:
            # Transition from code to markdown
            if current_cell_type == "code" and current_cell_lines:
                # Strip trailing blank lines from code cell
                while current_cell_lines and not current_cell_lines[-1].strip():
                    current_cell_lines.pop()
                # Strip leading blank lines
                while current_cell_lines and not current_cell_lines[0].strip():
                    current_cell_lines.pop(0)
                if current_cell_lines:
                    cells.append({
                        "cell_type": "code",
                        "metadata": {},
                        "source": [l + "\n" for l in current_cell_lines[:-1]] + [current_cell_lines[-1]],
                        "execution_count": None,
                        "outputs": []
                    })
                current_cell_lines = []
            
            current_cell_type = "markdown"
            # Strip the leading "# " or "#"
            md_content = line[2:] if line.startswith("# ") else ""
            current_cell_lines.append(md_content)
        else:
            # Transition from markdown to code
            if current_cell_type == "markdown" and current_cell_lines:
                # Strip trailing/leading empty lines
                while current_cell_lines and not current_cell_lines[-1].strip():
                    current_cell_lines.pop()
                while current_cell_lines and not current_cell_lines[0].strip():
                    current_cell_lines.pop(0)
                if current_cell_lines:
                    cells.append({
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": [l + "\n" for l in current_cell_lines[:-1]] + [current_cell_lines[-1]]
                    })
                current_cell_lines = []
            
            current_cell_type = "code"
            current_cell_lines.append(line)
            
    # Flush remaining
    if current_cell_lines:
        while current_cell_lines and not current_cell_lines[-1].strip():
            current_cell_lines.pop()
        while current_cell_lines and not current_cell_lines[0].strip():
            current_cell_lines.pop(0)
        if current_cell_lines:
            if current_cell_type == "code":
                cells.append({
                    "cell_type": "code",
                    "metadata": {},
                    "source": [l + "\n" for l in current_cell_lines[:-1]] + [current_cell_lines[-1]],
                    "execution_count": None,
                    "outputs": []
                })
            else:
                cells.append({
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [l + "\n" for l in current_cell_lines[:-1]] + [current_cell_lines[-1]]
                })
            
    # Package in notebook format
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3.13",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.13.13"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Created notebook: {nb_path}")

def main():
    root = Path(__file__).resolve().parent
    convert_py_to_notebook(root / "python_ifelse_probe.py", root / "python_ifelse_probe.ipynb")
    convert_py_to_notebook(root / "java_ifelse_probe.py", root / "java_ifelse_probe.ipynb")
    convert_py_to_notebook(root / "cpp_class_struct_probe.py", root / "cpp_class_struct_probe.ipynb")
    convert_py_to_notebook(root / "csharp_class_struct_probe.py", root / "csharp_class_struct_probe.ipynb")

if __name__ == "__main__":
    main()
