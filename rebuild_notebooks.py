"""Rebuild ipynb files in a simpler format and export .py copies."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def export_py(cells, out_path: Path, title: str) -> None:
    lines = [f'"""{title}"""\n']
    for cell in cells:
        if cell["cell_type"] == "markdown":
            src = "".join(cell.get("source", []))
            lines.append("\n# " + src.replace("\n", "\n# ").strip() + "\n")
        elif cell["cell_type"] == "code":
            src = "".join(cell.get("source", []))
            if src.strip():
                lines.append("\n" + src.rstrip() + "\n")
    out_path.write_text("".join(lines), encoding="utf-8")


def rebuild_notebook(src_path: Path) -> None:
    nb = json.loads(src_path.read_text(encoding="utf-8"))
    cells = nb.get("cells", [])

    # Drop trailing empty code cells
    while cells and cells[-1]["cell_type"] == "code" and not "".join(cells[-1].get("source", [])).strip():
        cells.pop()

    clean = {
        "cells": [],
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
        "nbformat_minor": 4,
    }

    for cell in cells:
        new_cell = {
            "cell_type": cell["cell_type"],
            "metadata": {},
            "source": cell.get("source", []),
        }
        if cell["cell_type"] == "code":
            new_cell["execution_count"] = None
            new_cell["outputs"] = []
        clean["cells"].append(new_cell)

    src_path.write_text(json.dumps(clean, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    export_py(clean["cells"], src_path.with_suffix(".py"), src_path.stem)


if __name__ == "__main__":
    for name in ["python_probe.ipynb", "java_probe.ipynb", "python_ifelse_probe.ipynb", "java_ifelse_probe.ipynb", "cpp_class_struct_probe.ipynb", "csharp_class_struct_probe.ipynb"]:
        path = ROOT / name
        rebuild_notebook(path)
        print(f"rebuilt {name} -> {path.with_suffix('.py')}")
