"""Utility script to insert a programmatic pip installation cell in the Jupyter Notebooks."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

INSTALL_CELL = {
    "cell_type": "code",
    "metadata": {},
    "source": [
        "import sys\n",
        "import subprocess\n",
        "\n",
        "# Programmatically install dependencies if missing\n",
        "def install_dependencies():\n",
        "    required = {'numpy', 'pandas', 'torch', 'transformers', 'scikit-learn', 'matplotlib'}\n",
        "    try:\n",
        "        installed = {pkg.split('==')[0].lower() for pkg in subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']).decode().split()}\n",
        "    except Exception:\n",
        "        installed = set()\n",
        "    missing = required - installed\n",
        "    if missing:\n",
        "        print(f\"Installing missing dependencies: {missing}\")\n",
        "        subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])\n",
        "        print(\"All dependencies installed successfully.\")\n",
        "\n",
        "install_dependencies()\n"
    ],
    "execution_count": None,
    "outputs": []
}

def add_install_cell_to_nb(nb_path: Path):
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    cells = nb.get("cells", [])
    
    # Check if the install cell is already present to avoid duplicate insertions
    has_install = False
    for cell in cells:
        source_str = "".join(cell.get("source", []))
        if "install_dependencies" in source_str:
            has_install = True
            break
            
    if not has_install:
        insert_idx = 0
        for i, cell in enumerate(cells):
            if cell["cell_type"] == "code":
                source_str = "".join(cell.get("source", []))
                if "import " in source_str:
                    insert_idx = i
                    break
        cells.insert(insert_idx, INSTALL_CELL)
        nb["cells"] = cells
        nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Added install cell to {nb_path.name}")
    else:
        print(f"Install cell already present in {nb_path.name}")

if __name__ == "__main__":
    for name in ["python_probe.ipynb", "java_probe.ipynb", "python_ifelse_probe.ipynb", "java_ifelse_probe.ipynb"]:
        path = ROOT / name
        if path.exists():
            add_install_cell_to_nb(path)
