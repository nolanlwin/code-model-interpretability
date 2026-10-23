#!/usr/bin/env python3
"""Build a self-contained arXiv source bundle for one paper directory.

arXiv compiles from a flat upload with no access to the repository tree, so
figures referenced as ``../../results/...`` have to travel with the source and
be rewritten to a local path. arXiv also does not run BibTeX for every
submission, so the bundle ships the committed ``main.bbl``.

The bundle is compiled in a scratch directory before it is written out, and
the page count is compared against the paper's own PDF. A bundle that does
not reproduce the paper is an error, not a warning.

The archive format follows the output extension. arXiv accepts both .zip
and .tar.gz.

Usage:
    python3 scripts/make_arxiv_bundle.py paper/lp4fm_short out/lp4fm_arxiv.zip
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Matches the path inside \includegraphics{...}, including the \detokenize{...}
# wrapper the appendix generator emits for paths with underscores.
GRAPHICS = re.compile(r"(\\includegraphics(?:\[[^\]]*\])?\{)(\\detokenize\{)?([^}]+)")


def external_figures(tex: str) -> set[str]:
    return {
        m.group(3) for m in GRAPHICS.finditer(tex) if m.group(3).startswith("../")
    }


def rewrite(tex: str) -> str:
    """Point every out-of-tree figure at the bundle's flat figures/ directory."""

    def sub(m: re.Match[str]) -> str:
        path = m.group(3)
        if not path.startswith("../"):
            return m.group(0)
        return f"{m.group(1)}{m.group(2) or ''}figures/{Path(path).name}"

    return GRAPHICS.sub(sub, tex)


def page_count(pdf: Path) -> int:
    import pymupdf

    with pymupdf.open(pdf) as doc:
        return doc.page_count


def build(paper: Path, out: Path) -> int:
    paper = paper.resolve()
    tex_files = sorted(paper.glob("*.tex"))
    if not (paper / "main.tex").exists():
        sys.exit(f"{paper} has no main.tex")
    bbl = paper / "main.bbl"
    if not bbl.exists():
        sys.exit(f"{bbl} is missing. Build the paper first so arXiv gets a .bbl")

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "bundle"
        (stage / "figures").mkdir(parents=True)

        copied: set[str] = set()
        for tex in tex_files:
            body = tex.read_text()
            for rel in external_figures(body):
                src = (paper / rel).resolve()
                if not src.exists():
                    sys.exit(f"{tex.name} references a missing figure: {rel}")
                shutil.copy2(src, stage / "figures" / src.name)
                copied.add(src.name)
            (stage / tex.name).write_text(rewrite(body))

        for extra in (bbl, paper / "neurips_2026.sty"):
            if extra.exists():
                shutil.copy2(extra, stage / extra.name)
        if (paper / "figures").is_dir():
            for fig in (paper / "figures").iterdir():
                if fig.is_file():
                    shutil.copy2(fig, stage / "figures" / fig.name)

        # arXiv runs pdflatex, not latexmk with a bibtex pass, so prove the
        # bundle builds the same way arXiv will.
        for _ in range(3):
            proc = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
                cwd=stage,
                capture_output=True,
                text=True,
            )
        if proc.returncode != 0:
            tail = "\n".join(proc.stdout.splitlines()[-25:])
            sys.exit(f"the bundle does not compile:\n{tail}")

        # A hyperlink split across a page break aborts pdfTeX part-way and
        # leaves a truncated PDF behind. The page-count check below catches
        # that too, but only after a confusing failure, so name it here.
        if "pdfendlink" in proc.stdout:
            sys.exit(
                "a hyperlink straddles a page break, so pdfTeX wrote a "
                "truncated PDF. Keep the offending citation or URL on one "
                "page, for example with \\Needspace before its paragraph."
            )

        expected = page_count(paper / "main.pdf")

        out.parent.mkdir(parents=True, exist_ok=True)
        # Only the built main.pdf and the LaTeX by-products are dropped. PDF
        # figures must travel, so the exclusion cannot be by suffix alone.
        byproducts = {".aux", ".log", ".out", ".fls", ".fdb_latexmk", ".blg"}
        payload = []
        for item in sorted(stage.rglob("*")):
            if not item.is_file() or item.suffix in byproducts:
                continue
            if item.parent == stage and item.name == "main.pdf":
                continue
            payload.append((item, str(item.relative_to(stage))))

        as_zip = out.suffix.lower() == ".zip"
        if as_zip:
            with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
                for item, arcname in payload:
                    zf.write(item, arcname)
        else:
            with tarfile.open(out, "w:gz") as tar:
                for item, arcname in payload:
                    tar.add(item, arcname=arcname)

        # Compile what actually ships, not what was staged. An earlier version
        # checked the staging directory and so missed a figure the tar step
        # was dropping.
        check = Path(tmp) / "unpacked"
        check.mkdir()
        if as_zip:
            with zipfile.ZipFile(out) as zf:
                zf.extractall(check)
        else:
            with tarfile.open(out) as tar:
                tar.extractall(check)
        for _ in range(3):
            proc = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
                cwd=check,
                capture_output=True,
                text=True,
            )
        if proc.returncode != 0 or not (check / "main.pdf").exists():
            out.unlink(missing_ok=True)
            tail = "\n".join(proc.stdout.splitlines()[-25:])
            sys.exit(f"the packed bundle does not compile:\n{tail}")
        built = page_count(check / "main.pdf")
        if built != expected:
            out.unlink(missing_ok=True)
            sys.exit(
                f"the packed bundle builds {built} pages but the paper has {expected}"
            )

        print(f"{out}  ({built} pages, {len(copied)} figures pulled in from results/)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paper", type=Path, help="paper directory, e.g. paper/lp4fm_short")
    ap.add_argument("out", type=Path, help="output .zip or .tar.gz path")
    args = ap.parse_args()
    return build(args.paper, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
