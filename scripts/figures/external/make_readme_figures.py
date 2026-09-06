#!/usr/bin/env python
"""Compile the standalone readme_fig<N>.tex files in this directory into README figure
screenshots (figures/external/readme_fig<N>.png).

Each readme_fig<N>.tex replicates the corresponding paper figure's layout (same panel
PNGs, same tabular structure) with a shortened, cross-reference-free caption.
Rerun after replacing the underlying figure PNGs (they are included from
`paper/figures/...` via \\graphicspath) to refresh the README images.

Requires `pdflatex` on PATH (any standard LaTeX distribution, e.g. TeX Live)
and `pymupdf` for the PDF->PNG render.

    python scripts/figures/external/make_readme_figures.py [--figs 1 2 3 4] [--dpi 200]
"""

import argparse
import shutil
import subprocess
from pathlib import Path

import pymupdf

HERE = Path(__file__).parent
OUT_DIR = HERE.parents[2] / "figures" / "external"


def find_pdflatex() -> str:
    pdflatex = shutil.which("pdflatex")
    if pdflatex is None:
        raise SystemExit("pdflatex not found: install a LaTeX distribution (e.g. TeX Live)")
    return pdflatex


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figs", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--dpi", type=int, default=380)
    args = ap.parse_args()

    pdflatex = find_pdflatex()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for fig in args.figs:
        tex = HERE / f"readme_fig{fig}.tex"
        subprocess.run(
            [pdflatex, "-interaction=nonstopmode", "-halt-on-error", tex.name],
            check=True, cwd=HERE, stdout=subprocess.DEVNULL,
        )
        pdf = HERE / f"readme_fig{fig}.pdf"
        page = pymupdf.open(pdf)[0]
        zoom = args.dpi / 72
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
        out = OUT_DIR / f"readme_fig{fig}.png"
        pix.save(out)
        for ext in (".pdf", ".aux", ".log", ".out"):
            (HERE / f"readme_fig{fig}{ext}").unlink(missing_ok=True)
        print(f"{tex.name} -> {out.relative_to(HERE.parents[2])}  {pix.width}x{pix.height}")


if __name__ == "__main__":
    main()
