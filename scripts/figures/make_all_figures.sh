#!/usr/bin/env bash
# Regenerates every paper figure (body + appendix) from the committed
# saved_runs_data/ bundles, into figures/ and figures/appendix/, then the
# README figure screenshots (external/make_readme_figures.py) into
# figures/external/ - that last step needs pdflatex and is skipped with a
# notice when it is not installed (the committed PNGs remain valid).
#
# Usage (from anywhere):
#     bash scripts/figures/make_all_figures.sh
#
# Verifies the exact figure inventory at the end: 18 body + 53 appendix
# (+ 4 external when pdflatex is available) PNGs, all (re)written by this run
# (totals pinned, in sync with test/system/dry/test_figure_scripts.py).

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-$PROJECT_DIR/.venv/bin/python}"

EXPECTED_BODY=17
EXPECTED_APPENDIX=54
EXPECTED_EXTERNAL=4

SCRIPTS=(
    make_grid_figures.py
    make_scaling_curve_figures.py
    make_mech_interp_figures.py
    appendix/make_large_vocab_figures.py
    appendix/make_repeats_figures.py
    appendix/make_seeds_analysis_figures.py
    appendix/make_circuit_ablation_figure.py
)

start_stamp="$(mktemp)"
trap 'rm -f "$start_stamp"' EXIT

for script in "${SCRIPTS[@]}"; do
    echo
    echo "=== scripts/figures/$script ==="
    "$PYTHON" "$PROJECT_DIR/scripts/figures/$script"
done

echo
echo "=== scripts/figures/external/make_readme_figures.py ==="
readme_ran=0
if command -v pdflatex >/dev/null 2>&1; then
    "$PYTHON" "$PROJECT_DIR/scripts/figures/external/make_readme_figures.py"
    readme_ran=1
else
    echo "skipped: pdflatex not installed - committed figures/external/readme_fig*.png left as is"
fi

count_pngs() {  # <dir> [-newer]
    local dir="$1" newer="${2:-}"
    if [[ -n "$newer" ]]; then
        find "$dir" -maxdepth 1 -name '*.png' -newer "$start_stamp" | wc -l
    else
        find "$dir" -maxdepth 1 -name '*.png' | wc -l
    fi
}

body_new=$(count_pngs "$PROJECT_DIR/figures" -newer)
appendix_new=$(count_pngs "$PROJECT_DIR/figures/appendix" -newer)
external_new=$(count_pngs "$PROJECT_DIR/figures/external" -newer)
body_total=$(count_pngs "$PROJECT_DIR/figures")
appendix_total=$(count_pngs "$PROJECT_DIR/figures/appendix")
external_total=$(count_pngs "$PROJECT_DIR/figures/external")

echo
echo "written this run: $body_new body + $appendix_new appendix + $external_new external = $((body_new + appendix_new + external_new))"
echo "figures/ now holds: $body_total body + $appendix_total appendix + $external_total external = $((body_total + appendix_total + external_total))"

expected_external_new=$((readme_ran ? EXPECTED_EXTERNAL : 0))
if [[ "$body_new" -ne "$EXPECTED_BODY" || "$appendix_new" -ne "$EXPECTED_APPENDIX" || "$external_new" -ne "$expected_external_new" ]]; then
    echo "ERROR: expected exactly $EXPECTED_BODY body + $EXPECTED_APPENDIX appendix + $expected_external_new external figures written" >&2
    exit 1
fi
if [[ "$body_total" -ne "$EXPECTED_BODY" || "$appendix_total" -ne "$EXPECTED_APPENDIX" || "$external_total" -ne "$EXPECTED_EXTERNAL" ]]; then
    echo "ERROR: figures/ holds stale extras beyond the $EXPECTED_BODY + $EXPECTED_APPENDIX + $EXPECTED_EXTERNAL inventory" >&2
    exit 1
fi

if [[ "$readme_ran" -eq 1 ]]; then
    echo "figure inventory OK: $EXPECTED_BODY body + $EXPECTED_APPENDIX appendix + $EXPECTED_EXTERNAL external"
else
    echo "figure inventory OK: $EXPECTED_BODY body + $EXPECTED_APPENDIX appendix (external skipped, $EXPECTED_EXTERNAL committed)"
fi
