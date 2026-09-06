#!/usr/bin/env bash
# Sets up a local pip venv for this project (the canonical environment).
#
# Works on both CUDA machines and CPU-only machines (e.g. a mac laptop):
# on a CUDA machine the full stack is installed and smoke-tested; without CUDA
# the GPU-only parts (causal-conv1d/mamba-ssm wheels) are skipped with a notice
# and everything else is set up identically.
#
# Installs, in order:
#   1. build-essential (gcc/g++) - needed by Triton to JIT-compile kernels used inside mamba_ssm
#      (Linux/apt only; skipped elsewhere)
#   2. a pinned torch version with a matching prebuilt causal-conv1d/mamba-ssm wheel. Installing
#      mamba-ssm normally (with deps) drags in unrelated newer-torch requirements (via its
#      tilelang/quack-kernels/apache-tvm-ffi dependencies) that silently upgrade torch and break
#      that ABI match, so torch is pinned explicitly here first.
#   3. requirements.txt (unchanged)
#   4. CUDA only: causal-conv1d + mamba-ssm prebuilt wheels, installed with --no-deps so their
#      unrelated transitive deps (tilelang/quack-kernels/apache-tvm-ffi) can't silently upgrade
#      torch and break ABI compatibility with the prebuilt .so files. This is the optional
#      CUDA-accelerated Mamba path (5 of 50 configs); if it fails to install (or there is no
#      CUDA), the pure-PyTorch "mamba_tiny" path used by the rest of the configs is unaffected.
#
# Usage: bash scripts/setup_venv.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv"

MAMBA_SSM_VERSION="2.3.2.post1"
CAUSAL_CONV1D_VERSION="1.6.2.post1"
PY_TAG="cp312"
TORCH_VERSION="2.10.0"
TORCH_TAG="torch2.10"  # must match TORCH_VERSION above

if ! command -v gcc >/dev/null 2>&1; then
    if command -v apt-get >/dev/null 2>&1; then
        echo "== installing build-essential (gcc/g++) =="
        sudo apt-get update -qq
        sudo apt-get install -y --no-install-recommends build-essential
    else
        echo "!! gcc not found and apt-get unavailable - skipping (only needed for the CUDA mamba_ssm path)."
    fi
fi

echo "== creating venv at $VENV_DIR =="
python3.12 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip

echo "== installing pinned torch (torch==$TORCH_VERSION) =="
pip install --no-cache-dir "torch==$TORCH_VERSION"

echo "== checking CUDA / GPU availability =="
HAS_CUDA="$(python -c 'import torch; print(1 if torch.cuda.is_available() else 0)')"

if [ "$HAS_CUDA" = "1" ]; then
    python - <<'PY'
import torch
print("torch:", torch.__version__, "cuda build:", torch.version.cuda)
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
print("GPU:", name, "compute capability:", cap)
x = torch.randn(1024, 1024, device="cuda")
(x @ x).sum().item()
print("GPU matmul smoke test OK")
PY
else
    python -c 'import torch; print("torch:", torch.__version__, "cuda build:", torch.version.cuda)'
    echo "!! No CUDA GPU detected - continuing with a CPU-only setup."
    echo "!! Training is not supported in this setup; figure scripts are functional."
fi

echo "== installing base requirements.txt (pinned via constraints.txt) =="
pip install --no-cache-dir -r "$ROOT_DIR/requirements.txt" -c "$ROOT_DIR/constraints.txt"

if [ "$HAS_CUDA" = "1" ]; then
    # torch.version.cuda determines which causal-conv1d/mamba-ssm wheel tag to use.
    CUDA_MAJOR="$(python -c 'import torch; print(torch.version.cuda.split(".")[0])')"
    CU_TAG="cu${CUDA_MAJOR}"
    echo "== detected CUDA major version: $CUDA_MAJOR (wheel tag: ${CU_TAG}${TORCH_TAG}) =="

    echo "== installing causal-conv1d + mamba-ssm (optional CUDA-accelerated Mamba, --no-deps) =="
    CAUSAL_CONV1D_WHEEL="causal_conv1d-${CAUSAL_CONV1D_VERSION}+${CU_TAG}${TORCH_TAG}cxx11abiTRUE-${PY_TAG}-${PY_TAG}-linux_x86_64.whl"
    MAMBA_SSM_WHEEL="mamba_ssm-${MAMBA_SSM_VERSION}+${CU_TAG}${TORCH_TAG}cxx11abiTRUE-${PY_TAG}-${PY_TAG}-linux_x86_64.whl"

    CAUSAL_CONV1D_URL="https://github.com/Dao-AILab/causal-conv1d/releases/download/v${CAUSAL_CONV1D_VERSION}/${CAUSAL_CONV1D_WHEEL}"
    MAMBA_SSM_URL="https://github.com/state-spaces/mamba/releases/download/v${MAMBA_SSM_VERSION}/${MAMBA_SSM_WHEEL}"

    set +e
    pip install --no-cache-dir --no-deps "$CAUSAL_CONV1D_URL" && pip install --no-cache-dir --no-deps "$MAMBA_SSM_URL"
    MAMBA_SSM_STATUS=$?
    set -e

    if [ "$MAMBA_SSM_STATUS" -ne 0 ]; then
        echo "!! causal-conv1d/mamba-ssm install failed - continuing without it."
        echo "!! configs with \"class\": \"mamba_ssm\" will not run, but the pure-PyTorch"
        echo "!! \"mamba_tiny\" path (used by most configs) is unaffected."
    else
        python - <<'PY'
import torch
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig
cfg = MambaConfig(d_model=64, n_layer=2, vocab_size=100)
model = MambaLMHeadModel(cfg, device="cuda", dtype=torch.float32)
x = torch.randint(0, 100, (2, 16), device="cuda")
model(x).logits.sum().backward()
print("mamba_ssm forward+backward smoke test OK")
PY
    fi
fi

echo
echo "== done =="
echo
echo "Activate the environment:"
echo "  source .venv/bin/activate"
echo
echo "Then, inside the .venv - "
echo
echo "Render figures with:"
echo "  python scripts/figures/make_grid_figures.py"
echo "  python scripts/figures/make_scaling_curve_figures.py"
echo "  python scripts/figures/make_mech_interp_figures.py"
echo
echo "Log in to Weights & Biases: (required for training and wet tests; you may skip otherwise)"
echo "  wandb login"
echo
echo "Run training with, for example: (only if host has available GPUs)"
echo "  python src/main.py -c config/MQAR__D_N__linear_trained__example.json5  # small example grid, linear Mamba model"
echo "  python src/main.py -c config/MQAR__D_N__full_trained__example.json5    # small example grid, full Mamba model"
echo "  python src/main.py -c config/MQAR__D_N__full_trained__regime_0.json5   # actual paper grid; long run. see more under config/*"
echo
echo "Run tests with: (note that on a CPU-only host, a few tests are skipped)"
echo "  pytest"
