"""
Test-suite conventions
======================

Tests are organized along two axes, encoded in the directory layout::

    test/<level>/<wetness>/test_*.py

- level:   ``unit`` (single function/class) or ``system`` (end-to-end pipeline)
- wetness: ``dry`` (no GPU, no training, no wandb - runs anywhere in seconds)
           or ``wet`` (trains on a CUDA GPU; wandb stays disabled)

Markers (``dry``/``wet``/``unit``/``system``) are applied automatically from the
path, so ``pytest test/ -m "unit and dry"`` etc. work without per-file boilerplate.

A bare ``pytest`` routes itself: dry tests always run; wet tests run whenever a
usable CUDA GPU is found and are skipped (with the reason) otherwise - except wet
tests marked ``cpu_ok`` (the ones training the simplified linear model, which needs
no CUDA kernels), which run on CPU instead of skipping:

    pytest                       # everything the machine can run
    TEST_CUDA_DEVICE=7 pytest    # same, wet tests pinned to cuda:7
    pytest -m "not wet"          # dry only
    pytest -m wet                # wet only

The wet GPU is TEST_CUDA_DEVICE if set (validated against the machine), else
the preferred device (``PREFERRED_TEST_GPU``) when it exists with enough free
memory, else the CUDA device with the most free memory; it must have >= 5 GiB
headroom (so in-flight research runs are never disturbed).
"""
import os
import subprocess
import warnings
import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# keep every test hermetic: no wandb traffic, headless matplotlib.
# (note: "dry" still allows importing utils.config, which try-imports mamba_ssm's
# CUDA extension .so - a load, not a CUDA context init)
os.environ.setdefault('WANDB_MODE', 'disabled')
os.environ.setdefault('MPLBACKEND', 'Agg')

MIN_FREE_GPU_MEM_MIB = 5 * 1024
PREFERRED_TEST_GPU = 0  # default wet-test device; used when present with enough free memory


def pytest_configure(config):
    config.addinivalue_line('markers', 'dry: no GPU, no training, no wandb')
    config.addinivalue_line('markers', 'wet: requires a CUDA GPU and trains models')
    config.addinivalue_line('markers', 'unit: single function/class level')
    config.addinivalue_line('markers', 'system: end-to-end pipeline level')
    config.addinivalue_line(
        'markers', 'cpu_ok: wet test that trains the simplified linear model - '
                   'runs on CPU instead of skipping when no GPU is found')


def _wet_route() -> tuple[int | None, str]:
    """(cuda index, detail) when wet tests can run on this machine, else (None, why not).
    TEST_CUDA_DEVICE must name an existing device - a bad pin fails loudly."""
    forced = os.environ.get('TEST_CUDA_DEVICE')
    gpus = _query_gpus_by_free_memory()
    if forced is not None:
        indices = sorted(index for index, _ in gpus)
        if not forced.isdigit() or int(forced) not in indices:
            raise pytest.UsageError(
                f'TEST_CUDA_DEVICE={forced!r}, but the available CUDA devices are {indices}')
        return int(forced), f'wet tests pinned to cuda:{forced} (TEST_CUDA_DEVICE)'
    if not gpus:
        return None, 'no CUDA device found'
    free_by_index = dict(gpus)
    preferred_free = free_by_index.get(PREFERRED_TEST_GPU, 0)
    if preferred_free >= MIN_FREE_GPU_MEM_MIB:
        return PREFERRED_TEST_GPU, (f'wet tests get cuda:{PREFERRED_TEST_GPU} '
                                    f'(preferred, {preferred_free} MiB free)')
    index, free = gpus[0]
    if free < MIN_FREE_GPU_MEM_MIB:
        return None, (f'no CUDA device with >= {MIN_FREE_GPU_MEM_MIB} MiB free '
                      f'(freest: cuda:{index}, {free} MiB)')
    return index, (f'wet tests get cuda:{index} ({free} MiB free; '
                   f'preferred cuda:{PREFERRED_TEST_GPU} unavailable or busy)')


_wet_route_cache: tuple[int | None, str] | None = None


def _get_wet_route() -> tuple[int | None, str]:
    global _wet_route_cache
    if _wet_route_cache is None:
        _wet_route_cache = _wet_route()
    return _wet_route_cache


def _wandb_configured() -> bool:
    if os.environ.get('WANDB_API_KEY'):
        return True
    netrc_path = Path(os.environ.get('NETRC') or Path.home() / '.netrc')
    return netrc_path.is_file() and 'api.wandb.ai' in netrc_path.read_text()


@pytest.fixture(scope='session', autouse=True)
def warn_if_wandb_unconfigured():
    """wandb is force-disabled for every test, so an unconfigured wandb never fails
    the suite - but training runs do need it, so surface a warning."""
    if not _wandb_configured():
        warnings.warn('wandb is not configured (no WANDB_API_KEY / ~/.netrc entry): '
                      'tests run with wandb disabled, but training runs require `wandb login`')
    yield


def pytest_report_header(config):
    gpu, detail = _get_wet_route()
    n_gpus = len(_query_gpus_by_free_memory())
    found = f'{n_gpus} CUDA device(s) found' if n_gpus else 'no CUDA device found'
    lines = [f'gpu: {found} - {detail if gpu is not None else "wet tests will be skipped"}']
    if not _wandb_configured():
        lines.append('wandb: not configured - tests run with wandb disabled; training runs need `wandb login`')
    return lines


def pytest_report_collectionfinish(config, start_path, items):
    def summarize(selected):
        unit = sum(1 for item in selected if 'unit' in Path(str(item.fspath)).parts)
        return len(selected), unit, len(selected) - unit

    dry_items = [item for item in items if 'dry' in Path(str(item.fspath)).parts]
    wet_items = [item for item in items if 'wet' in Path(str(item.fspath)).parts]
    n_dry, dry_unit, dry_system = summarize(dry_items)

    gpu_found = _get_wet_route()[0] is not None
    running_wet = [item for item in wet_items if gpu_found or 'cpu_ok' in item.keywords]
    skipped_wet = [item for item in wet_items if item not in running_wet]

    lines = [f'running {n_dry} dry tests (including {dry_unit} unit, {dry_system} system)']
    if running_wet:
        n_run, run_unit, run_system = summarize(running_wet)
        on = 'GPU' if gpu_found else 'CPU (cpu_ok)'
        lines.append(f'running {n_run} wet tests on {on} (including {run_unit} unit, {run_system} system)')
    if skipped_wet:
        n_skip, skip_unit, skip_system = summarize(skipped_wet)
        lines.append(f'skipping {n_skip} wet tests - need a GPU, none found '
                     f'(including {skip_unit} unit, {skip_system} system)')
    return lines


def pytest_collection_modifyitems(config, items):
    wet_gpu, wet_detail = _get_wet_route()
    skip_wet = pytest.mark.skip(reason=f'wet test: {wet_detail}')
    for item in items:
        parts = Path(str(item.fspath)).parts
        for marker in ('unit', 'system', 'dry', 'wet'):
            if marker in parts:
                item.add_marker(getattr(pytest.mark, marker))
        if 'wet' in parts and wet_gpu is None and 'cpu_ok' not in item.keywords:
            item.add_marker(skip_wet)
    # run every dry test before the first wet one (stable within each group)
    items.sort(key=lambda item: 'wet' in Path(str(item.fspath)).parts)


# ---------- shared fixtures ----------

@pytest.fixture(autouse=True)
def clean_recorder():
    """utils.recorder keeps module-global state plus an env flag; leaked state
    changes model-code behavior across tests, so always reset it."""
    yield
    if 'utils.recorder' in sys.modules:
        sys.modules['utils.recorder'].empty()
    os.environ.pop('ENABLE_RECORDING', None)


@pytest.fixture()
def cpu_default_device(monkeypatch):
    """mamba_tiny puts A/D on _get_default_device() at construction time, which
    picks CUDA whenever available - mixed-device params on a GPU host. Dry model
    tests force the default to CPU."""
    import torch
    import mamba_tiny.model as mamba_tiny_model
    monkeypatch.setattr(mamba_tiny_model, '_get_default_device', lambda: torch.device('cpu'))


def _query_gpus_by_free_memory() -> list[tuple[int, int]]:
    """[(cuda_index, free_MiB)] sorted by free memory, descending; [] if no nvidia-smi."""
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
            text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        index, free = line.split(',')
        gpus.append((int(index), int(free)))
    return sorted(gpus, key=lambda pair: -pair[1])


@pytest.fixture(scope='session')
def free_gpu() -> int:
    """CUDA index for wet tests: TEST_CUDA_DEVICE if set (e.g. TEST_CUDA_DEVICE=7),
    otherwise the device with the most free memory (>= 5 GiB headroom)."""
    gpu, detail = _get_wet_route()
    if gpu is None:
        pytest.skip(detail)
    return gpu


@pytest.fixture(scope='session')
def training_device() -> str:
    """'cuda:<free_gpu>' when a usable GPU exists, else 'cpu' - never skips. For
    cpu_ok wet tests (the simplified linear model, which needs no CUDA kernels)."""
    gpu, _ = _get_wet_route()
    return f'cuda:{gpu}' if gpu is not None else 'cpu'
