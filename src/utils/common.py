import inspect
import os
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


# global paths

# project root dir
project_dir = Path(__file__).resolve().parents[2]  # shouldn't change, but adapt if required

# project dirs
config_dir = project_dir / "config"
results_dir = project_dir / "results"
figures_dir = project_dir / "figures"
figure_scripts_dir = project_dir / "scripts" / "figures"

# test dirs
test_dir = project_dir / "test"
test_results_dir = test_dir / "results"
test_config_dir = test_dir / "config"

# wandb artifact dirs (under the repo root, gitignored);
# overridable per run config via wandb.cache_dir / wandb.output_dir
wandb_cache_dir = project_dir / "wandb" / "cache"
wandb_output_dir = project_dir / "wandb" / "output"


def set_seed(seed=123, verbose=False):

    if verbose:
        caller = inspect.currentframe().f_back.f_code.co_name
        print(f"{caller}: setting {seed=}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)



def _named_arrays(*names, include_globals=True):
    # allow named_arrays(["A","B"]) or named_arrays(("A","B"))
    if len(names) == 1 and isinstance(names[0], Iterable) and not isinstance(names[0], (str, bytes)):
        names = tuple(names[0])

    # Start at the immediate caller, then walk up until we leave this module.
    frame = inspect.currentframe().f_back
    try:
        this_mod = __name__
        while frame and frame.f_globals.get('__name__') == this_mod:
            frame = frame.f_back

        scope = {}
        if frame:
            scope = dict(frame.f_locals)
            if include_globals:
                scope = {**frame.f_globals, **scope}

        # keep only string names present in scope
        return {k: scope[k] for k in names if isinstance(k, str) and k in scope}
    finally:
        del frame  # avoid reference cycles

def _print_shapes(**arrays):
    print("\nshapes:\n")
    for name, M in arrays.items():
        print(f"{name:<12}\t{getattr(M, 'shape', None)}")
    print("\n" + "-"*25)

def print_array_shapes(*names, include_globals=True):
    _print_shapes(**_named_arrays(*names, include_globals=include_globals))