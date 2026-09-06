import json5
import netrc
import os
from pathlib import Path
from dataclasses import dataclass, field, fields
from typing import Mapping

import numpy as np
import torch
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from torch.utils.data import DataLoader

from utils.config import get_dataset_config_by_split_from_dataset_config
from mqar import MqarDimensions
from mqar.dataloaders import get_mqar_dynamic_dataloaders_by_split
from utils.common import results_dir, test_results_dir


@dataclass
class GridAxis:
    axis: np.ndarray
    name: str

@dataclass
class GridAxes:
    x: GridAxis
    y: GridAxis

@dataclass(frozen=True)
class GridConstants:
    data: Mapping[str, int]
    aliases: Mapping[str, str] = field(default_factory=lambda: {"N_facts": "Nf"})

    # constants hidden from names/titles when at their default value
    DEFAULT_SKIP = (("N_k", 1), ("N_q", 1))

    def as_dict(self) -> dict[str, int]:
        return dict(self.data)

    def _named_items(self):
        skip = dict(self.DEFAULT_SKIP)
        return [(k, v) for k, v in self.data.items() if not (k in skip and v == skip[k])]

    def _key(self, k: str) -> str:
        return self.aliases.get(k, k)

    def _field_name(self, k, v, safe=False) -> str:
        if isinstance(v, int):
            s = f"{self._key(k)}={int(v):d}"
        elif v is None:
            s = f"{self._key(k)}"
        else:
            raise TypeError
        if safe:
            s = s.replace("=", "")
        return s

    @property
    def name(self) -> str:
        return ", ".join(self._field_name(k, v) for k, v in self._named_items())

    @property
    def safe_name(self) -> str:
        return "_".join(self._field_name(k, v, safe=True) for k, v in self._named_items())


def _wandb_api_key_available() -> bool:
    if os.environ.get("WANDB_API_KEY"):
        return True
    netrc_path = os.environ.get("NETRC") or os.path.expanduser("~/.netrc")
    try:
        authenticators = netrc.netrc(netrc_path).authenticators("api.wandb.ai")
    except (FileNotFoundError, netrc.NetrcParseError):
        return False
    return authenticators is not None


def _ensure_wandb_authenticated() -> None:
    # checked directly (WANDB_API_KEY / ~/.netrc) instead of calling wandb.login(),
    # since wandb.login() prompts interactively whenever a tty is attached (even if
    # nobody can answer it, e.g. `docker run -d -it`), which hangs instead of failing
    if not _wandb_api_key_available():
        raise RuntimeError(
            "wandb.activate is true in the run config, but no wandb API key is configured "
            "(checked WANDB_API_KEY env var and ~/.netrc). "
            "Run `.venv/bin/wandb login --relogin` once before launching the grid run, "
            "or set wandb.activate to false in the config to disable logging."
        )


def initialize_grid_run(run_config: dict[str, Any], config_path: str | Path = None):

    if run_config["wandb"].get("activate", False):
        _ensure_wandb_authenticated()

    model_config = run_config["model"]
    grid_config = run_config["grid"]
    run_config.setdefault("io", {})  # configs need not carry the io section

    # time
    time_zone_name = run_config["io"].get("time_zone", None)
    time_zone = ZoneInfo(time_zone_name) if time_zone_name else None  # None -> system local time
    time_now = datetime.now(time_zone)
    title_timestamp = time_now.strftime("%d.%m.%Y %H-%M-%S")
    dir_timestamp = time_now.strftime("%d_%m_%Y__%H_%M_%S")

    model_class = model_config['class']
    model_variant = model_config['variant']

    # descriptors, matching the paper's terminology
    experiment_name = run_config['runtime']['experiment_name']
    task_type = 'AR' if run_config.get('dataset', {}).get('kwargs', {}).get('reduce_to_AR', False) else 'MQAR'
    paper_variant = 'full' if model_class == 'mamba_ssm' else 'linear'
    trained_str = 'trained' if run_config['runtime'].get('should_train', True) else 'non-trained'

    # grid constants
    grid_axes, grid_constants = get_axes_from_grid_config(grid_config)

    # names
    grid_axes_name = f"{grid_axes.x.name}, {grid_axes.y.name}"
    grid_axes_safe_name = f"{grid_axes.x.name}_{grid_axes.y.name}"

    # wandb rejects project names longer than 128 characters: keep it compact
    # (paper_variant already encodes the model class; Λ instead of Lambda), and
    # as a last resort elide the axes/constants tail
    MAX_WANDB_PROJECT_NAME_LEN = 128
    name_prefix = f"{experiment_name} | {task_type}, {paper_variant}, {trained_str} | "
    name_suffix = f" | {title_timestamp}"
    name_middle = f"{grid_axes_name}, {grid_constants.name}".replace('Lambda', 'Λ')
    budget = MAX_WANDB_PROJECT_NAME_LEN - len(name_prefix) - len(name_suffix)
    if len(name_middle) > budget:
        name_middle = name_middle[:max(budget - 3, 0)] + '...'
    grid_run_name = (name_prefix + name_middle + name_suffix)[:MAX_WANDB_PROJECT_NAME_LEN]

    grid_run_dir_name = \
        (f"{model_class}_{model_variant}__"
         f"{grid_axes_safe_name}_{grid_constants.safe_name}__"
         f"{dir_timestamp}")

    # dirs
    is_test = run_config["runtime"].get("is_test", False)
    results_dir_to_use = results_dir if not is_test else test_results_dir
    run_results_dir = results_dir_to_use / experiment_name / grid_run_dir_name
    run_config["io"]["run_results_dir"] = str(run_results_dir)

    # make dirs
    run_results_dir.mkdir(parents=True, exist_ok=True)

    # save the run config: a byte-identical copy of the source config file, so the
    # archived artifact matches config/ exactly (runtime-filled fields such as
    # runtime.experiment_name and io.run_results_dir live only in memory); the
    # json5 dump is the fallback for programmatically-built configs
    saved_config = run_results_dir / "run_config.json5"
    if config_path is not None:
        saved_config.write_bytes(Path(config_path).read_bytes())
    else:
        saved_config.write_text(json5.dumps(run_config, indent=2))

    print(f"\n\nstarting grid run:\n{grid_run_name}\n")
    print(f"results are saved to: {run_results_dir}\n")

    return grid_run_name


def prepare_grid_pairs_to_iterate(
        grid_axes: GridAxes,
        grid_constants: GridConstants,
        grid_options: dict[str, Any]
) -> tuple[list[tuple[int, int]], MqarDimensions]:

    dims = MqarDimensions(**grid_constants.as_dict())

    x_axis = grid_axes.x.axis
    y_axis = grid_axes.y.axis

    xy_pairs: list[tuple[int, int]] = []

    for x in x_axis:
        for y in y_axis:
            xy_pairs.append((x, y))

    return xy_pairs, dims


def get_parallel_settings(run_config: dict[str, Any]):

    parallel_config = run_config.get("parallel", None)
    is_wandb_activated = run_config["wandb"].get("activate", False)

    parallel_enabled = parallel_config is not None

    if parallel_enabled:
        num_processes_per_device = int(parallel_config.get("num_processes_per_device", 1))
        num_cpu_threads_per_process = int(parallel_config.get("num_cpu_threads_per_process", 1))
        if num_cpu_threads_per_process > 1:
            assert not is_wandb_activated, \
                (f"wandb is not supported with multi-threading; "
                 f"set num_cpu_threads_per_process to 1 (currently {num_cpu_threads_per_process})")
    else:
        num_processes_per_device = 1
        num_cpu_threads_per_process = 1

    # choose devices
    cuda_available = _cuda_available()
    if parallel_enabled:
        req = parallel_config.get("devices_to_use")  # e.g. [0,1,2], ["cuda:0"], None, or ["cpu"]
        if not req:  # auto: all GPUs or CPU
            if cuda_available:
                used_devices = [_select_device(i) for i in range(torch.cuda.device_count())]
            else:
                print("no GPU device available; using CPU instead")
                used_devices = ["cpu"]
        else:
            if not cuda_available and not all(_is_explicit_cpu_token(d) for d in req):
                print("no GPU device available; using CPU instead")
            used_devices = [_select_device(d) for d in req] or ["cpu"]
            used_devices = list(dict.fromkeys(used_devices))  # dedupe; drop if you want multiple CPU workers
    else:  # sequential
        device_token = run_config.get("runtime", {}).get("device")
        if not cuda_available and not _is_explicit_cpu_token(device_token):
            print("no GPU device available; using CPU instead")
        used_devices = [_select_device(device_token)]

    return used_devices, num_processes_per_device, num_cpu_threads_per_process


def _cuda_available() -> bool:
    try:
        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    except Exception:
        return False


def _is_explicit_cpu_token(token) -> bool:
    return isinstance(token, str) and token.strip().lower() == "cpu"


def _select_device(token=None) -> str:
    """
    Normalize a device token to 'cpu' or 'cuda:<idx>'.
    Accepts: None, int, '3', 'cuda:3', 'cpu'.
    Never raises; falls back to 'cpu' if CUDA is unusable/out-of-range.
    """

    if token is None:
        return "cuda:0" if _cuda_available() else "cpu"

    s = str(token).strip().lower()
    if s == "cpu":
        return "cpu"

    if s.isdigit():
        idx = int(s)
    elif s.startswith("cuda:"):
        try:
            idx = int(s.split(":", 1)[1])
        except Exception:
            return "cpu"
    else:
        return "cpu"

    if _cuda_available() and 0 <= idx < torch.cuda.device_count():
        return f"cuda:{idx}"
    return "cpu"


def get_axes_from_grid_config(grid_config: dict[str, Any]) -> tuple[GridAxes, GridConstants]:

    x_name = grid_config['x_axis']
    y_name = grid_config['y_axis']

    x_axis = eval(grid_config[x_name]).astype(int)
    y_axis = eval(grid_config[y_name]).astype(int)

    x = GridAxis(name=x_name, axis=x_axis)
    y = GridAxis(name=y_name, axis=y_axis)

    dim_names = [f.name for f in fields(MqarDimensions)]
    axes_names = [x_name, y_name]

    constants = {name: value for name, value in grid_config.items() if ((name in dim_names) and (name not in axes_names))}

    grid_constants = GridConstants(constants)
    grid_axes = GridAxes(x=x, y=y)

    return grid_axes, grid_constants


def build_dataloaders(dims: MqarDimensions, run_config: dict[str, Any]) -> dict[str, DataLoader]:

    dataloaders_seed = run_config['runtime']['seed']

    # build dataloaders *inside* the child (since they are dynamic)
    dataset_config = run_config['dataset']
    train_config = run_config['training']

    # fill dims
    dataset_config['V'] = dims.V
    dataset_config['L'] = dims.L
    dataset_config['N_facts'] = dims.N_facts
    # N_k / N_q as grid dims override the static dataset kwargs (repeated keys/queries)
    if getattr(dims, 'N_k', None):
        dataset_config['kwargs']['num_key_repeats'] = int(dims.N_k)
    if getattr(dims, 'N_q', None):
        dataset_config['kwargs']['num_query_repeats'] = int(dims.N_q)

    # optional: determine train set size
    if dataset_config['split_size'].get('train', None) is None:  # by default, scale by training number of steps
        train_split_size = train_config['max_num_steps'] * dataset_config['batch_size']
        dataset_config['split_size']['train'] = train_split_size

    # prepare dataloaders
    dataset_config_by_split = get_dataset_config_by_split_from_dataset_config(dataset_config, seed=dataloaders_seed)
    dataloaders = get_mqar_dynamic_dataloaders_by_split(dataset_config_by_split)

    return dataloaders
