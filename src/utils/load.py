import glob
from pathlib import Path

import json5
import torch

from utils.config import get_model_from_config
from experiments.grid_runs import GridPoint
from experiments.grid_runs_utils import get_axes_from_grid_config, prepare_grid_pairs_to_iterate, build_dataloaders
from utils.common import results_dir


def load_model_and_dataloaders_from_saved_config(
        relative_run_dir: str,
        x_point: GridPoint, y_point: GridPoint,
):
    run_dir = results_dir / relative_run_dir

    run_config, dims, grid_axes, grid_constants = load_run_metadata(run_dir)

    dataloaders = load_dataloaders_from_saved_config(run_dir)

    model = load_model_from_saved_config(run_dir, x_point, y_point)

    return model, dataloaders, run_config, dims


def load_run_metadata(run_dir: Path):

    run_config_path = run_dir / 'run_config.json5'
    run_config = json5.loads(run_config_path.read_text())

    grid_config = run_config['grid']
    grid_options = run_config['grid_options']

    grid_axes, grid_constants = get_axes_from_grid_config(grid_config)
    xy_pairs, dims = prepare_grid_pairs_to_iterate(grid_axes, grid_constants, grid_options)

    return run_config, dims, grid_axes, grid_constants


def load_dataloaders_from_saved_config(run_dir: Path):

    run_config, dims, grid_axes, grid_constants = load_run_metadata(run_dir)

    dataset_config = run_config['dataset']
    dataset_config['V'] = dims.V

    dataloaders = build_dataloaders(dims=dims, run_config=run_config)

    return dataloaders


def load_model_from_saved_run(
        relative_run_name: str,
        x: int,
        y: int,
):
    relative_run_dir = results_dir / relative_run_name
    run_config, dims, grid_axes, grid_constants = load_run_metadata(relative_run_dir)
    grid_config = run_config['grid']

    x_name = grid_config['x_axis']
    y_name = grid_config['y_axis']

    x_point = GridPoint(name=x_name, value=x)
    y_point = GridPoint(name=y_name, value=y)

    # load
    model, dataloaders, run_config, dims = load_model_and_dataloaders_from_saved_config(
        relative_run_dir, x_point, y_point,
    )

    dims.__setattr__(x_name, x)
    dims.__setattr__(y_name, y)

    return model, dataloaders, run_config, dims


def load_model_from_saved_config(
        run_dir: Path,
        x_point: GridPoint,
        y_point: GridPoint,
        device: str = None,  # default: cuda:0 if available, else cpu
        models_subdir: str = 'best_models',  # or 'model_checkpoints'
        verbose: bool = False,
):
    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    run_config, dims, grid_axes, grid_constants = load_run_metadata(run_dir)


    assert grid_axes.x.name == x_point.name
    assert grid_axes.x.name == x_point.name

    setattr(dims, x_point.name, x_point.value)
    setattr(dims, y_point.name, y_point.value)

    model_config = run_config['model']

    # designed/non-trained model
    if not run_config['runtime']['should_train']:
        model = get_model_from_config(dims=dims, model_config=model_config)
        model.to(device)
        print("note: a non-trained model weights are not loaded but built from configuration")
        return model

    # trained model: do not set weights (they are loaded anyway)
    model_config['kwargs']['set_designed_mqar_weights'] = False

    prefix_for_loaded_model = f"{x_point.safe_name}_{y_point.safe_name}"

    matching_models = [
        Path(x).name for x in glob.glob(str(run_dir / models_subdir / '*'))
        if Path(x).name.startswith(prefix_for_loaded_model)]

    if not matching_models:
        raise FileNotFoundError(f"no match found for '{prefix_for_loaded_model}'")

    model_name = matching_models[0]

    saved_model_path = run_dir / models_subdir / model_name


    # load weights
    model = get_model_from_config(dims=dims, model_config=model_config)
    model.to(device)

    if verbose:
        print(f"loading weights from:\n{saved_model_path}\n{model_name}\n")
    checkpoint = torch.load(saved_model_path, map_location=device)
    model.load_state_dict(checkpoint)
    if verbose:
        print("model loaded")

    return model
