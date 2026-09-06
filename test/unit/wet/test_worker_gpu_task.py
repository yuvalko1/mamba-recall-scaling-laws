"""Worker-level contract on a real GPU: _run_worker (called in-process, no
spawn) trains one tiny grid cell end-to-end on the wet-test device and reports
COMPLETED with a sane accuracy - the single-task version of the micro grid."""
import multiprocessing as mp

import shutil

import json5
import pytest

from experiments.grid_runs import GridPoint, Task, TaskStatus, Worker, _run_worker
from experiments.grid_runs_utils import get_axes_from_grid_config, initialize_grid_run
from utils.common import test_dir, test_results_dir

MICRO_GRID_CONFIG_PATH = test_dir / 'fixtures' / 'micro_grid.json5'


@pytest.fixture()
def clean_experiment_dir():
    yield
    shutil.rmtree(test_results_dir / 'micro_grid', ignore_errors=True)


@pytest.mark.cpu_ok
def test_single_gpu_task_completes_through_worker(training_device, clean_experiment_dir):
    run_config = json5.loads(MICRO_GRID_CONFIG_PATH.read_text())
    run_config['training']['max_num_steps'] = 20
    run_config['training']['evaluate_any_num_steps'] = 10
    run_config['parallel']['devices_to_use'] = [training_device]
    initialize_grid_run(run_config)  # fills io.run_results_dir (under test/results/)

    grid_axes, grid_constants = get_axes_from_grid_config(run_config['grid'])
    task = Task(x=GridPoint('D', 16), y=GridPoint('N', 8), constants=grid_constants, seed=0)

    ctx = mp.get_context('spawn')
    task_queue, result_queue = ctx.Queue(), ctx.Queue()
    logs_queue, progress_queue = ctx.Queue(), ctx.Queue()
    task_queue.put(task)
    task_queue.put(None)

    worker = Worker(device=training_device, num_cpu_threads=1)
    _run_worker(worker, task_queue, result_queue, logs_queue, progress_queue, run_config)

    task_id, status_int, result = result_queue.get(timeout=1)
    assert task_id == task.id
    assert TaskStatus(status_int) == TaskStatus.COMPLETED
    assert 0.0 <= result['accuracy'] <= 1.0
