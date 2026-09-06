"""Regression test for the hung-zombie failure mode: a worker that dies mid-run
must surface as a RuntimeError in execute_grid_run, not hang the parent forever
waiting on a task that will never report.

The grid runs in a subprocess (so a regression to the old hang cannot hang the
suite - it just times out and fails); one of its worker processes is SIGKILLed
mid-training, and the runner must exit promptly with the died-mid-run error.
"""
import os
import shutil
import signal
import subprocess
import sys
import time

import pytest

from utils.common import project_dir, test_dir, test_results_dir

RUNNER_CODE = """
import json5, sys
from pathlib import Path
from experiments.grid_runs import execute_grid_run
from experiments.grid_runs_utils import initialize_grid_run

config_path = Path(sys.argv[1])
run_config = json5.loads(config_path.read_text())
run_config['parallel']['devices_to_use'] = [sys.argv[2]]
run_config['training']['max_num_steps'] = 2000  # long enough to be killed mid-task

grid_run_name = initialize_grid_run(run_config)
execute_grid_run(grid_run_name, run_config)
print('COMPLETED-NORMALLY')
"""


def _child_pids(pid: int) -> list[int]:
    # portable across Linux/macOS ps (GNU '--ppid' is Linux-only)
    out = subprocess.run(['ps', '-eo', 'pid=,ppid='], capture_output=True, text=True).stdout
    return [int(fields[0]) for fields in (line.split() for line in out.splitlines())
            if len(fields) == 2 and int(fields[1]) == pid]


def _cmdline(pid: int) -> str:
    # portable across Linux/macOS ps (no '/proc' on macOS)
    return subprocess.run(['ps', '-o', 'command=', '-p', str(pid)],
                          capture_output=True, text=True).stdout


@pytest.fixture()
def clean_experiment_dir():
    experiment_dir = test_results_dir / 'micro_grid'
    yield experiment_dir
    shutil.rmtree(experiment_dir, ignore_errors=True)


@pytest.mark.cpu_ok
def test_killed_worker_raises_instead_of_hanging(training_device, clean_experiment_dir):
    config_path = test_dir / 'fixtures' / 'micro_grid.json5'
    env = dict(os.environ, PYTHONPATH=str(project_dir / 'src'))

    runner = subprocess.Popen(
        [sys.executable, '-c', RUNNER_CODE, str(config_path), training_device],
        env=env, cwd=project_dir, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        # wait for the logger + workers to spawn (workers are launched last, so
        # the highest child pid is a worker), then kill one worker mid-task
        deadline = time.time() + 90
        workers = []
        while time.time() < deadline and len(workers) < 3:  # logger + 2 workers
            workers = _child_pids(runner.pid)
            time.sleep(0.5)
        assert len(workers) >= 3, f'worker processes never appeared: {workers}'
        time.sleep(5)  # let training actually start
        os.kill(max(workers), signal.SIGKILL)

        # the fixed runner must exit promptly with the died-mid-run error;
        # the old code would hang here forever (bounded by the timeout)
        stdout, stderr = runner.communicate(timeout=90)
    finally:
        if runner.poll() is None:
            runner.kill()
            runner.communicate()

    assert runner.returncode != 0, f'runner unexpectedly succeeded: {stdout}'
    assert 'died mid-run' in stderr, f'expected the died-mid-run RuntimeError, got:\n{stderr[-2000:]}'
    assert 'COMPLETED-NORMALLY' not in stdout


@pytest.mark.cpu_ok
def test_logger_death_does_not_hang_or_fail_grid(training_device, clean_experiment_dir):
    """The logger is telemetry, not control flow: killing it mid-run must not
    hang the workers (its queue is unbounded) or fail the grid - the run
    completes, only the saved accuracy grids stop updating."""
    config_path = test_dir / 'fixtures' / 'micro_grid.json5'
    env = dict(os.environ, PYTHONPATH=str(project_dir / 'src'))
    fast_runner = RUNNER_CODE.replace("max_num_steps'] = 2000", "max_num_steps'] = 60")

    runner = subprocess.Popen(
        [sys.executable, '-c', fast_runner, str(config_path), training_device],
        env=env, cwd=project_dir, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        # the logger is the first spawn child (workers launch after it); skip the
        # multiprocessing resource_tracker, which is not a spawn_main child
        deadline = time.time() + 90
        spawn_children = []
        while time.time() < deadline and len(spawn_children) < 3:
            spawn_children = [
                pid for pid in _child_pids(runner.pid)
                if 'resource_tracker' not in _cmdline(pid)
            ]
            time.sleep(0.5)
        assert len(spawn_children) >= 3, f'logger + workers never appeared: {spawn_children}'
        os.kill(min(spawn_children), signal.SIGKILL)  # the logger

        stdout, stderr = runner.communicate(timeout=240)
    finally:
        if runner.poll() is None:
            runner.kill()
            runner.communicate()

    assert runner.returncode == 0, f'grid must survive a dead logger:\n{stderr[-2000:]}'
    assert 'COMPLETED-NORMALLY' in stdout
