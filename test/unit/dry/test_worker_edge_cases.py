"""Worker-level contracts of the grid runner (in-process, no spawn, no GPU):
every task outcome must be REPORTED on the result queue, and a failing task
must not kill the worker - the two invariants the parent's premature-death
check relies on (a worker that exits during collection is treated as a lost
task and aborts the grid)."""
import multiprocessing as mp
from queue import Empty

from experiments.grid_runs import GridPoint, Task, TaskStatus, Worker, _run_worker
from experiments.grid_runs_utils import GridConstants

CONSTANTS = GridConstants(data={'V': 64, 'L': 32, 'N_facts': 4, 'Lambda': 1, 'M': 1, 'N_k': 1, 'N_q': 1})


def _make_task(D=16, N=8, seed=0) -> Task:
    return Task(x=GridPoint('D', D), y=GridPoint('N', N), constants=CONSTANTS, seed=seed)


def _make_worker_env():
    ctx = mp.get_context('spawn')
    queues = {name: ctx.Queue() for name in ('task', 'result', 'logs', 'progress')}
    worker = Worker(device='cpu', num_cpu_threads=1)
    return worker, queues


def _run(worker, queues, run_config):
    _run_worker(worker, queues['task'], queues['result'], queues['logs'], queues['progress'], run_config)


def _theory_config() -> dict:
    return {
        'runtime': {'seed': 0},
        'training': {'print_non_tqdm_debug_messages': False, 'view_per_gridpoint_tqdm': True},
        'wandb': {},
        'model': {'class': 'mamba_theory', 'task_type': 'MQAR'},
    }


def test_worker_exits_cleanly_on_sentinel():
    worker, queues = _make_worker_env()
    queues['task'].put(None)

    _run(worker, queues, _theory_config())  # returns; a hang would time the test out

    try:
        leftover = queues['result'].get(timeout=0.2)
        raise AssertionError(f'no result expected for a sentinel-only queue, got {leftover}')
    except Empty:
        pass


def test_theory_task_reports_completed_with_logged_accuracy():
    worker, queues = _make_worker_env()
    task = _make_task()
    queues['task'].put(task)
    queues['task'].put(None)

    _run(worker, queues, _theory_config())

    task_id, status_int, result = queues['result'].get(timeout=1)
    assert task_id == task.id
    assert TaskStatus(status_int) == TaskStatus.COMPLETED
    assert 0.0 <= result['accuracy'] <= 1.0

    log_task_id, split, accuracy = queues['logs'].get(timeout=1)
    assert (log_task_id, split) == (task.id, 'val')
    assert accuracy == result['accuracy']


def test_failing_tasks_report_failed_and_worker_survives(capsys):
    """A task exception must be reported as FAILED - and must NOT kill the worker:
    both queued tasks fail, both get reported, and the worker still consumes its
    sentinel and returns (a worker suicide after the first failure would leave
    the second task unreported and abort the whole grid at the parent)."""
    worker, queues = _make_worker_env()
    tasks = [_make_task(D=16, N=8), _make_task(D=32, N=16)]
    for task in tasks:
        queues['task'].put(task)
    queues['task'].put(None)

    broken_config = _theory_config()
    del broken_config['model']['task_type']  # every task raises KeyError

    _run(worker, queues, broken_config)  # must return, not raise

    reported = {}
    for _ in tasks:
        task_id, status_int, result = queues['result'].get(timeout=1)
        reported[task_id] = (TaskStatus(status_int), result)
    assert reported == {t.id: (TaskStatus.FAILED, {}) for t in tasks}
    assert 'task_type' in capsys.readouterr().err  # the traceback was printed


def test_wandb_configured_detection(monkeypatch, tmp_path):
    """The suite must not fail without wandb credentials - conftest only warns.
    Pin the detection both ways."""
    import conftest

    monkeypatch.setenv('WANDB_API_KEY', 'k-something')
    assert conftest._wandb_configured()

    monkeypatch.delenv('WANDB_API_KEY', raising=False)
    monkeypatch.setenv('NETRC', str(tmp_path / 'netrc'))
    assert not conftest._wandb_configured()

    (tmp_path / 'netrc').write_text('machine api.wandb.ai\n  login user\n  password k\n')
    assert conftest._wandb_configured()
