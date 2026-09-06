import time
import traceback
from dataclasses import dataclass
from enum import IntEnum, auto
from pathlib import Path
from queue import Empty
from typing import Any

import os, copy, uuid, multiprocessing as mp

import numpy as np
import torch
from tqdm import tqdm

from utils.config import get_model_from_config
from experiments.grid_runs_utils import (
    prepare_grid_pairs_to_iterate, get_parallel_settings, get_axes_from_grid_config, GridConstants, build_dataloaders)
from experiments.wandb_grid import AccuracyGridMonitor
from mqar import MqarDimensions
from experiments.train import run_train_loop, send_logs
from theory.approximate_scaling_laws import predict_theoretical_recall_accuracy
from utils.common import set_seed


# ---------- classes ----------

@dataclass
class TaskStatus(IntEnum):
    COMPLETED = auto()
    FAILED = auto()


@dataclass(frozen=True)
class GridPoint:
    name: str
    value: int

    def as_dict(self) -> dict[str, int]:
        return {self.name: self.value}

    def __str__(self) -> str:
        return f"{self.name}={str(self.value).zfill(4)}"

    @property
    def safe_name(self):
        return str(self).replace("=", "_")


class Task:

    def __init__(
            self,
            x: GridPoint,
            y: GridPoint,
            constants: GridConstants,
            seed: int,
    ):
        self.x = x
        self.y = y
        self.id: str = str(uuid.uuid4())
        self.dims: MqarDimensions = MqarDimensions(**x.as_dict(), **y.as_dict(), **constants.as_dict())
        self.seed: int = seed

        self.name = f"{str(x)}, {str(y)}, {seed=}"


class Worker:

    def __init__(
            self,
            device: str,
            num_cpu_threads: int,
    ):
        self.id = uuid.uuid4()
        self.device = device
        self.num_cpu_threads = num_cpu_threads
        self.name = f"{self.device}-0x{self.id.hex[:4]}"


# ---------- parent: process scheduler ----------

def execute_grid_run(
        grid_run_name: str,
        run_config: dict[str, Any],
) -> dict[str, Any]:
    """
    Process-parallel grid runner using multiprocessing 'spawn'.
    - Multi-GPU, with per-device concurrency.
    - Each worker builds its own dataloaders.
    """

    grid_config = run_config["grid"]
    grid_options = run_config["grid_options"]
    parent_seed = run_config['runtime']['seed']

    set_seed(parent_seed)

    training_config = run_config.get('training', {})
    debug_prints = training_config.get('print_non_tqdm_debug_messages', False)
    view_per_gridpoint_tqdm = training_config.get('view_per_gridpoint_tqdm', True)

    # wandb
    if run_config['wandb']['activate']:
        run_config['wandb']['project_name'] = grid_run_name

    # -------- parallel settings --------

    used_devices, num_processes_per_device, num_cpu_threads_per_process = get_parallel_settings(run_config)
    num_devices = len(used_devices)

    # build device tokens with per-device concurrency
    workers_list: list[Worker] = []
    for _ in range(max(1, num_processes_per_device)):
        for device in used_devices:
            _worker = Worker(
                device=device,
                num_cpu_threads=num_cpu_threads_per_process,
            )
            workers_list.append(_worker)
    num_workers = len(workers_list)
    num_threads = int(num_workers * num_cpu_threads_per_process)

    print(f"using {num_devices} devices: {used_devices}")
    print(f"using {num_workers} workers ({num_processes_per_device} per device)")

    if num_cpu_threads_per_process > 1:
        assert not run_config["wandb"].get("activate", False), "wandb is not supported with multi-threading"
        os.environ.setdefault("WANDB_MODE", "disabled")
        print(f"using {num_threads} threads ({num_cpu_threads_per_process} per worker process)")

    print()

    # -------- multiprocessing setup --------

    # create queues (tasks, results and logs)
    start_method = "spawn"
    ctx = mp.get_context(start_method)
    task_queue: mp.Queue = ctx.Queue()
    result_queue: mp.Queue = ctx.Queue()
    logs_queue: mp.Queue = ctx.Queue()
    progress_queue: mp.Queue = ctx.Queue()  # per-batch training progress (grouped tqdm view)

    # seeds
    seeds_to_run = _get_seeds_to_run(seeds_config=run_config['seeds'])

    # set up grid/lines pairs
    grid_axes, grid_constants = get_axes_from_grid_config(grid_config)
    xy_pairs, dims = prepare_grid_pairs_to_iterate(grid_axes, grid_constants, grid_options)

    # optionally shuffle grid iteration order
    if grid_options.get('shuffle', False):
        rng = np.random.default_rng(parent_seed)  # reproducible
        xy_pairs = rng.permutation(xy_pairs).tolist()

    # build tasks
    tasks = []
    for seed in seeds_to_run:
        tasks_per_seed = []
        for (x, y) in xy_pairs:
            x_pt = GridPoint(name=grid_axes.x.name, value=int(x))
            y_pt = GridPoint(name=grid_axes.y.name, value=int(y))
            task = Task(x=x_pt, y=y_pt, constants=grid_constants, seed=seed)
            # optionally scale Nf with L:
            if grid_config.get('scale_Nf_with_L', False):
                task.dims.N_facts = int(task.dims.L / 4.)
            # optionally fix total state capacity Lambda*N = Lambda_N:
            if grid_config.get('scale_N_with_Lambda', False):
                task.dims.N = int(grid_config['Lambda_N'] / task.dims.Lambda)
            tasks_per_seed.append(task)
        tasks += tasks_per_seed
    tasks_by_id: dict[str, Task] = {t.id: t for t in tasks}

    # filter out infeasible grid points up front, so the root progress bar reflects
    # the actual number of grid points that will be trained (not the raw grid size)
    skip_flags = [_should_skip_grid_point(dims=t.dims) for t in tasks]
    tasks_pre_skipped = [t for t, skip in zip(tasks, skip_flags) if skip]
    tasks_to_run = [t for t, skip in zip(tasks, skip_flags) if not skip]
    print(f"{len(tasks_pre_skipped)} of {len(tasks)} grid points skipped (infeasible dims); "
          f"{len(tasks_to_run)} to train")
    if debug_prints:
        for t in tasks_pre_skipped:
            print(f"task {t.name} skipped")

    # enqueue tasks
    for t in tasks_to_run:
        task_queue.put(t)
    num_enqueued = len(tasks_to_run)

    # root grid-points progress, two display modes:
    #
    # - grouped view (view_per_gridpoint_tqdm: false): workers render nothing; they stream
    #   batch progress into progress_queue, and the parent (the only process writing to
    #   the tty) renders one pinned global bar plus one group-average bar per active wave
    #   of num_workers tasks. Single writer => no bar collisions, by construction.
    #
    # - per-worker view (view_per_gridpoint_tqdm: true): each worker animates its own per-task
    #   bar, and the parent prints the global counter as plain newline-terminated lines.
    #   Printed (never \r-animated) on purpose: a live bar here would just get overwritten
    #   by the workers' refreshes, and the leading "\n" guarantees each line starts fresh
    #   even if the cursor is mid-way through an unfinished worker bar line.
    grid_points_start_time = time.monotonic()

    def _print_grid_progress(n: int) -> None:
        print("\n" + tqdm.format_meter(
            n=n, total=num_enqueued,
            elapsed=time.monotonic() - grid_points_start_time,
            prefix="grid points", unit="pt",
        ), flush=True)

    # -------- multiprocessing start --------

    # launch a single background logger (listens to worker logs)
    monitor_ready = ctx.Event()
    logger_args = (logs_queue, tasks_by_id, run_config, monitor_ready)
    logger = ctx.Process(target=_run_logger, args=logger_args)
    print(flush=True)  # blank line before the wandb monitor's console block
    logger.start()

    # let the logger's wandb monitor finish initializing (and printing its login/run
    # lines) before drawing any bars, so those prints can't land inside the bar block
    monitor_ready.wait(timeout=120)
    print(flush=True)  # blank line after it

    if view_per_gridpoint_tqdm:
        grouped_view = None
        _print_grid_progress(0)
    else:
        grouped_view = GroupedGridProgress(
            tasks_to_run, group_size=num_workers,
            default_total=training_config.get('max_num_steps', 1),
        )

    # launch workers
    procs: list[mp.Process] = []

    for i, worker in enumerate(workers_list):
        if debug_prints:
            print(f"launching worker {i}: {worker.name}")

        worker_args = (worker, task_queue, result_queue, logs_queue, progress_queue, run_config)

        p = ctx.Process(target=_run_worker, args=worker_args)
        p.daemon = False
        p.start()

        procs.append(p)

        sleep_interval = 1.  # 0.25
        time.sleep(sleep_interval)  # optional

    # collect results; manage retries here
    num_received = 0

    try:
        while num_received < num_enqueued:

            if grouped_view is not None:
                grouped_view.drain_and_refresh(progress_queue)

            # wait for a result, with a timeout so worker liveness is re-checked:
            # a blocking get() would wait forever on a task whose worker died
            try:
                (task_id, status_int, result) = result_queue.get(timeout=0.5)
                num_received += 1  # count every finished task

            except Empty:
                # workers only exit after their post-collection sentinel, so ANY
                # exit here (exit code 0 included) is a premature death whose
                # in-flight task will never report - surface it instead of hanging
                for i, (p, w) in enumerate(zip(procs, workers_list)):
                    if p.exitcode is not None:
                        raise RuntimeError(
                            f"worker {i} ({w.name}) died mid-run (exit code {p.exitcode}) "
                            f"with {num_enqueued - num_received} task(s) still unreported")
                continue

            # get the task
            finished_task = tasks_by_id[task_id]
            if debug_prints:
                print(f"\n{finished_task.name}: completed with status {TaskStatus(status_int).name}", flush=True)
            if grouped_view is not None:
                grouped_view.record_completion(task_id)
            else:
                _print_grid_progress(num_received)

    except Exception:
        # don't leave surviving workers busy-waiting on the task queue (or the
        # logger blocked on its queue) behind a raised error
        for p in procs:
            if p.is_alive():
                p.terminate()
        if logger.is_alive():
            logger.terminate()
        raise

    if grouped_view is not None:
        grouped_view.close()

    # all tasks (including retries) accounted for; now release workers
    for _ in range(num_workers):
        task_queue.put(None)  # sentinel per worker

    # join children
    for i, (p, w) in enumerate(zip(procs, workers_list)):
        p.join()
        if p.exitcode not in (0, None):
            raise RuntimeError(f"worker {i} ({w.name}) exited with error: code {p.exitcode}")

    print("all done; closing...")
    time.sleep(5)

    # stop the logger
    logs_queue.put(None)
    print("done")

    logger.join()


def _get_seeds_to_run(seeds_config: dict[str, dict]) -> range:
    n_seeds = seeds_config.get('n_seeds', 1)
    start_seed = seeds_config.get('start_seed', 0)
    return range(start_seed, start_seed + n_seeds)


# ---------- parent: grouped progress view ----------

class GroupedGridProgress:
    """
    Single-writer terminal progress for a grid run (view_per_gridpoint_tqdm: false).

    Tasks are split, in enqueue order, into groups of `group_size` (= the number of
    workers, i.e. one wave of concurrency). The display is a CONSTANT small block of
    pinned rows, so it fits any terminal, and since no bar is ever created, closed,
    or moved mid-run, rows can never reshuffle or override each other:

        total grid points: 152/752 [...]
        completed groups:    7/38  [...]
        group 08/38:  1132.4/1500 [..., completed=14/20]   <- slot bar (relabeled)
        group 09/38:   204.2/1500 [..., completed=0/20]    <- slot bar (relabeled)
        group 10/38:      0.0/1500 [..., completed=0/20]   ...
        group 11/38:      0.0/1500 [..., completed=0/20]
        group 12/38:      0.0/1500 [..., completed=0/20]

    The slot bars are persistent and always display the lowest-numbered unfinished
    groups, upcoming ones included (a group bar shows the group-average trained
    batches, with not-yet-started members counted as 0); the window slides forward
    as groups finish, so slots go idle only near the very end. A periodic full
    redraw self-heals the block after foreign prints (e.g. the wandb monitor's
    startup lines).
    """

    _NUM_SLOTS = 5
    _FULL_REDRAW_INTERVAL = 5.0  # seconds; self-heal after foreign prints

    def __init__(
            self,
            tasks_to_run: list[Task],
            group_size: int,
            default_total: int = 1,
            refresh_interval: float = 0.25,
    ):
        group_size = max(1, int(group_size))
        self._num_groups = (len(tasks_to_run) + group_size - 1) // group_size

        self._group_of_task: dict[str, int] = {}
        self._group_members: dict[int, list[str]] = {}
        for i, task in enumerate(tasks_to_run):
            group = i // group_size
            self._group_of_task[task.id] = group
            self._group_members.setdefault(group, []).append(task.id)

        # per task: fraction trained (0..1) and total batches once known
        self._fraction: dict[str, float] = {t.id: 0.0 for t in tasks_to_run}
        self._total: dict[str, int | None] = {t.id: None for t in tasks_to_run}
        self._completed: set[str] = set()

        self._finished_groups: set[int] = set()

        self._default_total = max(1, int(default_total))
        self._refresh_interval = refresh_interval
        self._last_refresh = 0.0
        self._last_full_redraw = time.monotonic()

        print(f"splitting {len(tasks_to_run)} grid points into "
              f"{self._num_groups} groups (up to {group_size} each)")

        self._global_bar = tqdm(
            total=len(tasks_to_run), desc="total grid points", unit="pt",
            position=0, dynamic_ncols=True, leave=True,
        )
        self._groups_bar = tqdm(
            total=self._num_groups, desc="completed groups", unit="group",
            position=1, dynamic_ncols=True, leave=True,
        )

        # persistent slot bars, relabeled to the currently active groups;
        # n is a group-average float, padded to 2 decimals (134.60/1500) so its
        # width stays fixed and the line doesn't flicker
        # note: {desc} already carries the ': ' appended by set_description
        slot_bar_format = ('{desc}{percentage:3.0f}%|{bar}| {n:.2f}/{total_fmt} '
                           '[{elapsed}<{remaining}, {rate_fmt}{postfix}]')
        self._slot_bars: list[tqdm] = [
            tqdm(
                total=self._default_total, desc="(idle)", unit="batch",
                position=2 + slot, dynamic_ncols=True, leave=False,
                bar_format=slot_bar_format,
            )
            for slot in range(self._NUM_SLOTS)
        ]
        self._slot_group: list[int | None] = [None] * self._NUM_SLOTS
        self._slot_shown: list[float | None] = [None] * self._NUM_SLOTS

        self._refresh_slots(force=True)  # label the slots with the first groups right away

    def drain_and_refresh(self, progress_queue: mp.Queue) -> None:
        while True:
            try:
                task_id, batches_done, total_batches = progress_queue.get_nowait()
            except Empty:
                break
            if task_id not in self._fraction or task_id in self._completed:
                continue
            self._total[task_id] = int(total_batches)
            fraction = min(1.0, batches_done / max(1, total_batches))
            self._fraction[task_id] = max(self._fraction[task_id], fraction)  # ignore out-of-order messages

        now = time.monotonic()
        if now - self._last_refresh >= self._refresh_interval:
            full = (now - self._last_full_redraw) >= self._FULL_REDRAW_INTERVAL
            self._refresh_slots(force=full)
            if full:
                self._global_bar.refresh()
                self._groups_bar.refresh()
                self._last_full_redraw = now
            self._last_refresh = now

    def record_completion(self, task_id: str) -> None:
        if task_id in self._fraction:
            self._completed.add(task_id)
            self._fraction[task_id] = 1.0
            group = self._group_of_task[task_id]
            if group not in self._finished_groups:
                members = self._group_members[group]
                if all(t in self._completed for t in members):
                    self._finish_group(group)
        self._global_bar.update(1)
        self._refresh_slots()

    def close(self) -> None:
        # slot bars are transient (leave=False); the summary bars stay. Close in
        # reverse row order so the cursor unwinds cleanly.
        for bar in reversed(self._slot_bars):
            bar.close()
        self._groups_bar.close()
        self._global_bar.close()

    def _finish_group(self, group: int) -> None:
        self._finished_groups.add(group)
        self._groups_bar.update(1)

    def _refresh_slots(self, force: bool = False) -> None:

        # slots display the lowest-numbered unfinished groups, upcoming ones included
        # (a not-yet-started group simply shows 0.0 average batches); the window
        # slides forward as groups finish, so slots go idle only near the very end
        unfinished = [g for g in range(self._num_groups) if g not in self._finished_groups]
        displayed = unfinished[:self._NUM_SLOTS]

        for slot, bar in enumerate(self._slot_bars):
            group = displayed[slot] if slot < len(displayed) else None

            if group != self._slot_group[slot]:
                self._slot_group[slot] = group
                self._slot_shown[slot] = None  # forces a redraw with the new label

            if group is None:
                if force or self._slot_shown[slot] != -1.0:
                    bar.set_description("(idle)", refresh=False)
                    bar.set_postfix({}, refresh=False)
                    bar.total = self._default_total
                    bar.n = 0
                    bar.refresh()
                    self._slot_shown[slot] = -1.0  # sentinel: rendered as idle
                continue

            members = self._group_members[group]
            totals = [self._total[t] for t in members if self._total[t] is not None]
            bar_total = max(totals) if totals else self._default_total
            avg_batches = round(float(np.mean([self._fraction[t] for t in members]) * bar_total), 2)

            if (not force) and (avg_batches == self._slot_shown[slot]):
                continue  # unchanged; don't touch its row

            bar.set_description(f"group {group + 1:02d}/{self._num_groups}", refresh=False)
            bar.total = bar_total
            bar.n = avg_batches
            num_done = sum(t in self._completed for t in members)
            bar.set_postfix({'completed': f"{num_done}/{len(members)}"}, refresh=False)
            bar.refresh()
            self._slot_shown[slot] = avg_batches


# ---------- child: logger process ----------

def _run_logger(
    logs_queue: mp.Queue,
    tasks_by_id: dict[str, Task],
    run_config: dict[str, Any],
    monitor_ready: mp.Event = None,
):
    splits = ['train', 'val', 'test']
    seeds = _get_seeds_to_run(seeds_config=run_config['seeds'])
    empty_grid, x_axis, y_axis = _initialize_grid(grid_config=run_config['grid'])
    accuracy_grids = {split: {seed: empty_grid * 1 for seed in seeds} for split in splits}

    seeds_seen = []
    splits_seen = []
    num_log_events = 0
    grid_monitor = AccuracyGridMonitor(run_config)

    # signal the parent that wandb init (and its console prints) is done,
    # so it can safely start drawing progress bars
    if monitor_ready is not None:
        monitor_ready.set()

    while True:

        log_item = logs_queue.get()  # blocks

        if log_item is None:
            grid_monitor.finish(
                splits_seen=splits_seen,
                seeds_seen=seeds_seen,
                num_log_events=num_log_events,
            )
            break

        task_id, split, accuracy = log_item
        task = tasks_by_id[task_id]
        seed = task.seed
        num_log_events += 1

        if seed not in seeds_seen:
            seeds_seen.append(seed)
        if split not in splits_seen:
            splits_seen.append(split)


        # if new accuracy value is better than existing, update grid:
        grid = accuracy_grids[split][seed]
        i = np.where(x_axis == task.x.value)[0]
        j = np.where(y_axis == task.y.value)[0]
        accuracy_ij = grid[i, j]
        if np.isnan(accuracy_ij) or (accuracy > accuracy_ij):
            accuracy_grids[split][seed][i, j] = accuracy

        # save
        save_dir = Path(run_config["io"]["run_results_dir"])
        for split_seen in splits_seen:
            # stack seeds
            array = np.stack([accuracy_grids[split_seen][seen_seed] for seen_seed in seeds_seen])
            array_name = f"accuracy_grid__{split_seen}"
            save_path = save_dir / f"{array_name}.npy"
            np.save(save_path, array)

        grid_monitor.log_if_due(
            split=split,
            splits_seen=splits_seen,
            seeds_seen=seeds_seen,
            num_log_events=num_log_events,
        )


def _initialize_grid(grid_config):

    # set up grid
    grid_axes, grid_constants = get_axes_from_grid_config(grid_config)

    x_axis = grid_axes.x.axis.astype(int)
    y_axis = grid_axes.y.axis.astype(int)

    empty_grid = np.full((len(x_axis), len(y_axis)), np.nan)

    return empty_grid, x_axis, y_axis


# ---------- child: worker process ----------

def _run_worker(
    worker: Worker,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    logs_queue: mp.Queue,
    progress_queue: mp.Queue,
    base_run_config: dict,
):
    """
    One process per worker.
    Builds its own dataloaders and runs tasks pulled from task_queue.
    Sends TaskResult objects back via result_queue.
    """

    debug_prints = base_run_config.get('training', {}).get('print_non_tqdm_debug_messages', False)
    view_per_gridpoint_tqdm = base_run_config.get('training', {}).get('view_per_gridpoint_tqdm', True)

    if debug_prints:
        print(f"worker {worker.name}: started")
    time.sleep(1)  # optional (to make print less mixed up)

    # keep each process lean on CPU threads (avoid oversubscription)
    num_cpu_threads = max(1, worker.num_cpu_threads)
    os.environ.setdefault("OMP_NUM_THREADS", str(num_cpu_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(num_cpu_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(num_cpu_threads))
    torch.set_num_threads(num_cpu_threads)

    # runtime device and unique run name
    local_run_config = copy.deepcopy(base_run_config)
    local_run_config.setdefault('runtime', {})
    local_run_config['runtime']['device'] = worker.device  # always 'cpu' or 'cuda:i'
    local_run_config.setdefault('wandb', {})

    # grouped progress view: the parent renders all bars, so workers must stay silent
    if not view_per_gridpoint_tqdm:
        local_run_config.setdefault('training', {})
        local_run_config['training']['view_train_tqdm'] = False
        local_run_config['training']['view_evaluation_tqdm'] = False

    # set device in this process
    if worker.device.startswith("cuda"):
        torch.cuda.set_device(int(worker.device.split(":")[1]))

    while True:

        # get a task from the queue
        task = task_queue.get()

        # end of queue (reached sentinel)
        if task is None:
            break

        if debug_prints:
            print(f"worker {worker.name}: received task {task.name}")

        def _exit_with_status(task_status: TaskStatus, result=None):
            if debug_prints:
                print(f"worker {worker.name}: exited with status {task_status.name}, {result=}")
            if result == None:
                result = {}
            output = (task.id, task_status.value, result)
            result_queue.put(output)

        # we should run task

        # MOST IMPORTANT! set task seed
        local_run_config['runtime']['seed'] = task.seed  # to be used inside build_and_train_model
        set_seed(task.seed)  # just to be on the safe side

        # build, train and save model
        try:
            if debug_prints:
                print(f"\nstarting task {task.name}")
            run_name = f"{task.name} @ {worker.name}"
            local_run_config['wandb']['run_name'] = run_name

            # compute theoretical prediction
            if local_run_config['model']['class'] == 'mamba_theory':
                task_type = local_run_config['model']['task_type']
                theoretical_accuracy = predict_theoretical_recall_accuracy(dims=task.dims, task_type=task_type)
                run_result = {'accuracy': theoretical_accuracy}
                send_logs(logs=run_result, logs_queue=logs_queue, task_id=task.id, split='val')
                time.sleep(0.5)

            # build model and train/evaluate
            else:
                run_result = _build_and_train_model(
                    dims=task.dims,
                    run_config=local_run_config,
                    run_name=run_name,
                    logs_queue=logs_queue,
                    task_id=task.id,
                    progress_queue=progress_queue,
                )

            _exit_with_status(TaskStatus.COMPLETED, result=run_result)

        except Exception:
            print(f"\n\nworker {worker.name} encountered an error:\n\n")
            traceback.print_exc()
            _exit_with_status(TaskStatus.FAILED)
            # keep the worker alive for the remaining tasks: the failure is fully
            # reported (traceback above, FAILED result, grid cell stays NaN), and a
            # worker exit here would abort the whole grid via the parent's
            # premature-death check
            continue


def _should_skip_grid_point(dims: MqarDimensions) -> bool:

    # avoid model dims d_state > d_model
    if dims.N > dims.D:
        return True

    # avoid MQAR dims assertion error (zoology)
    if dims.L >= dims.V:
        return True
    if dims.L < (4 * dims.N_facts):
        return True

    return False


def _build_and_train_model(
        dims: MqarDimensions,
        run_config: dict[str, Any],
        run_name: str = None,
        logs_queue: mp.Queue = None,
        task_id: str = None,
        progress_queue: mp.Queue = None,
):

    set_seed(run_config['runtime']['seed'])

    model_config = run_config['model']

    # build model
    model = get_model_from_config(
        dims=dims,
        model_config=model_config,
    )

    # now build dataloaders (using optionally scaled batch size)
    dataloaders = build_dataloaders(dims=dims, run_config=run_config)

    device = run_config['runtime']['device']

    # sync and clear cache
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.set_float32_matmul_precision("high")

    # train model
    results = run_train_loop(
        model=model, dataloaders=dataloaders,
        run_config=run_config, debug_text=run_name,
        logs_queue=logs_queue,
        task_id=task_id,
        progress_queue=progress_queue,
    )

    # cleanup
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    return results
