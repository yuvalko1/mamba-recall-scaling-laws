import os
import time
import traceback
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wandb

from experiments.grid_runs_utils import get_axes_from_grid_config
from utils.common import wandb_output_dir
from utils.plot import _load_and_aggregate_accuracy_grid, plot_accuracy_grid


class AccuracyGridMonitor:

    def __init__(self, run_config: dict[str, Any]):
        self._run_config = run_config
        self._run = None
        self._plot_step = 0
        self._plots_dirty = False
        self._last_plot_time = 0.0

        plot_config = run_config['wandb'].get('accuracy_grid_plots', {})
        self._plot_interval = max(
            1.0,
            float(plot_config.get('log_any_num_seconds', 30)),
        )

        try:
            self._run = initialize_accuracy_grid_monitor(run_config)
        except Exception:
            print('accuracy-grid W&B monitor failed to initialize; continuing without it')
            traceback.print_exc()

    def log_if_due(
            self,
            split: str,
            splits_seen: list[str],
            seeds_seen: list[int],
            num_log_events: int,
    ) -> None:
        if split in ('train', 'val'):
            self._plots_dirty = True

        self._log(
            splits_seen=splits_seen,
            seeds_seen=seeds_seen,
            num_log_events=num_log_events,
            force=False,
        )

    def finish(
            self,
            splits_seen: list[str],
            seeds_seen: list[int],
            num_log_events: int,
    ) -> None:
        self._log(
            splits_seen=splits_seen,
            seeds_seen=seeds_seen,
            num_log_events=num_log_events,
            force=True,
        )

        if self._run is None:
            return

        try:
            self._run.finish()
        except Exception:
            print('accuracy-grid W&B monitor failed to finish cleanly')
            traceback.print_exc()

    def _log(
            self,
            splits_seen: list[str],
            seeds_seen: list[int],
            num_log_events: int,
            force: bool,
    ) -> None:
        if self._run is None or not self._plots_dirty:
            return

        now = time.monotonic()
        if not force and now - self._last_plot_time < self._plot_interval:
            return

        try:
            self._plot_step = log_accuracy_grid_plots(
                monitor_run=self._run,
                run_config=self._run_config,
                splits_seen=splits_seen,
                seeds_seen=seeds_seen,
                plot_step=self._plot_step,
                num_log_events=num_log_events,
            )
            self._plots_dirty = False
        except Exception:
            print('accuracy-grid W&B log failed; will retry after the logging interval')
            traceback.print_exc()
        finally:
            self._last_plot_time = now


def initialize_accuracy_grid_monitor(run_config: dict[str, Any]):
    wandb_config = run_config['wandb']
    plot_config = wandb_config.get('accuracy_grid_plots', {})
    if not (wandb_config.get('activate', False) and plot_config.get('activate', False)):
        return None

    output_dir = str(wandb_config.get('output_dir', wandb_output_dir))
    os.makedirs(output_dir, exist_ok=True)

    return wandb.init(
        project=wandb_config['project_name'],
        name='Accuracy grid monitor',
        dir=output_dir,
        config=run_config,
        job_type='grid-monitor',
        reinit='finish_previous',
    )


def _render_accuracy_grid_image(
        accuracy_grid,
        grid_axes,
        grid_constants,
        run_config: dict[str, Any],
        split: str,
        seed_aggregation: str | None,
        plot_config: dict[str, Any],
        title_prefix: str,
        caption: str,
):
    plot_accuracy_grid(
        accuracy_grid=accuracy_grid,
        grid_axes=grid_axes,
        grid_constants=grid_constants,
        run_config=run_config,
        split=split,
        seed_aggregation=seed_aggregation,
        show_text=plot_config.get('show_text', False),
        show_title=plot_config.get('show_title', True),
        show_cbar=plot_config.get('show_cbar', True),
        figsize=tuple(plot_config.get('figsize', [10, 6])),
        cmap=plot_config.get('cmap', 'inferno'),
        title_prefix=title_prefix,
    )

    fig = plt.gcf()
    fig.tight_layout()
    try:
        return wandb.Image(fig, caption=caption)
    finally:
        plt.close(fig)


def log_accuracy_grid_plots(
        monitor_run,
        run_config: dict[str, Any],
        splits_seen: list[str],
        seeds_seen: list[int],
        plot_step: int,
        num_log_events: int,
) -> int:
    if monitor_run is None:
        return plot_step

    plot_config = run_config['wandb']['accuracy_grid_plots']
    save_dir = Path(run_config['io']['run_results_dir'])
    grid_axes, grid_constants = get_axes_from_grid_config(run_config['grid'])
    seed_aggregation = plot_config.get('seed_aggregation', 'max')
    images = {}
    num_seeds = int(run_config['seeds'].get('n_seeds', len(seeds_seen)))

    for split in ('train', 'val'):
        if split not in splits_seen:
            continue

        array_path = save_dir / f'accuracy_grid__{split}.npy'
        if not array_path.exists():
            continue

        accuracy_grids = _load_and_aggregate_accuracy_grid(
            array_path,
            seed_aggregation=None,
        )
        if len(accuracy_grids) != len(seeds_seen):
            raise RuntimeError(
                f'{array_path} has {len(accuracy_grids)} seed grids, '
                f'but {len(seeds_seen)} seeds were recorded'
            )

        for seed_index, seed in enumerate(seeds_seen):
            caption = (
                f'{split} accuracy for seed {seed}; '
                f'{len(seeds_seen)}/{num_seeds} seeds observed; '
                f'{num_log_events} grid log events received'
            )
            images[f'accuracy_grid/{split}/seed_{seed}'] = _render_accuracy_grid_image(
                accuracy_grid=accuracy_grids[seed_index],
                grid_axes=grid_axes,
                grid_constants=grid_constants,
                run_config=run_config,
                split=split,
                seed_aggregation=None,
                plot_config=plot_config,
                title_prefix=f'{split.title()} accuracy, seed {seed}',
                caption=caption,
            )

        aggregated_grid = _load_and_aggregate_accuracy_grid(
            array_path,
            seed_aggregation=seed_aggregation,
        )
        aggregate_caption = (
            f'{split} accuracy; {seed_aggregation} over observed seeds {seeds_seen} '
            f'({len(seeds_seen)}/{num_seeds}); '
            f'{num_log_events} grid log events received'
        )
        aggregate_key = f'accuracy_grid/{split}/{seed_aggregation}_of_{num_seeds}_seeds'
        images[aggregate_key] = _render_accuracy_grid_image(
            accuracy_grid=aggregated_grid,
            grid_axes=grid_axes,
            grid_constants=grid_constants,
            run_config=run_config,
            split=split,
            seed_aggregation=seed_aggregation,
            plot_config=plot_config,
            title_prefix=f'{split.title()} accuracy, {seed_aggregation} over seeds',
            caption=aggregate_caption,
        )

    if images:
        images['accuracy_grid/events_received'] = num_log_events
        images['accuracy_grid/seeds_seen'] = len(seeds_seen)
        monitor_run.log(images, step=plot_step)
        return plot_step + 1

    return plot_step
