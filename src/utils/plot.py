import matplotlib

from experiments.grid_runs_utils import GridAxes, GridConstants
import numpy as np
import matplotlib.pyplot as plt


DPI = 300


def configure_matplotlib(dpi: int = DPI, ax_labelsize=36, tick_labelsize=24):

    matplotlib.rcParams.update({
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "font.size": 14,          # base font size
        "axes.labelsize": ax_labelsize,     # x/y labels
        "axes.titlesize": 18,     # title
        "xtick.labelsize": tick_labelsize,  # 14,
        "ytick.labelsize": tick_labelsize,  # 14,
        "legend.fontsize": 14,
    })


def save_figure_to_png(fig, save_dir, save_name, dpi=DPI, tight=True, verbose=True):

    if tight:
        fig.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)

    out_png = save_dir / f"{save_name}.png"

    fig.savefig(
        out_png,
        dpi=dpi, facecolor='white', bbox_inches='tight',
        pad_inches=0.05, transparent=False,
    )

    if verbose:
        print(f"figure saved to: {out_png}")



def _load_and_aggregate_accuracy_grid(array_path, seed_aggregation='max'):

    array = np.load(array_path)
    array_aggregated = None

    all_nan_mask = np.all(np.isnan(array), axis=0)

    match seed_aggregation:
        case 'max':
            array_nums = np.nan_to_num(array, nan=-np.inf)
            array_aggregated = np.max(array_nums, axis=0, keepdims=True)
            array_aggregated[:, all_nan_mask] = np.nan
            array_aggregated = array_aggregated[0]  # optional
        case 'mean':
            all_nan_mask = np.all(np.isnan(array), axis=0)
            sum_vals = np.nansum(array, axis=0, keepdims=True)
            count_vals = np.sum(~np.isnan(array), axis=0, keepdims=True)
            with np.errstate(invalid='ignore'):  # 0/0 on all-NaN cells, masked just below
                array_aggregated = sum_vals / count_vals
            array_aggregated[:, all_nan_mask] = np.nan
            array_aggregated = array_aggregated[0]  # optional
        case None:
            array_aggregated = array
        case _:
            raise RuntimeError(f"Unrecognized seed_aggregation: {seed_aggregation}")
    
    return array_aggregated


def plot_accuracy_grid(
        accuracy_grid: np.ndarray = None,
        grid_axes: GridAxes = None,
        grid_constants: GridConstants = None,
        run_config: dict = None,
        # kwargs
        split='val',
        seed_aggregation='max',
        show_text=True,
        show_title=True,
        show_cbar=True,
        figsize=(10, 7),
        cmap='inferno',
        x_label=None,
        y_label=None,
        make_square=True,
        ticks_type=None,
        title_prefix=None,
):
    assert accuracy_grid is not None
    assert grid_axes is not None
    assert grid_constants is not None
    assert run_config is not None

    x_axis = grid_axes.x.axis
    y_axis = grid_axes.y.axis

    if x_label is None:
        x_label = grid_axes.x.name
    if y_label is None:
        y_label = grid_axes.y.name

    if grid_axes.x.name == 'V':
        # log-scale the V axis; use a math-mode label unless the caller passed a custom one
        x_axis = np.log10(x_axis)
        if x_label in (None, 'V', '$V$'):
            x_label = r'$\log_{10}{V}$'


    plt.figure(figsize=figsize)
    ax = plt.gca()

    plt.pcolormesh(x_axis, y_axis, accuracy_grid.T, cmap=cmap, vmin=0, vmax=1)
    data_xlim = ax.get_xlim()  # the mesh's own edges, before any tick styling

    if make_square:
        ax.set_box_aspect(1)

    if show_cbar:
        plt.colorbar()

    plt.xlabel(x_label)
    plt.ylabel(
        y_label,
        rotation=0,
        labelpad=15,
    )

    if ticks_type == 'quarter':
        set_quarter_ticks(ax, x_axis, y_axis)
    elif ticks_type == 'all':
        set_all_ticks(ax, x_axis, y_axis)

    if grid_axes.x.name == 'V':
        # quarter/all ticks use integer rounding, which puts out-of-range ticks on the
        # log10 axis and expands the view - restore the mesh's own limits first
        ax.set_xlim(data_xlim)
        # denser readable ticks on the log10(V) axis: every half decade in range
        x_ticks = np.arange(np.floor(x_axis.min() * 2) / 2, x_axis.max() + 0.25, 0.5)
        x_ticks = x_ticks[(x_ticks >= data_xlim[0]) & (x_ticks <= data_xlim[1])]
        ax.set_xticks(x_ticks, labels=[f'{t:g}' for t in x_ticks])

    # saved bundle configs are byte-copies of the source config and carry no
    # runtime-filled experiment_name; it is only used for optional titles
    experiment_name = run_config['runtime'].get('experiment_name', '')
    meta = ", ".join([f'{k}={v}' for k, v in grid_constants.data.items()])
    title = (f'{experiment_name}\n'
             f'{meta}')
    if title_prefix:
        title = f"{title_prefix}\n{title}"


    if show_title:
        plt.title(title)

    if show_text:
        nx, ny = accuracy_grid.shape
        for i in range(nx):
            for j in range(ny):
                value = accuracy_grid[i, j]
                if not np.isfinite(value):  # empty (unrun) cells carry no text
                    continue
                x = x_axis[i]  # center in x
                y = y_axis[j]  # center in y
                color = 'k' if value > 0.5 else 'w'
                plt.text(
                    x, y,
                    f"{value:.2f}",
                    ha='center', va='center', color=color, size=10)

    return accuracy_grid, grid_axes


def set_quarter_ticks(ax, x_axis, y_axis):

    x_min, x_max = x_axis.min(), x_axis.max()
    y_min, y_max = y_axis.min(), y_axis.max()

    norm_positions = np.array([0, 0.25, 0.5, 0.75, 1])

    x_ticks = x_min + norm_positions * (x_max - x_min)
    y_ticks = y_min + norm_positions * (y_max - y_min)

    x_ticks = (2 * (x_ticks // 2)).astype(int)
    y_ticks = (2 * (y_ticks // 2)).astype(int)

    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    ax.set_xticklabels(x_ticks)
    ax.set_yticklabels(y_ticks)


def set_all_ticks(ax, x_axis, y_axis):

    x_ticks = x_axis.astype(int)
    y_ticks = y_axis.astype(int)

    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    ax.set_xticklabels(x_ticks)
    ax.set_yticklabels(y_ticks)
