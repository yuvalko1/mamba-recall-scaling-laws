import numpy as np

from experiments.grid_runs_utils import GridAxes, GridConstants


def flatten_D_N_grid_data(accuracy_grid: np.ndarray, grid_axes: GridAxes, grid_constants: GridConstants):

    assert grid_axes.x.name == 'D'
    assert grid_axes.y.name == 'N'

    D_axis, N_axis = grid_axes.x.axis, grid_axes.y.axis

    accuracy_list = []
    D_list = []
    N_list = []

    for i, d in enumerate(D_axis):
        for j, n in enumerate(N_axis):

            if n > d:
                continue

            accuracy = accuracy_grid[i, j]

            D_list.append(d)
            N_list.append(n)
            accuracy_list.append(accuracy)

    D, N, accuracy = np.array(D_list), np.array(N_list), np.array(accuracy_list)

    return D, N, accuracy


def flatten_N_Lambda_grid_data(accuracy_grid: np.ndarray, grid_axes: GridAxes, grid_constants: GridConstants):

    assert grid_axes.x.name == 'N'
    assert grid_axes.y.name == 'Lambda'

    D = grid_constants['D']

    N_axis, Lambda_axis = grid_axes.x.axis, grid_axes.y.axis

    accuracy_list = []
    N_list = []
    Lambda_list = []

    for i, n in enumerate(N_axis):
        for j, lm in enumerate(Lambda_axis):

            # optional
            if lm*n > D:  # Lambda*N > D
                continue

            accuracy = accuracy_grid[i, j]

            N_list.append(n)
            Lambda_list.append(lm)
            accuracy_list.append(accuracy)

    N, Lambda, accuracy = np.array(N_list), np.array(Lambda_list), np.array(accuracy_list)

    return N, Lambda, accuracy
