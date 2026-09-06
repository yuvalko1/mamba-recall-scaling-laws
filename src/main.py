import warnings
warnings.filterwarnings("error", category=UserWarning)  # turn warnings into exceptions

import argparse
import json5

from pathlib import Path

from experiments.grid_runs import execute_grid_run
from experiments.grid_runs_utils import initialize_grid_run


def main():

    # parse args
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", type=Path, required=True,
                   help="run config, e.g. config/MQAR__D_N__linear_trained__regime_0.json5")
    args = p.parse_args()

    # load configs
    run_config_path = args.config
    print(f"\nloading run config from: \n{run_config_path}")
    run_config = json5.loads(run_config_path.read_text())

    # init
    run_config['runtime']['experiment_name'] = run_config_path.stem
    grid_run_name = initialize_grid_run(run_config=run_config, config_path=run_config_path)

    # run
    execute_grid_run(grid_run_name, run_config=run_config)

if __name__ == "__main__":
    main()
