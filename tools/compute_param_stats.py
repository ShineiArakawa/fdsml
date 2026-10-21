"""Compute the mean and standard deviation of parameters from parameter CSV files.

License
-------

Copyright (c) 2026 Shinei Arakawa
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
"""

import glob
import os

import click
import numpy as np
import tqdm

import torch_utils


@click.command(context_settings=dict(show_default=True, max_content_width=200))
@click.option('--param_dir', type=click.Path(exists=True, file_okay=False), required=True, help='Directory containing parameter files.')
def main(**kwargs):
    """Compute mean and std of parameters from parameter CSV files."""

    opts = torch_utils.EasyDict(kwargs)

    csv_files = sorted(glob.glob(os.path.join(opts.param_dir, '**', '*.csv'), recursive=True))

    moments = np.zeros((2, 6), dtype=np.float64)  # for average and squared average

    for csv_file in tqdm.tqdm(csv_files, desc='Processing parameter files', unit='file'):
        params = np.loadtxt(csv_file, dtype=np.float64, delimiter=',', skiprows=1)  # skip header
        weighted_params = params[-1, :]  # last row

        moments[0] += weighted_params
        moments[1] += weighted_params ** 2

    n = len(csv_files)
    mean = moments[0] / n
    std = np.sqrt((moments[1] / n) - (mean ** 2))

    print('Parameter statistics:')
    print(f'  Mean : {tuple(mean.tolist())}')
    print(f'  Std  : {tuple(std.tolist())}')


if __name__ == '__main__':
    main()
