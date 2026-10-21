"""Create a parameter cache file (params.pt) by reading the last 6 parameters from each CSV file in the specified directory.

License
-------

Copyright (c) 2026 Shinei Arakawa
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
"""

import glob
import os
import pathlib
import re

import click
import numpy as np
import pandas as pd
import torch
import tqdm


def load_csv_tail6(csv_path: pathlib.Path) -> torch.Tensor:
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        raise ValueError(f"CSV has no rows: {csv_path}")

    last = df.iloc[-1]
    wa_cols = ["wa", "wb", "wc", "wd", "we", "wf"]
    if all(c in df.columns for c in wa_cols):
        vals = last[wa_cols].values.astype(np.float32)
        return torch.from_numpy(vals)

    raise ValueError(f"Cannot extract 6 params from last row in {csv_path}. "
                     f"Require columns wa..wf or at least 6 numeric cols in the last row.")


@click.command(context_settings=dict(show_default=True, max_content_width=200))
@click.option('--param_dir', type=click.Path(exists=True, file_okay=False), required=True, help='Directory containing parameter files.')
def main(param_dir: str) -> None:
    # Read csv
    param_dir = os.path.abspath(param_dir)
    param_data: dict[tuple[int, int], np.ndarray] = {}

    csv_file_name_template = re.compile(r"^(?P<cat>\d+)_(?P<wid>\d+)\.csv$")
    csv_files = glob.glob(os.path.join(param_dir, '**', '*.csv'), recursive=True)
    for csv_file in tqdm.tqdm(csv_files, desc='Loading parameter CSV files', unit='file'):
        file_name = os.path.basename(csv_file)
        match = csv_file_name_template.search(file_name)
        if not match:
            continue

        cat = int(match.group("cat"))
        wid = int(match.group("wid"))
        key = (cat, wid)

        param = load_csv_tail6(pathlib.Path(csv_file))

        param_data[key] = param

    print(f'Loaded {len(param_data)} parameter entries from {param_dir}.')

    out_file = os.path.join(param_dir, 'params.pt')
    torch.save(param_data, out_file)
    print(f'Saved parameter data to {out_file}.')

    print('done.')


if __name__ == '__main__':
    main()
