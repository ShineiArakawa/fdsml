"""Compute and plot the norm distribution of metric learning features.

License
-------

Copyright (c) 2026 Shinei Arakawa
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
"""

import os

import click
import matplotlib.pyplot as plt
import numpy as np

import torch_utils.util as util


@click.command(context_settings=dict(show_default=True, max_content_width=200))
@click.option('--npz', help='Path to the input .npz file', required=True, type=click.Path(exists=True))
@click.option('--out', help='Path to the output .png file', required=True, type=click.Path())
def main(**opts):
    opts = util.EasyDict(opts)  # for convenience

    # ---------------------------------------------------------------------------
    # Load the data.

    data = np.load(opts.npz)

    embs: np.ndarray = data['embs']

    # ---------------------------------------------------------------------------
    # Compute norms.

    norms = np.linalg.norm(embs, axis=1, ord=2)

    # ---------------------------------------------------------------------------
    # Plot histogram.

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.hist(norms, bins=50, color='blue', alpha=0.7)
    ax.set_title('Histogram of Embedding Norms')
    ax.set_xlabel('L2 Norm')
    ax.set_ylabel('Frequency')
    ax.grid(True)
    fig.tight_layout()

    out_png_path = os.path.abspath(opts.out)
    os.makedirs(os.path.dirname(out_png_path), exist_ok=True)
    fig.savefig(out_png_path)

    print(f'Saved histogram plot to {out_png_path}')


if __name__ == '__main__':
    main()
