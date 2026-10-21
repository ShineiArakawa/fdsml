"""Compute and plot PCA results for metric learning features.

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
import sklearn.decomposition
import torch

import torch_utils.util as util


@click.command(context_settings=dict(show_default=True, max_content_width=200))
@click.option('--npz', help='Path to the input .npz file', required=True, type=click.Path(exists=True))
@click.option('--out', help='Path to the output .png file', required=True, type=click.Path())
@click.option('--seed', help='Random seed', type=int, default=0)
@click.option('--num', help='Number of samples to use. When specified, all samples will be shuffled and then truncated to this number', type=int, default=None)
@click.option('--num_cls', help='Number of classes to use. If not set, all classes are used', type=int, default=None)
def main(**opts):
    opts = util.EasyDict(opts)  # for convenience

    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)

    rnd_state = np.random.RandomState(opts.seed)

    # ---------------------------------------------------------------------------
    # Load the data.

    data = np.load(opts.npz)

    embs: np.ndarray = data['embs']
    labels: np.ndarray = data['labels']

    num_classes_avail = len(np.unique(labels))
    print(f'Found {len(embs)} samples with {num_classes_avail} classes in {opts.npz}')

    if opts.num_cls is not None and opts.num_cls < num_classes_avail:
        unique_classes = np.unique(labels)
        selected_classes = unique_classes[:opts.num_cls]
        selected_mask = np.isin(labels, selected_classes)
        embs = embs[selected_mask]
        labels = labels[selected_mask]
        print(f'Selected {opts.num_cls} classes, resulting in {len(embs)} samples')

    if opts.num is not None:
        orig_len = len(embs)
        indices = rnd_state.permutation(len(embs))[:min(opts.num, orig_len)]
        embs = embs[indices]
        labels = labels[indices]
        print(f'Selected {len(embs)} samples (from original {orig_len})')

    pca = sklearn.decomposition.PCA(
        n_components=None,  # Keep all components
        random_state=opts.seed,
        svd_solver='auto',
    )
    pca.fit(embs)
    pca_results = pca.transform(embs)

    explained_variance = pca.explained_variance_ratio_
    cumsum_explained_variance = np.cumsum(explained_variance)
    print(f'Explained variance ratio (cumulative): {cumsum_explained_variance}')

    # ---------------------------------------------------------------------------
    # Plot the results.

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    ax.plot(cumsum_explained_variance, marker='o', linestyle='-', color='blue', markersize=1)
    ax.set_title('Cumulative Explained Variance by PCA Components')
    ax.set_xlabel('Number of PCA Components')
    ax.set_ylabel('Cumulative Explained Variance Ratio')
    ax.set_ylim(-0.05, 1.05)  # Set y-axis limits to ensure visibility
    ax.grid(True)
    fig.tight_layout()

    out_png_path = os.path.abspath(opts.out)
    os.makedirs(os.path.dirname(out_png_path), exist_ok=True)
    fig.savefig(out_png_path)

    print(f'Saved PCA plot to {out_png_path}')


if __name__ == '__main__':
    main()
