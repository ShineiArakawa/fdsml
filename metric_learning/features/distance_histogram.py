"""Compute and plot distance histograms for metric learning features.

License
-------

Copyright (c) 2026 Shinei Arakawa
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
"""

import os
import pprint

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm

import torch_utils.util as util


@click.command(context_settings=dict(show_default=True, max_content_width=200))
@click.option('--npz', help='Path to the input .npz file', required=True, type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option('--out', help='Path to the output directory', required=True, type=click.Path(file_okay=False, dir_okay=True, writable=True))
@click.option('--seed', help='Random seed', type=int, default=0)
@click.option('--num', help='Number of samples to use. When specified, all samples will be shuffled and then truncated to this number', type=int, default=None)
@click.option('--num_cls', help='Number of classes to use. If not set, all classes are used', type=int, default=None)
@click.option('--xlim', help='X-axis limit for the histogram plots', type=click.Tuple([float, float]), default=(0.0, 50.0))
@click.option('--norm_embs', help='Whether to normalize embeddings to unit length before computing distances', is_flag=True, default=False)
@click.option('--feat_key', help='Key in the .npz file for the feature embeddings', type=str, default='embs')
def main(**opts):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    opts = util.EasyDict(opts)  # for convenience

    print(f'\nOptions:\n{pprint.pformat(opts)}\n')

    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)

    rnd_state = np.random.RandomState(opts.seed)

    # ---------------------------------------------------------------------------
    # Load the data.

    data = np.load(opts.npz)

    embs: np.ndarray = data[opts.feat_key]
    labels: np.ndarray = data['labels']

    unique_classes = np.unique(labels)
    num_classes_avail = len(unique_classes)
    print(f'Found {len(embs)} samples with {num_classes_avail} classes in {opts.npz}')

    # Randomly select a subset of classes
    if opts.num_cls is not None and opts.num_cls < num_classes_avail:
        indices = rnd_state.permutation(num_classes_avail)[:opts.num_cls]
        selected_classes = unique_classes[indices]
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

    # Recompute unique classes after filtering
    unique_classes = np.unique(labels)

    embs_tensor = torch.from_numpy(embs).to(device)
    labels_tensor = torch.from_numpy(labels).to(device)

    if opts.norm_embs:
        print('Normalizing embeddings to unit length')
        embs_tensor = torch.nn.functional.normalize(embs_tensor, p=2, dim=1)

    # ---------------------------------------------------------------------------
    # Compute distance matrix and plot

    print(f'Outputting histograms to {opts.out}')

    for i_class in tqdm.trange(len(unique_classes), desc='Processing classes'):
        cls_id = unique_classes[i_class]

        cls_embs = embs_tensor[labels_tensor == cls_id]        # (M, D)
        diff = cls_embs[:, None, :] - embs_tensor[None, :, :]  # (M, N, D)

        dist = diff.square().sum(dim=-1).sqrt()  # (M, N)

        non_diagonal_mask = ~torch.eye(dist.size(0), device=device).bool()
        same_cls_dist = dist[:, labels_tensor == cls_id][non_diagonal_mask].flatten().cpu().numpy()
        diff_cls_dist = dist[:, labels_tensor != cls_id].flatten().cpu().numpy()

        fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
        ax.hist(same_cls_dist, bins=50, alpha=0.5, label='Same Class', color='blue', density=True)
        ax.hist(diff_cls_dist, bins=50, alpha=0.5, label='Different Class', color='red', density=True)
        ax.legend()
        ax.set_title(f'L2 Distance Histogram for Class {cls_id}')
        ax.set_xlabel('Distance')
        ax.set_ylabel('Density')

        if opts.xlim is not None:
            ax.set_xlim(opts.xlim)

        file_path = os.path.join(opts.out, f'class_{cls_id}.png')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        fig.savefig(file_path)

        fig.clf()
        plt.close(fig)
        del fig

    print('done.')


if __name__ == '__main__':
    main()
