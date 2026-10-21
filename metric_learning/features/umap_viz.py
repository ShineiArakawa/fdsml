"""Compute and plot UMAP results for metric learning features.

License
-------

Copyright (c) 2026 Shinei Arakawa
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
"""

import os

import click
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import umap

import metric_learning.features.feature_utils as feature_utils
import torch_utils.util as util


@click.command(context_settings=dict(show_default=True, max_content_width=200))
@click.option('--npz', help='Path to the input .npz file', required=True, type=click.Path(exists=True))
@click.option('--out', help='Path to the output .png file', required=True, type=click.Path())
@click.option('--seed', help='Random seed', type=int, default=0)
@click.option('--num', help='Number of samples to use. When specified, all samples will be shuffled and then truncated to this number', type=int, default=None)
@click.option('--num_cls', help='Number of classes to use. If not set, all classes are used', type=int, default=None)
@click.option('--cls_ids', help='Comma-separated list of class IDs to use. If set, overrides --num_cls', type=str, default=None)
@click.option('--imagenet', help='Use ImageNet dataset', is_flag=True, default=False)
@click.option('--norm_embs', help='Whether to normalize embeddings to unit length before UMAP', is_flag=True, default=False)
@click.option('--feat_key', help='Key in the .npz file for the feature embeddings', type=str, default='embs')
def main(**opts):
    opts = util.EasyDict(opts)  # for convenience

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

    if opts.cls_ids is not None:
        # Use only specified class IDs
        str_cls_ids: str = opts.cls_ids
        str_cls_ids = str_cls_ids.strip().lstrip('(').rstrip(')')
        cls_id_list = [int(s.strip()) for s in str_cls_ids.split(',')]

        selected_mask = np.isin(labels, cls_id_list)
        embs = embs[selected_mask]
        labels = labels[selected_mask]
        print(f'Selected classes {cls_id_list}, resulting in {len(embs)} samples')
    elif opts.num_cls is not None and opts.num_cls < num_classes_avail:
        # Randomly select a subset of classes
        indices = rnd_state.permutation(num_classes_avail)[:opts.num_cls]
        selected_classes = unique_classes[indices]
        selected_mask = np.isin(labels, selected_classes)
        embs = embs[selected_mask]
        labels = labels[selected_mask]
        print(f'Selected {opts.num_cls} classes, resulting in {len(embs)} samples')

    # Optionally subsample number of samples
    if opts.num is not None:
        orig_len = len(embs)
        indices = rnd_state.permutation(len(embs))[:min(opts.num, orig_len)]
        embs = embs[indices]
        labels = labels[indices]
        print(f'Selected {len(embs)} samples (from original {orig_len})')

    # Recompute unique classes after filtering
    unique_classes = np.unique(labels) if opts.cls_ids is None else cls_id_list
    num_classes = len(unique_classes)

    # ---------------------------------------------------------------------------
    # Normalize embeddings if specified.

    if opts.norm_embs:
        print('Normalizing embeddings to unit length')
        embs_norm = np.linalg.norm(embs, axis=1, keepdims=True)
        embs = embs / embs_norm

    # ----------------------------------------------    -----------------------------
    # Run UMAP.

    print('Running UMAP...')
    reducer = umap.UMAP(
        n_components=2,
        random_state=opts.seed,
        verbose=True,
    )
    umap_results = reducer.fit_transform(embs)

    # ---------------------------------------------------------------------------
    # Plot the results.

    fig = plt.figure(figsize=(12, 10), dpi=300)
    axes = fig.add_subplot(1, 1, 1)

    if opts.imagenet:
        imagenet_meta = feature_utils.load_imagenet_meta_data()
        human_readable_labels = []
        for i in range(imagenet_meta.num_classes):
            label = '{l_en} ({l_jp})'.format(
                l_en=imagenet_meta.get_human_label(i),
                l_jp=imagenet_meta.get_human_label_jp(i),
            )
            human_readable_labels.append(label)
        imagenet_labels = np.array(human_readable_labels)

    use_tab20 = num_classes <= 20

    if use_tab20:
        print('Using tab20 colormap')

        cls_to_idx = {cls_id: idx for idx, cls_id in enumerate(unique_classes)}
        labels_mapped = np.vectorize(cls_to_idx.get)(labels)

        base_cmap = plt.get_cmap('tab20')
        colors = base_cmap.colors[:num_classes]
        cmap = mcolors.ListedColormap(colors)

        norm = mcolors.BoundaryNorm(
            np.arange(-0.5, num_classes + 0.5, 1),
            cmap.N,
        )

        scatter = axes.scatter(
            umap_results[:, 0], umap_results[:, 1],
            c=labels_mapped,
            cmap=cmap,
            norm=norm,
            s=2,
        )
    else:
        scatter = axes.scatter(
            umap_results[:, 0], umap_results[:, 1],
            c=labels,
            cmap='jet',
            s=2,
        )

    legend_handles = []
    legend_labels = []
    classes_in_plot = np.unique(labels) if opts.cls_ids is None else cls_id_list
    for cls_id in classes_in_plot:
        if use_tab20:
            idx = cls_to_idx[cls_id]
            handle_color = cmap(idx)
        else:
            handle_color = scatter.cmap(scatter.norm(cls_id))

        legend_handles.append(
            plt.Line2D(
                [0], [0],
                marker='o',
                color='w',
                markerfacecolor=handle_color,
                markersize=6,
                linestyle='',
            )
        )
        legend_labels.append(
            imagenet_labels[int(cls_id)] if opts.imagenet else str(cls_id)
        )

    legend = axes.legend(
        legend_handles,
        legend_labels,
        title="Classes",
        loc='upper left',
        bbox_to_anchor=(1.05, 1),
        borderaxespad=0,
        fontsize='x-small'
    )
    axes.add_artist(legend)

    axes.set_title('UMAP Visualization of Embeddings')
    axes.set_xlabel('Component 1')
    axes.set_ylabel('Component 2')

    fig.subplots_adjust(right=0.6)  # make space for legend

    png_path = os.path.abspath(opts.out)
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    fig.savefig(png_path)  # ensure legend anchored outside axes is included
    plt.close(fig)

    print(f'Saved the figure to {png_path}')


if __name__ == '__main__':
    main()
