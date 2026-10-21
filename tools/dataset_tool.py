"""Dataset tool for converting a dataset of images and parameters into a single zip file.

License
-------

Copyright (c) 2026 Shinei Arakawa
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
"""

import glob
import io
import json
import multiprocessing as mp
import os
import re
import zipfile

import click
import numpy as np
import pandas as pd
import PIL.Image
import torch
import tqdm

import torch_utils

# ------------------------------------------------------------------------------------------------------------------------
# Utility functions


def load_csv_tail6(csv_path: str) -> torch.Tensor:
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        raise ValueError(f"CSV has no rows: {csv_path}")

    last = df.iloc[-1]
    # If the last row has columns wa, wb, wc, wd, we, wf, use them directly
    wa_cols = ["wa", "wb", "wc", "wd", "we", "wf"]
    if all(c in df.columns for c in wa_cols):
        vals = last[wa_cols].values.astype(np.float32)
        return torch.from_numpy(vals)

    # Otherwise, try to find the last 6 numeric columns in the last row
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) >= 6:
        cols = numeric_cols[-6:]
        vals = last[cols].values.astype(np.float32)
        return torch.from_numpy(vals)

    raise ValueError(f"Cannot extract 6 params from last row in {csv_path}. Require columns wa..wf or at least 6 numeric cols in the last row.")


def load_param(args: torch_utils.EasyDict) -> tuple[tuple[int, int], torch.Tensor] | None:
    file_name = os.path.basename(args.csv_file)
    match = args.csv_file_name_template.search(file_name)
    if not match:
        return None

    cat = int(match.group("cat"))
    wid = int(match.group("wid"))
    key = (cat, wid)

    param = load_csv_tail6(args.csv_file)

    return key, param


def convert_and_write_image(args: torch_utils.EasyDict) -> bytes:
    """Convert an image to PNG format and return it as a BytesIO object."""

    img = PIL.Image.open(args.img_path).convert('RGB')  # Ensure image is in RGB format

    if 'transform' in args and args.transform is not None:
        img = args.transform(img)

    img_bits = io.BytesIO()
    img.save(img_bits, format='PNG', compress_level=5, optimize=True)

    return img_bits.getvalue()


# ------------------------------------------------------------------------------------------------------------------------
# Multi-processing


def launch_jobs(func, args_list, desc='Processing', num_workers=None):
    """Launch multiple jobs in parallel using multiprocessing."""

    if num_workers is None:
        num_workers = mp.cpu_count()

    with mp.get_context('spawn').Pool(processes=num_workers) as pool:
        results = list(tqdm.tqdm(pool.imap(func, args_list), total=len(args_list), desc=desc))

    return results

# ------------------------------------------------------------------------------------------------------------------------
# Command line interface


@click.group(context_settings=dict(show_default=True, max_content_width=200))
def cmdline() -> None:
    """Dataset tool."""
    pass


@cmdline.command(context_settings=dict(show_default=True, max_content_width=200))
@click.option('--source', type=click.Path(exists=True, file_okay=False), required=True, help='Path to dataset directory.')
@click.option('--dest', type=click.Path(file_okay=True, dir_okay=False), required=True, help='Path to output zip file.')
@click.option('--param_dir', type=click.Path(exists=True, file_okay=False), default=None, help='Directory containing parameter files.')
@click.option('--num_workers', type=int, default=os.cpu_count(), help='Number of worker processes for loading parameters.')
def convert(**kwargs) -> None:
    """Convert dataset to a single zip file."""

    PIL.Image.init()
    opts = torch_utils.EasyDict(kwargs)

    # Find images
    source_dir = os.path.abspath(opts.source)
    dirs = glob.glob(os.path.join(source_dir, '*'))
    dirs = [dir_path for dir_path in dirs if os.path.isdir(dir_path)]
    dirs = sorted(dirs)

    dset_stats = torch_utils.EasyDict()
    dset_stats.num_classes = len(dirs)
    dset_stats.img_paths = []
    dset_stats.cls_labels = []
    dset_stats.unique_labels_str = []
    dset_stats.num_instances_per_class = -1

    for dir_path in dirs:
        assert os.path.isdir(dir_path), f'Expected directory, but got file: {dir_path}'

        img_paths = glob.glob(os.path.join(dir_path, '*'))
        img_paths = [os.path.relpath(img_path, source_dir) for img_path in img_paths if os.path.isfile(img_path)]
        assert len(img_paths) > 0, f'No images found in directory: {dir_path}'

        if dset_stats.num_instances_per_class == -1:
            dset_stats.num_instances_per_class = len(img_paths)
        else:
            assert dset_stats.num_instances_per_class == len(img_paths), f'Inconsistent number of instances per class. Expected {dset_stats.num_instances_per_class}, but found {len(img_paths)} in directory: {dir_path}'

        dset_stats.img_paths.extend(img_paths)
        dset_stats.cls_labels.extend([len(dset_stats.unique_labels_str)] * len(img_paths))

        dset_stats.unique_labels_str.append(os.path.basename(dir_path))

    assert len(dset_stats.img_paths) > 0, 'No images found in the dataset.'
    assert len(dset_stats.img_paths) == len(dset_stats.cls_labels), 'Mismatch between number of images and labels.'

    # Load parameter data if param_dir is provided
    if opts.param_dir is not None:
        print(f'Loading parameters from CSV files in: {opts.param_dir}', flush=True)

        # Read csv
        param_dir = os.path.abspath(opts.param_dir)
        csv_files = glob.glob(os.path.join(param_dir, '**', '*.csv'), recursive=True)
        csv_file_name_template = re.compile(r"^(?P<cat>\d+)_(?P<wid>\d+)\.csv$")

        mp_args = []
        for csv_file in csv_files:
            mp_args.append(torch_utils.EasyDict(csv_file=csv_file, csv_file_name_template=csv_file_name_template))

        results = launch_jobs(load_param, mp_args, desc='Loading parameters', num_workers=opts.num_workers)

        param_data: dict[tuple[int, int], torch.Tensor] = {}
        for key, param in results:
            param_data[key] = param

        print(f'Loaded parameters for {len(param_data)} (category, weightID) pairs from {param_dir}.', flush=True)

    # Save dataset to zip file
    dest_path = os.path.abspath(opts.dest)
    print(f'Saving dataset to zip file: {dest_path}', flush=True)

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    with zipfile.ZipFile(dest_path, mode='w') as zfile:
        # Convert and write images in parallel
        mp_args = []
        for img_path in dset_stats.img_paths:
            full_img_path = os.path.join(source_dir, img_path)
            mp_args.append(torch_utils.EasyDict(img_path=full_img_path))

        converted_images = launch_jobs(convert_and_write_image, mp_args, desc='Converting images', num_workers=opts.num_workers)

        # Write images
        for img_path, img_bytes in zip(dset_stats.img_paths, converted_images):
            zfile.writestr(img_path, img_bytes)

        # Write parameters if available
        if opts.param_dir is not None:
            arc_param_path = 'params.pt'
            with zfile.open(arc_param_path, 'w') as param_file:
                torch.save(param_data, param_file)
            dset_stats.param_path = arc_param_path

        # Write dataset statistics as 'dataset.json'
        with zfile.open('dataset.json', 'w') as json_file:
            json.dump(dset_stats.to_stddict(), json_file, indent=4, ensure_ascii=False)

    print(f'Dataset successfully saved to {dest_path}', flush=True)


if __name__ == "__main__":
    cmdline()
