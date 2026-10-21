"""FineFractalDB generation scripts: rendering of fractal images from IFS parameters.

Given a pickle file of iterated function system (IFS) parameters produced by ``python -m finefractaldb.ifs``, this
module renders a fixed number of image instances per system (class). Each instance is obtained by randomly perturbing
one transform of the base system, generating a point cloud with :func:`finefractaldb.ifs.iterate`, and rasterizing it
with :func:`finefractaldb.ifs.render`. The perturbed parameters are saved alongside each image as a CSV file.

License
-------

Copyright (c) 2026 Shinei Arakawa
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

This code is a derivative work based on code from the catalys1/fractal-pretraining repository, whose original code is licensed under the MIT License. The original code can be found at https://github.com/catalys1/fractal-pretraining.
"""

import hashlib
import multiprocessing as mp
import os
import pickle
import pprint
import textwrap

import click
import cv2
import numpy as np
import tqdm

import finefractaldb.ifs as ifs
import torch_utils

# ---------------------------------------------------------------------------------------------------------------------------
# Utilities


def hashnize(*args, **kwargs) -> int:
    """Compute a deterministic integer hash of arbitrary arguments.

    The arguments are converted to strings, concatenated, and hashed with SHA-256. This is used to derive a
    per-image random seed from the global seed and the image indices.

    Parameters
    ----------
    *args
        Positional values to include in the hash. Each is converted with ``str``.
    **kwargs
        Keyword values to include in the hash. Each is converted to ``'key:value'``.

    Returns
    -------
    int
        The SHA-256 digest interpreted as a (256-bit) integer.
    """

    str_repr = 'hash'
    for arg in args:
        str_repr += f'_{str(arg)}'
    for k, v in kwargs.items():
        str_repr += f'_{k}:{str(v)}'
    hash_value = hashlib.sha256(str_repr.encode('utf-8')).hexdigest()
    return int(hash_value, 16)


# ---------------------------------------------------------------------------------------------------------------------------
# Image generation


def generate_image(opts: torch_utils.EasyDict) -> None:
    """Render one fractal image instance and save it together with its IFS parameters.

    A single transform of ``opts.base_system`` is scaled by a random factor drawn from ``opts.noise_scale``. The
    perturbed system is rejected and resampled if any transform is no longer contractive (largest singular value
    greater than 1) or if the rendered image is too sparse (fraction of non-empty pixels below
    ``opts.pixel_threshold``). The accepted image is written as a 3-channel PNG, and the perturbed parameters are
    written as a CSV file.

    This function is intended to run in a worker process; all randomness is seeded from ``opts.seed`` so that the
    result is reproducible regardless of scheduling.

    Parameters
    ----------
    opts : torch_utils.EasyDict
        Per-image settings with the following keys.

        base_system : numpy.ndarray
            Array of shape ``(n, 2, 3)`` with the unperturbed affine transforms of the class.
        num_points : int
            Number of points to generate with :func:`finefractaldb.ifs.iterate`.
        image_size : int
            Side length of the output image in pixels.
        out_dir : str
            Dataset root directory. Images go to ``<out_dir>/images/<cls>/`` and parameters to
            ``<out_dir>/params/<cls>/``.
        cls_id : int
            Class index (index of the base system).
        inst_id : int
            Instance index within the class.
        idx : int
            Global image index, used in the image file name.
        seed : int
            Random seed for this image.
        noise_scale : tuple of float
            ``(low, high)`` range from which the perturbation scale is drawn uniformly.
        pixel_threshold : float
            Minimum fraction of non-empty pixels required to accept the image.
        patch : bool
            If ``True``, render each point as a random ``3 x 3`` patch instead of a single pixel.

    Returns
    -------
    None

    Notes
    -----
    The CSV file has columns ``wa, wb, wc, wd, we, wf`` corresponding to the affine parameters
    ``a, b, c, d, e, f`` of each transform, where the transform maps ``(x, y)`` to
    ``(a*x + b*y + e, c*x + d*y + f)``. It contains ``n + 1`` rows: one per transform, followed by the expectation
    of the parameters under the selection probabilities (proportional to the absolute determinants).
    """
    np.random.seed(opts.seed)

    num_params = opts.base_system.shape[0]

    color_image = None
    perturbed_system = None

    while color_image is None:
        perturb_at = np.random.randint(0, num_params)
        perturb_scale = np.random.uniform(opts.noise_scale[0], opts.noise_scale[1])

        perturbed_system = opts.base_system.copy()
        perturbed_system[perturb_at] *= perturb_scale

        # Check if the perturbed system is valid
        svd = np.linalg.svd(perturbed_system[:, :, :2], compute_uv=False)
        if svd.max() > 1.0:
            # invalid, try again
            continue

        # Render
        rnds = np.random.rand(opts.num_points)
        points = ifs.iterate(perturbed_system, rnds)

        # Occupancy test
        gray_image = ifs.render(points, s=opts.image_size, patch=False)
        num_pixels = opts.image_size * opts.image_size
        num_nonempty = np.count_nonzero(gray_image)
        if num_nonempty / num_pixels < opts.pixel_threshold:
            # too sparse, try again
            print(f'Underneath the threshold of pixel rate: {num_nonempty / num_pixels * 100.0:.3f} [%]', flush=True)
            continue

        # Rerender with patch if needed
        gray_image = ifs.render(points, s=opts.image_size, patch=opts.patch)

        color_image = gray_image[..., None].repeat(3, axis=-1) * 127

    # Save the image
    file_path = os.path.join(opts.out_dir, 'images', f'{opts.cls_id:05d}', f'{opts.cls_id:05d}_{opts.inst_id:05d}_{opts.idx:08d}.png')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    cv2.imwrite(file_path, cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 0])

    # Save the parameters
    param_file_path = os.path.join(opts.out_dir, 'params', f'{opts.cls_id:05d}', f'{opts.cls_id:05d}_{opts.inst_id:05d}.csv')
    os.makedirs(os.path.dirname(param_file_path), exist_ok=True)

    cols = ['wa', 'wb', 'wc', 'wd', 'we', 'wf']
    contiguous_params = np.zeros((num_params, 3, 2), dtype=perturbed_system.dtype)
    contiguous_params[:, 0, 0] = perturbed_system[:, 0, 0]  # a
    contiguous_params[:, 0, 1] = perturbed_system[:, 0, 1]  # b
    contiguous_params[:, 1, 0] = perturbed_system[:, 1, 0]  # c
    contiguous_params[:, 1, 1] = perturbed_system[:, 1, 1]  # d
    contiguous_params[:, 2, 0] = perturbed_system[:, 0, 2]  # e
    contiguous_params[:, 2, 1] = perturbed_system[:, 1, 2]  # f
    contiguous_params = contiguous_params.reshape((num_params, 6))  # (N, 6)

    dets = np.abs(contiguous_params[:, 0] * contiguous_params[:, 3] - contiguous_params[:, 1] * contiguous_params[:, 2])
    probs = dets / dets.sum()  # (N,)

    expected_params = np.sum(contiguous_params.copy() * probs[:, None], axis=0)  # (6,)

    to_save = np.vstack([contiguous_params, expected_params])  # (N+1, 6)

    np.savetxt(param_file_path, to_save, delimiter=',', header=','.join(cols), comments='')


# ---------------------------------------------------------------------------------------------------------------------------
# Main function


def main(**kwargs) -> None:
    """Render FineFractalDB images from a file of IFS parameters using a pool of worker processes."""

    opts = torch_utils.EasyDict(kwargs)
    print(f'Arguments: {pprint.pformat(opts, indent=4)}', flush=True)

    seed = opts.seed
    np.random.seed(seed)

    with open(opts.param_file, 'rb') as f:
        params = pickle.load(f)['params']
    print(f'Loaded {len(params)} IFS parameters from {opts.param_file}.', flush=True)

    out_dir = os.path.abspath(opts.out_dir)

    print(f'Generating {len(params) * opts.insts_per_cls} images into {out_dir}...', flush=True)
    mp_args = []
    for cls_idx, param in enumerate(tqdm.tqdm(params, desc='Sampling perturbed instances', unit='cls')):
        system = param['system']  # (N, 2, 3)
        for inst_idx in range(opts.insts_per_cls):
            mp_arg = torch_utils.EasyDict()

            mp_arg.base_system = system.copy()
            mp_arg.num_points = opts.num_points
            mp_arg.image_size = opts.image_size
            mp_arg.out_dir = out_dir
            mp_arg.cls_id = cls_idx
            mp_arg.inst_id = inst_idx
            mp_arg.idx = cls_idx * opts.insts_per_cls + inst_idx
            mp_arg.seed = hashnize(seed, cls_idx, inst_idx) % (2**32)
            mp_arg.noise_scale = opts.noise_scale
            mp_arg.pixel_threshold = opts.pixel_threshold
            mp_arg.patch = opts.patch
            mp_args.append(mp_arg)

    # Generate images in parallel
    with mp.get_context('spawn').Pool(processes=opts.num_workers) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(generate_image, mp_args), total=len(mp_args), desc='Generating images', unit='img'):
            pass

    print('done.', flush=True)


# ---------------------------------------------------------------------------------------------------------------------------
# Command line interface


# autopep8: off
@click.command(context_settings=dict(show_default=True, max_content_width=200))
# I/O settings
@click.option('--param_file',       type=click.Path(exists=True, dir_okay=False), required=True, help='Path to the pickle file containing IFS parameters, as produced by `python -m finefractaldb.ifs`.')
@click.option('--out_dir',          type=click.Path(file_okay=False), required=True, help='Dataset root directory. Images are saved under <out_dir>/images/<class>/ and parameter CSV files under <out_dir>/params/<class>/.')
# Rendering settings
@click.option('--num_points',       type=int, default=100000, help='Number of points to iterate for each fractal.')
@click.option('--noise_scale',      type=click.Tuple([float, float]), default=(0.6, 1.4), help='Range (low, high) of the scale factor applied to one randomly chosen transform to create each instance.')
@click.option('--image_size',       type=int, default=362, help='Side length of the output images in pixels (image_size x image_size).')
@click.option('--insts_per_cls',    type=int, default=1000, help='Number of image instances to render per class (per IFS).')
@click.option('--pixel_threshold',  type=float, default=0.0005, help='Minimum fraction of non-empty pixels required to accept a rendered image; sparser images are resampled.')
@click.option('--patch',            is_flag=True, default=False, help='Render each point as a random 3x3 patch instead of a single pixel.')
# Misc settings
@click.option('--seed',             type=int, default=0, help='Global random seed. Each image is seeded deterministically from this seed and its class/instance indices.')
@click.option('--num_workers',      type=int, default=os.cpu_count(), help='Number of parallel worker processes for image generation.')
# autopep8: on
def cmdline(**kwargs) -> None:
    """Render FineFractalDB images from a file of IFS parameters."""

    main(**kwargs)


def create_help_message() -> str:
    return textwrap.dedent("""Render FineFractalDB images from a file of IFS parameters.

For every IFS (class) in the parameter file, `insts_per_cls` images are rendered by randomly perturbing one transform
of the system. Images are written to <out_dir>/images/<class>/<class>_<inst>_<idx>.png and the perturbed parameters to
<out_dir>/params/<class>/<class>_<inst>.csv. Rendering is parallelized over `num_workers` processes.

Examples:

\b
# Render 1,000 images per class (1,000,000 images for 1,000 classes) with 3x3 patch rendering
python -m finefractaldb.render \\
    --param_file=datasets/finefractaldb/ifs_params.pkl \\
    --out_dir=datasets/finefractaldb \\
    --patch
"""
                           )


cmdline.help = create_help_message()

# ---------------------------------------------------------------------------------------------------------------------------
# Entry point


if __name__ == '__main__':
    cmdline()

# ---------------------------------------------------------------------------------------------------------------------------
