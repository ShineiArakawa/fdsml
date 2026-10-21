"""FineFractalDB generation scripts: sampling of iterated function systems (IFS).

This module provides utilities to

1. sample random, contractive 2D affine iterated function systems (:func:`sample_system`),
2. generate point clouds of the attractor of an IFS by random iteration (:func:`iterate`), and
3. rasterize these point clouds into square binary images (:func:`render`).

Run this module as a script to sample a set of IFS parameters and save them to a pickle file, which can then be
rendered into images with ``python -m finefractaldb.render``.

License
-------

Copyright (c) 2026 Shinei Arakawa
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

This code is a derivative work based on code from the catalys1/fractal-pretraining repository, whose original code is licensed under the MIT License. The original code can be found at https://github.com/catalys1/fractal-pretraining.
"""

import os
import pickle
import pprint
import textwrap

import click
import numba
import numpy as np
import tqdm

import torch_utils

# ---------------------------------------------------------------------------------------------------------------------------
# Sampling of IFS parameters


def sample_svs(n, a, rng=None):
    r"""Sample ``n`` pairs of singular values whose weighted sum equals ``a``.

    The ``2n`` singular values :math:`s_{i,1}, s_{i,2}` (``i = 0, ..., n-1``) are sampled so that

    1. :math:`0 \le s_{i,2} \le s_{i,1} \le 1`, i.e. every pair describes a contraction, and
    2. :math:`\sum_{i=0}^{n-1} (s_{i,1} + 2 s_{i,2}) = a`.

    The pairs are sampled sequentially; the feasible interval of each value is derived from the remaining
    budget so that the constraint can still be satisfied by the remaining pairs.

    Parameters
    ----------
    n : int
        Number of pairs of singular values to sample (i.e. the number of affine transforms in the system).
    a : float
        Target value of the weighted sum of all singular values. Must lie in ``[0, 3 * n]``; values outside this
        range are clipped.
    rng : numpy.random.Generator, optional
        Random number generator. If ``None`` (default), ``numpy.random.default_rng()`` is used.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n, 2)``. Row ``i`` holds the singular values ``(s_{i,1}, s_{i,2})`` of the ``i``-th
        transform, sorted in descending order.
    """
    if rng is None:
        rng = np.random.default_rng()
    if a < 0:
        a = 0
    elif a > 3*n:
        a = 3*n
    s = np.empty((n, 2))
    p = a
    q = a - 3 * n + 3
    # sample the first 2*(n-1) singular values (first n-1 pairs)
    for i in range(n - 1):
        s1 = rng.uniform(max(0, q/3), min(1, p))
        q -= s1
        p -= s1
        s2 = rng.uniform(max(0, q/2), min(s1, p/2))
        q = q - 2 * s2 + 3
        p -= 2 * s2
        s[i, :] = s1, s2
    # sample the last pair of singular values
    s2 = rng.uniform(max(0, (p-1)/2), p/3)
    s1 = p - 2*s2
    s[-1, :] = s1, s2

    return s


sample_fn_catalog = {
    'sample_svs': sample_svs,
}
"""Mapping from name to function for sampling singular values. Used to resolve the ``--sample_fn`` CLI option."""


def sample_system(n=None, constrain=True, bval=1, rng=None, beta=None, sample_fn=None):
    r"""Sample a random iterated function system (IFS) of ``n`` 2D affine transforms.

    Each transform is represented by a ``2 x 3`` matrix ``[A | b]`` that maps a point ``x`` to ``A @ x + b``.
    When ``constrain`` is ``True``, the linear part ``A`` of each transform is built from its singular value
    decomposition ``A = U @ diag(s) @ V`` with random orthonormal ``U``, ``V`` and singular values ``s < 1``, so
    that every transform is a contraction and the attractor of the system is well-defined.

    Parameters
    ----------
    n : int, range, tuple of int, list of int, or None, optional
        Number of transforms in the system. If an ``int``, it is used as is. If a ``range`` or a ``(low, high)``
        tuple/list, ``n`` is drawn uniformly from ``[low, high)``. If ``None`` (default), ``n`` is drawn from
        ``range(2, 8)``.
    constrain : bool, optional
        If ``True`` (default), enforce contractivity of the transforms via the singular value constraint. If
        ``False``, all ``2 x 3`` entries are drawn i.i.d. from ``Uniform(-1, 1)`` (the original FractalDB
        behaviour), and ``bval``, ``beta`` and ``sample_fn`` are ignored.
    bval : float, optional
        Maximum magnitude of the translation parameters ``b``, which are drawn from ``Uniform(-bval, bval)``.
        Translation does not affect contractivity. Default is ``1``.
    rng : numpy.random.Generator, optional
        Random number generator. If ``None`` (default), ``numpy.random.default_rng()`` is used.
    beta : tuple of float, optional
        Range ``(low, high)`` from which the target weighted sum of singular values ``a`` is drawn uniformly
        (see :func:`sample_svs`). If ``None`` (default), ``((5 + n) / 2, (6 + n) / 2)`` is used.
    sample_fn : callable, optional
        Function used to sample singular values. It must have the signature ``sample_fn(n, a, rng)`` and return
        an array of shape ``(n, 2)``. If ``None`` (default), :func:`sample_svs` is used.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n, 2, 3)`` containing the ``n`` affine transforms ``[A | b]``.
    """
    if rng is None:
        rng = np.random.default_rng()
    if n is None:
        n = rng.integers(2, 8)
    elif isinstance(n, range):
        n = rng.integers(n.start, n.stop)
    elif isinstance(n, (tuple, list)):
        n = rng.integers(*n)

    if beta is None:
        beta = ((5 + n) / 2, (6 + n) / 2)

    if sample_fn is None:
        sample_fn = sample_svs

    if constrain:
        # sample a matrix with singular values < 1 (a contraction)
        # 1. sample the singular vectors--random orthonormal matrices--by randomly rotating the standard basis
        base = np.sign(rng.random((2*n, 2, 1)) - 0.5) * np.eye(2)
        angle = rng.uniform(-np.pi, np.pi, 2*n)
        ss = np.sin(angle)
        cc = np.cos(angle)
        rmat = np.empty((2 * n, 2, 2))
        rmat[:, 0, 0] = cc
        rmat[:, 0, 1] = -ss
        rmat[:, 1, 0] = ss
        rmat[:, 1, 1] = cc
        uv = rmat @ base
        u, v = uv[:n], uv[n:]
        # 2. sample the singular values
        a = rng.uniform(*beta)
        s = sample_fn(n, a, rng)
        # 3. sample the translation parameters from Uniform(-bval, bval) and create the transformation matrix
        m = np.empty((n, 2, 3))
        m[:, :, :2] = u * s[:, None, :] @ v
        m[:, :, 2] = rng.uniform(-bval, bval, (n, 2))
    else:
        m = rng.uniform(-1, 1, (n, 2, 3))

    return m


def random_systems(num_systems, n=(2, 5), bval=None, beta=None, sample_fn=None, rng=None):
    """Sample multiple random iterated function systems.

    Parameters
    ----------
    num_systems : int
        Number of systems to sample.
    n : int or tuple of int, optional
        Number of transforms per system, or a ``(low, high)`` range to draw it from (``high`` is exclusive).
        See :func:`sample_system`. Default is ``(2, 5)``.
    bval : float, optional
        Maximum magnitude of the translation parameters. See :func:`sample_system`.
    beta : tuple of float, optional
        Range for the weighted sum of singular values. See :func:`sample_system`.
    sample_fn : callable, optional
        Function used to sample singular values. See :func:`sample_system`.
    rng : numpy.random.Generator, optional
        Random number generator shared by all samples. If ``None`` (default), ``numpy.random.default_rng()`` is
        used.

    Returns
    -------
    list of dict
        List of length ``num_systems``. Each element is a dict ``{'system': numpy.ndarray}`` whose array has
        shape ``(n, 2, 3)``.
    """
    if rng is None:
        rng = np.random.default_rng()
    systems = []
    for i in tqdm.trange(num_systems):
        s = sample_system(n, bval=bval, beta=beta, sample_fn=sample_fn, rng=rng)
        systems.append({'system': s})
    return systems


# ---------------------------------------------------------------------------------------------------------------------------
# Evaluation of IFS (point cloud generation)


@numba.njit(cache=True)
def iterate(sys, rnds, ps=None):
    """Generate points on the attractor of an IFS by random iteration (the "chaos game").

    Starting from the fixed point of the first transform, one transform is chosen at each step according to the
    probabilities ``ps`` and applied to the current point. The sequence of visited points approximates the
    attractor of the system.

    Parameters
    ----------
    sys : numpy.ndarray
        Array of shape ``(n, 2, 3)`` containing the ``n`` affine transforms ``[A | b]`` of the system.
    rnds : numpy.ndarray
        1D array of uniform random numbers in ``[0, 1)``. One transform is selected per element, so the number
        of iterations (and of returned points) equals ``len(rnds)``.
    ps : numpy.ndarray, optional
        Array of shape ``(n,)`` with the selection probability of each transform. If ``None`` (default), the
        probabilities are proportional to the absolute determinants of the linear parts ``A``.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(len(rnds), 2)`` containing the ``(x, y)`` coordinates of the generated points. If the
        iteration diverges (non-contractive system), the iteration stops early and the remaining rows are left
        uninitialized.
    """
    n_iter = len(rnds)

    det = sys[:, 0, 0] * sys[:, 1, 1] - sys[:, 0, 1] * sys[:, 1, 0]
    if ps is None:
        ps = np.abs(det)
        ps = ps / ps.sum()
    ps = np.cumsum(ps)
    coords = np.empty((n_iter, 2))

    # starting point is $v = (I-A_1)^(-1) b$ since this point is gaurenteed to be in the set
    # (assuming that A_1 is contractive) (A_1 = sys[0])
    s = 1 / (1 + det[0] - sys[0, 0, 0] - sys[0, 1, 1])
    x = s * ((1 - sys[0, 1, 1]) * sys[0, 0, 2] + sys[0, 0, 1] * sys[0, 1, 2])
    y = s * ((1 - sys[0, 0, 0]) * sys[0, 1, 2] + sys[0, 1, 0] * sys[0, 0, 2])

    for i in range(n_iter):
        r = rnds[i]
        for k in range(len(ps)):
            if r < ps[k]:
                break
        a, b, e, c, d, f = sys[k].ravel()
        xt = x
        x = a * xt + b * y + e
        y = c * xt + d * y + f
        coords[i] = x, y
        if not np.isfinite(x) or not np.isfinite(y):
            break  # if contractivity is satisfied, can remove this check

    return coords


# ---------------------------------------------------------------------------------------------------------------------------
# Rendering


@numba.njit(cache=True)
def minmax(coords):
    """Compute the column-wise minimum and maximum of a point array in a single pass.

    This is faster than calling ``numpy.min`` and ``numpy.max`` separately because the array is traversed
    only once.

    Parameters
    ----------
    coords : numpy.ndarray
        Array of shape ``(n, 2)`` containing ``(x, y)`` coordinates.

    Returns
    -------
    mins : numpy.ndarray
        Array of shape ``(2,)`` holding ``(min_x, min_y)``.
    maxs : numpy.ndarray
        Array of shape ``(2,)`` holding ``(max_x, max_y)``.
    """
    mins = np.full(2, np.inf)
    maxs = np.full(2, -np.inf)
    for i in range(len(coords)):
        x, y = coords[i]
        if x < mins[0]:
            mins[0] = x
        if y < mins[1]:
            mins[1] = y
        if x > maxs[0]:
            maxs[0] = x
        if y > maxs[1]:
            maxs[1] = y
    return mins, maxs


@numba.njit(cache=True)
def _extent(region):
    """Return the width and height of a region, guarding against zero-size spans.

    Parameters
    ----------
    region : numpy.ndarray
        Array of shape ``(4,)`` holding ``(min_x, min_y, max_x, max_y)``.

    Returns
    -------
    xspan : float
        ``max_x - min_x``, or ``1`` if the span is not positive.
    yspan : float
        ``max_y - min_y``, or ``1`` if the span is not positive.
    """
    x1, y1, x2, y2 = region
    xspan = x2 - x1
    xspan = xspan if xspan > 0 else 1
    yspan = y2 - y1
    yspan = yspan if yspan > 0 else 1
    return xspan, yspan


@numba.njit(cache=True)
def _render_binary_patch(coords, s, region, patch):
    """Rasterize points into a square binary image, drawing a ``3 x 3`` patch at each point.

    For every point, the ``3 x 3`` neighbourhood of its pixel is first cleared and then the pixels selected by
    ``patch`` are set to ``1``. Later points therefore overwrite earlier ones.

    Parameters
    ----------
    coords : numpy.ndarray
        Array of shape ``(n, 2)`` containing ``(x, y)`` coordinates.
    s : int
        Side length of the output image in pixels.
    region : numpy.ndarray
        Array of shape ``(4,)`` holding ``(min_x, min_y, max_x, max_y)``. This region is mapped onto the image;
        points outside of it are ignored.
    patch : numpy.ndarray
        Integer array of shape ``(k, 2)`` listing the ``(row, col)`` offsets, each in ``{0, 1, 2}``, of the
        pixels to set within the ``3 x 3`` neighbourhood (``(1, 1)`` is the centre).

    Returns
    -------
    numpy.ndarray
        Binary image of dtype ``uint8`` and shape ``(s, s)``.
    """
    imgb = np.zeros((s, s), dtype=np.uint8)
    xspan, yspan = _extent(region)
    xscale = (s-1) / xspan
    yscale = (s-1) / yspan
    xmin, ymin = region[0], region[1]
    for i in range(len(coords)):
        rr = int((coords[i, 0] - xmin) * xscale)
        cc = int((coords[i, 1] - ymin) * yscale)

        for j in range(3):
            for k in range(3):
                r = rr + j - 1
                c = cc + k - 1
                if r >= 0 and r < s and c >= 0 and c < s:
                    imgb[r, c] = 0

        for j in range(len(patch)):
            r = rr + patch[j, 0] - 1
            c = cc + patch[j, 1] - 1
            if r >= 0 and r < s and c >= 0 and c < s:
                imgb[r, c] = 1
    return imgb


@numba.njit(cache=True)
def _render_binary(coords, s, region):
    """Rasterize points into a square binary image, setting one pixel per point.

    Parameters
    ----------
    coords : numpy.ndarray
        Array of shape ``(n, 2)`` containing ``(x, y)`` coordinates.
    s : int
        Side length of the output image in pixels.
    region : numpy.ndarray
        Array of shape ``(4,)`` holding ``(min_x, min_y, max_x, max_y)``. This region is mapped onto the image;
        points outside of it are ignored.

    Returns
    -------
    numpy.ndarray
        Binary image of dtype ``uint8`` and shape ``(s, s)``.
    """
    imgb = np.zeros((s, s), dtype=np.uint8)
    xspan, yspan = _extent(region)
    xscale = (s-1) / xspan
    yscale = (s-1) / yspan
    xmin, ymin = region[0], region[1]
    for i in range(len(coords)):
        r = int((coords[i, 0] - xmin) * xscale)
        c = int((coords[i, 1] - ymin) * yscale)
        if r >= 0 and r < s and c >= 0 and c < s:
            imgb[r, c] = 1
    return imgb


def render(coords, s=256, patch=False):
    """Render a point cloud as a square binary image.

    The bounding box of ``coords`` is mapped onto the full image, so the fractal always fills the frame.

    Parameters
    ----------
    coords : numpy.ndarray
        Array of shape ``(n, 2)`` containing ``(x, y)`` coordinates, e.g. the output of :func:`iterate`.
    s : int, optional
        Side length of the output image in pixels. Default is ``256``.
    patch : bool, optional
        If ``False`` (default), each point is drawn as a single pixel. If ``True``, each point is drawn as a
        random ``3 x 3`` patch; the patch shape is sampled once per call (each of the 9 pixels is included with
        probability 0.5) using the global ``numpy.random`` state. An all-empty patch is rejected and resampled,
        so at least one pixel is always drawn per point.

    Returns
    -------
    numpy.ndarray
        Binary image of dtype ``uint8`` and shape ``(s, s)``.
    """
    region = np.concatenate(minmax(coords))
    if patch:
        mask = np.random.randint(0, 2, (9,), dtype=bool)
        while not mask.any():
            # reject an empty patch, which would otherwise produce an all-black image
            mask = np.random.randint(0, 2, (9,), dtype=bool)
        p = np.stack(np.divmod(np.arange(9)[mask], 3), 1)
        return _render_binary_patch(coords, s, region, p)
    else:
        return _render_binary(coords, s, region)


# ---------------------------------------------------------------------------------------------------------------------------
# Main function


def main(**kwargs) -> None:
    """Sample random IFS parameters and optionally save them to a pickle file."""

    opts = torch_utils.EasyDict(kwargs)
    print(f'Arguments: {pprint.pformat(opts, indent=4)}', flush=True)

    rng = np.random.default_rng(opts.seed)
    np.random.seed(opts.seed)

    beta = None
    if opts.beta_min is not None:
        beta_max = opts.beta_max if opts.beta_max is not None else opts.beta_min
        beta = (opts.beta_min, beta_max)

    sample_fn = sample_fn_catalog[opts.sample_fn]

    hparams = dict(
        num_systems=opts.num_systems,
        n=(opts.min_n, opts.max_n + 1),
        bval=opts.bval,
        beta=beta,
        sample_fn=sample_fn,
    )
    systems = random_systems(**hparams, rng=rng)

    # Save the sampled systems to a pickle file
    save_path = os.path.abspath(opts.save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    hparams['sample_fn'] = opts.sample_fn  # store the name rather than the function object
    hparams['seed'] = opts.seed
    with open(save_path, 'wb') as f:
        pickle.dump({'params': systems, 'hparams': hparams}, f)
    print(f'Saved {len(systems)} IFS parameters to {save_path}', flush=True)


# ---------------------------------------------------------------------------------------------------------------------------
# Command line interface


# autopep8: off
@click.command(context_settings=dict(show_default=True, max_content_width=200))
# I/O settings
@click.option('--save_path',    type=click.Path(dir_okay=False), required=True, help='Path to the output pickle file.')
# Sampling settings
@click.option('--num_systems',  type=int, default=1000, help='Number of IFS (i.e. classes) to sample.')
@click.option('--min_n',        type=int, default=2, help='Minimum number of affine transforms per system (inclusive).')
@click.option('--max_n',        type=int, default=4, help='Maximum number of affine transforms per system (inclusive).')
@click.option('--bval',         type=float, default=1.0, help='Maximum magnitude of the translation parameters, which are drawn from Uniform(-bval, bval).')
@click.option('--beta_min',     type=float, default=None, help='Lower bound of the range for the weighted sum of singular values. If not set, the default range ((5 + n) / 2, (6 + n) / 2) is used.')
@click.option('--beta_max',     type=float, default=None, help='Upper bound of the range for the weighted sum of singular values. Ignored unless --beta_min is set; defaults to --beta_min.')
@click.option('--sample_fn',    type=click.Choice(list(sample_fn_catalog.keys())), default='sample_svs', help='Function used to sample the singular values of each transform.')
# Misc settings
@click.option('--seed',         type=int, default=0, help='Random seed for reproducibility.')
# autopep8: on
def cmdline(**kwargs) -> None:
    """Sample random iterated function system (IFS) parameters for FineFractalDB."""

    main(**kwargs)


def create_help_message() -> str:
    return textwrap.dedent("""Sample random iterated function system (IFS) parameters for FineFractalDB.

Each system consists of 2 to 4 (by default) contractive 2D affine transforms. The sampled systems are saved to a pickle
file as a dict {'params': [{'system': ndarray(n, 2, 3)}, ...], 'hparams': {...}}, which is consumed by
`python -m finefractaldb.render` to generate fractal images.

Examples:

\b
# Sample 1,000 IFS parameters and save them to a pickle file
python -m finefractaldb.ifs \\
    --save_path=datasets/finefractaldb/ifs_params.pkl \\
    --num_systems=1000
"""
                           )


cmdline.help = create_help_message()

# ---------------------------------------------------------------------------------------------------------------------------
# Entry point


if __name__ == '__main__':
    cmdline()

# ---------------------------------------------------------------------------------------------------------------------------
