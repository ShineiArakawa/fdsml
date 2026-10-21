"""
IFS Functions
=============

A fast implementation of iterated function systems.

License
-------

Copyright (c) 2026 Shinei Arakawa
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
"""

import numba
import numpy as np

# ------------------------------------------------------------------------------------------------------------------------------------


@numba.njit(cache=True, fastmath=True, nogil=True)
def compute_points(
    A: np.ndarray,
    B: np.ndarray,
    p_cumsum: np.ndarray,
    rnds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:  # Enable JIT Compile with numba
    """Compute the IFS points. The original python native implementation was not durable for interactive use, so we use `numba` to speed it up.

    Parameters
    ----------
    A : np.ndarray
        The affine transformation matrices. Shape: `(num_params, 2, 2)`
    B : np.ndarray
        The translation vectors. Shape: `(num_params, 2)`
    p_cumsum : np.ndarray
        The cumulative sum of the probabilities. The components must be monotically increasing and the last component must be 1. Shape: `(num_params,)`
    rnds : np.ndarray
        The random numbers for selecting the affine transformations. The values must be in [0, 1). Shape: `(num_points,)`

    Returns
    -------
    np.ndarray, np.ndarray
        1. The computed points. Shape: `(num_points, 2)`
        2. The selected indices. Shape: `(num_points,)`
    """

    num_points = rnds.shape[0]

    points = np.zeros((num_points, 2), dtype=A.dtype)
    selected_indices: np.ndarray = np.zeros((num_points,), dtype=np.int64)

    # Decompose A and B for faster access
    A00, A01 = A[:, 0, 0], A[:, 0, 1]
    A10, A11 = A[:, 1, 0], A[:, 1, 1]
    B0, B1 = B[:, 0], B[:, 1]

    for i in range(num_points - 1):
        r = rnds[i]

        # Use binary search to find the index
        idx = np.searchsorted(p_cumsum, r, side='right')

        # Compute affine transformation
        points[i + 1, 0] = A00[idx] * points[i, 0] + A01[idx] * points[i, 1] + B0[idx]
        points[i + 1, 1] = A10[idx] * points[i, 0] + A11[idx] * points[i, 1] + B1[idx]

        selected_indices[i + 1] = idx

    return points, selected_indices

# ------------------------------------------------------------------------------------------------------------------------------------


@numba.njit(cache=True, fastmath=True, nogil=True)  # Enable JIT Compile with numba
def render_patch_pattern(
    img_size: int,
    points: np.ndarray,
    patch_pattern: np.ndarray,
) -> np.ndarray:
    """Render the patch pattern on the image. The color is fixed to white.

    Parameters
    ----------
    img_size : int
        The size of the image (assumed to be square).
    points : np.ndarray
        The point coordinates to render. Shape: `(num_points, 2)`
    patch_pattern : np.ndarray
        Boolean 2D array indicating the patch pattern. Shape: `(3, 3)`

    Returns
    -------
    np.ndarray
        The rendered image. Shape: `(img_size, img_size, 3)`
    """

    assert patch_pattern.shape == (3, 3), 'patch_pattern must be a 3x3 boolean array'

    img = np.zeros((img_size, img_size, 3), dtype=np.float32)

    num_points = points.shape[0]

    for i_point in range(num_points):
        center_x = int(points[i_point, 0])
        center_y = int(points[i_point, 1])

        for del_y in [-1, 0, 1]:
            y = center_y + del_y
            for del_x in [-1, 0, 1]:
                x = center_x + del_x
                if patch_pattern[del_y + 1][del_x + 1] and 0 <= x < img_size and 0 <= y < img_size:  # (patch_pattern?) and (in width?) and (in height?)
                    img[y, x, :] = 1.0  # White

    return img


@numba.njit(cache=True, fastmath=True, nogil=True)  # Enable JIT Compile with numba
def render_points_colored(
    img_size: int,
    points: np.ndarray,
    colors: np.ndarray,
) -> np.ndarray:
    """Render the points on the image with colors.

    Parameters
    ----------
    img_size : int
        The size of the image (assumed to be square).
    points : np.ndarray
        The point coordinates to render. Shape: `(num_points, 2)`
    colors : np.ndarray
        The colors for each point. Shape: `(num_points, 3)`

    Returns
    -------
    np.ndarray
        The rendered image. Shape: `(img_size, img_size, 3)`
    """

    img = np.zeros((img_size, img_size, 3), dtype=np.float32)

    num_points = points.shape[0]

    for i_point in range(num_points):
        x = int(points[i_point, 0])
        y = int(points[i_point, 1])
        color = colors[i_point]

        if 0 <= x < img_size and 0 <= y < img_size:
            img[y, x, :] = color

    return img


@numba.njit(cache=True, fastmath=True, nogil=True)  # Enable JIT Compile with numba
def render_patch_pattern_colored(
    img_size: int,
    points: np.ndarray,
    colors: np.ndarray,
    patch_pattern_array: np.ndarray,
    patch_pattern_indices: np.ndarray,
) -> np.ndarray:
    """Render the patch pattern on the image.

    Parameters
    ----------
    img_size : int
        The size of the image (assumed to be square).
    points : np.ndarray
        The point coordinates to render. Shape: `(num_points, 2)`
    colors : np.ndarray
        The colors for each point. Shape: `(num_points, 3)`
    patch_pattern_array : np.ndarray
        Boolean 2D array indicating the patch pattern. Shape: `(P, 3, 3)`
    patch_pattern_indices : np.ndarray
        The indices of the patch patterns for each point. Shape: `(num_points,)`

    Returns
    -------
    np.ndarray
        The rendered image. Shape: `(img_size, img_size, 3)`
    """

    img = np.zeros((img_size, img_size, 3), dtype=np.float32)

    num_points = points.shape[0]

    for i_point in range(num_points):
        patch_pattern_idx = patch_pattern_indices[i_point]
        center_x = int(points[i_point, 0])
        center_y = int(points[i_point, 1])
        color = colors[i_point]

        for del_y in [-1, 0, 1]:
            y = center_y + del_y
            for del_x in [-1, 0, 1]:
                x = center_x + del_x
                if 0 <= x < img_size and 0 <= y < img_size:
                    if patch_pattern_array[patch_pattern_idx, del_y + 1, del_x + 1]:
                        img[y, x, :] = color
                    else:
                        img[y, x, :] = 0.0  # Black

    return img


# ------------------------------------------------------------------------------------------------------------------------------------

def generate_image(
    A: np.ndarray,
    B: np.ndarray,
    rnds: np.ndarray,
    img_size: int,
    padding: int,
    patch_pattern: np.ndarray,
) -> np.ndarray | None:
    # Compute probabilities from determinants
    dets = np.linalg.det(A)
    probs = dets / np.sum(dets)
    p_cumsum = np.cumsum(probs)

    # Compute points
    points, _ = compute_points(A, B, p_cumsum, rnds)

    # Check nan or inf
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return None

    # Rescale points to fit in the image
    points_min = np.min(points, axis=0, keepdims=True)
    points_max = np.max(points, axis=0, keepdims=True)
    points: np.ndarray = (points - points_min) / (points_max - points_min + 1e-10)  # [0, 1]
    points = points * (img_size - 1 - 2 * padding) + padding  # Add the same amount of padding to all sides
    points = points.astype(np.int64)  # Convert to int for indexing

    # Render the image
    return render_patch_pattern(img_size, points, patch_pattern)

# ------------------------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    import contextlib
    import time

    import tqdm

    class _Timer:
        def __init__(self):
            self._elapsed_times = {}
            self._counter = {}

        @contextlib.contextmanager
        def timeit(self, name: str):
            start_time = time.perf_counter()
            yield
            end_time = time.perf_counter()

            elapsed_time = end_time - start_time

            if name not in self._elapsed_times:
                self._elapsed_times[name] = 0.0
                self._counter[name] = 0

            self._elapsed_times[name] += elapsed_time
            self._counter[name] += 1

        def print(self):
            for name, elapsed_time in self._elapsed_times.items():
                elapsed_time /= self._counter[name]
                print(f'{name}: {elapsed_time} [sec], {elapsed_time * 1000:.6f} [ms], {1.0 / elapsed_time:.2f} [FPS]')

    num_tries = 100
    num_iters = 200000
    img_size = 224
    padding = 6
    patch_pattern = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)  # Cross pattern
    timer = _Timer()

    for i_try in tqdm.trange(num_tries):
        rnd_gen = np.random.default_rng(seed=i_try)
        num_params = rnd_gen.integers(2, 9)

        A = rnd_gen.uniform(-1.0, 1.0, size=(num_params, 2, 2)).astype(np.float32)
        B = rnd_gen.uniform(-1.0, 1.0, size=(num_params, 2)).astype(np.float32)
        rnds = rnd_gen.uniform(0.0, 1.0, size=(num_iters,)).astype(np.float32)

        with timer.timeit('generate_image'):
            _ = generate_image(A, B, rnds, img_size, padding, patch_pattern)

    timer.print()
