"""
Grid IFS Visualizer
===================

A grid visualizer for iterated function systems (IFS) inspired by FractalDB and Improved IFS. This tool allows users to explore how different IFS parameters affect the resulting fractal patterns by visualizing them in a grid layout.

See Also
---

- [FractalDB: Pre-training without Natural Images](https://arxiv.org/abs/2101.08515)
- [Improving Fractal Pre-training](https://arxiv.org/abs/2110.03091)

Memo
----
- 2025-09-25
    - In the FractalDB paper, the order of matrix elements is shown in column-major order, but in the official implementation in ifs_function.py (lines 64 and 65), it is actually in row-major order.

License
-------

Copyright (c) 2026 Shinei Arakawa
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
"""

# autopep8: off
# isort: skip_file

from __future__ import annotations

import argparse
import concurrent.futures as futures
import copy
import dataclasses
import datetime
import functools
import hashlib
import itertools
import json
import math
import os
import pathlib
import time
import typing

import torch  # NOTE: Make sure to import torch before pyviewer_extended to avoid potential conflicts

import cachetools
import cmap
import ifs_func
import natsort
import nfdpy
import numba
import numpy as np
import OpenGL.GL as gl
import PIL.Image
import PIL.PngImagePlugin
import pydantic
import resize_right
import typing_extensions
import ui.async_pool as async_pool
import ui.docking_viewer as docking_viewer
import ui.drawable_window as drawable_window
import ui.utils as ui_utils
from imgui_bundle import hello_imgui, imgui  # type: ignore

# autopep8: off

# ------------------------------------------------------------------------------------------------------------
# Logger

logger = ui_utils.get_logger()

# --------------------------------------------------------------------------------------------------------------------------------
# Constants

WINDOW_TITLE = 'Grid IFS Visualizer'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
compute_dtype = np.float64

# Available color maps
AVAILABLE_CMAPS: typing.Final[list[str]] = natsort.natsorted(list(cmap.available_color_maps()))
AVAILABLE_CMAPS = ['constant', 'gray'] + [c for c in AVAILABLE_CMAPS]

PATCH_PATTERNS: typing.Final[np.ndarray] = np.stack([
    np.array(pattern, dtype=bool).reshape((3, 3)) for pattern in list(itertools.product([0, 1], repeat=9)) if np.any(np.array(pattern, dtype=bool) == True)
], axis=0)  # Shape: (P, 3, 3) where P is the number of patterns (excluding all-zero pattern)

SQRT_INT64_MAX: typing.Final[int] = math.floor(math.sqrt(2**63 - 1)) # 3037000499

DEFAULT_IMG_SAVE_DIR: typing.Final[pathlib.Path] = pathlib.Path('./grid_ifs_viz_imgs').resolve()


# --------------------------------------------------------------------------------------------------------------------------------
# Enums and Data classes

class SeedGridMode(ui_utils.StrEnum):
    random = 'Random'
    fixed_x = 'Fixed along X'
    fixed_y = 'Fixed along Y'
    fixed = 'Fixed everywhere'

class ParamAxis(ui_utils.StrEnum):
    none = 'None'
    lerp_sigma_factor = 'Lerp Sigma Factor'
    interleave_num_params = 'Number of Parameters'
    lerp_angle = 'Lerp Angle'

class IFSMethod(ui_utils.StrEnum):
    original = 'FractalDB'
    improved = 'Improved IFS'

class SigmaFactorMethod(ui_utils.StrEnum):
    auto = 'Auto'
    manual = 'Manual'

class NumParamsMode(ui_utils.StrEnum):
    range_based = 'Range Based'
    fixed = 'Fixed'

@pydantic.dataclasses.dataclass
class UIState:
    # autopep8: off
    H                             : int               = 8
    W                             : int               = 8
    x_axis                        : ParamAxis         = ParamAxis.none
    y_axis                        : ParamAxis         = ParamAxis.none
    seed_mode                     : SeedGridMode      = SeedGridMode.random
    ifs_method                    : IFSMethod         = IFSMethod.original
    sigma_factor_method           : SigmaFactorMethod = SigmaFactorMethod.auto
    b_range                       : float             = 1.0
    min_sigma_factor              : float             = 4.0
    max_sigma_factor              : float             = 4.5
    seed                          : int               = 100
    img_size                      : int               = 362
    num_steps                     : int               = 200000
    num_params_mode               : NumParamsMode     = NumParamsMode.range_based
    min_num_params                : int               = 2
    max_num_params                : int               = 8
    padding                       : int               = 6
    enable_patch_pattern          : bool              = True
    enable_filling_rate_test      : bool              = False
    filling_rate_threshold        : float             = 0.01
    filling_rate_test_resolution  : int               = 512
    max_num_attempts              : int               = 100
    cmap                          : str               = AVAILABLE_CMAPS[-1]  # maybe 'viridis'
    debug_mode                    : bool              = False
    # autopep8: on

    def deterministic_hash(self) -> str:
        """Compute a deterministic hash of the UI state, which is stable across sessions.
        """

        state_dict = self.asdict()
        state_bytes = json.dumps(state_dict).encode()

        return hashlib.sha256(state_bytes).hexdigest()

    # ------------------------------------------------------------------------------------
    # Serialization

    def asdict(self) -> dict:
        return dataclasses.asdict(self)

    @staticmethod
    def fromdict(state_dict: dict) -> UIState:
        # Filtering
        valid_fields = {field.name for field in dataclasses.fields(UIState)}
        state_dict = {k: v for k, v in state_dict.items() if k in valid_fields}

        return UIState(**state_dict)


@pydantic.dataclasses.dataclass(config=pydantic.config.ConfigDict(arbitrary_types_allowed=True))
class RenderState:
    # autopep8: off
    state_hash                         : str

    img_size                           : list[int]

    order                              : list[tuple[int, int]]
    i                                  : int
    start_idx                          : int
    end_idx                            : int

    start_time                         : float
    end_time                           : float | None

    grid                               : torch.Tensor
    # autopep8: on


@pydantic.dataclasses.dataclass(config=pydantic.config.ConfigDict(arbitrary_types_allowed=True))
class RenderResult:
    img_tensor: torch.Tensor
    filling_rate: float | None
    num_params: int
    x: int
    y: int
    seed: int
    num_attempts: int
    sigma_factor: float | None
    avg_theta: float | None = None
    avg_phi: float | None = None

    def copy(self) -> RenderResult:
        return RenderResult(
            img_tensor=self.img_tensor.clone(),
            filling_rate=self.filling_rate,
            num_params=self.num_params,
            x=self.x,
            y=self.y,
            seed=self.seed,
            num_attempts=self.num_attempts,
            sigma_factor=self.sigma_factor,
            avg_theta=self.avg_theta,
            avg_phi=self.avg_phi,
        )

# --------------------------------------------------------------------------------------------------------------------------------
# Rendering functions


@numba.njit(cache=True, fastmath=True, nogil=True)  # Enable JIT Compile with numba
def render_points(
    img_size: int,
    points: np.ndarray,
    fill_value: float,
) -> np.ndarray:
    """Render the patch pattern on the image. The color is fixed to white.

    Parameters
    ----------
    img_size : int
        The size of the image (width, height).
    points : np.ndarray
        The point coordinates to render. Shape: `(num_points, 2)`
    fill_value : float
        The value to fill in the patch pixels.

    Returns
    -------
    np.ndarray
        The rendered image. Shape: `(img_size, img_size)`
    """

    img = np.zeros((img_size, img_size), dtype=np.float32)

    num_points = points.shape[0]

    for i_point in range(num_points):
        center_x = int(points[i_point, 0])  # width direction
        center_y = int(points[i_point, 1])  # height direction

        if 0 <= center_x < img_size and 0 <= center_y < img_size:  # within the image?
            img[center_y, center_x] = fill_value

    return img


@functools.lru_cache(maxsize=8)
def create_diagonal_stripes(width: int = 1920, height: int = 1080, stripe_width: int = 128, color_a: tuple[float, float, float] = (1., 1., 1.), color_b: tuple[float, float, float] = (1., 0., 0.), anti_aliasing: bool = True, mag_factor: int = 4) -> np.ndarray:
    """Create a 45° diagonal stripe image alternating color_a and color_b. Stripes run along lines of constant (x + y).

    Parameters
    ----------
    width : int
        The width of the image.
    height : int
        The height of the image.
    stripe_width : int
        The width of each stripe.
    color_a : tuple[float, float, float]
        The RGB color for the first stripe.
    color_b : tuple[float, float, float]
        The RGB color for the second stripe.
    anti_aliasing : bool
        Whether to apply anti-aliasing using super-sampling.
    mag_factor : int
        Super-sampling factor for anti-aliasing.

    Returns
    -------
    np.ndarray
        The generated stripe image of shape (height, width, 3).
    """

    if not anti_aliasing:
        mag_factor = 1

    mag_width = width * mag_factor
    mag_height = height * mag_factor

    x = np.arange(mag_width)[None, :]           # shape (1, W)
    y = np.arange(mag_height)[:, None]          # shape (H, 1)
    bands = ((x + y) // stripe_width) % 2   # 0 or 1 alternating diagonally

    img = np.empty((mag_height, mag_width, 3), dtype=np.float32)

    img[bands == 0] = color_a
    img[bands == 1] = color_b

    if anti_aliasing:
        img = resize_right.resize(img, out_shape=(height, width, 3), interp_method=resize_right.interp_methods.lanczos3)

    return img

# --------------------------------------------------------------------------------------------------------------------------------
# Utilities


def space_integers(low: int, high: int, num: int) -> np.ndarray:
    range_size = high - low + 1
    if num >= range_size and num % range_size == 0:
        reps = num // range_size
        # Create interleaved array
        spaced_integers = np.arange(low, high + 1, dtype=np.int64).reshape((range_size, 1)).repeat(reps, axis=1).flatten()
    else:
        spaced_integers = np.linspace(low, high, num=num, dtype=np.int64, endpoint=True)

    return spaced_integers

# --------------------------------------------------------------------------------------------------------------------------------
# Visualizer class


class GridIFSViz(docking_viewer.MultiTexturesDockingViewer):
    # ------------------------------------------------------------------------------------
    # Constants

    # autopep8: off
    KEY_GRID_WINDOW     : typing.Final[str] = 'Grid window'

    MIN_IMG_SIZE        : int               = 5    # it has to be larger than 4
    MAX_IMG_SIZE        : int               = 2048 # 256 is already heavy with padding even if using CUDA backend ...

    MAX_NUM_PARAMS      : int               = 8    # according to the original FractalDB paper

    STATES_CACHE_PATH   : pathlib.Path      = pathlib.Path('.cache/grid_ifs_viz_params.json')
    NUM_STATES_CACHES   : int               = 9    # should be < 10 due to the limit of number keys
    # autopep8: on

    # ------------------------------------------------------------------------------------

    def __init__(self, cache_states: bool = True, cache_ui_layout: bool = True, enable_vsync: bool = True, **kwargs):
        self.cur_cache_param_id = 0
        self.cur_selected_ifs_param_id = -1  # No selection
        self.cached_states: list[UIState] = [UIState() for _ in range(self.NUM_STATES_CACHES)]
        self.cache_params_to_file = cache_states
        self.cache_ui_layout = cache_ui_layout

        self.status_msg = ''
        self.error_msg = ''

        self.worker_pool = async_pool.MultiThreadWorkerPool(num_workers=os.cpu_count())

        # -----------------------------------------------------------------------------

        if not self.cache_ui_layout:
            # Remove ini file ... ugly
            # .ini for window and app state saving'

            ini_filename = WINDOW_TITLE.lower().strip().replace(' ', '_') + '.ini'
            ini_path = pathlib.Path(hello_imgui.ini_folder_location(hello_imgui.IniFolderType.app_user_config_folder)) / ini_filename
            if ini_path.is_file():
                os.remove(str(ini_path))
                logger.info(f'Removed {ini_path}')

        # -----------------------------------------------------------------------------

        super(GridIFSViz, self).__init__(
            name=WINDOW_TITLE,
            texture_names=[self.KEY_GRID_WINDOW,],
            enable_vsync=enable_vsync,
            hightlight_on_hover_mode=[drawable_window.HighlightMode.grid_cells,],
            full_screen_mode=hello_imgui.FullScreenMode.full_monitor_work_area,
            **kwargs
        )

    # ------------------------------------------------------------------------------------
    # Parameter cache handling

    def _load_param_caches(self, file_path: pathlib.Path) -> None:
        try:
            with open(file_path, mode='r') as file:
                state_dict = json.load(file)
        except Exception as e:
            state_dict = None

        if state_dict is not None and isinstance(state_dict, dict):
            # Load params
            params = state_dict.get('params', [])
            if isinstance(params, list):
                for i_param, param_dict in enumerate(params):
                    if i_param >= len(self.cached_states):
                        continue
                    self.cached_states[i_param] = UIState.fromdict(param_dict)

                self.state = self.cached_states[self.cur_cache_param_id]

            # Load common settings
            common_dict = state_dict.get('common', {})
            if isinstance(common_dict, dict):
                # Load common settings
                self.img_save_dir = pathlib.Path(common_dict.get('img_save_dir', str(DEFAULT_IMG_SAVE_DIR))).resolve()
                if 'attrib_txt_color' in common_dict:
                    self.texture_windows[self.KEY_GRID_WINDOW].attrib_txt_color = common_dict['attrib_txt_color']

            logger.info(f'Loaded parameters from {file_path}')
        else:
            logger.warning(f'Failed to load parameters from {file_path}')

    def _save_param_caches(self, file_path: pathlib.Path) -> None:
        params: list[dict] = []
        for cached_param in self.cached_states:
            params.append(cached_param.asdict())

        state_dict = {
            'params': params,
            'common': {
                'img_save_dir': str(self.img_save_dir),
                'attrib_txt_color': self.texture_windows[self.KEY_GRID_WINDOW].attrib_txt_color,
            }
        }

        cache_file = file_path.resolve()
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(cache_file, mode='w', encoding='utf-8') as file:
                json.dump(state_dict, file, indent=4, ensure_ascii=False)
            logger.info(f'Saved parameters to {cache_file}')
        except Exception as e:
            logger.error(f'Failed to save parameters to {cache_file}')
            logger.error(e)

    # ------------------------------------------------------------------------------------
    # Utilities

    def _get_lds(self, W: int, H: int) -> list[tuple[int, int]]:
        """Low discrepancy ordering"""

        def phi(d, iter=20) -> float:
            x = 2.0
            for _ in range(iter):
                x = pow(1+x, 1/(d+1))
            return x

        phis = [1 / phi(2), 1/phi(2)**2]  # inverse powers of plastic number

        def R2(i: int, dim: int) -> float:
            return (phis[dim] * i) % 1

        pos = []  # ordering, (x, y)
        occ = np.zeros([W, H], dtype=np.int8)
        i = 0
        while len(pos) < W * H:
            x = np.floor(W * R2(i, 0)).astype(np.int32)  # strictly < W
            y = np.floor(H * R2(i, 1)).astype(np.int32)
            i += 1
            if not occ[x, y]:
                occ[x, y] = 1
                pos.append((x, y))

        return pos

    # ------------------------------------------------------------------------------------
    # Overrides

    @typing_extensions.override
    def draw_status_bar(self):
        return super().draw_status_bar()

    @typing_extensions.override
    def setup_state(self):
        # Set highlight color
        self.texture_windows[self.KEY_GRID_WINDOW].highlight_color_pos = drawable_window.DrawableTextureWindowUnit.HIGHTLIGHT_COLOR_NEG
        self.texture_windows[self.KEY_GRID_WINDOW].attrib_txt_color = drawable_window.DrawableTextureWindowUnit.HIGHTLIGHT_COLOR_NEG

        # Initialize state
        self.state = UIState()

        self.render_cache: dict[str, RenderResult] = cachetools.FIFOCache(maxsize=1 << 12)  # Cache up to 4096 rendered images
        self.render_state: RenderState = None

        self.elapsed_time_cache: dict[str, float] = cachetools.FIFOCache(maxsize=256)  # Cache up to 256 elapsed times

        self.params_cache_file = self.STATES_CACHE_PATH

        self.img_save_dir = DEFAULT_IMG_SAVE_DIR

        self.gl_max_tex_size: int = gl.glGetIntegerv(gl.GL_MAX_TEXTURE_SIZE)

        # Load params
        if self.cache_params_to_file and self.params_cache_file.is_file():
            self._load_param_caches(self.params_cache_file)

        # Update param cache
        self.cached_states[self.cur_cache_param_id] = self.state

    @typing_extensions.override
    def save_settings(self) -> None:
        if self.cache_params_to_file:
            self._save_param_caches(self.params_cache_file)

    @typing_extensions.override
    def drag_and_drop_callback(self, paths):
        if paths is not None and len(paths) > 0:
            file_path: pathlib.Path = paths[0]
            if not file_path.is_file():
                # Abort
                return

            if file_path.suffix.lower() == '.json':
                # Load params
                self._load_param_caches(file_path)
            elif file_path.suffix.lower() == '.png':
                # Load metadata from PNG
                try:
                    img_pil = PIL.Image.open(file_path)
                    png_info = img_pil.info
                    params_json = png_info.get('Parameters', None)
                    if params_json is not None:
                        state_dict = json.loads(params_json)
                        state = UIState.fromdict(state_dict)
                        self.cached_states[self.cur_cache_param_id] = state
                        self.state = self.cached_states[self.cur_cache_param_id]
                        logger.info(f'Loaded parameters from PNG metadata: {file_path}')
                    else:
                        logger.warning(f'No parameter metadata found in PNG: {file_path}')
                except Exception as e:
                    logger.warning(f'Failed to load parameters from PNG metadata: {file_path}')
                    logger.error(e)

    @typing_extensions.override
    def setup_docking_layout(self, layout_funcs):
        """Generate a graceful initial layout."""

        splits = [
            # autopep8: off
            # Split columns first
            hello_imgui.DockingSplit('MainDockSpace', 'Dock01', imgui.Dir.right, ratio_=5/6),
            # Then split rows
            hello_imgui.DockingSplit('MainDockSpace', 'Dock10', imgui.Dir.down, ratio_=12.5/15),
            hello_imgui.DockingSplit('Dock10', 'Dock20', imgui.Dir.down, ratio_=7.5/12.5),
            hello_imgui.DockingSplit('Dock20', 'Dock30', imgui.Dir.down, ratio_=5.5/7.5),
            hello_imgui.DockingSplit('Dock30', 'Dock40', imgui.Dir.down, ratio_=3.5/5.5),
            hello_imgui.DockingSplit('Dock40', 'Dock50', imgui.Dir.down, ratio_=1/3.5),
            # autopep8: on
        ]

        title_to_dockspace_name = {
            # autopep8: off
            'toolbar_param_cache'        : 'MainDockSpace', self.KEY_GRID_WINDOW: 'Dock01',
            'toolbar_universal_params'   : 'Dock10',
            'toolbar_sampling_method'    : 'Dock20',
            'toolbar_filling_rate_test'  : 'Dock30',
            'toolbar_view_options'       : 'Dock40',
            'toolbar_messages'           : 'Dock50',
            # autopep8: on
        }

        windows = [hello_imgui.DockableWindow(f._title, title_to_dockspace_name[f._title], f, can_be_closed_=True) for f in layout_funcs]

        return splits, windows

    @typing_extensions.override
    def compute(self):
        state = copy.deepcopy(self.state)  # Detach parameters

        if state.num_steps <= 0:
            return None

        state_hash = state.deterministic_hash()

        # ---------------------------------------------------------------------------------------------------------------------------------------
        # Compute task

        def compute_task(ui_state: UIState, render_state: RenderState, order_idx: int, *, cancel_event=None):
            # Do not use state inside this function directly!

            def interrupted() -> bool:
                return cancel_event is not None and cancel_event.is_set()

            if interrupted():
                return None

            assert order_idx < len(render_state.order)
            coord = render_state.order[order_idx]

            x, y = coord
            idx = y * ui_state.W + x

            img_hash = f'{ui_state.deterministic_hash()}_{idx}'
            if img_hash not in self.render_cache:
                if interrupted():
                    return None

                if order_idx == 0:
                    # Show black image
                    self.texture_windows.update_image(**{self.KEY_GRID_WINDOW: render_state.grid})

                # ------------------------------------------------------------------------------------
                # Setup random generator

                master_rnd_gen = np.random.default_rng(ui_state.seed)

                if state.seed_mode == SeedGridMode.random:
                    grid_seed = master_rnd_gen.integers(low=0, high=2**31 - 1, size=(ui_state.W, ui_state.H), dtype=np.int64)
                elif ui_state.seed_mode == SeedGridMode.fixed_x:
                    # Deterministic along x-axis
                    grid_seed = master_rnd_gen.integers(low=0, high=2**31 - 1, size=(1, ui_state.H), dtype=np.int64).repeat(ui_state.W, axis=0)
                elif ui_state.seed_mode == SeedGridMode.fixed_y:
                    # Deterministic along y-axis
                    grid_seed = master_rnd_gen.integers(low=0, high=2**31 - 1, size=(ui_state.W, 1), dtype=np.int64).repeat(ui_state.H, axis=1)
                elif ui_state.seed_mode == SeedGridMode.fixed:
                    # Deterministic everywhere
                    grid_seed = master_rnd_gen.integers(low=0, high=2**31 - 1, size=(1, 1), dtype=np.int64).repeat(ui_state.W, axis=0).repeat(ui_state.H, axis=1)

                rng_seed = grid_seed[x, y]

                rnd_gen = np.random.default_rng(rng_seed)

                # ------------------------------------------------------------------------------------
                # Sample number of parameters

                if ui_state.x_axis == ParamAxis.interleave_num_params and ui_state.y_axis == ParamAxis.interleave_num_params:
                    raise NotImplementedError('Both axes cannot be num_params.')
                elif ui_state.x_axis == ParamAxis.interleave_num_params and ui_state.y_axis != ParamAxis.interleave_num_params:
                    # identical along y-axis
                    grid_num_params = space_integers(ui_state.min_num_params, ui_state.max_num_params, num=ui_state.W).reshape((ui_state.W, 1)).repeat(ui_state.H, axis=1).astype(np.int64)
                elif ui_state.x_axis != ParamAxis.interleave_num_params and ui_state.y_axis == ParamAxis.interleave_num_params:
                    # identical along x-axis
                    grid_num_params = space_integers(ui_state.min_num_params, ui_state.max_num_params, num=ui_state.H).reshape((1, ui_state.H)).repeat(ui_state.W, axis=0).astype(np.int64)
                else:
                    # independent
                    grid_num_params = rnd_gen.integers(low=ui_state.min_num_params, high=ui_state.max_num_params, dtype=np.int64, endpoint=True, size=(ui_state.W, ui_state.H))

                num_params = grid_num_params[x, y]

                cur_iter = 0
                max_filling_rate, final_filling_rate, points, selected_indices = 0.0, 0.0, None, None
                avg_theta, avg_phi = None, None

                while cur_iter < ui_state.max_num_attempts:
                    if interrupted():
                        return None

                    cur_iter += 1

                    # ------------------------------------------------------------------------------------
                    # Create parameter array

                    sigma_factor = None

                    if ui_state.ifs_method == IFSMethod.original:
                        params_A = rnd_gen.uniform(low=-1.0, high=1.0, size=(num_params, 2, 2))
                        params_B = rnd_gen.uniform(low=-1.0, high=1.0, size=(num_params, 2))
                        sigma_factor = None  # Not used in the original method
                    elif ui_state.ifs_method == IFSMethod.improved:
                        if ui_state.sigma_factor_method == SigmaFactorMethod.auto:
                            min_sigma_factor = (5.0 + float(num_params)) / 2.0
                            max_sigma_factor = (6.0 + float(num_params)) / 2.0
                        elif ui_state.sigma_factor_method == SigmaFactorMethod.manual:
                            min_sigma_factor = min(ui_state.min_sigma_factor, 3.0 * float(num_params))
                            max_sigma_factor = min(ui_state.max_sigma_factor, 3.0 * float(num_params))

                        # Sample alpha
                        if ui_state.x_axis == ParamAxis.lerp_sigma_factor and ui_state.y_axis == ParamAxis.lerp_sigma_factor:
                            raise NotImplementedError('Sigma factor axis is not implemented yet.')
                        elif ui_state.x_axis == ParamAxis.lerp_sigma_factor and ui_state.y_axis != ParamAxis.lerp_sigma_factor:
                            # identical along y-axis
                            grid_alpha = np.linspace(min_sigma_factor, max_sigma_factor, num=ui_state.W).reshape((ui_state.W, 1)).repeat(ui_state.H, axis=1)
                        elif ui_state.x_axis != ParamAxis.lerp_sigma_factor and ui_state.y_axis == ParamAxis.lerp_sigma_factor:
                            # identical along x-axis
                            grid_alpha = np.linspace(min_sigma_factor, max_sigma_factor, num=ui_state.H).reshape((1, ui_state.H)).repeat(ui_state.W, axis=0)
                        else:
                            # independent
                            grid_alpha = rnd_gen.uniform(low=min_sigma_factor, high=max_sigma_factor, size=(ui_state.W, ui_state.H))

                        alpha = grid_alpha[x, y]
                        sigma_factor = float(alpha)  # alpha means the sigma factor

                        # Sample singular values
                        sigma = np.zeros((num_params, 2), dtype=compute_dtype)  # the array of singular values
                        b_l = alpha - 3.0 * float(num_params) + 3.0             # sampling lower bound
                        b_u = alpha                                             # sampling upper bound

                        for k in range(num_params - 1):
                            sigma_k_1 = rnd_gen.uniform(low=max(0.0, b_l / 3.0), high=min(1.0, b_u))

                            b_l = b_l - sigma_k_1
                            b_u = b_u - sigma_k_1

                            sigma_k_2 = rnd_gen.uniform(low=max(0.0, b_l / 2.0), high=min(sigma_k_1, b_u / 2.0))

                            b_l = b_l - 2.0 * sigma_k_2 + 3.0
                            b_u = b_u - 2.0 * sigma_k_2

                            sigma[k, 0] = sigma_k_1
                            sigma[k, 1] = sigma_k_2

                        sigma[num_params - 1, 1] = rnd_gen.uniform(low=max(0.0, (b_u - 1.0) / 2.0), high=b_u / 3.0)
                        sigma[num_params - 1, 0] = b_u - 2.0 * sigma[num_params - 1, 1]

                        # Sample rotation rotation angle theta
                        grid_thetas = np.zeros((ui_state.W, ui_state.H, num_params), dtype=compute_dtype)
                        grid_phis = np.zeros((ui_state.W, ui_state.H, num_params), dtype=compute_dtype)

                        if ui_state.x_axis == ParamAxis.lerp_angle and ui_state.y_axis == ParamAxis.lerp_angle:
                            raise NotImplementedError('Both axes cannot be lerp_angle.')
                        elif ui_state.x_axis == ParamAxis.lerp_angle and ui_state.y_axis != ParamAxis.lerp_angle:
                            # identical along y-axis
                            grid_thetas = rnd_gen.uniform(low=-np.pi, high=np.pi, size=(1, ui_state.H, num_params)).repeat(ui_state.W, axis=0)
                            phase = np.linspace(-np.pi, np.pi, num=ui_state.W).reshape((ui_state.W, 1, 1)).repeat(ui_state.H, axis=1)
                            grid_thetas += phase  # Add linear phase along x-axis

                            grid_phis = rnd_gen.uniform(low=-np.pi, high=np.pi, size=(1, 1, num_params)).repeat(ui_state.W, axis=0).repeat(ui_state.H, axis=1)  # Identical phi
                        elif ui_state.x_axis != ParamAxis.lerp_angle and ui_state.y_axis == ParamAxis.lerp_angle:
                            # identical along x-axis
                            grid_thetas = rnd_gen.uniform(low=-np.pi, high=np.pi, size=(ui_state.W, 1, num_params)).repeat(ui_state.H, axis=1)
                            phase = np.linspace(-np.pi, np.pi, num=ui_state.H).reshape((1, ui_state.H, 1)).repeat(ui_state.W, axis=0)
                            grid_thetas += phase  # Add linear phase along y-axis

                            grid_phis = rnd_gen.uniform(low=-np.pi, high=np.pi, size=(1, 1, num_params)).repeat(ui_state.W, axis=0).repeat(ui_state.H, axis=1)  # Identical phi
                        else:
                            # independent
                            grid_thetas = rnd_gen.uniform(low=-np.pi, high=np.pi, size=(ui_state.W, ui_state.H, num_params))
                            grid_phis = rnd_gen.uniform(low=-np.pi, high=np.pi, size=(ui_state.W, ui_state.H, num_params))

                        thetas = grid_thetas[x, y]
                        avg_theta = float(np.mean(thetas))

                        rot_mat_theta = np.zeros((num_params, 2, 2), dtype=compute_dtype)
                        rot_mat_theta[:, 0, 0] = np.cos(thetas)
                        rot_mat_theta[:, 0, 1] = -np.sin(thetas)
                        rot_mat_theta[:, 1, 0] = np.sin(thetas)
                        rot_mat_theta[:, 1, 1] = np.cos(thetas)

                        # Create scaling matrix
                        scale_mat = np.zeros((num_params, 2, 2), dtype=compute_dtype)
                        scale_mat[:, 0, 0] = sigma[:, 0]
                        scale_mat[:, 1, 1] = sigma[:, 1]

                        # Sample rotation rotation angle phi
                        phis = grid_phis[x, y]
                        avg_phi = float(np.mean(phis))
                        rot_mat_phi = np.zeros((num_params, 2, 2), dtype=compute_dtype)
                        rot_mat_phi[:, 0, 0] = np.cos(phis)
                        rot_mat_phi[:, 0, 1] = -np.sin(phis)
                        rot_mat_phi[:, 1, 0] = np.sin(phis)
                        rot_mat_phi[:, 1, 1] = np.cos(phis)

                        sign = rnd_gen.choice(a=np.array([-1.0, 1.0], dtype=compute_dtype), size=(num_params, 2))
                        sign_mat = np.zeros((num_params, 2, 2), dtype=compute_dtype)
                        sign_mat[:, 0, 0] = sign[:, 0]
                        sign_mat[:, 1, 1] = sign[:, 1]

                        params_A = rot_mat_theta @ scale_mat @ rot_mat_phi @ sign_mat  # (num_params, 2, 2)

                        params_B = rnd_gen.uniform(low=-ui_state.b_range, high=ui_state.b_range, size=(num_params, 2))

                    dets = np.array([np.abs(np.linalg.det(A)) for A in params_A])
                    probs = dets / np.sum(dets)  # (num_params,)
                    prob_cumsum = np.cumsum(probs)

                    # ------------------------------------------------------------------------------------
                    # Create IFS function

                    # Compute random numbers for parameter selection
                    rnds = rnd_gen.random(ui_state.num_steps)

                    # Compute IFS
                    test_points, test_selected_indices = ifs_func.compute_points(params_A, params_B, prob_cumsum, rnds)
                    if interrupted():
                        return None

                    # ------------------------------------------------------------------------------------
                    # Rescaling
                    #     This approach is identical to the official FractalDB implementation.

                    filling_rate_test_resolution = 512

                    points_min = np.min(test_points, axis=0, keepdims=True)
                    points_max = np.max(test_points, axis=0, keepdims=True)
                    test_points: np.ndarray = (test_points - points_min) / (points_max - points_min + 1e-10)  # [0, 1]
                    ftest_points = test_points * (filling_rate_test_resolution - 1 - 2 * ui_state.padding) + ui_state.padding  # Add the same amount of padding to all sides
                    ftest_points = ftest_points.astype(np.int64)  # Convert to int for indexing

                    # -------------------------------------------------------------------------------------------------------
                    # Filling rate test

                    test_img = render_points(
                        img_size=filling_rate_test_resolution,
                        points=ftest_points,
                        fill_value=0.5,
                    )  # shape (H, W)

                    test_filling_rate = np.count_nonzero(test_img) / (test_img.shape[0] * test_img.shape[1])
                    max_filling_rate = max(max_filling_rate, test_filling_rate)

                    if not ui_state.enable_filling_rate_test or test_filling_rate >= ui_state.filling_rate_threshold:
                        # Passed

                        test_points = test_points * (ui_state.img_size - 1 - 2 * ui_state.padding) + ui_state.padding  # Scale to the final image size
                        test_points = test_points.astype(np.int64)
                        points = test_points

                        selected_indices = test_selected_indices
                        final_filling_rate = test_filling_rate
                        break

                if points is not None and selected_indices is not None:
                    assert final_filling_rate == max_filling_rate

                    # Render the final image

                    # ------------------------------------------------------------------------------------
                    # Coloring

                    if ui_state.cmap == 'constant':
                        colors = np.ones((points.shape[0], 3), dtype=np.float32)  # white
                    elif ui_state.cmap == 'gray':
                        quant = float(np.max(selected_indices)) + 1.0  # to avoid displaying black points
                        gray_vals = (selected_indices.astype(np.float32) + 1.0) / quant  # (N,)
                        colors = np.repeat(gray_vals[:, np.newaxis], 3, axis=1)  # (N, 3)
                    else:
                        assert ui_state.cmap in AVAILABLE_CMAPS, f'Invalid colormap: {ui_state.cmap}'
                        vals = selected_indices.astype(np.float32) / max(1.0, float(num_params - 1))  # (N,)
                        colors = cmap.colorize(vals, ui_state.cmap)  # (N, 3)

                    # ------------------------------------------------------------------------------------
                    # Render image

                    if ui_state.enable_patch_pattern:
                        patch_pattern_indices = rnd_gen.integers(low=0, high=PATCH_PATTERNS.shape[0], size=(points.shape[0],))
                        img_np = ifs_func.render_patch_pattern_colored(
                            img_size=ui_state.img_size,
                            points=points,
                            colors=colors,
                            patch_pattern_array=PATCH_PATTERNS,
                            patch_pattern_indices=patch_pattern_indices,
                        )
                    else:
                        img_np = ifs_func.render_points_colored(
                            img_size=ui_state.img_size,
                            points=points,
                            colors=colors,
                        )
                else:
                    img_np = create_diagonal_stripes(
                        width=ui_state.img_size,
                        height=ui_state.img_size,
                        stripe_width=ui_state.img_size // 5,
                    )

                if interrupted():
                    return None

                img = torch.from_numpy(img_np).to(device=device, dtype=torch.float32)

                # ------------------------------------------------------------------------------------
                # Store to cache

                if interrupted():
                    return None

                self.render_cache[img_hash] = RenderResult(
                    img_tensor=img,
                    filling_rate=max_filling_rate,
                    num_params=num_params,
                    x=x,
                    y=y,
                    seed=int(rng_seed),
                    num_attempts=cur_iter,
                    sigma_factor=sigma_factor,
                    avg_theta=avg_theta,
                    avg_phi=avg_phi,
                )

            if interrupted():
                return None

            return self.render_cache[img_hash].copy()

        # ---------------------------------------------------------------------------------------------------------------------------------------
        # Update render state

        if self.render_state is None or self.render_state.state_hash != state_hash:
            # Compute grid size
            entire_h = state.H * state.img_size
            entire_w = state.W * state.img_size
            if entire_h > self.gl_max_tex_size or entire_w > self.gl_max_tex_size:
                self.status_msg = ''
                self.error_msg = f'The entire image size ({entire_h}, {entire_w}) exceeded OpenGL maximum texture size ({self.gl_max_tex_size}, {self.gl_max_tex_size}). Please reduce the number of images.'
                return None
            else:
                self.status_msg = ''
                self.error_msg = ''

            # Compute the order of rendering
            order = self._get_lds(state.W, state.H)

            # Update render state
            self.render_state = RenderState(
                state_hash=state_hash,
                img_size=[state.img_size, state.img_size],
                order=order,
                i=0,
                start_idx=0,
                end_idx=state.W * state.H,
                start_time=time.perf_counter(),
                end_time=None,
                grid=torch.zeros((entire_h, entire_w, 3), device=device, dtype=torch.float32),
            )

            # Set window states (ugly ...)
            self.texture_windows[self.KEY_GRID_WINDOW].grid_size = (state.H, state.W)
            self.texture_windows[self.KEY_GRID_WINDOW].attribs = [['Rendering ...' for _ in range(state.W)] for _ in range(state.H)]

        # Check if rendering is complete
        if self.render_state.i >= len(self.render_state.order):
            return {self.KEY_GRID_WINDOW: self.render_state.grid}

        assert self.worker_pool.num_active() > 0, 'No workers online'
        futs: list[futures.Future] = []
        for i in range(self.render_state.i, len(self.render_state.order)):
            futs.append(self.worker_pool.submit(compute_task, state, self.render_state, i))

        # Busy loop in order to react to UI changes quickly
        n_total = len(futs)
        while len(futs) > 0:
            progress = 1 - len(futs) / n_total
            self.set_window_title(f'Rendering: {progress:.1%}')

            # UI has changed => early out
            if self.render_state.state_hash != self.state.deterministic_hash():
                self.worker_pool.clear_tasks(block=False)  # non-blocking, stray results removed in next iteration
                return None

            # Next finished task
            f = next((f for f in futs if f.done()), None)
            if f is None:
                time.sleep(20 / 1000)  # 20ms
                continue

            futs.remove(f)
            ret: RenderResult = f.result()

            if ret is not None:
                # Insert to grid
                w, h = self.render_state.img_size
                self.render_state.grid[ret.y * h:(ret.y + 1) * h, ret.x * w:(ret.x + 1) * w, :] = ret.img_tensor
                self.texture_windows.update_image(**{self.KEY_GRID_WINDOW: self.render_state.grid})

                # Set grid states (ugly ...)
                attrib_txt = ''
                if ret.filling_rate is not None:
                    attrib_txt += f'Filling rate: {(ret.filling_rate * 100.0):.2f} [%]'
                if ret.sigma_factor is not None:
                    attrib_txt += f'\nSigma factor: {(ret.sigma_factor):.2f}'
                attrib_txt += f'\nNum params: {ret.num_params}'
                if state.debug_mode:
                    attrib_txt = f'X: {ret.x}, Y: {ret.y}\nSeed: {ret.seed}\nNum attempts: {ret.num_attempts}\n' + attrib_txt
                    attrib_txt += f'\nAvg theta: {ret.avg_theta:.2f}\nAvg phi: {ret.avg_phi:.2f}'
                self.texture_windows[self.KEY_GRID_WINDOW].attribs[ret.y][ret.x] = attrib_txt

        # Reset window title
        self.set_window_title(WINDOW_TITLE)

        # Sanity
        assert self.worker_pool.task_queue.empty(), 'Task queue is not empty after rendering'

        # All done
        self.render_state.i = len(self.render_state.order)
        if state_hash not in self.elapsed_time_cache:
            self.render_state.end_time = time.perf_counter()
            self.elapsed_time_cache[state_hash] = self.render_state.end_time - self.render_state.start_time

        self.texture_windows.update_image(**{self.KEY_GRID_WINDOW: self.render_state.grid})

        self.status_msg = f'Rendering completed in {(self.elapsed_time_cache[state_hash]):.2f} seconds.'

        return {self.KEY_GRID_WINDOW: self.render_state.grid}

    # ------------------------------------------------------------------------------------
    # UI components

    @ui_utils.dockable
    def toolbar_param_cache(self):
        highlight_color = (1.0, 1.0, 0.0, 1.0)

        # ---------------------------------------------------------------------------------------------------
        # Key callback

        # Detect arrow key
        is_any_item_active = imgui.is_any_item_active()
        if not is_any_item_active:  # Except when inputting values ...
            if self.keyhit(imgui.Key.right_arrow):
                if imgui.is_key_down(imgui.Key.left_shift):
                    # Duplicate current state to the next slot
                    next_id = (self.cur_cache_param_id + 1) % self.NUM_STATES_CACHES
                    self.cached_states[next_id] = copy.deepcopy(self.state)
                self.cur_cache_param_id = (self.cur_cache_param_id + 1) % self.NUM_STATES_CACHES
            elif self.keyhit(imgui.Key.left_arrow):
                if imgui.is_key_down(imgui.Key.left_shift):
                    # Duplicate current state to the previous slot
                    prev_id = (self.cur_cache_param_id - 1) % self.NUM_STATES_CACHES
                    self.cached_states[prev_id] = copy.deepcopy(self.state)
                self.cur_cache_param_id = (self.cur_cache_param_id - 1) % self.NUM_STATES_CACHES

            # Detect number key
            for i_cache in range(self.NUM_STATES_CACHES):
                key = getattr(imgui.Key, f'_{i_cache + 1}')
                if self.keyhit(key):
                    self.cur_cache_param_id = i_cache
                    break

        # ---------------------------------------------------------------------------------------------------
        # Parameter caching

        imgui.separator_text('Current Param')
        for i_cache in range(self.NUM_STATES_CACHES):
            if i_cache != 0:
                imgui.same_line()

            if imgui.radio_button(f'##Param{i_cache}', self.cur_cache_param_id == i_cache) or self.cur_cache_param_id == i_cache:
                self.cur_cache_param_id = i_cache
                self.state = self.cached_states[self.cur_cache_param_id]

        imgui.text_colored(highlight_color, 'Tips:')
        imgui.text_colored(highlight_color, '  - switch params with "arrow" or "number" keys')
        imgui.text_colored(highlight_color, '  - duplicate current params to the next/prev slot with "shift + arrow" keys')

    @ui_utils.dockable
    def toolbar_universal_params(self):
        # ---------------------------------------------------------------------------------------------------
        # Universal parameters

        imgui.separator_text('Parameters')

        self.state.H = imgui.slider_int('H', self.state.H, 1, 32)[1]
        self.state.W = imgui.slider_int('W', self.state.W, 1, 32)[1]
        self.state.seed = imgui.input_int('Random Seed', self.state.seed)[1]
        self.state.seed_mode = self.state.seed_mode.combo('Seed Mode')
        self.state.img_size = max(min(imgui.input_int('Image Size', self.state.img_size)[1], self.MAX_IMG_SIZE), self.MIN_IMG_SIZE)
        self.state.num_steps = max(imgui.input_int('Num Steps', self.state.num_steps)[1], 0)
        self.state.b_range = max(min(imgui.slider_float('B Range (translation params will be sampled from U(-B, B))', self.state.b_range, 0.0, 1.0)[1], 1.0), 0.0)
        self.state.padding = max(min(imgui.input_int('Padding', self.state.padding)[1], self.state.img_size // 2 - 1), 0)
        self.state.enable_patch_pattern = imgui.checkbox('Enable Patch Pattern', self.state.enable_patch_pattern)[1]

        self.state.num_params_mode = self.state.num_params_mode.combo('Num Params Mode')
        if self.state.num_params_mode == NumParamsMode.fixed:
            num_params = imgui.input_int('Num Params', self.state.min_num_params)[1]
            self.state.min_num_params = max(min(num_params, self.MAX_NUM_PARAMS), 2)
            self.state.max_num_params = self.state.min_num_params
        elif self.state.num_params_mode == NumParamsMode.range_based:
            self.state.min_num_params = max(min(imgui.input_int('Min Num Params', self.state.min_num_params)[1], self.state.max_num_params), 2)
            self.state.max_num_params = max(min(imgui.input_int('Max Num Params', self.state.max_num_params)[1], self.MAX_NUM_PARAMS), self.state.min_num_params)

    @ui_utils.dockable
    def toolbar_sampling_method(self):
        # ---------------------------------------------------------------------------------------------------
        # Parameter sampling method

        imgui.separator_text('Sampling Method')

        self.state.ifs_method = self.state.ifs_method.combo('IFS Method')

        if self.state.ifs_method == IFSMethod.improved:
            self.state.x_axis = self.state.x_axis.combo('X-Axis')
            y_axis = self.state.y_axis.combo('Y-Axis')

            # Avoid some invalid combinations
            if self.state.x_axis == ParamAxis.lerp_sigma_factor and y_axis == ParamAxis.lerp_sigma_factor:
                y_axis = ParamAxis.none
            elif self.state.x_axis == ParamAxis.interleave_num_params and y_axis == ParamAxis.interleave_num_params:
                y_axis = ParamAxis.none
            elif self.state.x_axis == ParamAxis.lerp_angle and y_axis == ParamAxis.lerp_angle:
                y_axis = ParamAxis.none

            self.state.y_axis = y_axis

            self.state.sigma_factor_method = self.state.sigma_factor_method.combo('Sigma Factor Method')

            if self.state.sigma_factor_method == SigmaFactorMethod.manual:
                self.state.min_sigma_factor = min(max(imgui.input_float('Min Sigma Factor', self.state.min_sigma_factor)[1], 0.0), self.state.max_sigma_factor)
                self.state.max_sigma_factor = min(max(imgui.input_float('Max Sigma Factor', self.state.max_sigma_factor)[1], self.state.min_sigma_factor), 3.0 * self.state.max_num_params)

    @ui_utils.dockable
    def toolbar_filling_rate_test(self):
        # ---------------------------------------------------------------------------------------------------
        # Filling rate test

        imgui.separator_text('Filling Rate Test')

        self.state.enable_filling_rate_test = imgui.checkbox('Enable Filling Rate Test', self.state.enable_filling_rate_test)[1]
        if self.state.enable_filling_rate_test:
            self.state.filling_rate_test_resolution = min(max(imgui.input_int('Filling Rate Test Resolution', self.state.filling_rate_test_resolution)[1], 32), 4096)
            self.state.filling_rate_threshold = min(max(imgui.slider_float('Filling Rate Threshold', self.state.filling_rate_threshold, 0.0, 1.0)[1], 0.0), 1.0)
            self.state.max_num_attempts = max(imgui.input_int('Max Num Attempts', self.state.max_num_attempts)[1], 1)

    @ui_utils.dockable
    def toolbar_view_options(self):
        # ---------------------------------------------------------------------------------------------------
        # View options

        imgui.separator_text('View Options')

        cur_cmap_idx = AVAILABLE_CMAPS.index(self.state.cmap)
        new_cmap_idx = imgui.combo('Colormap', cur_cmap_idx, AVAILABLE_CMAPS)[1]
        self.state.cmap = AVAILABLE_CMAPS[new_cmap_idx]

        self.state.debug_mode = imgui.checkbox('Enable Debug Mode', self.state.debug_mode)[1]

        if self.texture_windows[self.KEY_GRID_WINDOW].attrib_txt_color is not None:
            self.texture_windows[self.KEY_GRID_WINDOW].attrib_txt_color = imgui.color_edit4('Text Color', self.texture_windows[self.KEY_GRID_WINDOW].attrib_txt_color)[1]

        # Save image
        if imgui.button('Save Image', size=(-1, 0)):
            if self.render_state is not None and self.render_state.grid is not None:
                status_msg = f'Saving image and metadata to disk ...'
                self.status_msg = status_msg
                self.error_msg = ''

                grid_img = self.render_state.grid.clone().cpu().numpy()

                def img_saving_task(img_np: np.ndarray, save_dir: pathlib.Path, *, cancel_event=None):
                    def interrupted() -> bool:
                        return cancel_event is not None and cancel_event.is_set()

                    if interrupted():
                        return

                    img_np = (np.clip(img_np, 0.0, 1.0) * 255.0).astype(np.uint8)
                    img_pil = PIL.Image.fromarray(img_np)

                    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                    save_path = (save_dir / f'{timestamp}.png').resolve()
                    save_path.parent.mkdir(parents=True, exist_ok=True)

                    png_info = PIL.PngImagePlugin.PngInfo()
                    png_info.add_text('Generated by', 'Grid IFS Visualizer')
                    png_info.add_text('Parameters', json.dumps(self.state.asdict(), indent=4))

                    if interrupted():
                        return

                    img_pil.save(save_path, pnginfo=png_info, optimize=True, compress_level=9)

                    status_msg = f'Saved image and metadata to:\n\"{save_path}\"'
                    self.status_msg = status_msg
                    logger.info(status_msg)

                self.worker_pool.submit(img_saving_task, grid_img, self.img_save_dir)
            else:
                self.status_msg = ''
                self.error_msg = 'No image to save. Please render an image first.'

        # Image save directory
        avail_w, _ = imgui.get_content_region_avail()
        imgui.set_next_item_width(avail_w * 0.8)
        img_save_dir = imgui.input_text('##Image Save Directory', str(self.img_save_dir))[1]
        imgui.same_line()
        imgui.set_next_item_width(-1)
        if imgui.button('Browse##img_save_dir'):
            if self.img_save_dir.is_dir():
                default_path = str(self.img_save_dir)
            elif self.img_save_dir.parent.is_dir():
                default_path = str(self.img_save_dir.parent)
            else:
                default_path = str(pathlib.Path.cwd())
            ret = nfdpy.pick_folder_dialog(default_path=default_path)
            if ret is not None:
                img_save_dir = ret
        self.img_save_dir = pathlib.Path(img_save_dir)

    @ui_utils.dockable
    def toolbar_messages(self):
        if len(self.status_msg) > 0:
            imgui.text_colored(drawable_window.DrawableTextureWindowUnit.HIGHTLIGHT_COLOR_POS, self.status_msg)
        if len(self.error_msg) > 0:
            imgui.text_colored((1.0, 0.0, 0.0, 1.0), f'Error: {self.error_msg}')


def main():
    parser = argparse.ArgumentParser(description='IFS Visualizer', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--enable_vsync', action='store_true', help='Enable display vertical synchronization (vsync)')
    parser.add_argument('--no_cache', action='store_true', help='Do not cache UI states to files')
    parser.add_argument('--no_cache_ui', action='store_true', help='Do not cache UI layouts to files')
    args = parser.parse_args()

    viz = GridIFSViz(
        enable_vsync=args.enable_vsync,
        cache_states=not args.no_cache,
        cache_ui_layout=not args.no_cache_ui,
    )

    logger.info('done.')


if __name__ == '__main__':
    main()
