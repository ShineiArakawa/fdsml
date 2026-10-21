"""
Distance-based IFS Visualizer
=============================

A visualizer for Iterated Function Systems (IFS) based on parameter distance.

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


import argparse
import concurrent.futures as futures
import copy
import dataclasses
import hashlib
import itertools
import json
import os
import pathlib
import random
import time
import typing

import torch  # NOTE: Make sure to import torch before pyviewer_extended to avoid potential conflicts

import cachetools
import cmap
import ifs_func
import natsort
import nfdpy
import numpy as np
import OpenGL.GL as gl
import pandas as pd
import pydantic
import typing_extensions
import ui.async_pool as async_pool
import ui.docking_viewer as docking_viewer
import ui.drawable_window as drawable_window
import ui.utils as ui_utils
from imgui_bundle import hello_imgui, imgui  # type: ignore

# autopep8: off

# --------------------------------------------------------------------------------------------------------------------------------
# Constants

WINDOW_TITLE = 'Distance-based IFS Visualizer'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
compute_dtype = np.float64

# Available color maps
AVAILABLE_CMAPS: typing.Final[list[str]] = natsort.natsorted(list(cmap.available_color_maps()))
AVAILABLE_CMAPS = ['constant', 'gray'] + [c for c in AVAILABLE_CMAPS]

PATCH_PATTERNS: typing.Final[np.ndarray] = np.stack([
    np.array(pattern, dtype=bool).reshape((3, 3)) for pattern in list(itertools.product([0, 1], repeat=9)) if np.any(np.array(pattern, dtype=bool) == True)
], axis=0)  # Shape: (P, 3, 3) where P is the number of patterns (excluding all-zero pattern)


# --------------------------------------------------------------------------------------------------------------------------------
# Data classes

class GridLayout(ui_utils.StrEnum):
    random = 'Random'
    monotonic = 'Monotonic-increasing'

@pydantic.dataclasses.dataclass(config=pydantic.config.ConfigDict(arbitrary_types_allowed=True))
class IFSParam:
    # autopep8: off
    A          : np.ndarray
    B          : np.ndarray
    p          : float
    selected   : bool = False
    # autopep8: on

    # ------------------------------------------------------------------------------------
    # Serialization

    def asdict(self) -> dict:
        return {
            'A': self.A.tolist(),
            'B': self.B.tolist(),
            'p': self.p,
            'selected': self.selected,
        }

    @staticmethod
    def fromdict(state_dict: dict):
        return IFSParam(
            A=np.array(state_dict['A']).reshape((2, 2)),
            B=np.array(state_dict['B']).reshape((2,)),
            p=state_dict['p'],
            selected=state_dict['selected'],
        )


@pydantic.dataclasses.dataclass
class IFSParamFile:
    # autopep8: off
    params : list[IFSParam]
    # autopep8: on

    # ------------------------------------------------------------------------------------
    # Serialization

    def asdict(self) -> dict:
        return {'params': [p.asdict() for p in self.params]}

    @staticmethod
    def fromdict(state_dict: dict):
        return IFSParamFile(params=[IFSParam.fromdict(p) for p in state_dict['params']])


@pydantic.dataclasses.dataclass
class UIState:
    # autopep8: off
    H                    : int               = 8
    W                    : int               = 8
    seed                 : int               = 100
    img_size             : int               = 362
    num_steps            : int               = 200000
    enable_patch_pattern : bool              = True
    dist_p               : int               = 2
    dist_max             : float             = 0.01
    compute_dist_exp     : bool              = False
    sample_uniformly     : bool              = False
    grid_layout          : GridLayout        = GridLayout.random
    clip_zero_one        : bool              = True
    ifs_params           : list[IFSParam]    = dataclasses.field(default_factory=lambda: [])
    padding              : int               = 6
    load_from_file       : bool              = False
    fix_probabilities    : bool              = False
    cmap                 : str               = AVAILABLE_CMAPS[-1]  # maybe 'viridis'
    # autopep8: on

    def deterministic_hash(self) -> str:
        """Compute a deterministic hash of the UI state, which is stable across sessions.
        """
        state_dict = {}
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if field.name == 'ifs_params':
                # Convert list of IFSParam to list of dicts
                value = [p.asdict() for p in value]
            state_dict[field.name] = value

        state_bytes = json.dumps(state_dict).encode()
        return hashlib.sha256(state_bytes).hexdigest()

    # ------------------------------------------------------------------------------------
    # Serialization

    def asdict(self) -> dict:
        state_dict = {}
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if field.name == 'ifs_params':
                # Convert list of IFSParam to list of dicts
                value = [p.asdict() for p in value]
            state_dict[field.name] = value
        return state_dict

    @staticmethod
    def fromdict(state_dict: dict):
        ifs_params = [IFSParam.fromdict(p) for p in state_dict.get('ifs_params', [])]
        state_dict.update({'ifs_params': ifs_params})
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

    img                                : torch.Tensor
    grid                               : torch.Tensor
    # autopep8: on

# --------------------------------------------------------------------------------------------------------------------------------


class DistanceViz(docking_viewer.MultiTexturesDockingViewer):
    # ------------------------------------------------------------------------------------
    # Constants

    # autopep8: off
    KEY_MAIN_WINDOW     : typing.Final[str] = 'Main window'
    KEY_GRID_WINDOW     : typing.Final[str] = 'Grid window'

    MIN_IMG_SIZE        : int               = 5    # it has to be larger than 4
    MAX_IMG_SIZE        : int               = 2048 # 256 is already heavy with padding even if using CUDA backend ...

    MAX_NUM_PARAMS      : int               = 8    # according to the original FractalDB paper

    STATES_CACHE_PATH   : pathlib.Path      = pathlib.Path('.cache/distance_ifs_viz_params.json')
    NUM_STATES_CACHES   : int               = 9    # should be < 10 due to the limit of number keys
    # autopep8: on

    # ------------------------------------------------------------------------------------

    def __init__(self, cache_states: bool = True, cache_ui_layout: bool = True, enable_vsync: bool = True, **kwargs):
        self.cur_cache_param_id = 0
        self.cur_selected_ifs_param_id = -1  # No selection
        self.cached_states: list[UIState] = [UIState() for _ in range(self.NUM_STATES_CACHES)]
        self.cache_params_to_file = cache_states
        self.cache_ui_layout = cache_ui_layout
        self.rnd_counter = 0

        self.params_dir = os.path.join(os.getcwd(), 'data', 'weighted_vector_avg')
        self.param_file_list: list[str] = []
        self.param_class_start_idx: list[int] = []
        self.param_file_cursor = 0
        self.param_file_cache: dict[str, IFSParamFile] = cachetools.FIFOCache(maxsize=512)
        self.load_weighted_params = True

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
                print(f'Removed {ini_path}')

        # -----------------------------------------------------------------------------

        super(DistanceViz, self).__init__(
            name=WINDOW_TITLE,
            texture_names=[
                self.KEY_MAIN_WINDOW,
                self.KEY_GRID_WINDOW,
            ],
            enable_vsync=enable_vsync,
            hightlight_on_hover_mode=[
                drawable_window.HighlightMode.entire_image,  # self.KEY_MAIN_WINDOW
                drawable_window.HighlightMode.grid_cells    # self.KEY_GRID_WINDOW
            ],
            full_screen_mode=hello_imgui.FullScreenMode.full_monitor_work_area,
            with_font_awesome=True,
            **kwargs
        )

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
                self.params_dir = common_dict.get('params_dir', self.params_dir)
                self.param_file_cursor = common_dict.get('param_file_cursor', self.param_file_cursor)
                self.load_weighted_params = common_dict.get('load_weighted_params', self.load_weighted_params)

            print(f'Loaded parameters from {file_path}')
        else:
            print(f'Failed to load parameters from {file_path}')

    def _save_param_caches(self, file_path: pathlib.Path) -> None:
        params: list[dict] = []
        for cached_param in self.cached_states:
            params.append(cached_param.asdict())

        state_dict = {
            'params': params,
            'common': {
                'params_dir': self.params_dir,
                'param_file_cursor': self.param_file_cursor,
                'load_weighted_params': self.load_weighted_params,
            }
        }

        cache_file = file_path.resolve()
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(cache_file, mode='w', encoding='utf-8') as file:
                json.dump(state_dict, file, indent=4, ensure_ascii=False)
            print(f'Saved parameters to {cache_file}')
        except Exception as e:
            print(f'[ERROR] Failed to save parameters to {cache_file}')
            print(e)

    def _update_param_file_list(self, dir_path: str) -> None:
        if os.path.isdir(dir_path):
            all_files = os.listdir(dir_path)
            self.param_file_list = [f for f in all_files if f.lower().endswith('.csv')]
            self.param_file_list = natsort.natsorted(self.param_file_list)
            self.param_class_start_idx = []
            tmp_class = None
            for i_file, fname in enumerate(self.param_file_list):
                try:
                    class_name = fname.split('_')[0]
                    if tmp_class != class_name:
                        self.param_class_start_idx.append(i_file)
                        tmp_class = class_name
                except Exception:
                    pass
        else:
            self.param_file_list = []
        self.param_file_cursor = 0

    def _load_ifs_param_from_file(self, file_path: str, load_weighted_params: bool) -> None:
        if not os.path.isfile(file_path):
            print(f'File not found: {file_path}')
            return

        key = f'{file_path}_{load_weighted_params}'
        if key not in self.param_file_cache:
            params: list[IFSParam] = []

            df = pd.read_csv(file_path)

            # Count the number of parameters which is not nan in 'a' column
            num_params = int(df['a'].notna().sum())

            keys = ['a', 'b', 'c', 'd', 'e', 'f']
            if load_weighted_params:
                for i in range(len(keys)):
                    keys[i] = 'w' + keys[i]
            for i in range(num_params):
                p = IFSParam(
                    A=np.array([[float(df.loc[i, keys[0]]), float(df.loc[i, keys[1]])],
                                [float(df.loc[i, keys[2]]), float(df.loc[i, keys[3]])]], dtype=compute_dtype),
                    B=np.array([float(df.loc[i, keys[4]]),
                                float(df.loc[i, keys[5]])], dtype=compute_dtype),
                    p=float(df.loc[i, 'prob']),
                )
                params.append(p)

            self.param_file_cache[key] = IFSParamFile(params=params)

        self.state.ifs_params = copy.deepcopy(self.param_file_cache[key].params)

    def _update_ifs_param_from_file(self) -> None:
        if self.param_file_cursor < 0 or self.param_file_cursor >= len(self.param_file_list):
            return

        cur_file = self.param_file_list[self.param_file_cursor]

        file_path = os.path.join(self.params_dir, cur_file)
        self._load_ifs_param_from_file(file_path, self.load_weighted_params)

    def _get_lds(self, W, H):
        """Low discrepancy ordering"""
        def phi(d, iter=20):
            x = 2.0
            for _ in range(iter):
                x = pow(1+x, 1/(d+1))
            return x
        phis = [1/phi(2), 1/phi(2)**2]  # inverse powers of plastic number

        def R2(i: int, dim: int):
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

    @typing_extensions.override
    def draw_status_bar(self):
        return super().draw_status_bar()

    @typing_extensions.override
    def setup_state(self):
        self.state = UIState()

        self.render_cache: dict[str, (float, torch.Tensor)] = cachetools.FIFOCache(maxsize=512)
        self.render_state: RenderState = None

        self.params_cache_file = self.STATES_CACHE_PATH

        self.gl_max_tex_size: int = gl.glGetIntegerv(gl.GL_MAX_TEXTURE_SIZE)

        # Load params
        if self.cache_params_to_file and self.params_cache_file.is_file():
            self._load_param_caches(self.params_cache_file)

        # Update param cache
        self.cached_states[self.cur_cache_param_id] = self.state

        self._update_param_file_list(self.params_dir)

        # Set highlight color
        self.texture_windows[self.KEY_GRID_WINDOW].highlight_color_pos = drawable_window.DrawableTextureWindowUnit.HIGHTLIGHT_COLOR_NEG

    @typing_extensions.override
    def save_settings(self) -> None:
        if self.cache_params_to_file:
            self._save_param_caches(self.params_cache_file)

    @typing_extensions.override
    def drag_and_drop_callback(self, paths):
        if paths is not None and len(paths) > 0:
            cache_path: pathlib.Path = paths[0]
            if cache_path.is_file():
                self._load_param_caches(cache_path)

    @typing_extensions.override
    def setup_docking_layout(self, layout_funcs):
        # Graceful initial layout ----------------------------------------------------------------------------------------------------------------
        # Window layout
        #   + ---------------------------- + ---------------------------- + ---------------------------- +
        #   |                              |                              |                              |
        #   +                              +                              +                              |
        #   |                              |                              |                              |
        #   + 'toolbar'                    + 'Main Window'                + 'Grid Window'                |
        #   |                              |                              |                              |
        #   |                              |                              |                              |
        #   |                              |                              |                              |
        #   + ---------------------------- + ---------------------------- + ---------------------------- +
        #
        # Dockspace names
        #   + ---------------------------- + ---------------------------- + ---------------------------- +
        #   |                              |                              |                              |
        #   +                              +                              +                              |
        #   |                              |                              |                              |
        #   + 'MainDockSpace'              + 'Dock01'                     + 'Docke02'                    |
        #   |                              |                              |                              |
        #   |                              |                              |                              |
        #   |                              |                              |                              |
        #   + ---------------------------- + ---------------------------- + ---------------------------- +

        splits = [
            # autopep8: off
            # Split colomns first
            hello_imgui.DockingSplit('MainDockSpace', 'Dock01', imgui.Dir.right, ratio_=5/6),
            hello_imgui.DockingSplit(       'Dock01', 'Dock02', imgui.Dir.right, ratio_=3/4),
            # autopep8: on
        ]

        title_to_dockspace_name = {
            # autopep8: off
            'toolbar' : 'MainDockSpace', self.KEY_MAIN_WINDOW: 'Dock01', self.KEY_GRID_WINDOW: 'Dock02',
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
                # Create flattened parameter array

                num_params = len(ui_state.ifs_params)
                param_list = []
                if num_params > 0:
                    for i_param in range(num_params):
                        param_list.append(ui_state.ifs_params[i_param].A.flatten())
                        param_list.append(ui_state.ifs_params[i_param].B.flatten())
                else:
                    param_list.append(np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=compute_dtype))  # identity
                    num_params += 1

                params = np.concatenate(param_list, axis=0)  # (num_params * 6,)
                ndim = params.shape[0]

                # ------------------------------------------------------------------------------------
                # Sample a new parameter set

                rnd_gen = np.random.default_rng(ui_state.seed + order_idx + 1)

                # Generate directional vector in n-dimensional space
                if ui_state.dist_p == 2:
                    e = rnd_gen.normal(size=params.shape)
                    n = e / np.linalg.norm(e, ord=2)
                elif ui_state.dist_p == 1:
                    e = rnd_gen.exponential(scale=1.0, size=params.shape)
                    e = e / np.sum(e)
                    sign = rnd_gen.choice([-1.0, 1.0], size=params.shape)
                    n = sign * e
                else:
                    raise ValueError('dist_p must be 1 or 2')

                # Sample distance
                if ui_state.grid_layout == GridLayout.random:
                    pho = rnd_gen.random()  # Generate a random number from U(0, 1)
                    if ui_state.sample_uniformly:
                        pho = pho ** (1.0 / ndim)  # to sample uniformly for volume in n-dimensional space
                    dist = ui_state.dist_max * pho
                elif ui_state.grid_layout == GridLayout.monotonic:
                    t = idx / (float(ui_state.W * ui_state.H) - 1.0)  # 0.0, ..., 1.0
                    dist = ui_state.dist_max * t
                else:
                    raise ValueError('Invalid grid layout')

                # Compute sampled parameters
                sampled_params: np.ndarray = params + dist * n
                if ui_state.clip_zero_one:
                    sampled_params = np.clip(sampled_params, a_min=-1.0, a_max=1.0)  # Clamp to [-1, 1]

                # Reshape sampled parameters to (num_params, 2, 2) and (num_params, 2)
                sampled_params_reshaped = sampled_params.reshape((num_params, 6))
                sampled_A = sampled_params_reshaped[:, 0:4].reshape((num_params, 2, 2))
                sampled_B = sampled_params_reshaped[:, 4:6].reshape((num_params, 2))

                # Compute selection probabilities
                dets = np.array([np.abs(np.linalg.det(A)) for A in sampled_A])
                probs = dets / np.sum(dets)  # (num_params,)
                prob_cumsum = np.cumsum(probs)

                # Compute random numbers for parameter selection
                rnds = rnd_gen.random(ui_state.num_steps)

                # Compute IFS
                points, selected_indices = ifs_func.compute_points(sampled_A, sampled_B, prob_cumsum, rnds)
                if interrupted():
                    return None

                # ------------------------------------------------------------------------------------
                # Rescaling
                #     This approach is identical to the official FractalDB implementation.

                points_min = np.min(points, axis=0, keepdims=True)
                points_max = np.max(points, axis=0, keepdims=True)
                points: np.ndarray = (points - points_min) / (points_max - points_min + 1e-10)  # [0, 1]
                points = points * (state.img_size - 1 - 2 * state.padding) + state.padding  # Add the same amount of padding to all sides
                points = points.astype(np.int64)  # Convert to int for indexing

                # ------------------------------------------------------------------------------------
                # Coloring

                if state.cmap == 'constant':
                    colors = np.ones((points.shape[0], 3), dtype=np.float32)  # white
                elif state.cmap == 'gray':
                    quant = float(np.max(selected_indices)) + 1.0  # to avoid displaying black points
                    gray_vals = (selected_indices.astype(np.float32) + 1.0) / quant  # (N,)
                    colors = np.repeat(gray_vals[:, np.newaxis], 3, axis=1)  # (N, 3)
                else:
                    assert state.cmap in AVAILABLE_CMAPS, f'Invalid colormap: {state.cmap}'
                    num_params = len(state.ifs_params)
                    vals = selected_indices.astype(np.float32) / max(1.0, float(num_params - 1))  # (N,)
                    colors = cmap.colorize(vals, state.cmap)  # (N, 3)

                # ------------------------------------------------------------------------------------
                # Render image

                if state.enable_patch_pattern:
                    patch_pattern_indices = rnd_gen.integers(low=0, high=PATCH_PATTERNS.shape[0], size=(points.shape[0],))
                    img_np = ifs_func.render_patch_pattern_colored(
                        img_size=state.img_size,
                        points=points,
                        colors=colors,
                        patch_pattern_array=PATCH_PATTERNS,
                        patch_pattern_indices=patch_pattern_indices,
                    )
                else:
                    img_np = ifs_func.render_points_colored(
                        img_size=state.img_size,
                        points=points,
                        colors=colors,
                    )

                img = torch.from_numpy(img_np).to(device=device, dtype=torch.float32)

                # ------------------------------------------------------------------------------------
                # Recompute distance

                if state.compute_dist_exp:
                    params_A = params.reshape((num_params, 3, 2))[:, 0:2, :]  # (n, 2, 2)
                    params_dets = np.array([np.abs(np.linalg.det(A)) for A in params_A])
                    params_probs = params_dets / np.sum(params_dets)  # (num_params,)
                    params_expected = np.sum(params_probs[:, np.newaxis] * params.reshape((num_params, 6)), axis=0)  # (6,)

                    sampled_params_expected = np.sum(probs[:, np.newaxis] * sampled_params_reshaped, axis=0)  # (6,)

                    # Compute expected distance
                    actual_dist = np.linalg.norm(sampled_params_expected - params_expected, ord=ui_state.dist_p)
                else:
                    actual_dist = np.linalg.norm(sampled_params - params, ord=ui_state.dist_p)

                # ------------------------------------------------------------------------------------
                # Store to cache

                if interrupted():
                    return None

                self.render_cache[img_hash] = (actual_dist, img.detach())

                if interrupted():
                    return None

            if interrupted():
                return None

            actual_dist, img = self.render_cache[img_hash]
            img = img.to(device=device)

            return actual_dist, img, coord

        # ---------------------------------------------------------------------------------------------------------------------------------------
        # Update render state

        if self.render_state is None or self.render_state.state_hash != state_hash:
            # Compute grid size
            entire_h = state.H * state.img_size
            entire_w = state.W * state.img_size
            if entire_h > self.gl_max_tex_size or entire_w > self.gl_max_tex_size:
                self.error_msg = f'The entire image size ({entire_h}, {entire_w}) exceeded OpenGL maximum texture size ({self.gl_max_tex_size}, {self.gl_max_tex_size}). Please reduce the number of images.'
                return None
            else:
                self.error_msg = ''

            # Compute the order of rendering
            order = self._get_lds(state.W, state.H)

            # Render anchor image
            anchor_img_hash = state_hash + '_anchor'
            if anchor_img_hash not in self.render_cache:
                anchor_img = self.generate_image(state)
                if anchor_img is not None:
                    self.render_cache[anchor_img_hash] = (0.0, anchor_img)

            if anchor_img_hash in self.render_cache:
                _, anchor_img = self.render_cache[anchor_img_hash]
            else:
                anchor_img = torch.zeros((state.img_size, state.img_size, 3), device=device, dtype=torch.float32)

            # Update render state
            self.render_state = RenderState(
                state_hash=state_hash,
                img_size=[state.img_size, state.img_size],
                order=order,
                i=0,
                start_idx=0,
                end_idx=state.W * state.H,
                img=anchor_img,
                grid=torch.zeros((entire_h, entire_w, 3), device=device, dtype=torch.float32),
            )

            # Set window states (ugly ...)
            self.texture_windows[self.KEY_GRID_WINDOW].grid_size = (state.H, state.W)
            self.texture_windows[self.KEY_GRID_WINDOW].attribs = [['Rendering ...' for _ in range(state.W)] for _ in range(state.H)]

        # Update anchor image immediately
        self.texture_windows.update_image(**{self.KEY_MAIN_WINDOW: self.render_state.img})

        # Check if rendering is complete
        if self.render_state.i >= len(self.render_state.order):
            return {
                self.KEY_MAIN_WINDOW: self.render_state.img,
                self.KEY_GRID_WINDOW: self.render_state.grid,
            }

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
            ret = f.result()

            if ret is not None:
                actual_dist: float = ret[0]
                img: torch.Tensor = ret[1]
                x, y = ret[2]

                # Insert to grid
                w, h = self.render_state.img_size
                self.render_state.grid[y * h:(y + 1) * h, x * w:(x + 1) * w, :] = img
                self.texture_windows.update_image(**{self.KEY_GRID_WINDOW: self.render_state.grid})

                # Set grid states (ugly ...)
                self.texture_windows[self.KEY_GRID_WINDOW].attribs[y][x] = f'Distance: {actual_dist:.4f}'

        # Reset window title
        self.set_window_title(WINDOW_TITLE)

        # Sanity
        assert self.worker_pool.task_queue.empty(), 'Task queue is not empty after rendering'

        # All done
        self.render_state.i = len(self.render_state.order)

        return {
            self.KEY_MAIN_WINDOW: self.render_state.img,
            self.KEY_GRID_WINDOW: self.render_state.grid,
        }

    # ------------------------------------------------------------------------------------
    # Main processing functions

    def generate_image(self, state: UIState) -> torch.Tensor:
        """Process the UIState and generate a RenderState.

        Parameters
        ----------
        state : UIState
            UI state to process.

        Returns
        -------
        torch.Tensor | None
            The generated render state or None if processing failed or skipped.
        """

        if state.num_steps <= 0:
            return None

        # ------------------------------------------------------------------------------------
        # Main loop

        # Generate random numbers in [0, 1)
        rnd_gen = np.random.default_rng(state.seed)
        rnds = rnd_gen.random(state.num_steps)

        p_cumsum = np.cumsum([ifs_param.p for ifs_param in state.ifs_params])

        num_params = len(state.ifs_params)
        if num_params > 0:
            A = np.stack([s.A for s in state.ifs_params], axis=0)  # (n, 2, 2)
            B = np.stack([s.B for s in state.ifs_params], axis=0)  # (n, 2)

            # Call numba jit function
            points, selected_indices = ifs_func.compute_points(A, B, p_cumsum, rnds)
        else:
            points = np.zeros((state.num_steps, 2), dtype=compute_dtype)
            selected_indices = np.zeros((state.num_steps,), dtype=np.int64)

        # ------------------------------------------------------------------------------------
        # Rescaling
        #     This approach is identical to the official FractalDB implementation.

        points_min = np.min(points, axis=0, keepdims=True)
        points_max = np.max(points, axis=0, keepdims=True)
        points: np.ndarray = (points - points_min) / (points_max - points_min + 1e-10)  # [0, 1]
        points = points * (state.img_size - 1 - 2 * state.padding) + state.padding  # Add the same amount of padding to all sides
        points = points.astype(np.int64)  # Convert to int for indexing

        # ------------------------------------------------------------------------------------
        # Coloring

        if state.cmap == 'constant':
            colors = np.ones((points.shape[0], 3), dtype=np.float32)  # white
        elif state.cmap == 'gray':
            quant = float(np.max(selected_indices)) + 1.0  # to avoid displaying black points
            gray_vals = (selected_indices.astype(np.float32) + 1.0) / quant  # (N,)
            colors = np.repeat(gray_vals[:, np.newaxis], 3, axis=1)  # (N, 3)
        else:
            assert state.cmap in AVAILABLE_CMAPS, f'Invalid colormap: {state.cmap}'
            num_params = len(state.ifs_params)
            vals = selected_indices.astype(np.float32) / max(1.0, float(num_params - 1))  # (N,)
            colors = cmap.colorize(vals, state.cmap)  # (N, 3)

        # ------------------------------------------------------------------------------------
        # Render image

        if state.enable_patch_pattern:
            patch_pattern_indices = rnd_gen.integers(low=0, high=PATCH_PATTERNS.shape[0], size=(points.shape[0],))
            img = ifs_func.render_patch_pattern_colored(
                img_size=state.img_size,
                points=points,
                colors=colors,
                patch_pattern_array=PATCH_PATTERNS,
                patch_pattern_indices=patch_pattern_indices,
            )
            self.rnd_counter += 1
        else:
            img = ifs_func.render_points_colored(
                img_size=state.img_size,
                points=points,
                colors=colors,
            )

        img = torch.from_numpy(img).to(device=device, dtype=torch.float32)

        return img

    # ------------------------------------------------------------------------------------
    # UI components

    @ui_utils.dockable
    def toolbar(self):
        """Build the toolbar UI for the visualizer.
        """

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
            elif self.keyhit(imgui.Key.up_arrow):
                self.param_file_cursor = (self.param_file_cursor - 1) % max(1, len(self.param_file_list))
                self._update_ifs_param_from_file()
            elif self.keyhit(imgui.Key.down_arrow):
                self.param_file_cursor = (self.param_file_cursor + 1) % max(1, len(self.param_file_list))
                self._update_ifs_param_from_file()

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

        # ---------------------------------------------------------------------------------------------------
        # Universal parameters

        imgui.separator_text('Parameters')

        self.state.H = imgui.slider_int('H', self.state.H, 1, 32)[1]
        self.state.W = imgui.slider_int('W', self.state.W, 1, 32)[1]
        self.state.seed = imgui.input_int('Random Seed', self.state.seed)[1]
        self.state.img_size = max(min(imgui.input_int('Image Size', self.state.img_size)[1], self.MAX_IMG_SIZE), self.MIN_IMG_SIZE)
        self.state.num_steps = max(imgui.input_int('Num Steps', self.state.num_steps)[1], 0)
        self.state.padding = max(min(imgui.input_int('Padding', self.state.padding)[1], self.state.img_size // 2 - 1), 0)
        self.state.enable_patch_pattern = imgui.checkbox('Enable Patch Pattern', self.state.enable_patch_pattern)[1]

        # ---------------------------------------------------------------------------------------------------
        # Distance parameters

        imgui.separator_text('Distance Sampling Parameters')

        self.state.dist_p = imgui.combo('Distance p-norm', self.state.dist_p - 1, ['1', '2'])[1] + 1
        self.state.dist_max = imgui.slider_float('Distance Max', self.state.dist_max, 0.0, 10.0)[1]
        self.state.grid_layout = self.state.grid_layout.combo('Grid Layout')
        if self.state.grid_layout == GridLayout.random:
            self.state.sample_uniformly = imgui.checkbox('Sample Uniformly (per volume) in n-Dim Space', self.state.sample_uniformly)[1]
        self.state.clip_zero_one = imgui.checkbox('Clip Parameters to [-1, 1]', self.state.clip_zero_one)[1]
        self.state.compute_dist_exp = imgui.checkbox('Compute Distance on Expectation', self.state.compute_dist_exp)[1]

        # ---------------------------------------------------------------------------------------------------
        # IFS parameters

        # Load from file
        imgui.separator_text('Load IFS Params from File')
        load_from_file = imgui.checkbox('Load from file', self.state.load_from_file)[1]
        if load_from_file:
            params_dir = imgui.input_text('##params_dir', self.params_dir)[1]
            imgui.same_line()
            if imgui.button('Open'):
                default_path = params_dir if os.path.isdir(params_dir) else os.getcwd()
                params_dir = nfdpy.pick_folder_dialog(default_path=default_path)
            if params_dir is not None and os.path.isdir(params_dir):
                if self.params_dir != params_dir:
                    self._update_param_file_list(params_dir)
                self.params_dir = params_dir
            imgui.text(f'Found {len(self.param_file_list)} csv files')

            if len(self.param_file_list) > 0:
                self.load_weighted_params = imgui.checkbox('Load Weighted Params', self.load_weighted_params)[1]

                cur_file = self.param_file_list[self.param_file_cursor]
                new_file = imgui.input_text('Current File', cur_file)[1]
                if new_file in self.param_file_list:
                    self.param_file_cursor = self.param_file_list.index(new_file)
                    if cur_file != new_file:
                        self._update_ifs_param_from_file()
                    cur_file = new_file

                if imgui.button('Load', size=(-1, 0)):
                    self._update_ifs_param_from_file()

                if len(self.param_class_start_idx) > 0:
                    avail_width = imgui.get_content_region_avail()[0]
                    if imgui.button('Prev class', size=(avail_width // 2, 0)):
                        # Find the previous class start index
                        prev_cursor = None
                        for class_start_idx in reversed(self.param_class_start_idx):
                            if class_start_idx < self.param_file_cursor:
                                prev_cursor = class_start_idx
                                break
                        if prev_cursor is None:
                            prev_cursor = self.param_class_start_idx[-1]  # Loop back to the last class
                        self.param_file_cursor = prev_cursor
                        self._update_ifs_param_from_file()
                    imgui.same_line()
                    if imgui.button('Next class', size=(avail_width // 2, 0)):
                        # Find the next class start index
                        next_cursor = None
                        for class_start_idx in self.param_class_start_idx:
                            if class_start_idx > self.param_file_cursor:
                                next_cursor = class_start_idx
                                break
                        if next_cursor is None:
                            next_cursor = self.param_class_start_idx[0]  # Loop back to the first class
                        self.param_file_cursor = next_cursor
                        self._update_ifs_param_from_file()

                imgui.text_colored(highlight_color, 'Tips: use "arrow" keys to switch files')

                # Initialize ifs_params when enabling the option
                if load_from_file != self.state.load_from_file:
                    self._update_ifs_param_from_file()

        self.state.load_from_file = load_from_file

        imgui.separator_text('IFS Parameters')

        avail_width = imgui.get_content_region_avail()[0]
        if imgui.button('Add New Param', size=(avail_width // 2, 0)) and len(self.state.ifs_params) < self.MAX_NUM_PARAMS:
            self.rnd_counter += 1
            rnd_gen = random.Random(self.state.seed + self.rnd_counter)  # Make it deterministic
            self.state.ifs_params.append(
                IFSParam(
                    A=np.array([[rnd_gen.uniform(-1.0, 1.0) for _ in range(2)] for _ in range(2)], dtype=compute_dtype),
                    B=np.array([rnd_gen.uniform(-1.0, 1.0) for _ in range(2)], dtype=compute_dtype),
                    p=1.0
                )
            )
        imgui.same_line()
        if imgui.button('Remove Selected Params', size=(avail_width // 2, 0)):
            for i_param in range(len(self.state.ifs_params)):
                if self.state.ifs_params[i_param].selected:
                    self.state.ifs_params[i_param] = None

            # Remove None
            self.state.ifs_params = [p for p in self.state.ifs_params if p is not None]

        avail_width = imgui.get_content_region_avail()[0]
        if imgui.button('Select All', size=(avail_width // 2, 0)):
            for i_param in range(len(self.state.ifs_params)):
                self.state.ifs_params[i_param].selected = True
        imgui.same_line()
        if imgui.button('Deselect All', size=(avail_width // 2, 0)):
            for i_param in range(len(self.state.ifs_params)):
                self.state.ifs_params[i_param].selected = False

        self.state.fix_probabilities = imgui.checkbox('Fix Probabilities', self.state.fix_probabilities)[1]

        n_cols = 4  # checkbox + A + B + p
        if imgui.begin_table(
            'IFS Param Table',
            columns=n_cols,
            flags=imgui.TableFlags_.borders | imgui.TableFlags_.scroll_y | imgui.TableFlags_.row_bg,
            outer_size=(-1, 300)
        ):
            avail_width = imgui.get_content_region_avail()[0]
            stride = avail_width // (1 + 2 + 1 + 1)   # checkbox + A(2) + B(1) + p(1)

            imgui.table_setup_column(' ', imgui.TableColumnFlags_.width_fixed, stride)  # checkbox
            imgui.table_setup_column('A', imgui.TableColumnFlags_.width_fixed, stride * 2)  # A(2)
            imgui.table_setup_column('B', imgui.TableColumnFlags_.width_fixed, stride)
            imgui.table_setup_column('Probability', imgui.TableColumnFlags_.width_fixed, stride)
            imgui.table_headers_row()

            # Table rows
            v_speed = 0.005
            v_min = -1.0
            v_max = 1.0
            for i_param, ifs_param in enumerate(self.state.ifs_params):
                imgui.table_next_row()

                if i_param == 0:
                    for i in range(n_cols):
                        imgui.table_set_column_index(i)
                        imgui.push_item_width(-1)

                # Column 1: checkbox
                imgui.table_next_column()
                ifs_param.selected = imgui.checkbox(f'##select_{i_param}', ifs_param.selected)[1]

                # Column 2: matrix A
                imgui.table_next_column()
                if imgui.begin_table(f'##mat_A_{i_param}', 2, imgui.TableFlags_.none):
                    for i_row in range(2):
                        imgui.table_next_row()
                        for i_col in range(2):
                            imgui.table_next_column()
                            self.state.ifs_params[i_param].A[i_row, i_col] = imgui.drag_float(f'##a_{i_param}_{i_row}_{i_col}', self.state.ifs_params[i_param].A[i_row, i_col], v_min=v_min, v_max=v_max, v_speed=v_speed)[1]
                    imgui.end_table()

                # Column 3: vector B
                imgui.table_next_column()
                if imgui.begin_table(f'##vec_B_{i_param}', 1, imgui.TableFlags_.none):
                    for i_row in range(2):
                        imgui.table_next_row()
                        imgui.table_next_column()
                        self.state.ifs_params[i_param].B[i_row] = imgui.drag_float(f'##b_{i_param}_{i_row}', self.state.ifs_params[i_param].B[i_row], v_min=v_min, v_max=v_max, v_speed=v_speed)[1]
                    imgui.end_table()

                # Column 3: vector p
                imgui.table_next_column()
                imgui.text(f'{ifs_param.p:.4f}')

            imgui.end_table()

        # Clip parameters
        for i_param in range(len(self.state.ifs_params)):
            self.state.ifs_params[i_param].A = np.clip(self.state.ifs_params[i_param].A, -1.0, 1.0)
            self.state.ifs_params[i_param].B = np.clip(self.state.ifs_params[i_param].B, -1.0, 1.0)

        # ---------------------------------------------------------------------------------------------------
        # Compute probability sum

        if not self.state.fix_probabilities:
            dets = []
            for ifs_param in self.state.ifs_params:
                dets.append(abs(np.linalg.det(ifs_param.A)))
            sum_dets = sum(dets)
            for i_param in range(len(self.state.ifs_params)):
                p = dets[i_param] / sum_dets
                assert p >= 0.0, 'The sum of determinants should be non-negative!'
                self.state.ifs_params[i_param].p = p

        # ---------------------------------------------------------------------------------------------------
        # View options

        imgui.separator_text('View Options')

        cur_cmap_idx = AVAILABLE_CMAPS.index(self.state.cmap)
        new_cmap_idx = imgui.combo('Colormap', cur_cmap_idx, AVAILABLE_CMAPS)[1]
        self.state.cmap = AVAILABLE_CMAPS[new_cmap_idx]

        # ---------------------------------------------------------------------------------------------------
        imgui.separator()

        if len(self.error_msg) > 0:
            imgui.text_colored((1.0, 0.0, 0.0, 1.0), f'Error: {self.error_msg}')


def main():
    parser = argparse.ArgumentParser(description='IFS Visualizer', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--enable_vsync', action='store_true', help='Enable display vertical synchronization (vsync)')
    parser.add_argument('--no_cache', action='store_true', help='Do not cache UI states to files')
    parser.add_argument('--no_cache_ui', action='store_true', help='Do not cache UI layouts to files')
    args = parser.parse_args()

    viz = DistanceViz(
        enable_vsync=args.enable_vsync,
        cache_states=not args.no_cache,
        cache_ui_layout=not args.no_cache_ui,
    )

    print('done.')


if __name__ == '__main__':
    main()
