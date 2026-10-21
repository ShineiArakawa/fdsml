"""
IFS Visualizer
==============

A visualizer for Iterated Function Systems (IFS).

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
import copy
import dataclasses
import hashlib
import json
import os
import pathlib
import random
import typing

import torch # NOTE: Make sure to import torch before pyviewer_extended to avoid potential conflicts

import cachetools
import cmap
import natsort
import nfdpy
import numba
import numpy as np
import pandas as pd
import pydantic
import typing_extensions
import ui.utils as ui_utils
import ui.docking_viewer as docking_viewer
import ifs_func
from imgui_bundle import hello_imgui, imgui  # type: ignore

# autopep8: off

# ------------------------------------------------------------------------------------------------------------
# Logger

logger = ui_utils.get_logger()

# --------------------------------------------------------------------------------------------------------------------------------
# Constants

WINDOW_TITLE = 'IFS Visualizer'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
compute_dtype = np.float64

# Available color maps
AVAILABLE_CMAPS: typing.Final[list[str]] = natsort.natsorted(list(cmap.available_color_maps()))
AVAILABLE_CMAPS = ['constant', 'gray'] + [c for c in AVAILABLE_CMAPS]


# --------------------------------------------------------------------------------------------------------------------------------
# Data classes


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
    seed                 : int               = 100
    img_size             : int               = 362
    num_steps            : int               = 200000
    ifs_params           : list[IFSParam]    = dataclasses.field(default_factory=lambda: [])
    patch_pattern        : list[list[bool]]  = dataclasses.field(default_factory=lambda: [[False, False, False], [False, True, False], [False, False, False]])
    padding              : int               = 6
    load_from_file       : bool              = False
    fix_probabilities    : bool              = False
    edit_direction       : list[list[float]] = dataclasses.field(default_factory=lambda: [[0.0, 0.0] for _ in range(3)])  # A(2) + B(1)
    edit_amount          : float             = 0.0
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
    img                     : torch.Tensor = None
    # autopep8: on

# --------------------------------------------------------------------------------------------------------------------------------


class IFSViz(docking_viewer.MultiTexturesDockingViewer):
    # ------------------------------------------------------------------------------------
    # Constants

    # autopep8: off
    KEY_MAIN_WINDOW     : typing.Final[str] = 'Main window'

    WINDOW_KEYS = [
        KEY_MAIN_WINDOW,
    ]

    MIN_IMG_SIZE        : int               = 5    # it has to be larger than 4
    MAX_IMG_SIZE        : int               = 2048 # 256 is already heavy with padding even if using CUDA backend ...

    MAX_NUM_PARAMS      : int               = 8    # according to the original FractalDB paper

    STATES_CACHE_PATH   : pathlib.Path      = pathlib.Path('.cache/ifs_viz_params.json')
    PCA_COMP_PATH       : pathlib.Path      = pathlib.Path('data/pca_components.csv')
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

        self.params_dir = os.path.join(os.getcwd(), 'datasets', 'finefractaldb', 'params')
        self.param_file_list: list[str] = []
        self.param_class_start_idx: list[int] = []
        self.param_file_cursor = 0
        self.param_file_cache: dict[str, IFSParamFile] = cachetools.FIFOCache(maxsize=512)
        self.load_weighted_params = True
        self.auto_load_param_file = True
        self.enable_directional_editing = False
        self._directional_edit_origin: list[np.ndarray, np.ndarray] = []

        # Load PCA components
        self._pca_comps: np.ndarray | None = None
        self._pca_comps_inv: np.ndarray | None = None
        if self.PCA_COMP_PATH.is_file():
            df = pd.read_csv(self.PCA_COMP_PATH, index_col=0)
            pca_comps = df.values.astype(dtype=compute_dtype)  # (6, 6)

            self._pca_comps = pca_comps.transpose()
            self._pca_comps_inv = np.linalg.inv(self._pca_comps)

        self._edit_in_pca_space = False

        # -----------------------------------------------------------------------------
        if not self.cache_ui_layout:
            # Remove ini file ... ugly
            # .ini for window and app state saving'

            ini_filename = WINDOW_TITLE.lower().strip().replace(' ', '_') + '.ini'
            ini_path = pathlib.Path(hello_imgui.ini_folder_location(hello_imgui.IniFolderType.app_user_config_folder)) / ini_filename
            if ini_path.is_file():
                os.remove(str(ini_path))
                logger.info(f'Removed {ini_path}')

        super(IFSViz, self).__init__(
            name=WINDOW_TITLE,
            texture_names=self.WINDOW_KEYS,
            enable_vsync=enable_vsync,
            full_screen_mode=hello_imgui.FullScreenMode.full_monitor_work_area,
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

            logger.info(f'Loaded parameters from {file_path}')
        else:
            logger.error(f'Failed to load parameters from {file_path}')

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
            logger.info(f'Saved parameters to {cache_file}')
        except Exception as e:
            logger.error(f'[ERROR] Failed to save parameters to {cache_file}')
            logger.error(e)

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
            logger.error(f'File not found: {file_path}')
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

    # ------------------------------------------------------------------------------------

    @typing_extensions.override
    def draw_status_bar(self):
        return super().draw_status_bar()

    @typing_extensions.override
    def setup_state(self):
        self.state = UIState()

        self.render_state_cache: dict[str, RenderState] = cachetools.FIFOCache(maxsize=512)
        self.render_state = RenderState()

        self.params_cache_file = self.STATES_CACHE_PATH

        # Load params
        if self.cache_params_to_file and self.params_cache_file.is_file():
            self._load_param_caches(self.params_cache_file)

        # Update param cache
        self.cached_states[self.cur_cache_param_id] = self.state

        self._update_param_file_list(self.params_dir)

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
    def compute(self):
        state = copy.deepcopy(self.state)  # Detach parameters

        state_hash = state.deterministic_hash()

        if state_hash not in self.render_state_cache:
            render_state = self.process(state)
            if render_state is None:
                return None
            self.render_state_cache[state_hash] = render_state

        render_state = self.render_state_cache[state_hash]

        return {
            self.KEY_MAIN_WINDOW: render_state.img,
        }

    # ------------------------------------------------------------------------------------
    # Main processing functions

    @staticmethod
    @numba.jit(nopython=True, cache=True, parallel=False)  # Enable JIT Compile with numba
    def _render_patch_pattern(img_size: int, points: np.ndarray, patch_pattern: np.ndarray, colors: np.ndarray) -> np.ndarray:
        """Render the patch pattern on the image.

        Parameters
        ----------
        img_size : int
            The size of the image (assumed to be square).
        points : np.ndarray
            The point coordinates to render. Shape: `(num_points, 2)`
        patch_pattern : np.ndarray
            Boolean 2D array indicating the patch pattern. Shape: `(3, 3)`
        colors : np.ndarray
            The colors for each point. Shape: `(num_points, 3)`

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
            color = colors[i_point]

            for del_y in [-1, 0, 1]:
                y = center_y + del_y
                for del_x in [-1, 0, 1]:
                    x = center_x + del_x
                    if patch_pattern[del_y + 1][del_x + 1] and 0 <= x < img_size and 0 <= y < img_size:  # (patch_pattern?) and (in width?) and (in height?)
                        img[y, x, :] = color

        return img

    def process(self, state: UIState) -> RenderState | None:
        """Process the UIState and generate a RenderState.

        Parameters
        ----------
        state : UIState
            UI state to process.

        Returns
        -------
        RenderState | None
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
        points = (points - points_min) / (points_max - points_min + 1e-10)  # [0, 1]
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

        img = IFSViz._render_patch_pattern(
            img_size=state.img_size,
            points=points,
            patch_pattern=np.asarray(state.patch_pattern, dtype=bool),
            colors=colors,
        )

        img = torch.from_numpy(img).to(device=device, dtype=torch.float32)

        return RenderState(
            img=img,
        )

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

        self.state.seed = imgui.input_int('Random Seed', self.state.seed)[1]
        self.state.img_size = max(min(imgui.input_int('Image Size', self.state.img_size)[1], self.MAX_IMG_SIZE), self.MIN_IMG_SIZE)
        self.state.num_steps = max(imgui.input_int('Num Steps', self.state.num_steps)[1], 0)
        self.state.padding = max(min(imgui.input_int('Padding', self.state.padding)[1], self.state.img_size // 2 - 1), 0)

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
        self._edit_in_pca_space = imgui.checkbox('Edit in PCA Space', self._edit_in_pca_space)[1]

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

            # Apply pca
            params_A: list[np.ndarray] = []
            params_B: list[np.ndarray] = []
            for ifs_param in self.state.ifs_params:
                if self._edit_in_pca_space and self._pca_comps is not None:
                    param_vec = np.array(
                        [
                            ifs_param.A[0, 0], ifs_param.A[0, 1],
                            ifs_param.A[1, 0], ifs_param.A[1, 1],
                            ifs_param.B[0], ifs_param.B[1]
                        ],
                        dtype=compute_dtype
                    )  # (6,)
                    param_vec = param_vec @ self._pca_comps  # (6,)
                    params_A.append(param_vec[0:4].reshape((2, 2)))
                    params_B.append(param_vec[4:6].reshape((2,)))
                else:
                    params_A.append(ifs_param.A)
                    params_B.append(ifs_param.B)

            # Table rows
            v_speed = 0.005
            v_min = -10.0 if self._edit_in_pca_space else -1.0
            v_max = 10.0 if self._edit_in_pca_space else 1.0
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
                            params_A[i_param][i_row, i_col] = imgui.drag_float(f'##a_{i_param}_{i_row}_{i_col}', params_A[i_param][i_row, i_col], v_min=v_min, v_max=v_max, v_speed=v_speed)[1]
                    imgui.end_table()

                # Column 3: vector B
                imgui.table_next_column()
                if imgui.begin_table(f'##vec_B_{i_param}', 1, imgui.TableFlags_.none):
                    for i_row in range(2):
                        imgui.table_next_row()
                        imgui.table_next_column()
                        params_B[i_param][i_row] = imgui.drag_float(f'##b_{i_param}_{i_row}', params_B[i_param][i_row], v_min=v_min, v_max=v_max, v_speed=v_speed)[1]
                    imgui.end_table()

                # Column 3: vector p
                imgui.table_next_column()
                imgui.text(f'{ifs_param.p:.4f}')

            imgui.end_table()

        # Directional Editing
        enable_directional_editing = imgui.checkbox('Directional Editing', self.enable_directional_editing)[1]

        if enable_directional_editing:
            # Buffer the origin when enabling the option
            if self.enable_directional_editing != enable_directional_editing:
                # Save the origin
                self._directional_edit_origin = []
                for i_param in range(len(params_A)):
                    self._directional_edit_origin.append((params_A[i_param].copy(), params_B[i_param].copy()))

            imgui.text_colored(highlight_color, 'Notice: Directional editing is enabled. The origins are fixed when enabling this option.')

            n_cols = 2  # A + B
            if imgui.begin_table(
                'Edit Direction',
                columns=n_cols,
                flags=imgui.TableFlags_.borders | imgui.TableFlags_.scroll_y | imgui.TableFlags_.row_bg,
                outer_size=(-1, 0)
            ):
                avail_width = imgui.get_content_region_avail()[0]
                stride = avail_width // (2 + 1)   # A(2) + B(1)

                imgui.table_setup_column('A', imgui.TableColumnFlags_.width_fixed, stride * 2)  # A(2)
                imgui.table_setup_column('B', imgui.TableColumnFlags_.width_fixed, stride)
                imgui.table_headers_row()

                # Single row
                imgui.table_next_row()

                # Column 1: matrix A
                imgui.table_next_column()
                if imgui.begin_table(f'##edit_mat_A', 2, imgui.TableFlags_.none):
                    for i_row in range(2):
                        imgui.table_next_row()
                        for i_col in range(2):
                            imgui.table_next_column()
                            self.state.edit_direction[i_row][i_col] = imgui.input_double(f'##edit_a_{i_row}_{i_col}', self.state.edit_direction[i_row][i_col])[1]
                    imgui.end_table()

                # Column 2: vector B
                imgui.table_next_column()
                if imgui.begin_table(f'##edit_vec_B', 1, imgui.TableFlags_.none):
                    for i_row in range(2):
                        imgui.table_next_row()
                        imgui.table_next_column()
                        self.state.edit_direction[2][i_row] = imgui.input_double(f'##edit_b_{i_row}', self.state.edit_direction[2][i_row])[1]
                    imgui.end_table()

                imgui.end_table()

            self.state.edit_amount = imgui.drag_float('Edit Amount', self.state.edit_amount, v_min=-1.0, v_max=1.0, v_speed=v_speed)[1]

            dA = np.array(self.state.edit_direction[0:2], dtype=compute_dtype)  # (2, 2)
            dB = np.array(self.state.edit_direction[2], dtype=compute_dtype).reshape((2,))  # (2, 1)

            for i_param, ifs_param in enumerate(self.state.ifs_params):
                if ifs_param.selected:
                    params_A[i_param] = (self._directional_edit_origin[i_param][0] + dA * self.state.edit_amount)
                    params_B[i_param] = (self._directional_edit_origin[i_param][1] + dB * self.state.edit_amount)

        self.enable_directional_editing = enable_directional_editing

        # ---------------------------------------------------------------------------------------------------
        # Apply back the edited params

        for i_param, ifs_param in enumerate(self.state.ifs_params):
            if self._edit_in_pca_space and self._pca_comps_inv is not None:
                param_vec = np.array(
                    [
                        params_A[i_param][0, 0], params_A[i_param][0, 1],
                        params_A[i_param][1, 0], params_A[i_param][1, 1],
                        params_B[i_param][0], params_B[i_param][1]
                    ],
                    dtype=compute_dtype
                )  # (6,)
                param_vec = param_vec @ self._pca_comps_inv  # (6,)
                ifs_param.A = param_vec[0:4].reshape((2, 2))
                ifs_param.B = param_vec[4:6].reshape((2,))
            else:
                ifs_param.A = params_A[i_param]
                ifs_param.B = params_B[i_param]

            ifs_param.A = np.clip(ifs_param.A, -1.0, 1.0)
            ifs_param.B = np.clip(ifs_param.B, -1.0, 1.0)

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
        # Patch pattern

        imgui.separator_text('Patch Pattern')

        if imgui.begin_table('Patch Pattern Table', columns=9, flags=imgui.TableFlags_.row_bg):
            n_cols = 3
            for i_row in range(n_cols):
                imgui.table_next_row()
                for i_col in range(n_cols):
                    if i_col == 0:
                        for i in range(n_cols):
                            imgui.table_set_column_index(i)
                            imgui.push_item_width(-1)
                    imgui.table_next_column()
                    if imgui.checkbox(f'##patch_{i_row}_{i_col}', self.state.patch_pattern[i_row][i_col])[1]:
                        self.state.patch_pattern[i_row][i_col] = True
                    else:
                        self.state.patch_pattern[i_row][i_col] = False
            imgui.end_table()

        # ---------------------------------------------------------------------------------------------------
        # View options

        imgui.separator_text('View Options')

        cur_cmap_idx = AVAILABLE_CMAPS.index(self.state.cmap)
        new_cmap_idx = imgui.combo('Colormap', cur_cmap_idx, AVAILABLE_CMAPS)[1]
        self.state.cmap = AVAILABLE_CMAPS[new_cmap_idx]


def main():
    parser = argparse.ArgumentParser(description='IFS Visualizer', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--enable_vsync', action='store_true', help='Enable display vertical synchronization (vsync)')
    parser.add_argument('--no_cache', action='store_true', help='Do not cache UI states to files')
    parser.add_argument('--no_cache_ui', action='store_true', help='Do not cache UI layouts to files')
    args = parser.parse_args()

    viz = IFSViz(
        enable_vsync=args.enable_vsync,
        cache_states=not args.no_cache,
        cache_ui_layout=not args.no_cache_ui,
    )

    logger.info('done.')


if __name__ == '__main__':
    main()
