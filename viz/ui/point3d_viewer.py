"""
point3d_viewer
==============

A module for viewing and interacting with 3D point clouds using a docking UI.

This source code is based on "pyviewer-extended" repository by Shinei Arakawa but has been heavily modified to enable drawing on textures. 

License
-------

Copyright (c) 2026 Shinei Arakawa
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

This code is a derivative work based on code from the harskish/pyviewer repository, whose original code is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0). The original code can be found at https://github.com/harskish/pyviewer.
"""

import functools
import pathlib
import threading
import time
import typing

import pyviewer
import pyviewer.imgui_themes as imgui_themes
import pyviewer.utils as utils
import ui.point3d_window as windows
import ui.utils as ui_utils
from imgui_bundle import glfw_utils, hello_imgui, imgui, immapp  # type: ignore
from imgui_bundle.demos_python.demo_utils import demos_assets_folder
from imgui_bundle.glfw_utils import \
    glfw  # NOTE: Import GLFW from imgui_bundle.glfw_utils to avoid multile definitions

# ------------------------------------------------------------------------------------------------------------
# Logger

logger = ui_utils.get_logger()


# ------------------------------------------------------------------------------------------------------------
# Why not Python backend?
# - No Nanovg (https://github.com/pthom/imgui_bundle/issues/259#issuecomment-2391258789)
# - No fps throttling ("sleep mode" on inactivity)
# - Not well supported/tested in general
# - Probably won't get non-glfw backends
# - CPP backend sometimes releases GIL, should be slightly more performant


# ------------------------------------------------------------------------------------------------------------
# Utilities


def file_drop_callback_wrapper(window: glfw._GLFWwindow, paths: list[pathlib.Path], callback: typing.Callable[[list[pathlib.Path]], bool], prev: typing.Callable):
    return callback([pathlib.Path(p) for p in paths]) or prev(window, paths)


# ------------------------------------------------------------------------------------------------------------
# MultiTextureDockingViewer


class Point3DDockingViewer:
    def __init__(
        self,
        name: str,
        enable_vsync=False,
        full_screen_mode=hello_imgui.FullScreenMode.no_full_screen,
        with_implot=True,
        with_implot3d=False,
        with_node_editor=False,
        with_node_editor_config=None,
        with_tex_inspect=False,
        with_font_awesome=False,
    ):
        # Immapp:
        #  immapp.run() python stub:   https://github.com/pthom/imgui_bundle/blob/v1.6.2/bindings/imgui_bundle/immapp/immapp_cpp.pyi#L162
        #  immapp.run() nanobind impl: https://github.com/pthom/imgui_bundle/blob/v1.6.2/external/immapp/bindings/pybind_immapp_cpp.cpp#L158
        #  immapp.run() CPP impl:      https://github.com/pthom/imgui_bundle/blob/v1.6.2/external/immapp/immapp/runner.cpp#L233

        # Hello-Imgui:
        #  hello_imgui.run() python stub:     https://github.com/pthom/imgui_bundle/blob/v1.6.2/bindings/imgui_bundle/hello_imgui.pyi#L3416
        #  hello_imgui.run() nanobind impl:   https://github.com/pthom/imgui_bundle/blob/v1.6.2/external/hello_imgui/bindings/pybind_hello_imgui.cpp#L1761
        #  hello_imgui.run() CPP impl:        https://github.com/pthom/hello_imgui/blob/c98503154f66/src/hello_imgui/impl/hello_imgui.cpp#L227
        #  AbstractRunner::Run():             https://github.com/pthom/hello_imgui/blob/c98503154f66/src/hello_imgui/internal/backend_impls/abstract_runner.cpp#L124
        #  PreNewFrame call:                  https://github.com/pthom/hello_imgui/blob/c98503154f66/src/hello_imgui/internal/backend_impls/abstract_runner.cpp#L1412
        #  SCOPED_RELEASE_GIL_ON_MAIN_THREAD: https://github.com/pthom/hello_imgui/blob/c98503154f66/src/hello_imgui/internal/backend_impls/abstract_runner.cpp#L1443
        #  fnReloadFontsIfDpiScaleChanged:    https://github.com/pthom/hello_imgui/blob/c98503154f66/src/hello_imgui/internal/backend_impls/abstract_runner.cpp#L947

        # Start compute thread asap
        self.start_event: threading.Event = threading.Event()
        self.stop_event: threading.Event = threading.Event()
        compute_thread = threading.Thread(target=self.compute_loop, args=[], daemon=True)
        compute_thread.start()

        # Installed by pip, includes two fonts
        hello_imgui.set_assets_folder(demos_assets_folder())

        runner_params = hello_imgui.RunnerParams()
        runner_params.app_window_params.window_title = name
        runner_params.app_window_params.window_geometry.size = (1200, 800)
        runner_params.app_window_params.window_geometry.full_screen_mode = full_screen_mode
        runner_params.app_window_params.restore_previous_geometry = True
        runner_params.fps_idling.enable_idling = False  # Disable fps idling to animate point cloud
        runner_params.fps_idling.fps_idle = 5.0

        # Normally setting no_mouse_input windows flags on containing window is enough,
        # but docking (presumably) seems to be capturing mouse input regardless.
        self._ui_scale = 1.0
        self.ui_locked = True  # resize in progress?
        self.first_frame = True  # e.g. imgui.set_scroll_size wonky on first frame

        # Main image (output of self.compute())
        self.state = ui_utils.EasyDict()

        self.initial_font_size = 15
        self.default_font: imgui.ImFont = None
        self.fonts: list[imgui.ImFont] = []
        self.window: glfw._GLFWwindow = None
        self._window_title = name
        self._orig_window_title = name
        self.load_font_awesome = with_font_awesome
        self.enable_vsync = enable_vsync

        # Check if HDR mode has been turned on
        from pyviewer import _macos_hdr_patch
        self.hdr = (_macos_hdr_patch.CUR_MODE == _macos_hdr_patch.Mode.PATCHED)

        # Create texture windows
        self.point3d_window = windows.Point3DWindow('point3d_window', (1200, 800))

        # For limiting OpenGL operations to UI thread
        self.ui_tid = threading.get_native_id()  # main thread

        def load_settings_cbk():
            try:
                self.ui_scale = float(hello_imgui.load_user_pref('ui_scale'))
            except:
                pass
            self.load_settings()

        def post_init_fun():
            glfw.make_context_current(self.window)

            if self.enable_vsync:
                glfw.swap_interval(1)  # Enable vsync

            # Create OpenGL texture
            self.point3d_window.initialize()

            # Setup
            load_settings_cbk()
            self.setup_state()
            self.start_event.set()

        def save_settings_cbk():
            hello_imgui.save_user_pref("ui_scale", f'{self.ui_scale}')
            self.save_settings()

        def before_exit():
            save_settings_cbk()
            self.stop_event.set()

            # Dispose OpenGL textures
            self.point3d_window.dispose()

        def add_backend_cbk(*args, **kwargs):
            # Set own glfw callbacks, will be chained by imgui
            self.window = glfw_utils.glfw_window_hello_imgui()  # why not glfw.get_current_context()?

            # Set callbacks for PannableArea
            self.point3d_window.set_callbacks(self.window)

            # Set callback for drop event
            def noop(*args, **kwargs):
                return False
            prev = glfw.set_drop_callback(self.window, None)
            glfw.set_drop_callback(self.window, functools.partial(file_drop_callback_wrapper, callback=self.drag_and_drop_callback, prev=(prev or noop)))

        def setup_theme_cbk():
            self.setup_theme()  # user-overridable
            self.scale_style_sizes()

            # Set theme for PannableArea
            self.point3d_window.set_theme()

        def pre_new_frame_wrapper():
            # Push default font at current size
            imgui.push_font(self.default_font, self.initial_font_size * self.ui_scale)
            self.pre_new_frame()

        def after_swap():
            self.first_frame = False
            glfw.set_window_title(self.window, self._window_title)  # from main thread
            imgui.pop_font()  # pop scaled default font pushed in `pre_new_frame_wrapper`

        runner_params.callbacks.post_init = post_init_fun
        runner_params.callbacks.before_exit = before_exit
        runner_params.callbacks.post_init_add_platform_backend_callbacks = add_backend_cbk
        runner_params.callbacks.load_additional_fonts = self.load_fonts
        runner_params.callbacks.setup_imgui_style = setup_theme_cbk
        runner_params.callbacks.pre_new_frame = pre_new_frame_wrapper
        runner_params.callbacks.after_swap = after_swap

        self.show_app_menu = False
        self.show_view_menu = True
        runner_params.imgui_window_params.show_menu_bar = True
        runner_params.imgui_window_params.show_menu_app = False  # called manually
        runner_params.imgui_window_params.show_menu_view = False  # called manually
        runner_params.callbacks.show_menus = lambda: self._draw_menu_wrapper(runner_params)

        # Status bar: fps etc.
        runner_params.imgui_window_params.remember_status_bar_settings = False  # don't cache
        if type(self).draw_status_bar != Point3DDockingViewer.draw_status_bar:  # overridden
            runner_params.imgui_window_params.show_status_bar = True  # off by default
            runner_params.callbacks.show_status = self.draw_status_bar

        # Create "MainDockSpace"
        runner_params.imgui_window_params.default_imgui_window_type = (
            hello_imgui.DefaultImGuiWindowType.provide_full_screen_dock_space
        )

        # Allow splitting into separate windows?
        runner_params.imgui_window_params.enable_viewports = True

        # Docking layout
        runner_params.docking_params = self.setup_layout()
        runner_params.docking_params.main_dock_space_node_flags |= imgui.DockNodeFlags_.auto_hide_tab_bar

        # .ini for window and app state saving'
        runner_params.ini_folder_type = hello_imgui.IniFolderType.app_user_config_folder
        runner_params.ini_filename = name.lower().strip().replace(' ', '_') + '.ini'
        ini_path = pathlib.Path(hello_imgui.ini_folder_location(runner_params.ini_folder_type)) / runner_params.ini_filename
        logger.info(f'INI path: {ini_path}')

        glfw.init()

        if self.hdr:
            glfw.window_hint(glfw.RED_BITS, 16)
            glfw.window_hint(glfw.GREEN_BITS, 16)
            glfw.window_hint(glfw.BLUE_BITS, 16)

        addons = immapp.AddOnsParams(
            with_implot=with_implot,
            with_implot3d=with_implot3d,
            with_node_editor=with_node_editor,
            with_node_editor_config=with_node_editor_config,
            with_tex_inspect=with_tex_inspect,
        )

        immapp.run(runner_params, add_ons_params=addons)

    @property
    def ui_scale(self):
        """UI scale getter"""
        return self._ui_scale

    @ui_scale.setter
    def ui_scale(self, value: float):
        """UI scale setter"""
        self.set_ui_scale(value)

    def set_ui_scale(self, scale: float):
        if self._ui_scale == scale:
            return  # setter might be called repeatedly e.g. via imgui.slider
        self._ui_scale = scale
        self.scale_style_sizes()

    def get_default_font(self):
        font = pathlib.Path(__file__).parent / 'MPLUSRounded1c-Medium.ttf'
        assert font.is_file(), f'Font file missing: "{font.resolve()}"'
        return str(font)

    # Includes keyboard (glfw.KEY_A) and mouse (glfw.MOUSE_BUTTON_LEFT)
    def keydown(self, key: typing.Union[int, str]):
        raise NotImplementedError()

    def keyhit(self, key: imgui.Key):
        return imgui.is_key_pressed(key, repeat=False)

    def scale_style_sizes(self):
        """More conservative alternative to imgui.get_style().scale_all_sizes()"""
        factor = self.ui_scale
        font_size = 9 * factor  # self._cur_font_size
        spacing = font_size * 0.3

        s = imgui.get_style()
        s.window_padding = [spacing, spacing]
        s.item_spacing = [spacing, spacing]
        s.item_inner_spacing = [spacing, spacing]
        s.columns_min_spacing = spacing
        s.indent_spacing = font_size
        s.scrollbar_size = font_size + 4

    def _draw_menu_wrapper(self, runner_params: hello_imgui.RunnerParams):
        if self.show_app_menu:
            hello_imgui.show_app_menu(runner_params)  # quit button
        if self.show_view_menu:
            hello_imgui.show_view_menu(runner_params)  # status bar show/hide, docking layout reset, etc.

        # User-provided
        self.draw_menu()

        # UI resize widget

        # Right-aligned button for locking / unlocking UI
        T = 'L' if self.ui_locked else 'U'
        C = [0.8, 0.0, 0.0] if self.ui_locked else [0.0, 1.0, 0.0]
        s = self.ui_scale

        # same_line() and negative sizes don't work within menu bar
        # => use invisible button instead
        # Tested: 16.3.2025, imgui_bundle==1.6.2
        max_x = imgui.get_window_width()
        cursor = imgui.get_cursor_pos()[0]

        # UI scale slider
        if not self.ui_locked:
            pad = max_x - cursor - 300 - 30*s
            imgui.invisible_button('##hidden', size=(pad, 1))
            max_scale = 5
            min_scale = 0.1

            imgui.set_next_item_width(300)  # size not dependent on s => prevents slider drift
            ch, val = imgui.slider_float('', s, min_scale, max_scale, format="%.1f")
            if imgui.is_item_hovered():
                if imgui.is_mouse_clicked(imgui.MouseButton_.right):
                    (ch, val) = (True, 1.0)

            if ch:
                self.set_ui_scale(val)
        else:
            pad = max_x - cursor - 25*s
            imgui.invisible_button('##hidden', size=(pad, 1))

        imgui.push_style_color(imgui.Col_.text, (*C, 1))
        if imgui.button(T, size=(20*s, 0)):
            self.ui_locked = not self.ui_locked
        imgui.pop_style_color()

    def pre_new_frame(self):
        """
        Called each frame before ImGui::NewFrame().
        Good place to add new dockable windows etc.
        """
        pass

    def setup_layout(self) -> hello_imgui.DockingParams:
        """
        Create initial dummy docking layout.
        Ignores @layout specifiers, creates horizontal split.
        Only used initially, subsequent runs will reload previous layout.
        """
        # Find all functions with @layout decorator
        # Use list of candidate names '_dockable_windows' to avoid calling getattr on properties before user state init
        # In case of multiple instances and name collisions: must check attribute _layout_pos as well
        fun_names, titles = zip(*ui_utils._dockable_windows)
        candidates = [getattr(self, name) for name in dir(self) if name in fun_names]

        candidates.append(self.point3d_window.point3d_window)  # ensure point3d_window is included

        layout_funcs = [f for f in candidates if hasattr(f, '_title')]
        assert len(titles) == len(set(titles)), 'Titles must be unique'

        # Creating layout from position tags seems non-trivial, ambiguous, and only affects first startup anyway.
        logger.info('Using dummy horizontal layout')

        # --------------------------------------------------------------------------------------------------------------------
        # N-1 splits

        splits, windows = self.setup_docking_layout(layout_funcs)

        return hello_imgui.DockingParams(splits, windows)

    def get_default_font_path(self):
        resource_dir = pathlib.Path(pyviewer.__file__).parent
        font = resource_dir / 'MPLUSRounded1c-Medium.ttf'
        assert font.is_file(), f'Font file missing: "{font.resolve()}"'
        return font.as_posix()

    def get_mono_font_path(self):
        resource_dir = pathlib.Path(pyviewer.__file__).parent
        font = resource_dir / 'Hack-Regular.ttf'
        assert font.is_file(), f'Font file missing: "{font.resolve()}"'
        return font.as_posix()

    def load_fonts(self):
        # BUNDLED:
        # Akronim-Regular.ttf, DroidSans.ttf, Font_Awesome_6_Free-Solid-900.otf, Inconsolata-Medium.ttf,
        # NotoEmoji-Regular.ttf, entypo.ttf, fontawesome-webfont.ttf, Playbox/Playbox-FREE.otf,
        # Roboto/Roboto-Bold.ttf, Roboto/Roboto-BoldItalic.ttf, Roboto/Roboto-Regular.ttf, Roboto/Roboto-RegularItalic.ttf

        # Load the main font
        size = self.initial_font_size * self.ui_scale
        external = hello_imgui.FontLoadingParams(inside_assets=False)
        self.default_font = hello_imgui.load_font(self.get_default_font_path(), size, external)
        self.fonts.append(self.default_font)

        # Merge with Font Awesome 6 (fontawesome.com/search?o=r&ic=free&s=solid&ip=classic)
        # Both font handles must be kept around for resizing
        if self.load_font_awesome:
            # TODO: if playing with UI scale slider: due to non-default range, will eventually trigger
            # github.com/pthom/hello_imgui/blob/c98503154f66/src/hello_imgui/impl/hello_imgui_font.cpp#L71
            font_loading_params_title_icons = hello_imgui.FontLoadingParams(merge_to_last_font=True, use_full_glyph_range=True)
            self.fonts.append(hello_imgui.load_font("fonts/Font_Awesome_6_Free-Solid-900.otf", size, font_loading_params_title_icons))

        self.code_font = hello_imgui.load_font(self.get_mono_font_path(), size, external)
        self.fonts.append(self.code_font)

    def get_window(self, name: str) -> hello_imgui.DockableWindow | None:
        windows = hello_imgui.get_runner_params().docking_params.dockable_windows
        for w in windows:
            if w.label == name:
                return w
        logger.info(f'No DockableWindow with label "{name}"')
        return None

    def toggle_window(self, name: str):
        w = self.get_window(name)
        if w is not None:
            w.is_visible = not w.is_visible

    def hide_window(self, name: str):
        w = self.get_window(name)
        if w is not None:
            w.is_visible = False

    def show_window(self, name: str):
        w = self.get_window(name)
        if w is not None:
            w.is_visible = True

    def toggle_menu(self):
        params = hello_imgui.get_runner_params().imgui_window_params
        params.show_menu_bar = not params.show_menu_bar

    def compute_loop(self):
        logger.info('Compute thread: waiting for start event')
        self.start_event.wait(timeout=None)
        logger.info('Compute thread: start event received')

        while not self.stop_event.is_set():
            ret = self.compute()

            if ret is not None:
                self.point3d_window.update_points(**ret)

            time.sleep(0.01)

        logger.info('Compute thread: received stop event, exiting')

    def reset_window_title(self):
        self._window_title = self._orig_window_title

    def set_window_title(self, title, suffix=False):
        """
        Set window title.
        If called from compute thread: updated later by UI thread.
        If called from UI thread: updated immediately.
        suffix: if True, append title as suffix to original.
        """
        if suffix:
            title = f'{self._orig_window_title} - {title}'
        self._window_title = title
        if threading.get_native_id() == self.ui_tid:
            # need glfw.make_context_current if calling from compute thread?
            glfw.set_window_title(self.window, title)

    #########################
    # User provided callbacks
    #########################

    def draw_status_bar(self):
        pass

    def draw_menu(self):
        pass

    def drag_and_drop_callback(self, paths: list[pathlib.Path]) -> bool:
        pass

    # Perform computation, returning single np/torch image, or None
    def compute(self) -> dict | None:
        pass

    # Program state init
    def setup_state(self):
        pass

    # Can be overridden
    def setup_theme(self):
        imgui_themes.theme_deep_dark()

    # Can be overridden
    def setup_docking_layout(self, layout_funcs: list[typing.Callable[[None], None]]) -> tuple[list[hello_imgui.DockingSplit], list[hello_imgui.DockableWindow]]:
        dock_space_names = ['MainDockSpace'] + [f'Dock{i}' for i in range(len(layout_funcs) - 1)]
        n_windows = len(layout_funcs)
        # NOTE: Should be splitted like: (N-1) / N, (N-2) / (N-1), (N-3) / (N-2), ..., 3 / 4, 2 / 3, 1 / 2
        splits = [hello_imgui.DockingSplit(prev, cur, imgui.Dir.right, ratio_=(n_windows - i_dock - 1) / (n_windows - i_dock)) for i_dock, (prev, cur) in enumerate(zip(dock_space_names[:-1], dock_space_names[1:]))]
        windows = [hello_imgui.DockableWindow(f._title, d, f, can_be_closed_=True) for f, d in zip(layout_funcs, dock_space_names)]
        return splits, windows

    def load_settings(self):
        """Load settings using `hello_imgui.load_user_pref(k: str)`"""
        pass

    def save_settings(self):
        """Save settings using `hello_imgui.save_user_pref(k: str, v: str)`"""
        pass

# ------------------------------------------------------------------------------------------------------------
