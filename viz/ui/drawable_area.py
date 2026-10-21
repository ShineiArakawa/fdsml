"""
drawable_area
=============

This module provides a drawable canvas class that extends the pannable area functionality.

License
-------

Copyright (c) 2026 Shinei Arakawa
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

This code is a derivative work based on code from the harskish/pyviewer repository, whose original code is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0). The original code can be found at https://github.com/harskish/pyviewer.
"""

import platform
import typing

import numpy as np
import pyviewer.utils as utils
import typing_extensions
import ui.utils as ui_utils
from imgui_bundle import imgui  # type: ignore

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# Helpers


def is_inside(coord: tuple[float, float]) -> bool:
    return coord[0] >= 0 and coord[1] >= 0 and coord[0] <= 1 and coord[1] <= 1

# ImGui widget that wraps an image and allows mouse pand & zoom controls.
# Does not transform texture coordinates; instead transforms a single textured quad.
# Panning produces no temporal aliasing: the pan follows the mouse, which moves in one pixel increments.


# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# DrawableCanvas class


class DrawableArea(utils.PannableArea):
    """A drawable canvas that extends the pannable area functionality.
    Drawing mode is activated when the `shift` key is pressed.
    """

    DRAWING_KEY: typing.Final[imgui.Key] = imgui.Key.left_shift

    def __init__(self, *args, drawing_enabled: bool = True, **kwargs) -> None:
        self._is_drawing = False
        self._drawing_enabled = drawing_enabled
        self._last_mouse_uv = None
        self._drawing_mouse_callback: typing.Callable[[tuple[float, float], tuple[float, float]], None] | None = None
        self._drawing_wheel_callback: typing.Callable[[float], None] | None = None
        super().__init__(*args, **kwargs)

    def set_drawing_mouse_callback(self, callback: typing.Callable[[tuple[float, float], tuple[float, float]], None]) -> None:
        """Set the drawing callback function for mouse movements.
        The callback function should take two arguments: start_uv and end_uv, which are the UV coordinates of the start and end points of the drawing.
        """
        self._drawing_mouse_callback = callback

    def set_drawing_wheel_callback(self, callback: typing.Callable[[float], None]) -> None:
        """Set the drawing callback function for wheel movements.
        """
        self._drawing_wheel_callback = callback

    def is_drawing_mode(self) -> bool:
        return self._drawing_enabled and imgui.is_key_down(self.DRAWING_KEY)

    @typing_extensions.override
    def handle_pan(self):
        if self.is_drawing_mode():
            # Drawing mode
            if imgui.is_mouse_down(imgui.MouseButton_.left):
                self._is_drawing = True
                mouse_uv = self.get_hovered_uv_image()
                if is_inside(mouse_uv) and (self._last_mouse_uv is not None) and is_inside(self._last_mouse_uv) and (self._drawing_mouse_callback is not None):
                    # Draw
                    with ui_utils.lock:
                        self._drawing_mouse_callback(self._last_mouse_uv, mouse_uv)
                self._last_mouse_uv = mouse_uv
            else:
                self._is_drawing = False
                self._last_mouse_uv = None
        else:
            # Pan mode
            self._is_drawing = False
            super().handle_pan()

    @typing_extensions.override
    def mouse_wheel_callback(self, window, x, y) -> None:
        if not self.force_mouse and imgui.get_io().want_capture_mouse:
            return self.prev_scroll_cbk(window, x, y)  # scroll imgui lists etc.

        if not (self.mouse_hovers_content() and self.zoom_enabled):
            return self.prev_scroll_cbk(window, x, y)

        if self.is_drawing_mode():
            if self._drawing_wheel_callback:
                self._drawing_wheel_callback(y)
        else:
            speed = 0.035 if platform.system() == 'Darwin' else 0.15
            # MacOS trackpad needs separate speed
            scale_fill_h = max(1, (self.tex_w / self.tex_h) * (self.canvas_h / self.canvas_w))
            zoom_max = 0.5 * scale_fill_h * self.tex_h  # canvas height convers 2 pixels
            zoom_min = 50 / min(self.canvas_h, self.canvas_w)  # canvas ~50x50 pixels
            new_zoom = min(zoom_max, max(zoom_min, self.zoom * (1.0 + y * speed)))

            # Zoom relative to mouse cursor
            # => keep same UV under cursor after zooming
            center_uv = 0.5 * sum(self.get_visible_box_canvas())
            mouse_pos = np.array(self.get_hovered_uv_canvas())
            dt = mouse_pos - center_uv  # center to mouse cursor
            dm = dt * ((self.zoom / new_zoom) - 1)  # movement of cursor UV due to zoom
            self.pan = np.array(self.pan) + dm * np.array([1, -1])  # to y up
            self.zoom = new_zoom


DrawableArea.__init__.__doc__ = utils.PannableArea.__doc__

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
