"""
point3d_window
==============

A component for displaying and interacting with 3D point clouds in a dedicated window using OpenGL and ImGui.

This source code is based on "pyviewer-extended" repository by Shinei Arakawa but has been heavily modified to enable drawing on textures. 

License
-------

Copyright (c) 2026 Shinei Arakawa
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

This code is a derivative work based on code from the harskish/pyviewer repository, whose original code is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0). The original code can be found at https://github.com/harskish/pyviewer.
"""


import ctypes
import enum
import math
import platform
import time
import types
import typing

import numpy as np
import OpenGL.GL as gl
import pyglm.glm as glm
import torch
import ui.cuda_gl_interop as cuda_gl_interop
import ui.gl_utils as gl_utils
import ui.utils as ui_utils
from imgui_bundle import imgui
from imgui_bundle.glfw_utils import glfw

# ------------------------------------------------------------------------------------------------------------
# Logger

logger = ui_utils.get_logger()

# --------------------------------------------------------------------------------------------------------------------------------

ArrayLike = typing.TypeVar('ArrayLike', np.ndarray, torch.Tensor)

# --------------------------------------------------------------------------------------------------------------------------------


class ArcBallMode(enum.Enum):
    NONE = enum.auto()
    ROTATE = enum.auto()
    TRANSLATE = enum.auto()
    SCALE = enum.auto()

# --------------------------------------------------------------------------------------------------------------------------------


class Point3DWindow:
    DOUBLE_CLICK_DURATION: typing.Final[float] = 0.3  # seconds

    def __init__(self, name: str, size: tuple[int, int]):
        self._name: str = name
        self._size: tuple[int, int] = size

        self._window: glfw._GLFWwindow | None = None  # Set by UI thread
        self._frame_buffer: gl_utils.FrameBuffer | None = None  # Lazy init

        self._clear_color = glm.vec4(0.0, 0.0, 0.0, 1.0)  # in [0, 1]

        self._output_pos_tl = glm.vec2(0.0, 0.0)  # Top-left corner of the output canvas, in absolute screen coordinates

        self._points: ArrayLike | None = None  # (N, 3) float32
        self._colors: ArrayLike | None = None  # (N, 3) float32
        self._point_count: int = 0
        self._point_dt: float = 0.0
        self._points_min: glm.vec3 = glm.vec3(0.0, 0.0, 0.0)
        self._points_max: glm.vec3 = glm.vec3(0.0, 0.0, 0.0)
        self._last_upload_ms: float = 0.0
        self._upload_time_ms: float = 0.1  # seconds

        self._point_shader: gl_utils.PointShader | None = None  # Lazy init

        self._vao: int | None = None  # Lazy init
        self._vbo: int | None = None  # Lazy init
        self._vao_size: int = 0  # Number of points in the VBO
        self._vbo_mapper: cuda_gl_interop.VBOMapper | None = None

        self._reset_camera_pose()

        self._point_size: int = 3

        self._is_dragging: bool = False
        self._arcball_mode: ArcBallMode = ArcBallMode.NONE

        self._mouse_clicked_time = time.monotonic()

        self._bbox: gl_utils.AxisAlignedBoundingBox = gl_utils.AxisAlignedBoundingBox()
        self._bbox_visible: bool = False
        self._bbox_color: glm.vec3 = glm.vec3(15/255, 230/255, 130/255)

        self._axes_cone: gl_utils.AxesCone = gl_utils.AxesCone()
        self._axes_cone_visible: bool = True

        self._rot_animation_enabled: bool = False
        self._rot_animation_speed: float = 0.8  # deg/frame

        # ------------------------------------------------------------------------------------------
        # Dynamically create a new independent copy of the window function

        orig_func = self.__class__.__dict__['point3d_window']
        copied_func = types.FunctionType(
            orig_func.__code__,
            orig_func.__globals__,
            name=orig_func.__name__,
            argdefs=orig_func.__defaults__,
            closure=orig_func.__closure__
        )
        dockable_window = ui_utils.dockable(copied_func, title=name)

        # Bind the copied function to this instance
        setattr(self, 'point3d_window', dockable_window.__get__(self, self.__class__))

    # ------------------------------------------------------------------------------------------
    # Properties

    @property
    def point_size(self) -> int:
        return self._point_size

    @point_size.setter
    def point_size(self, size: int) -> None:
        self._point_size = size

    @property
    def background_color(self) -> tuple[float, float, float]:
        return (self._clear_color.r, self._clear_color.g, self._clear_color.b)

    @background_color.setter
    def background_color(self, color: glm.vec3 | tuple[float, float, float] | list[float]) -> None:
        if not isinstance(color, glm.vec3):
            color = glm.vec3(*color)
        self._clear_color = glm.vec4(color.r, color.g, color.b, self._clear_color.a)

    @property
    def bbox_visible(self) -> bool:
        return self._bbox_visible

    @bbox_visible.setter
    def bbox_visible(self, visible: bool) -> None:
        self._bbox_visible = visible

    @property
    def bbox_color(self) -> glm.vec3:
        return self._bbox_color

    @bbox_color.setter
    def bbox_color(self, color: glm.vec3 | tuple[float, float, float] | list[float]) -> None:
        if not isinstance(color, glm.vec3):
            color = glm.vec3(*color)
        self._bbox_color = color

    @property
    def axes_cone_visible(self) -> bool:
        return self._axes_cone_visible

    @axes_cone_visible.setter
    def axes_cone_visible(self, visible: bool) -> None:
        self._axes_cone_visible = visible

    @property
    def rot_animation_enabled(self) -> bool:
        return self._rot_animation_enabled

    @rot_animation_enabled.setter
    def rot_animation_enabled(self, enabled: bool) -> None:
        self._rot_animation_enabled = enabled

    @property
    def rot_animation_speed(self) -> float:
        return self._rot_animation_speed

    @rot_animation_speed.setter
    def rot_animation_speed(self, speed: float) -> None:
        self._rot_animation_speed = speed

    # ------------------------------------------------------------------------------------------
    # Callbacks for UI thread

    def _resize_callback(self, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            return  # minimized window

        self._size = (width, height)

        # Update projection matrix
        self._proj_mat = glm.perspective(
            45.0,                                                          # fov
            self._size[0] / self._size[1] if self._size[1] != 0 else 1.0,  # aspect
            0.1,                                                           # near
            1000.0                                                         # far
        )

        # HDPI: affects framebuffer scale
        hdpi_factor = self._get_hdpi_scale()[0]
        fb_width = int(width * hdpi_factor)
        fb_height = int(height * hdpi_factor)

        # Resize FBO
        if self._frame_buffer is not None:
            self._frame_buffer.rescale(fb_width, fb_height)

    def _mouse_wheel_callback(self, window: glfw._GLFWwindow, x: float, y: float) -> None:
        if not self._mouse_hovers_content():
            return

        tex_w, tex_h = self._frame_buffer._width, self._frame_buffer._height
        canvas_w, canvas_h = self._size

        # MacOS trackpad needs separate speed
        speed = 0.035 if platform.system() == 'Darwin' else 0.15
        scale_fill_h = max(1, (tex_w / tex_h) * (canvas_h / canvas_w))
        scale_max = 0.5 * scale_fill_h * tex_h  # canvas height convers 2 pixels
        scale_min = 50 / min(canvas_h, canvas_w)  # canvas ~50x50 pixels
        self._arc_scale = min(scale_max, max(scale_min, self._arc_scale * (1.0 + y * speed)))

        self._update_arc_scale_mat()

    def _mouse_button_callback(self, window: glfw._GLFWwindow, button: int, action: int, mods: int) -> None:
        if self._mouse_hovers_content() and action == glfw.PRESS:
            # Detect double-click
            cur_time = time.monotonic()
            is_double_clicked = (cur_time - self._mouse_clicked_time) < self.DOUBLE_CLICK_DURATION
            self._mouse_clicked_time = cur_time

            if is_double_clicked:
                self._reset_camera_pose()
                return

            # Handle click actions
            if button == glfw.MOUSE_BUTTON_LEFT:
                self._arcball_mode = ArcBallMode.ROTATE
            elif button == glfw.MOUSE_BUTTON_MIDDLE:
                self._arcball_mode = ArcBallMode.SCALE
            elif button == glfw.MOUSE_BUTTON_RIGHT:
                self._arcball_mode = ArcBallMode.TRANSLATE

            if self._arcball_mode != ArcBallMode.NONE and not self._is_dragging:
                self._is_dragging = True
                self._last_mouse_pos = self._mouse_pos_canvas_norm
        else:
            self._arcball_mode = ArcBallMode.NONE
            self._is_dragging = False

    def _cursor_pos_callback(self, window: glfw._GLFWwindow, xpos: float, ypos: float) -> None:
        if not self._is_dragging:
            return

        new_mouse_pos = self._mouse_pos_canvas_norm

        delta = new_mouse_pos - self._last_mouse_pos
        length = delta.x * delta.x + delta.y * delta.y
        pix_width = min(1.0 / self._size[0], 1.0 / self._size[1])
        if length < pix_width * pix_width:
            return  # ignore small movement

        if self._arcball_mode == ArcBallMode.ROTATE:
            self._update_arc_rot_mat(new_mouse_pos, self._last_mouse_pos)
        elif self._arcball_mode == ArcBallMode.TRANSLATE:
            self._update_arc_trans_mat(new_mouse_pos, self._last_mouse_pos)
        elif self._arcball_mode == ArcBallMode.SCALE:
            tex_w, tex_h = self._frame_buffer._width, self._frame_buffer._height
            canvas_w, canvas_h = self._size
            scale_fill_h = max(1, (tex_w / tex_h) * (canvas_h / canvas_w))
            scale_max = 0.5 * scale_fill_h * tex_h  # canvas height convers 2 pixels
            scale_min = 50 / min(canvas_h, canvas_w)  # canvas ~50x50 pixels
            self._arc_scale = min(scale_max, max(scale_min, self._arc_scale * (1.0 + 10.0 * delta.y)))
            self._update_arc_scale_mat()

        self._last_mouse_pos = new_mouse_pos

    # ------------------------------------------------------------------------------------------
    # Internal methods

    def _reset_camera_pose(self) -> None:
        """Reset camera to default position."""

        self._camera_pos = glm.vec3(5.0, 0.0, 0.0)
        self._camera_lookat = glm.vec3(0.0, 0.0, 0.0)  # look at origin
        self._camera_up = glm.vec3(0.0, -1.0, 0.0)     # y-axis points down in screen space

        self._proj_mat = self._perspective(
            45.0,                                                          # fov
            self._size[0] / self._size[1] if self._size[1] != 0 else 1.0,  # aspect
            0.1,                                                           # near
            1000.0                                                         # far
        )

        self._arc_rot_mat = glm.mat4(1.0)
        self._arc_trans_mat = glm.mat4(1.0)
        self._arc_scale_mat = glm.mat4(1.0)
        self._arc_scale = 1.0

        self._view_mat = glm.lookAt(self._camera_pos, self._camera_lookat, self._camera_up)

        self._light_trans_mat = glm.mat4(1.0)
        self._light_pos = glm.vec3(10.0, 10.0, 10.0)

    def _rotate_model(self, delta_angle_deg: float, axis_in_world: glm.vec3) -> None:
        """Rotate the model by delta_angle_deg around the given axis (in world space).

        Parameters
        ----------
        delta_angle_deg : float
            The angle to rotate in degrees.
        axis_in_world : glm.vec3
            The axis to rotate around, in world space.
        """

        axis_in_world = glm.normalize(axis_in_world)
        quad = glm.angleAxis(glm.radians(delta_angle_deg), axis_in_world)
        rot_mat = glm.mat4_cast(quad)
        self._arc_rot_mat = rot_mat * self._arc_rot_mat

    def _update_arc_trans_mat(self, new_pos: glm.vec3, old_pos: glm.vec3) -> None:
        """Update the arcball translation matrix based on mouse movement.

        Parameters
        ----------
        new_pos : glm.vec2
            The new mouse position.
        old_pos : glm.vec2
            The old mouse position.
        """

        origin_on_screen = self._proj_mat * self._view_mat * glm.vec4(0.0, 0.0, 0.0, 1.0)
        origin_on_screen = origin_on_screen / origin_on_screen.w

        new_pos_on_screen = glm.vec4(
            2.0 * new_pos.x - 1.0,
            1.0 - 2.0 * new_pos.y,
            origin_on_screen.z,
            1.0
        )
        old_pos_on_screen = glm.vec4(
            2.0 * old_pos.x - 1.0,
            1.0 - 2.0 * old_pos.y,
            origin_on_screen.z,
            1.0
        )

        inv_mvp_mat = glm.inverse(self._proj_mat * self._view_mat)

        new_pos_in_obj_space = inv_mvp_mat * new_pos_on_screen
        new_pos_in_obj_space /= new_pos_in_obj_space.w
        old_pos_in_obj_space = inv_mvp_mat * old_pos_on_screen
        old_pos_in_obj_space /= old_pos_in_obj_space.w

        trans_world = glm.vec3(new_pos_in_obj_space - old_pos_in_obj_space)

        self._arc_trans_mat = glm.translate(trans_world) * self._arc_trans_mat

    def _update_arc_rot_mat(self, new_pos: glm.vec2, old_pos: glm.vec2) -> None:
        """Update the arcball rotation matrix based on mouse movement.

        Parameters
        ----------
        new_pos : glm.vec2
            The new mouse position.
        old_pos : glm.vec2
            The old mouse position.
        """

        new_pos = 2.0 * new_pos - 1.0
        old_pos = 2.0 * old_pos - 1.0

        c2w_max = glm.inverse(self._view_mat)

        dx = new_pos.x - old_pos.x
        dy = new_pos.y - old_pos.y

        horizontal_axis_in_world = glm.vec3(c2w_max[1])
        vertical_axis_in_world = glm.vec3(c2w_max[0])

        horizontal_angle = 2.0 * math.pi * dx
        vertical_angle = 2.0 * math.pi * dy

        horizontal_quad = glm.angleAxis(horizontal_angle, glm.normalize(horizontal_axis_in_world))
        vertical_quad = glm.angleAxis(vertical_angle, glm.normalize(vertical_axis_in_world))

        rot_mat = glm.mat4_cast(horizontal_quad * vertical_quad)

        self._arc_rot_mat = rot_mat * self._arc_rot_mat

    def _update_arc_scale_mat(self) -> None:
        """Update the arcball scale matrix based on the current scale factor."""

        self._arc_scale_mat = glm.scale(glm.vec3(self._arc_scale, self._arc_scale, self._arc_scale))

    def _perspective(self, fov_deg: float, aspect: float, near: float, far: float) -> glm.mat4:
        """Create a perspective projection matrix. This method creates the matrix based on the viewport size.

        Parameters
        ----------
        fov_deg : float
            Field of view in degrees.
        aspect : float
            Aspect ratio (width / height).
        near : float
            Near clipping plane.
        far : float
            Far clipping plane.

        Returns
        -------
        glm.mat4
            The perspective projection matrix.
        """

        angle = glm.radians(fov_deg)
        width, height = None, None
        if aspect >= 1.0:
            height = 2.0 * near * math.tan(angle / 2.0)
            width = height * aspect
        else:
            width = 2.0 * near * math.tan(angle / 2.0)
            height = width / aspect

        depth = far - near

        return glm.mat4(
            # autopep8: off
            2.0 * near / width,                       0.0,                                   0.0,   0.0,
                           0.0,       2.0 * near / height,                                   0.0,   0.0,
                           0.0,                       0.0,                 -(far + near) / depth,  -1.0,
                           0.0,                       0.0,             -2.0 * far * near / depth,   0.0,
            # autopep8: on
        )

    @property
    def _mouse_pos_abs(self) -> glm.vec2:
        return glm.vec2(*imgui.get_mouse_pos())

    @property
    def _mouse_pos_canvas_norm(self) -> glm.vec2:
        """
        Normalized mouse position on canvas, in [0, 1] relative to top-left.
        Does not consider current transformation.
        """
        if self._size[0] <= 0 or self._size[1] <= 0:
            return glm.vec2(-1, -1)  # no valid content
        pos = (self._mouse_pos_abs - self._output_pos_tl) / glm.vec2(*self._size)
        pos.y = 1.0 - pos.y  # flip y
        return pos

    def _mouse_hovers_content(self) -> bool:
        pos = self._mouse_pos_canvas_norm
        return (0 <= pos.x <= 1) and (0 <= pos.y <= 1)

    def _get_monitor(self):
        """Figure out which monitor window is currently on"""
        x, y = glfw.get_window_pos(self._window)
        for mon in glfw.get_monitors():
            x0, y0, w, h = glfw.get_monitor_workarea(mon)
            x1, y1 = (x0 + w, y0 + h)
            if (x0 <= x <= x1) and (y0 <= y <= y1):
                return mon
        return glfw.get_primary_monitor()

    def _get_hdpi_scale(self) -> tuple[float]:
        """
        Get HDPI framebuffer scale factor.
        OS scale returned forbuilt-in display on MacBook (otherwise output looks blurry).
        For other platforms, the native scale is used.
        """
        if platform.system() == 'Darwin':
            monitor = self._get_monitor()
            monitor_name = glfw.get_monitor_name(monitor).decode()
            if monitor_name == 'Built-in Retina Display':
                return glfw.get_monitor_content_scale(monitor)

        return (1.0, 1.0)

    def _allocate_vbo(self, num_points: int, enable_cuda_gl_interop: bool = False) -> None:
        """Allocate VBO for the given number of points.

        Parameters
        ----------
        num_points : int
            Number of points to allocate space for.
        enable_cuda_gl_interop : bool
            Whether to enable CUDA-OpenGL interop for uploading point data.
        """

        if self._vbo_mapper is not None:
            try:
                self._vbo_mapper.unregister()
            except Exception as exc:
                logger.warning(f'Failed to unregister CUDA VBO resource: {exc}')
            finally:
                self._vbo_mapper = None

        if self._vao is not None:
            gl.glDeleteVertexArrays(1, [self._vao])
            self._vao = None
        if self._vbo is not None:
            gl.glDeleteBuffers(1, [self._vbo])
            self._vbo = None

        self._vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self._vao)

        self._vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo)

        mem_pitch = 6 * 4  # 3 pos + 3 color, float32
        gl.glBufferData(gl.GL_ARRAY_BUFFER, num_points * mem_pitch, None, gl.GL_STATIC_DRAW)  # float32

        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, mem_pitch, ctypes.c_void_p(0))  # position

        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, mem_pitch, ctypes.c_void_p(3 * 4))  # color

        # No index buffer because we use glDrawArrays()

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindVertexArray(0)

        self._vao_size = num_points

        if enable_cuda_gl_interop:
            plugin = cuda_gl_interop.get_cuda_plugin()
            if plugin is None:
                self._vbo_mapper = None
                return

            try:
                self._vbo_mapper = cuda_gl_interop.VBOMapper(self._vbo)
            except Exception as exc:
                # Plugin might not be available; fall back to CPU uploads.
                logger.warning(f'CUDA VBO interop is unavailable: {exc}')
                self._vbo_mapper = None
        else:
            self._vbo_mapper = None

    def _upload_numpy_points(self, points: np.ndarray, colors: np.ndarray) -> None:
        """Upload points using numpy arrays.

        Parameters
        ----------
        points : np.ndarray
            A numpy array of shape (N, 3) containing the point coordinates.
        colors : np.ndarray
            A numpy array of shape (N, 3) containing the point colors.
        """

        assert points.ndim == 2 and points.shape[1] == 3, 'points must be of shape (N, 3)'
        assert points.dtype == np.float32, 'points must be float32 type'
        assert points.flags['C_CONTIGUOUS'], 'points must be a contiguous array'

        assert colors.ndim == 2 and colors.shape[1] == 3, 'colors must be of shape (N, 3)'
        assert colors.dtype == np.float32, 'colors must be float32 type'
        assert colors.flags['C_CONTIGUOUS'], 'colors must be a contiguous array'

        # Allocate VBO if not already done
        num_points = points.shape[0]
        if self._vao is None or self._vbo is None or num_points != self._vao_size:
            logger.info(f'Allocating VBO for {num_points} points')

            # Allocate new VAO/VBO
            self._allocate_vbo(num_points)

        # Create buffer data
        buffer_data = np.hstack((points, colors)).flatten()  # (6 * N,) float32

        # Upload to GPU
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo)
        gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, buffer_data.nbytes, buffer_data)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

    def _upload_torch_points(self, points: torch.Tensor, colors: torch.Tensor) -> bool:
        """Upload points using CUDA-OpenGL interop.

        Parameters
        ----------
        points : torch.Tensor
            A tensor of shape (N, 3) containing the point coordinates.
        colors : torch.Tensor
            A tensor of shape (N, 3) containing the point colors.

        Returns
        -------
        bool
            True if the upload was successful, False otherwise.
        """

        assert points.dim() == 2 and points.size(1) == 3, 'points must be of shape (N, 3)'
        assert points.dtype == torch.float32, 'points must be float32 type'
        assert points.device.type == 'cuda', 'points must be a CUDA tensor'

        assert colors.dim() == 2 and colors.size(1) == 3, 'colors must be of shape (N, 3)'
        assert colors.dtype == torch.float32, 'colors must be float32 type'
        assert colors.device.type == 'cuda', 'colors must be a CUDA tensor'

        # Make sure CUDA-OpenGL interop plugin is available
        if cuda_gl_interop.get_cuda_plugin() is None:
            return False

        # Allocate VBO if not already done
        num_points = points.size(0)
        if self._vao is None or self._vbo is None or num_points != self._vao_size:
            logger.info(f'Allocating CUDA VBO for {num_points} points')
            self._allocate_vbo(num_points, enable_cuda_gl_interop=True)

        # Create CUDA-OpenGL interop mapper if not already done
        if self._vbo_mapper is None:
            # We may have failed to create mapper earlier (e.g., plugin unavailable); try once more.
            try:
                self._vbo_mapper = cuda_gl_interop.VBOMapper(self._vbo)
            except Exception as exc:
                logger.error(f'Failed to set up CUDA interop for VBO: {exc}')
                self._vbo_mapper = None
                return False

        try:
            # Create contiguous buffer
            buffer_data = torch.hstack((points, colors)).flatten()  # (6 * N,) float32

            # Upload to GPU via CUDA-OpenGL interop
            self._vbo_mapper.upload(buffer_data)
        except Exception as exc:
            logger.error(f'CUDA VBO upload failed: {exc}')
            return False

        return True

    def _upload_points(self):
        """Upload point cloud data to GPU if it has been set."""

        if self._points is None:
            return

        if self._point_dt > self._last_upload_ms:
            cur_time = time.monotonic()

            # Allocate colors if not already done
            if self._colors is None or self._colors.shape[0] != self._points.shape[0]:
                if isinstance(self._points, np.ndarray):
                    self._colors = np.ones((self._points.shape[0], 3), dtype=np.float32)
                elif isinstance(self._points, torch.Tensor):
                    self._colors = torch.ones((self._points.size(0), 3), dtype=torch.float32, device=self._points.device)
                else:
                    raise TypeError('points must be either a numpy array or a torch tensor')

            assert type(self._colors) == type(self._points), 'points and colors must be of the same type'

            # Upload point coords to GPU
            if isinstance(self._points, np.ndarray):
                self._upload_numpy_points(self._points, self._colors)
            elif isinstance(self._points, torch.Tensor):
                is_succeed = False
                if self._points.device.type == 'cuda':
                    # Fall back to numpy if upload via CUDA interop fails
                    is_succeed = self._upload_torch_points(self._points, self._colors)

                if not is_succeed:
                    self._upload_numpy_points(self._points.cpu().numpy(), self._colors.cpu().numpy())
            else:
                raise TypeError('points must be either a numpy array or a torch tensor')

            # Update bounding box
            self._bbox.update_geometry(self._points_min, self._points_max)

            self._last_upload_ms = time.monotonic()
            self._upload_time_ms = (self._last_upload_ms - cur_time) * 1000.0  # in ms

    def _draw_to_fbo(self):
        """Draw the scene to the FBO."""

        if self._frame_buffer is None or self._vao is None:
            return

        if self._point_shader is None:
            # Compile shader on first use
            logger.info('Compiling point shader...')
            self._point_shader = gl_utils.PointShader()
            logger.info('done.')

        # Animate rotation
        if self._rot_animation_enabled and self._arcball_mode == ArcBallMode.NONE:
            self._rotate_model(self._rot_animation_speed, self._camera_up)

        model_mat = self._arc_trans_mat * self._arc_rot_mat * self._arc_scale_mat
        mv_mat = self._view_mat * model_mat
        mvp_mat = self._proj_mat * mv_mat

        # Render points
        with self._point_shader.ctx(
            gl_utils.PointShaderRenderingContext(
                mvp_mat=mvp_mat,
                point_size=float(self._point_size),
            )
        ):
            gl.glBindVertexArray(self._vao)
            gl.glDrawArrays(gl.GL_POINTS, 0, self._point_count)
            gl.glBindVertexArray(0)

        # Render bounding box
        if self._bbox_visible or self._is_dragging:
            self._bbox.draw(
                mvp_mat=mvp_mat,
                color=self._bbox_color,
            )

        # Render axes cone
        if self._axes_cone_visible:
            self._axes_cone.draw(
                rot_mat=self._arc_rot_mat,
                view_mat=self._view_mat,
                proj_mat=self._proj_mat,
                light_trans_mat=self._light_trans_mat,
                light_pos=self._light_pos,
            )

    def _draw_to_canvas(self, canvas_size: tuple[int, int]):
        """Draw the FBO texture to the ImGui canvas."""

        if self._frame_buffer is None:
            logger.info('Frame buffer is not initialized yet.')

        if self._size != canvas_size:
            self._resize_callback(canvas_size[0], canvas_size[1])

        # Keep track of content position
        self._output_pos_tl = glm.vec2(*imgui.get_cursor_screen_pos())  # (wmin.x + rmin.x, wmin.y + rmin.y)

        # Draw the FBO texture to the canvas
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)
        gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)

        with self._frame_buffer.ctx():
            gl.glViewport(0, 0, self._frame_buffer._width, self._frame_buffer._height)
            gl.glClearColor(*self._clear_color)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

            # Draw to the FBO
            self._draw_to_fbo()

    # ------------------------------------------------------------------------------------------
    # Window UI func

    def point3d_window(self):
        """ImGui window function to display the 3D point cloud."""

        # First upload arrays if it has been set
        self._upload_points()

        cW, cH = map(int, imgui.get_content_region_avail())

        # Draw the FBO texture to the canvas
        self._draw_to_canvas(canvas_size=(cW, cH))

        # Begin the ImGui window
        imgui.begin(self._name, True, imgui.WindowFlags_.no_collapse)

        if self._frame_buffer is not None:
            imgui.image(self._frame_buffer.texture_ref, (cW, cH))

        imgui.end()

    # ------------------------------------------------------------------------------------------
    # Public API

    def initialize(self) -> None:
        """Generate the OpenGL texture and framebuffer.
        """

        if self._frame_buffer is None:
            self._frame_buffer = gl_utils.FrameBuffer(self._size[0], self._size[1])

    def dispose(self) -> None:
        """Gracefully dispose all OpenGL resources.
        """

        if self._frame_buffer is not None:
            self._frame_buffer.dispose()
            self._frame_buffer = None

        if self._vbo_mapper is not None:
            try:
                self._vbo_mapper.unregister()
            except Exception as exc:
                logger.warning(f'Failed to unregister CUDA VBO resource during dispose: {exc}')
            finally:
                self._vbo_mapper = None

        if self._vbo is not None:
            gl.glDeleteBuffers(1, [self._vbo])
            self._vbo = None

        if self._vao is not None:
            gl.glDeleteVertexArrays(1, [self._vao])
            self._vao = None
            self._vao_size = 0

        # Dispose shader programs and buffers

        if self._point_shader is not None:
            self._point_shader.dispose()
            self._point_shader = None

        if self._bbox is not None:
            self._bbox.dispose()
            self._bbox = None

        if self._axes_cone is not None:
            self._axes_cone.dispose()
            self._axes_cone = None

    def set_callbacks(self, glfw_window: glfw._GLFWwindow) -> None:
        """Set GLFW callbacks for all texture windows.

        Parameters
        ----------
        glfw_window: glfw._GLFWwindow
            GLFW window to set callbacks for .
        """

        self._window = glfw_window
        glfw.set_scroll_callback(glfw_window, self._mouse_wheel_callback)
        glfw.set_mouse_button_callback(glfw_window, self._mouse_button_callback)
        glfw.set_cursor_pos_callback(glfw_window, self._cursor_pos_callback)

    def set_theme(self) -> None:
        """Set theme for window based on current ImGui style."""

        bg_color = imgui.get_style().color_(imgui.Col_.window_bg)
        self._clear_color = glm.vec4(bg_color.x, bg_color.y, bg_color.z, 1.0)

    def update_points(self, points: ArrayLike | None, colors: ArrayLike | None = None) -> None:
        """Update the points to be displayed in the window.

        Parameters
        ----------
        points : ArrayLike | None
            A (N, 3) float32 array or tensor representing the 3D points. If None, no update is performed.
        colors : ArrayLike | None
            A (M, 3) float32 array or tensor representing the RGB colors for each point. If None, the current color is used.

        Raises
        ------
        TypeError
            If the input is not a valid array or tensor.
        """

        if points is None:
            return

        # Type checking
        if isinstance(points, torch.Tensor):
            assert points.dtype == torch.float32, 'points must be float32 type'
            assert points.dim() == 2 and points.size(1) == 3, 'points must be of shape (N, 3)'
        elif isinstance(points, np.ndarray):
            assert points.dtype == np.float32, 'points must be float32 type'
            assert points.ndim == 2 and points.shape[1] == 3, 'points must be of shape (N, 3)'
        else:
            raise TypeError('points must be either a numpy array or a torch tensor')

        if colors is not None:
            if isinstance(colors, torch.Tensor):
                assert colors.dtype == torch.float32, 'colors must be float32 type'
                assert colors.dim() == 2 and colors.size(1) == 3, 'colors must be of shape (M, 3)'
                assert colors.size(0) == points.shape[0], 'colors must have the same number of rows as points'
            elif isinstance(colors, np.ndarray):
                assert colors.dtype == np.float32, 'colors must be float32 type'
                assert colors.ndim == 2 and colors.shape[1] == 3, 'colors must be of shape (M, 3)'
                assert colors.shape[0] == points.shape[0], 'colors must have the same number of rows as points'
            else:
                raise TypeError('colors must be either a numpy array or a torch tensor')

            self._colors = colors

        # Compute bounding box
        if isinstance(points, np.ndarray):
            points_min = points.min(axis=0)
            points_max = points.max(axis=0)
        else:  # torch.Tensor
            points_min = points.min(dim=0).values.cpu().numpy()
            points_max = points.max(dim=0).values.cpu().numpy()
        self._points_min = glm.vec3(points_min[0], points_min[1], points_min[2])
        self._points_max = glm.vec3(points_max[0], points_max[1], points_max[2])

        self._points = points
        self._colors = colors
        self._point_count = points.shape[0]
        self._point_dt = time.monotonic()

# --------------------------------------------------------------------------------------------------------------------------------
