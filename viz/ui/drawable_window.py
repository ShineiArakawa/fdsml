"""
drawable_window
===============

Module for creating dockable windows that display textures with optional masking capabilities.
Each window can display an image texture, and if masking is enabled, users can draw on the image to create a mask.
The mask can be used to hide or show parts of the image.

License
-------

Copyright (c) 2026 Shinei Arakawa
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

This code is a derivative work based on code from the harskish/pyviewer repository, whose original code is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0). The original code can be found at https://github.com/harskish/pyviewer.
"""

import math
import time
import types

import numpy as np
import OpenGL.GL as gl
import pyviewer.utils as utils
import torch
import ui.cuda_gl_interop as cuda_gl_interop
import ui.drawable_area as drawable_area
import ui.utils as ui_utils
from imgui_bundle import imgui  # type: ignore
from imgui_bundle.glfw_utils import \
    glfw  # NOTE: Import GLFW from imgui_bundle.glfw_utils to avoid multile definitions

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# Logger

logger = ui_utils.get_logger()


# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# Enums

class MaskMode(ui_utils.StrEnum):
    # initially hide everything (True), draw to show (False)
    draw_to_show = 'draw_to_show'
    # initially show everything (False), draw to hide (True)
    draw_to_hide = 'draw_to_hide'

    def canvas_value(self) -> bool:
        return self == MaskMode.draw_to_show

    def brush_value(self) -> bool:
        return self == MaskMode.draw_to_hide


class BrushType(ui_utils.StrEnum):
    square = 'square'
    circle = 'circle'


class PassValueType(ui_utils.StrEnum):
    constant = 'Constant'
    raw = 'Raw'


class HighlightMode(ui_utils.StrEnum):
    off = 'Off'
    entire_image = 'Entire Image'
    grid_cells = 'Grid Cells'

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# CUDA-GL interop


def cuda_synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class PTExtMapper:
    tex: int
    resource: int  # uint64_t

    def __init__(self, tex: int) -> None:
        self.tex = tex
        self.resource = cuda_gl_interop.get_cuda_plugin().register_texture(tex)

    def unregister(self) -> None:
        cuda_gl_interop.get_cuda_plugin().unregister_texture(self.resource)

    def upload(self, ptr: int, W: int, H: int, N: int) -> None:
        cuda_gl_interop.get_cuda_plugin().upload_texture(ptr, W, H, N, self.resource)  # map, copy, unmap

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# Texture unit


MIPMAP_MODES = [gl.GL_NEAREST_MIPMAP_NEAREST, gl.GL_LINEAR_MIPMAP_NEAREST, gl.GL_NEAREST_MIPMAP_LINEAR, gl.GL_LINEAR_MIPMAP_LINEAR]


class _texture:
    '''
    This class maps torch tensors to gl textures without a CPU roundtrip.
    '''

    def __init__(self, min_filter=gl.GL_LINEAR, mag_filter=gl.GL_LINEAR):
        # Can be shared between py and c++
        self.tex_2d = gl.glGenTextures(1)
        self.tex = self.tex_2d  # currently active
        self.type = gl.GL_TEXTURE_2D
        gl.glBindTexture(self.type, self.tex)  # need to bind to modify
        # sets repeat and filtering parameters; change the second value of any tuple to change the value
        for params in ((gl.GL_TEXTURE_WRAP_S, gl.GL_MIRRORED_REPEAT), (gl.GL_TEXTURE_WRAP_T, gl.GL_MIRRORED_REPEAT), (gl.GL_TEXTURE_MIN_FILTER, min_filter), (gl.GL_TEXTURE_MAG_FILTER, mag_filter)):
            gl.glTexParameteri(self.type, *params)
        self.min_filter = min_filter
        self.mag_filter = mag_filter
        self.mapper = None  # torch extension cudaGraphicsResource_t
        self.is_fp = True
        self.shape = [0, 0]

    # be sure to del textures if you create a forget them often (python doesn't necessarily call del on garbage collect)
    def __del__(self):
        if gl is not None:
            gl.glDeleteTextures(1, [self.tex])

    @property
    def needs_mipmap(self):
        return self.min_filter in MIPMAP_MODES or self.mag_filter in MIPMAP_MODES

    def set_interp(self, key, val):
        gl.glBindTexture(self.type, self.tex)
        gl.glTexParameteri(self.type, key, val)
        gl.glBindTexture(self.type, 0)
        if key == gl.GL_TEXTURE_MIN_FILTER:
            self.min_filter = val
        if key == gl.GL_TEXTURE_MAG_FILTER:
            self.mag_filter = val

    def generate_mipmaps(self):
        if self.type == gl.GL_TEXTURE_2D:
            prev = gl.glGetInteger(gl.GL_TEXTURE_BINDING_2D)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
            gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
            gl.glBindTexture(gl.GL_TEXTURE_2D, prev)

    def upload_np(self, image: np.ndarray):
        self.upload_iterable(image, image.shape, image.dtype.name)

    def upload_iterable(self, data, shape, dtype_str: str):
        has_alpha = shape[2] == 4
        is_fp = dtype_str in ['float32', 'float16']

        # Needed when alternating between CUDA and numpy
        if self.mapper is not None:
            self.mapper.unregister()
            self.mapper = None

        # See upload_ptr() for description of the formats
        internal_fmt = gl.GL_RGBA  # if has_alpha else gl.GL_RGB # how OGL stores data
        incoming_fmt = gl.GL_RGBA if has_alpha else gl.GL_RGB  # incoming channel format
        incoming_dtype = {
            'float32': gl.GL_FLOAT,
            'float16': gl.GL_HALF_FLOAT,
            'uint8': gl.GL_UNSIGNED_BYTE,
            'uint16': gl.GL_UNSIGNED_SHORT,
        }[dtype_str]

        # RGB UINT8 data: no guarantee of 4-byte row alignment
        # (F32 or RGBA alignment always divisible by 4)
        alignment = 4 if (is_fp or has_alpha) else 1
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, alignment)  # default: 4 bytes

        # Make sure tex2d is active
        self.tex = self.tex_2d
        self.type = gl.GL_TEXTURE_2D

        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        if shape[0] != self.shape[0] or shape[1] != self.shape[1] or self.is_fp != is_fp:
            # Reallocate
            self.shape = shape
            self.is_fp = is_fp
            gl.glTexImage2D(
                gl.GL_TEXTURE_2D,  # GLenum target
                0,                # GLint level
                internal_fmt,     # GLint internalformat
                shape[1],         # GLsizei width
                shape[0],         # GLsizei height
                0,                # GLint border
                incoming_fmt,     # GLenum format
                incoming_dtype,   # GLenum type
                data,             # const void * data
            )
        else:
            # Overwrite
            gl.glTexSubImage2D(
                gl.GL_TEXTURE_2D,   # GLenum target
                0,                  # GLint level
                0,                  # GLint xoffset
                0,                  # GLint yoffset
                shape[1],           # GLsizei width
                shape[0],           # GLsizei height
                incoming_fmt,       # GLenum format
                incoming_dtype,     # GLenum type
                data,               # const void * pixels
            )
        if self.needs_mipmap:
            self.generate_mipmaps()
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    def upload_torch(self, img):
        assert img.ndim == 3, "Please provide a HWC tensor"
        assert img.shape[2] < min(img.shape[0], img.shape[1]), "Please provide a HWC tensor"

        if cuda_gl_interop.get_cuda_plugin() is None:
            return self.upload_np(img.detach().cpu().numpy())

        assert img.dtype in [torch.float32, torch.uint8], 'CUDA interop: only fp32 and u8 supported'

        # OpenGL stores RGBA-strided data always
        # Must add alpha for gpu memcopy to work
        if img.shape[2] == 3:
            alpha = 255 if img.dtype == torch.uint8 else 1.0
            img = torch.cat((img, alpha*torch.ones_like(img[:, :, :1])), dim=-1)

        self.type = gl.GL_TEXTURE_2D
        img = img.contiguous()
        self.upload_ptr(img.data_ptr(), img.shape, img.dtype.is_floating_point)
        if self.needs_mipmap:
            self.generate_mipmaps()

    # Copy from cuda pointer
    def upload_ptr(self, ptr, shape, is_fp32):
        assert cuda_gl_interop.get_cuda_plugin() is not None, 'PT plugin needed for pointer upload'
        has_alpha = shape[-1] == 4

        # Reallocate if shape changed or data type changed from np to torch
        # Must also reallocate before mipmap (re)generation (https://stackoverflow.com/a/20359917)
        if self.needs_mipmap or shape[0] != self.shape[0] or shape[1] != self.shape[1] or self.is_fp != is_fp32 or self.mapper is None:
            self.shape = shape
            self.is_fp = is_fp32
            if self.mapper is not None:
                self.mapper.unregister()
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)

            # Internal format specifies how OpenGL stores the data
            #   RGB32F: data stored as 32bit floats internally, shader sees normal ieee floats
            #   RGB8: data stored in unsigned normalized format, shader sees equally spaced floats in [0, 1] ("compressed float")
            #   RGBA8UI: data stored in unsigned integer format, shader sees ints

            """
            And if you are interested, most GPUs like chunks of 4 bytes.
            In other words, GL_RGBA or GL_BGRA is preferred when each component is a byte.
            GL_RGB and GL_BGR is considered bizarre since most GPUs, most CPUs and any other kind of chip don't handle 24 bits.
            This means, the driver converts your GL_RGB or GL_BGR to what the GPU prefers, which typically is RGBA/BGRA.
            (https://www.khronos.org/opengl/wiki/Common_Mistakes)
            """
            # => just use RGBA for compatibility
            internal_fmt = gl.GL_RGBA32F if is_fp32 else gl.GL_RGBA8
            assert has_alpha, 'ptr upload needs alpha channel'

            # Incoming channel format and dtype
            incoming_fmt = gl.GL_RGBA if has_alpha else gl.GL_RGB
            incoming_dtype = gl.GL_FLOAT if is_fp32 else gl.GL_UNSIGNED_BYTE  # fp32 or u8

            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, internal_fmt, shape[1], shape[0], 0, incoming_fmt, incoming_dtype, None)
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            self.mapper = PTExtMapper(int(self.tex))

        # Cast to python integer type
        ptr_int = int(ptr)
        assert ptr_int == ptr, 'Device pointer overflow'

        N = 4 if is_fp32 else 1
        H, W, C = shape
        assert C == 4, 'Input data must be RGBA'  # OpenGL always stores alpha channel

        self.mapper.upload(ptr_int, W, H, C*N)
        cuda_synchronize()


# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# Window unit


class DrawableTextureWindowUnit:
    HIGHTLIGHT_COLOR_POS = (15.0 / 255.0, 230.0 / 255.0, 130.0 / 255.0, 1.0)  # Bright greenish color
    HIGHTLIGHT_COLOR_NEG = (230.0 / 255.0, 15.0 / 255.0, 115.0 / 255.0, 1.0)  # Complementary color

    def __init__(self, window, name: str, mask_enabled: bool = False, hightlight_on_hover_mode: HighlightMode = HighlightMode.entire_image, is_origin_symmetric: bool = False):
        self.top_window = window
        self.name = name
        self.mask_enabled = mask_enabled
        self.hightlight_on_hover_mode = hightlight_on_hover_mode
        self.is_origin_symmetric = is_origin_symmetric

        # Setup PannableArea for this window
        self._drawable_area = drawable_area.DrawableArea(force_mouse_capture=True, drawing_enabled=mask_enabled)

        # Setup OpenGL texture, lazily created
        self._texture: _texture | None = None

        # Buffer on RAM
        self.img_shape: list = [3, 4, 4]  # CHW (to match ToolbarViewer)
        self._image: np.ndarray | torch.Tensor | None = None  # HWC

        # Masking
        self.mask: np.ndarray | torch.Tensor | None = None  # optional mask for drawing
        self.draw_mode = MaskMode.draw_to_hide  # drawing mode
        self.brush_type = BrushType.circle  # brush type
        self.brush_size: float = 5  # in pixels
        if mask_enabled:
            self._drawable_area.set_drawing_mouse_callback(self._draw_mouse_callback)
            self._drawable_area.set_drawing_wheel_callback(self._draw_wheel_callback)

        # Grid highlighting
        self.grid_size: tuple[int, int] | None = None  # (num_rows, num_cols)
        self.attribs: list[list[str]] | None = None  # per-cell attributes for grid highlighting

        # Highlight colors
        self.highlight_color_pos = DrawableTextureWindowUnit.HIGHTLIGHT_COLOR_POS
        self.highlight_color_neg = DrawableTextureWindowUnit.HIGHTLIGHT_COLOR_NEG
        self.attrib_txt_color = None

        # Timings
        self.img_dt: float = 0.0
        self.last_upload_dt: float = 0
        self.tex_upload_ms: float = 0.0  # time taken to upload texture in ms

        self.mask_value: float = 0.0
        self.pass_value_type: PassValueType = PassValueType.raw

        # Dynamically create a new independent copy of the window function
        orig_func = self.__class__.__dict__['window']
        copied_func = types.FunctionType(
            orig_func.__code__,
            orig_func.__globals__,
            name=orig_func.__name__,
            argdefs=orig_func.__defaults__,
            closure=orig_func.__closure__
        )
        dockable_window = ui_utils.dockable(copied_func, title=name)

        # Bind the copied function to this instance
        setattr(self, 'window', dockable_window.__get__(self, self.__class__))

    @property
    def image(self):
        pass

    @image.getter
    def image(self) -> np.ndarray | torch.Tensor | None:
        return self._image

    @image.setter
    def image(self, img:  np.ndarray | torch.Tensor | None):
        self._image = img
        if self.mask is not None and torch.is_tensor(self._image) and torch.is_tensor(self.mask):
            self.mask = self.mask.to(device=self._image.device)

    # ---------------------------------------------------------------------------------------------------------------------
    # Masking

    def reset_mask(self) -> None:
        if self._image is None:
            return

        if self.mask is None or self.mask.shape != self._image.shape[:2]:
            # Create mask if it does not exist
            if isinstance(self._image, np.ndarray):
                self.mask = np.zeros(self.img_shape[1:], dtype=np.bool)
            elif torch.is_tensor(self._image):
                self.mask = torch.zeros(self.img_shape[1:], dtype=torch.bool, device=self._image.device)
            assert self.mask.shape == self._image.shape[:2], f'Unexpected mask shape: {self.mask.shape}, expected {self._image.shape[:2]}'

        self.mask[:, :] = self.draw_mode.canvas_value()

    def invert_mask(self) -> None:
        if self.mask is not None:
            self.mask[:, :] = np.logical_not(self.mask) if isinstance(self.mask, np.ndarray) else torch.logical_not(self.mask)

    def invert_draw_mode(self) -> None:
        self.draw_mode = MaskMode.draw_to_hide if self.draw_mode == MaskMode.draw_to_show else MaskMode.draw_to_show

    def _draw_mouse_callback(self, start_uv: tuple[float, float], end_uv: tuple[float, float]) -> None:
        if self._image is None:
            return

        if self.mask is None:
            self.reset_mask()

        H, W = self.mask.shape

        # Sweep linearly between start and end points with brush size
        start_uv = np.array(start_uv)
        end_uv = np.array(end_uv)

        delta_uv = end_uv - start_uv
        delta_pix = delta_uv * np.array([W, H])

        spacing = self.brush_size / 2.0
        maxd = float(np.abs(delta_pix).max())
        steps = 1 if maxd <= spacing else math.ceil(maxd / spacing)

        val = self.draw_mode.brush_value()
        bs = float(self.brush_size)

        for step in range(steps + 1):
            uv = start_uv + (step / steps) * delta_uv
            uv = np.clip(uv, 0.0, 1.0)

            px = uv[0] * W
            py = uv[1] * H

            x0 = max(0, int(math.floor(px - bs)))
            x1 = min(W, int(math.ceil(px + bs)))
            y0 = max(0, int(math.floor(py - bs)))
            y1 = min(H, int(math.ceil(py + bs)))

            if self.brush_type == BrushType.circle:
                xs = np.arange(x0, x1) + 0.5
                ys = np.arange(y0, y1) + 0.5
                X, Y = np.meshgrid(xs, ys)
                disk = (X - px)**2 + (Y - py)**2 <= bs**2
                roi = self.mask[y0:y1, x0:x1]
                roi[disk] = val

                if self.is_origin_symmetric:
                    yy0, yy1 = H - y1, H - y0
                    xx0, xx1 = W - x1, W - x0
                    mroi = self.mask[yy0:yy1, xx0:xx1]
                    mroi[np.flipud(np.fliplr(disk)).copy()] = val

            elif self.brush_type == BrushType.square:
                self.mask[y0:y1, x0:x1] = val
                if self.is_origin_symmetric:
                    self.mask[H - y1:H - y0, W - x1:W - x0] = val

    def _draw_wheel_callback(self, y: float) -> None:
        if self._image is not None:
            short_side = min(self._image.shape[:2])
            self.brush_size = max(min(self.brush_size + int(y), short_side), 0.1)

    def get_mask(self) -> np.ndarray | torch.Tensor | None:
        """Get the current mask.

        Returns
        -------
        np.ndarray | torch.Tensor | None
            The current mask, or None if no mask is set. The shape is (H, W) and dtype is bool.
        """

        if self.mask is None:
            return None

        if isinstance(self.mask, np.ndarray):
            return self.mask
        elif torch.is_tensor(self.mask):
            return self.mask
        else:
            return None

    def set_mask(self, mask: np.ndarray | torch.Tensor | None) -> None:
        """Set the current mask.

        Parameters
        ----------
        mask: np.ndarray | torch.Tensor | None
            The mask to set. The shape must be (H, W) and dtype must be bool. If None, the mask is reset.
        """

        if mask.shape == self._image.shape[:2]:
            if isinstance(mask, np.ndarray):
                self.mask = mask
            else:
                self.mask = mask
                if self._image is not None:
                    self.mask = self.mask.to(device=self._image.device)

    # ---------------------------------------------------------------------------------------------------------------------

    def is_hovered_on_canvas(self) -> bool:
        mouse_pos_uv = self._drawable_area.get_hovered_uv_image()
        return 0.0 <= mouse_pos_uv[0] <= 1.0 and 0.0 <= mouse_pos_uv[1] <= 1.0

    def is_hovered_on_window(self) -> bool:
        return self._drawable_area.mouse_hovers_content()

    def generate_texture(self) -> None:
        """Lazily create OpenGL texture.
        """

        if self._texture is None:
            self._texture = _texture(gl.GL_NEAREST, gl.GL_NEAREST)

    def set_callbacks(self, glfw_window: glfw._GLFWwindow) -> None:
        """Set callbacks for PannableArea. This must be called after glfw window is created.

        Parameters
        ----------
        glfw_window: glfw._GLFWwindow
            GLFW window to set callbacks for .
        """

        self._drawable_area.set_callbacks(glfw_window)

    def set_theme(self) -> None:
        """Set theme for PannableArea.
        """

        self._drawable_area.clear_color = imgui.get_style().color_(imgui.Col_.window_bg)  # match theme_deep_dark

    def dispose_texture(self) -> None:
        """Dispose OpenGL texture. This is called when the window is closed or the texture is no longer needed.
        """

        if self._texture is not None:
            del self._texture
            self._texture = None

    def get_font(self, target_size: int):
        sizes = list(self.top_window._imgui_fonts.keys())
        i_best = np.argmin(np.abs(np.array(sizes) - target_size))
        return self.top_window._imgui_fonts[sizes[i_best]]

    def upload_texture(self) -> None:
        """Upload image data to OpenGL texture.
        """

        if self._texture is None:
            logger.warning('No OpenGL texture to upload to')
            return

        if isinstance(self._image, np.ndarray):
            image = self._image.copy()
            mask = self.mask.copy() if self.mask is not None else None
        elif torch.is_tensor(self._image):
            image = self._image.clone()
            mask = self.mask.clone() if self.mask is not None else None
        else:
            return

        if self.mask_enabled and mask is not None and image is not None and mask.shape == image.shape[:2]:
            if isinstance(image, np.ndarray):
                pass_value = image if self.pass_value_type == PassValueType.raw else np.ones(shape=image.shape[:2], dtype=image.dtype)
                image = np.where(self.mask.unsqueeze(-1).tile(1, 1, image.shape[-1]), self.mask_value, pass_value)
            else:  # Tensor
                pass_value = image if self.pass_value_type == PassValueType.raw else torch.ones(shape=image.shape[:2], dtype=image.dtype, device=image.device)
                self.mask = self.mask.to(device=image.device)
                image = torch.where(self.mask.unsqueeze(-1).tile(1, 1, image.shape[-1]), self.mask_value, pass_value)

        if self.img_dt > self.last_upload_dt:
            t0 = time.monotonic()

            if isinstance(image, np.ndarray):
                self._texture.upload_np(image)
            elif image.device.type == 'cuda':
                self._texture.upload_torch(image)
            elif image.device.type == 'mps':
                # https://github.com/prabu-ram/Custom_PyTorch-Operations/blob/main/compiler.py
                self._texture.upload_mps(image)
            else:
                self._texture.upload_np(image.cpu().numpy())

            self.last_upload_dt = time.monotonic()

            self.tex_upload_ms = (self.last_upload_dt - t0) * 1000  # upload time stats

    def window(self):
        """Dockable window for displaying a texture.
        """

        # Callback
        if self.is_hovered_on_window() and self._drawable_area._drawing_enabled:
            if imgui.is_key_pressed(imgui.Key.i):
                self.invert_mask()
            elif imgui.is_key_pressed(imgui.Key.d):
                self.invert_draw_mode()
            elif imgui.is_key_pressed(imgui.Key.r):
                self.reset_mask()
            elif imgui.is_key_pressed(imgui.Key.b):
                self.brush_type = BrushType.square if self.brush_type == BrushType.circle else BrushType.circle

        # First upload texture if it has been set
        with ui_utils.lock:
            self.upload_texture()

        # Begin the ImGui window
        imgui.begin(self.name, True, imgui.WindowFlags_.no_collapse)

        if self._image is not None:
            tH, tW, _ = self._image.shape
            cW, cH = map(int, imgui.get_content_region_avail())

            # --------------------------------------------------------------------------------------------------
            # Draw the texture to the canvas
            canvas_tex = self._drawable_area.draw_to_canvas(self._texture.tex, tW, tH, cW, cH)
            imgui.image(canvas_tex, (cW, cH))

            # --------------------------------------------------------------------------------------------------
            # Overlay brush cursor
            if self.mask_enabled and imgui.is_key_down(imgui.Key.left_shift):
                # Overlay the brush cursor
                mouse_pos_screen = imgui.get_mouse_pos()
                pitch_in_uv = 1.0 / float(self._image.shape[0])
                pos_on_screen = self._drawable_area.uv_to_screen_xform() @ np.array([0.0, pitch_in_uv, 0.0])
                pitch_on_screen = np.linalg.norm(pos_on_screen[:2])  # in pixels

                if self.brush_type == BrushType.circle:
                    imgui.get_window_draw_list().add_circle_filled(
                        center=mouse_pos_screen,
                        radius=self.brush_size * pitch_on_screen,
                        col=imgui.get_color_u32((1, 0, 0, 0.5)),
                    )
                elif self.brush_type == BrushType.square:
                    imgui.get_window_draw_list().add_rect_filled(
                        p_min=(mouse_pos_screen[0] - self.brush_size * pitch_on_screen, mouse_pos_screen[1] - self.brush_size * pitch_on_screen),
                        p_max=(mouse_pos_screen[0] + self.brush_size * pitch_on_screen, mouse_pos_screen[1] + self.brush_size * pitch_on_screen),
                        col=imgui.get_color_u32((1, 0, 0, 0.5)),
                    )

            # --------------------------------------------------------------------------------------------------
            # Overlay highlight border

            if self.hightlight_on_hover_mode != HighlightMode.off:
                mouse_pos_uv = self._drawable_area.get_hovered_uv_image()
                if 0.0 <= mouse_pos_uv[0] <= 1.0 and 0.0 <= mouse_pos_uv[1] <= 1.0:
                    ul_uv = None
                    br_uv = None
                    attrib_text = None
                    H, W = 1, 1

                    if self.hightlight_on_hover_mode == HighlightMode.entire_image:
                        # Highlight the entire image
                        ul_uv = np.array([0.0, 0.0, 1.0])
                        br_uv = np.array([1.0, 1.0, 1.0])
                        H = 1
                        W = 1
                    elif self.hightlight_on_hover_mode == HighlightMode.grid_cells and self.grid_size is not None:
                        rows, cols = self.grid_size
                        cell_w = 1.0 / cols
                        cell_h = 1.0 / rows
                        col_idx = min(int(mouse_pos_uv[0] / cell_w), cols - 1)
                        row_idx = min(int(mouse_pos_uv[1] / cell_h), rows - 1)
                        ul_uv = np.array([col_idx * cell_w, row_idx * cell_h, 1.0])
                        br_uv = np.array([(col_idx + 1) * cell_w, (row_idx + 1) * cell_h, 1.0])
                        H = rows
                        W = cols
                        if self.attribs is not None and len(self.attribs) == rows and len(self.attribs[row_idx]) == cols:
                            attrib_text = str(self.attribs[row_idx][col_idx])

                    if ul_uv is not None and br_uv is not None:
                        uv_to_screen_mat = self._drawable_area.uv_to_screen_xform()
                        ul_pos_screen = (uv_to_screen_mat @ ul_uv)[:2].tolist()
                        br_pos_screen = (uv_to_screen_mat @ br_uv)[:2].tolist()

                        rect_color = imgui.get_color_u32(self.highlight_color_neg if self.draw_mode == MaskMode.draw_to_show else self.highlight_color_pos)
                        txt_color = imgui.get_color_u32(self.attrib_txt_color) if self.attrib_txt_color is not None else rect_color

                        imgui.get_window_draw_list().add_rect(
                            p_min=ul_pos_screen,
                            p_max=br_pos_screen,
                            col=rect_color,
                            thickness=3.0,
                        )

                        if attrib_text is not None:
                            img_width_uv = 1 / W
                            target_size = 150 * self.top_window.ui_scale * self._drawable_area.zoom * img_width_uv
                            imgui.get_window_draw_list().add_text(
                                (
                                    ul_pos_screen[0] + target_size * 0.2,
                                    ul_pos_screen[1] + target_size * 0.1,
                                ),
                                txt_color,
                                attrib_text,
                            )

        imgui.end()


class DrawableTextureWindows(dict[str, DrawableTextureWindowUnit]):
    def __init__(
        self,
        window,
        names: list[str],
        normalize: bool,
        mask_enabled: list[str] | None = None,
        hightlight_on_hover_mode: list[HighlightMode] | HighlightMode = HighlightMode.entire_image,
        is_origin_symmetric: bool = False
    ):
        mask_enabled = mask_enabled or [False for _ in names]
        hightlight_on_hover_mode = hightlight_on_hover_mode if isinstance(hightlight_on_hover_mode, list) else [hightlight_on_hover_mode for _ in names]
        windows = {str(name): DrawableTextureWindowUnit(window, name, mask, hightlight, is_origin_symmetric) for (name, mask, hightlight) in zip(names, mask_enabled, hightlight_on_hover_mode)}
        self.normalize = normalize
        super().__init__(windows)

    def generate_texture(self) -> None:
        """Generate OpenGL textures for all texture windows.
        """

        for window in self.values():
            window.generate_texture()

    def dispose_texture(self) -> None:
        """Dispose OpenGL textures for all texture windows.
        """

        for window in self.values():
            window.dispose_texture()

    def set_callbacks(self, glfw_window: glfw._GLFWwindow) -> None:
        """Set GLFW callbacks for all texture windows.

        Parameters
        ----------
        glfw_window: glfw._GLFWwindow
            GLFW window to set callbacks for .
        """

        for window in self.values():
            window.set_callbacks(glfw_window)

    def set_theme(self) -> None:
        """Set theme for all texture windows' PannableArea.
        """

        for window in self.values():
            window.set_theme()

    def update_image(self, **kwargs):
        kwargs = dict(kwargs)

        for key, img_hwc in kwargs.items():
            if img_hwc is None:
                continue

            assert key in self, f'An image data was assigned to a non-existing texture window: "{key}"'

            is_np = isinstance(img_hwc, np.ndarray)
            assert is_np or torch.is_tensor(img_hwc), 'Expected np.ndarray or torch.Tensor'

            is_fp = (img_hwc.dtype.kind == 'f') if is_np else img_hwc.dtype.is_floating_point
            is_signed = (img_hwc.dtype.kind == 'i') if is_np else img_hwc.dtype.is_signed
            dtype_bits = img_hwc.dtype.itemsize * 8
            H, W, C = img_hwc.shape

            # RGBA texture uploads are much faster on some drivers
            if img_hwc.shape[-1] == 3:
                maxval = 1 if is_fp else 2**(dtype_bits - int(is_signed))
                if is_np:
                    img_hwc = np.concatenate([img_hwc, maxval * np.ones((H, W, 1), dtype=img_hwc.dtype)], axis=-1)
                else:
                    img_hwc = torch.cat([img_hwc, ui_utils.alpha_ch_torch(H, W, maxval, img_hwc.dtype, img_hwc.device)], dim=2)

            img_hwc = utils.normalize_image_data(img_hwc, img_hwc.dtype) if self.normalize else img_hwc

            # Eventually uploaded by UI thread
            self[key]._image = img_hwc
            self[key].img_shape = [img_hwc.shape[2], *img_hwc.shape[:2]]  # shape in chw format
            self[key].img_dt = time.monotonic()
            if self[key].mask_enabled and (self[key].mask is None or self[key].mask.shape != img_hwc.shape[:2]):
                self[key].reset_mask()


# ----------------------------------------------------------------------------------------------------------------------------------------------------------
