"""OpenGL utilities for framebuffer management and shader compilation.

License
-------

Copyright (c) 2026 Shinei Arakawa
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
"""

import contextlib
import ctypes
import dataclasses
import enum
import math
import textwrap

import numpy as np
import OpenGL.GL as gl
import pyglm.glm as glm
import ui.utils as ui_utils
from imgui_bundle import imgui

# ------------------------------------------------------------------------------------------------------------
# Logger

logger = ui_utils.get_logger()


# --------------------------------------------------------------------------------------------------------------------------------
# Framebuffer object


class FrameBuffer:
    def __init__(self, width: int, height: int):
        """Create a framebuffer object with a texture and a renderbuffer.

        Parameters
        ----------
        width : int
            The width of the framebuffer.
        height : int
            The height of the framebuffer.

        Raises
        ------
        RuntimeError
            If the framebuffer is not complete.
        """

        self._fbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._fbo)

        self._texture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._texture)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, width, height, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
        self.texture_ref = imgui.ImTextureRef(self._texture)

        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self._texture, 0)
        self._rbo = gl.glGenRenderbuffers(1)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self._rbo)
        gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_DEPTH24_STENCIL8, width, height)
        gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_STENCIL_ATTACHMENT, gl.GL_RENDERBUFFER, self._rbo)

        if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError('Framebuffer is not complete')

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, 0)

        self._width = width
        self._height = height

    @contextlib.contextmanager
    def ctx(self):
        """Context manager for binding and unbinding the framebuffer.
        """

        self.bind()
        yield
        self.unbind()

    def bind(self):
        """Bind the framebuffer.
        """

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._fbo)

    def unbind(self):
        """Unbind the framebuffer.
        """

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

    def rescale(self, width: int, height: int):
        """Rescale the framebuffer to a new width and height.

        Parameters
        ----------
        width : int
            New width.
        height : int
            New height.

        Raises
        ------
        RuntimeError
            If the framebuffer is not complete.
        """

        self.bind()

        gl.glBindTexture(gl.GL_TEXTURE_2D, self._texture)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, width, height, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glGenerateMipmap(gl.GL_TEXTURE_2D)

        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self._texture, 0)

        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self._rbo)
        gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_DEPTH24_STENCIL8, width, height)
        gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_STENCIL_ATTACHMENT, gl.GL_RENDERBUFFER, self._rbo)

        if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError('Framebuffer is not complete')

        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, 0)

        self.unbind()

        self._width = width
        self._height = height

    def dispose(self):
        """Gracefully delete the framebuffer and its attachments.
        """

        gl.glDeleteFramebuffers(1, [self._fbo])
        gl.glDeleteTextures(1, [self._texture])
        gl.glDeleteRenderbuffers(1, [self._rbo])

# --------------------------------------------------------------------------------------------------------------------------------
# Shader compiler


def build_shader_program(vert_src: str, frag_src: str) -> int:
    """Build a shader program from vertex and fragment shader source code.

    Parameters
    ----------
    vert_src : str
        Vertex shader source code.
    frag_src : str
        Fragment shader source code.

    Returns
    -------
    int
        The ID of the created shader program.
    """

    program = gl.glCreateProgram()
    vert_shader = gl.glCreateShader(gl.GL_VERTEX_SHADER)
    frag_shader = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
    gl.glShaderSource(vert_shader, textwrap.dedent(vert_src))
    gl.glShaderSource(frag_shader, textwrap.dedent(frag_src))
    gl.glCompileShader(vert_shader)
    log = gl.glGetShaderInfoLog(vert_shader)
    if log:
        logger.error(log)
    gl.glCompileShader(frag_shader)
    log = gl.glGetShaderInfoLog(frag_shader)
    if log:
        logger.error(log)

    gl.glAttachShader(program, vert_shader)
    gl.glAttachShader(program, frag_shader)

    gl.glLinkProgram(program)
    gl.glDeleteShader(vert_shader)
    gl.glDeleteShader(frag_shader)

    return program


# --------------------------------------------------------------------------------------------------------------------------------
# Shaders


@dataclasses.dataclass
class PointShaderRenderingContext:
    mvp_mat: glm.mat4  # 4x4 float32
    point_size: float  # float32


class PointShader:
    VERTEX_SHADER = """
    #version 330 core
    layout(location = 0) in vec3 in_position;
    layout(location = 1) in vec3 in_color;

    uniform mat4 u_mvp_mat;
    uniform float u_point_size;

    out vec3 frag_color;

    void main() {
        gl_Position = u_mvp_mat * vec4(in_position, 1.0);
        gl_PointSize = u_point_size;
        frag_color = in_color;
    }
    """

    FRAGMENT_SHADER = """
    #version 330 core
    in vec3 frag_color;

    out vec4 out_color;

    void main() {
        out_color = vec4(frag_color, 1.0);
    }
    """

    def __init__(self):
        self._program = build_shader_program(self.VERTEX_SHADER, self.FRAGMENT_SHADER)

        field_names = [f'u_{name}' for name in PointShaderRenderingContext.__dataclass_fields__.keys()]
        self._uniform_locs = {name: gl.glGetUniformLocation(self._program, name) for name in field_names}

    @contextlib.contextmanager
    def ctx(self, ctx: PointShaderRenderingContext):
        """Context manager for using the shader program.

        Parameters
        ----------
        ctx : PointShaderRenderingContext
            The rendering context containing uniform values.
        """

        gl.glUseProgram(self._program)

        gl.glUniformMatrix4fv(self._uniform_locs['u_mvp_mat'], 1, gl.GL_FALSE, glm.value_ptr(ctx.mvp_mat))
        gl.glUniform1f(self._uniform_locs['u_point_size'], ctx.point_size)

        yield

        gl.glUseProgram(0)

    def dispose(self):
        """Gracefully delete the shader program."""
        gl.glDeleteProgram(self._program)
        self._program = 0


@dataclasses.dataclass
class LineShaderRenderingContext:
    mvp_mat: glm.mat4  # 4x4 float32
    color: glm.vec3  # 3 float32


class LineShader:
    VERTEX_SHADER = """
    #version 330 core
    layout(location = 0) in vec3 in_position;

    uniform mat4 u_mvp_mat;
    uniform vec3 u_color;

    out vec3 frag_color;

    void main() {
        gl_Position = u_mvp_mat * vec4(in_position, 1.0);
        frag_color = u_color;
    }
    """

    FRAGMENT_SHADER = """
    #version 330 core
    in vec3 frag_color;

    out vec4 out_color;

    void main() {
        out_color = vec4(frag_color, 1.0);
    }
    """

    def __init__(self):
        self._program = build_shader_program(self.VERTEX_SHADER, self.FRAGMENT_SHADER)

        field_names = [f'u_{name}' for name in LineShaderRenderingContext.__dataclass_fields__.keys()]
        self._uniform_locs = {name: gl.glGetUniformLocation(self._program, name) for name in field_names}

    @contextlib.contextmanager
    def ctx(self, ctx: LineShaderRenderingContext):
        """Context manager for using the shader program.

        Parameters
        ----------
        ctx : LineShaderRenderingContext
            The rendering context containing uniform values.
        """

        gl.glUseProgram(self._program)

        gl.glEnable(gl.GL_LINE_SMOOTH)
        gl.glEnable(gl.GL_BLEND)

        gl.glUniformMatrix4fv(self._uniform_locs['u_mvp_mat'], 1, gl.GL_FALSE, glm.value_ptr(ctx.mvp_mat))
        gl.glUniform3fv(self._uniform_locs['u_color'], 1, glm.value_ptr(ctx.color))

        yield

        gl.glDisable(gl.GL_LINE_SMOOTH)
        gl.glDisable(gl.GL_BLEND)

        gl.glUseProgram(0)

    def dispose(self):
        """Gracefully delete the shader program."""
        gl.glDeleteProgram(self._program)
        self._program = 0


class RenderingType(enum.IntEnum):
    SHADING_COLOR = 0


@dataclasses.dataclass
class ModelShaderRenderingContext:
    # autopep8: off
    mvp_mat                : glm.mat4 # 4x4 float32
    mv_mat                 : glm.mat4 # 4x4 float32
    normal_mat             : glm.mat4 # 4x4 float32
    light_pos_camera_space : glm.vec3 # 3 float32
    rendering_type         : float    # float32 (RenderingType)
    shiness                : float    # float32
    ambient_intensity      : float    # float32
    # autopep8: on


class ModelShader:
    VERTEX_SHADER = """
    # version 330 core
    
    layout(location = 0) in vec3 in_position;
    layout(location = 1) in vec3 in_color;
    layout(location = 2) in vec3 in_normal;

    uniform mat4 u_mvp_mat;
    uniform mat4 u_mv_mat;
    uniform mat4 u_normal_mat;

    out vec3 f_color;
    out vec3 f_normal_camera_space;
    out vec3 f_pos_camera_space;

    void main() {
        gl_Position = u_mvp_mat * vec4(in_position, 1.0);
        
        f_color = in_color;
        f_normal_camera_space = (u_normal_mat * vec4(in_normal, 0.0)).xyz;
        f_pos_camera_space = (u_mv_mat * vec4(in_position, 1.0)).xyz;
    }
    
    """

    FRAGMENT_SHADER = """
    # version 330 core
    
    in vec3 f_color;
    in vec3 f_normal_camera_space;
    in vec3 f_pos_camera_space;
    
    uniform float u_rendering_type;
    uniform float u_shiness;
    uniform float u_ambient_intensity;
    uniform vec3 u_light_pos_camera_space;
    
    out vec4 out_color;
    
    vec4 shade(vec3 diffuse_color, vec3 specular_color, vec3 ambient_color, vec3 N) {
        vec3 V = normalize(-f_pos_camera_space);                           // View direction
        vec3 L = normalize(u_light_pos_camera_space - f_pos_camera_space); // Light direction
        vec3 H = normalize(L + V);                                         // Halfway vector
        
        float NdotL = max(dot(N, L), 0.0);
        float NdotH = max(dot(N, H), 0.0);
        
        vec3 diffuse = diffuse_color * NdotL;
        vec3 specular = specular_color * pow(NdotH, u_shiness);
        vec3 ambient = ambient_color * u_ambient_intensity;
        
        return vec4(diffuse + specular + ambient, 1.0);
    }
    
    void main() {
        // Compute normal
        vec3 N = normalize(f_normal_camera_space);
        
        // Compute color
        if (-0.5 < u_rendering_type && u_rendering_type < 0.5) {
            // SHADING with color
            out_color = shade(f_color, vec3(1.0), f_color, N);
        }
    }
    """

    def __init__(self):
        self._program = build_shader_program(self.VERTEX_SHADER, self.FRAGMENT_SHADER)

        field_names = [f'u_{name}' for name in ModelShaderRenderingContext.__dataclass_fields__.keys()]
        self._uniform_locs = {name: gl.glGetUniformLocation(self._program, name) for name in field_names}

    @contextlib.contextmanager
    def ctx(self, ctx: ModelShaderRenderingContext):
        """Context manager for using the shader program.

        Parameters
        ----------
        ctx : ModelShaderRenderingContext
            The rendering context containing uniform values.
        """

        gl.glUseProgram(self._program)
        gl.glUniformMatrix4fv(self._uniform_locs['u_mvp_mat'], 1, gl.GL_FALSE, glm.value_ptr(ctx.mvp_mat))
        gl.glUniformMatrix4fv(self._uniform_locs['u_mv_mat'], 1, gl.GL_FALSE, glm.value_ptr(ctx.mv_mat))
        gl.glUniformMatrix4fv(self._uniform_locs['u_normal_mat'], 1, gl.GL_FALSE, glm.value_ptr(ctx.normal_mat))
        gl.glUniform3fv(self._uniform_locs['u_light_pos_camera_space'], 1, glm.value_ptr(ctx.light_pos_camera_space))
        gl.glUniform1f(self._uniform_locs['u_rendering_type'], float(ctx.rendering_type))
        gl.glUniform1f(self._uniform_locs['u_shiness'], ctx.shiness)
        gl.glUniform1f(self._uniform_locs['u_ambient_intensity'], ctx.ambient_intensity)

        yield

        gl.glUseProgram(0)

    def dispose(self):
        """Gracefully delete the shader program."""
        gl.glDeleteProgram(self._program)
        self._program = 0


# --------------------------------------------------------------------------------------------------------------------------------
# Object primitives


class AxisAlignedBoundingBox:
    INDICES = [
        0, 1,
        1, 2,
        2, 3,
        3, 0,
        0, 4,
        1, 5,
        2, 6,
        3, 7,
        4, 5,
        5, 6,
        6, 7,
        7, 4,
    ]

    def __init__(self):
        self._vao = None  # Lazy allocation
        self._vbo = None
        self._ebo = None

        self._num_vertices = 8
        self._num_indices = len(self.INDICES)

        self._line_shader: LineShader | None = None

    def _allocate_new_buffer(self):
        if self._vao is not None:
            self.dispose()

        self._vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self._vao)

        self._vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self._num_vertices * 3 * 4, None, gl.GL_DYNAMIC_DRAW)  # 3 float32 per vertex

        stride = 3 * 4  # 3 float32 per vertex
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(0))

        # Index buffer
        self._ebo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._ebo)
        index_data = (ctypes.c_uint * self._num_indices)(*self.INDICES)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, ctypes.sizeof(index_data), index_data, gl.GL_STATIC_DRAW)

        gl.glBindVertexArray(0)

    def update_geometry(self, min_pt: glm.vec3, max_pt: glm.vec3):
        if self._vao is None:
            self._allocate_new_buffer()

        vertices = glm.array(
            glm.vec3(min_pt.x, min_pt.y, min_pt.z),
            glm.vec3(max_pt.x, min_pt.y, min_pt.z),
            glm.vec3(max_pt.x, max_pt.y, min_pt.z),
            glm.vec3(min_pt.x, max_pt.y, min_pt.z),
            glm.vec3(min_pt.x, min_pt.y, max_pt.z),
            glm.vec3(max_pt.x, min_pt.y, max_pt.z),
            glm.vec3(max_pt.x, max_pt.y, max_pt.z),
            glm.vec3(min_pt.x, max_pt.y, max_pt.z),
        )

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo)
        gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, vertices.nbytes, vertices.ptr)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

    def dispose(self):
        """Gracefully delete OpenGL resources."""

        if self._vao is not None:
            gl.glDeleteVertexArrays(1, [self._vao])
            self._vao = None
        if self._vbo is not None:
            gl.glDeleteBuffers(1, [self._vbo])
            self._vbo = None
        if self._ebo is not None:
            gl.glDeleteBuffers(1, [self._ebo])
            self._ebo = None

        if self._line_shader is not None:
            self._line_shader.dispose()
            self._line_shader = None

    def draw(self, mvp_mat: glm.mat4, color: glm.vec3):
        if self._vao is None:
            return

        if self._line_shader is None:
            # Compile shader on first use
            logger.info('Compiling line shader...')
            self._line_shader = LineShader()
            logger.info('done.')

        with self._line_shader.ctx(LineShaderRenderingContext(mvp_mat, color)):

            gl.glBindVertexArray(self._vao)
            gl.glDrawElements(gl.GL_LINES, self._num_indices, gl.GL_UNSIGNED_INT, None)
            gl.glBindVertexArray(0)


class AxesCone:
    CONE_HEIGHT = 0.2
    CONE_RADIUS = 0.2
    CYLINDER_RADIUS = 0.1
    POS_IN_SCREEN_SPACE = glm.vec4(-0.8, 0.8, 0.99, 1.0)

    def __init__(
        self,
        n_divs: int = 32,
        scale: float = 1.0,
    ) -> None:
        self._n_divs = max(3, int(n_divs))

        self._scale = float(scale)

        self._vao: int | None = None
        self._vbo: int | None = None
        self._ebo: int | None = None

        self._num_vertices = 0
        self._num_indices = 0

        self._model_shader: ModelShader | None = None

    def _emit_vertex(
        self,
        out_vertices: list[float],
        position: glm.vec3,
        color: glm.vec3,
        normal: glm.vec3,
        rotation_mat: glm.mat4,
    ) -> None:
        pos4 = rotation_mat * glm.vec4(position, 1.0)
        transformed_pos = glm.vec3(pos4.x, pos4.y, pos4.z) * self._scale

        norm4 = rotation_mat * glm.vec4(normal, 0.0)
        transformed_norm = glm.vec3(norm4.x, norm4.y, norm4.z)
        norm_length = glm.length(transformed_norm)
        if norm_length > 0:
            transformed_norm = transformed_norm / norm_length

        out_vertices.extend(
            (
                transformed_pos.x,
                transformed_pos.y,
                transformed_pos.z,
                color.r,
                color.g,
                color.b,
                transformed_norm.x,
                transformed_norm.y,
                transformed_norm.z,
            )
        )

    def _append_axis_vertices(
        self,
        out_vertices: list[float],
        color: glm.vec3,
        rotation_mat: glm.mat4,
    ) -> None:
        delta_theta = 2.0 * math.pi / float(self._n_divs - 1)

        origin = glm.vec3(0.0, 0.0, 0.0)
        apex = glm.vec3(1.0, 0.0, 0.0)
        cone_bottom = glm.vec3(1.0 - self.CONE_HEIGHT, 0.0, 0.0)

        for i_theta in range(self._n_divs - 1):
            theta = i_theta * delta_theta
            theta_next = theta + delta_theta

            cone_bottom_pos0 = glm.vec3(
                cone_bottom.x,
                self.CONE_RADIUS * math.cos(theta),
                self.CONE_RADIUS * math.sin(theta),
            )
            cone_bottom_pos1 = glm.vec3(
                cone_bottom.x,
                self.CONE_RADIUS * math.cos(theta_next),
                self.CONE_RADIUS * math.sin(theta_next),
            )

            cone_normal0 = glm.vec3(0.0, math.cos(theta), math.sin(theta))
            cone_normal1 = glm.vec3(0.0, math.cos(theta_next), math.sin(theta_next))

            cylinder_top_pos0 = glm.vec3(
                cone_bottom.x,
                self.CYLINDER_RADIUS * math.cos(theta),
                self.CYLINDER_RADIUS * math.sin(theta),
            )
            cylinder_top_pos1 = glm.vec3(
                cone_bottom.x,
                self.CYLINDER_RADIUS * math.cos(theta_next),
                self.CYLINDER_RADIUS * math.sin(theta_next),
            )
            cylinder_bottom_pos0 = glm.vec3(
                0.0,
                self.CYLINDER_RADIUS * math.cos(theta),
                self.CYLINDER_RADIUS * math.sin(theta),
            )
            cylinder_bottom_pos1 = glm.vec3(
                0.0,
                self.CYLINDER_RADIUS * math.cos(theta_next),
                self.CYLINDER_RADIUS * math.sin(theta_next),
            )

            cylinder_normal0 = glm.vec3(0.0, math.cos(theta), math.sin(theta))
            cylinder_normal1 = glm.vec3(0.0, math.cos(theta_next), math.sin(theta_next))

            # Cone (front face)
            self._emit_vertex(out_vertices, cone_bottom_pos0, color, cone_normal0, rotation_mat)
            self._emit_vertex(out_vertices, cone_bottom_pos1, color, cone_normal1, rotation_mat)
            self._emit_vertex(out_vertices, apex, color, apex, rotation_mat)

            # Cone (back face)
            self._emit_vertex(out_vertices, cone_bottom_pos0, color, -apex, rotation_mat)
            self._emit_vertex(out_vertices, cone_bottom, color, -apex, rotation_mat)
            self._emit_vertex(out_vertices, cone_bottom_pos1, color, -apex, rotation_mat)

            # Cylinder side
            self._emit_vertex(out_vertices, cylinder_top_pos1, color, cylinder_normal1, rotation_mat)
            self._emit_vertex(out_vertices, cylinder_top_pos0, color, cylinder_normal0, rotation_mat)
            self._emit_vertex(out_vertices, cylinder_bottom_pos0, color, cylinder_normal0, rotation_mat)
            self._emit_vertex(out_vertices, cylinder_bottom_pos0, color, cylinder_normal0, rotation_mat)
            self._emit_vertex(out_vertices, cylinder_bottom_pos1, color, cylinder_normal1, rotation_mat)
            self._emit_vertex(out_vertices, cylinder_top_pos1, color, cylinder_normal1, rotation_mat)

            # Cylinder cap (back face)
            self._emit_vertex(out_vertices, cylinder_bottom_pos0, color, -apex, rotation_mat)
            self._emit_vertex(out_vertices, origin, color, -apex, rotation_mat)
            self._emit_vertex(out_vertices, cylinder_bottom_pos1, color, -apex, rotation_mat)

    def _build_geometry(self) -> tuple[np.ndarray, np.ndarray]:
        vertices: list[float] = []

        rotations = (
            glm.mat4(1.0),
            glm.rotate(glm.mat4(1.0), math.radians(90.0), glm.vec3(0.0, 0.0, 1.0)),
            glm.rotate(glm.mat4(1.0), math.radians(-90.0), glm.vec3(0.0, 1.0, 0.0)),
        )
        colors = (
            glm.vec3(1.0, 0.0, 0.0),
            glm.vec3(0.0, 1.0, 0.0),
            glm.vec3(0.0, 0.0, 1.0),
        )

        for rotation_mat, color in zip(rotations, colors):
            self._append_axis_vertices(vertices, color, rotation_mat)

        vertex_array = np.array(vertices, dtype=np.float32).reshape(-1, 9)
        index_array = np.arange(vertex_array.shape[0], dtype=np.uint32)

        return vertex_array, index_array

    def _initialize(self) -> None:
        if self._vao is not None:
            self.dispose()

        vertex_array, index_array = self._build_geometry()
        self._num_vertices = vertex_array.shape[0]
        self._num_indices = index_array.size

        self._vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self._vao)

        self._vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertex_array.nbytes, vertex_array, gl.GL_STATIC_DRAW)

        self._ebo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, index_array.nbytes, index_array, gl.GL_STATIC_DRAW)

        stride = 9 * 4  # 9 float32 per vertex (pos, color, normal)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(3 * 4))
        gl.glEnableVertexAttribArray(2)
        gl.glVertexAttribPointer(2, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(6 * 4))

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindVertexArray(0)

    def draw(
        self,
        rot_mat: glm.mat4,
        view_mat: glm.mat4,
        proj_mat: glm.mat4,
        light_trans_mat: glm.mat4,
        light_pos: glm.vec3,
        shiness: float = 32.0,
        ambient_intensity: float = 0.4,
    ) -> None:
        if self._vao is None:
            self._initialize()

        if self._model_shader is None:
            # Compile shader on first use
            logger.info('Compiling model shader...')
            self._model_shader = ModelShader()
            logger.info('done.')

        model_mat = rot_mat
        mv_mat = view_mat * model_mat
        mvp_mat = proj_mat * mv_mat
        light_mat = mv_mat * light_trans_mat

        pos_in_camera_space = glm.inverse(proj_mat) * self.POS_IN_SCREEN_SPACE
        pos_in_camera_space /= pos_in_camera_space.w
        pos_in_model_space = (glm.inverse(mv_mat) * pos_in_camera_space).xyz

        mvt_mat = mv_mat * glm.translate(pos_in_model_space)
        mvpt_mat = mvp_mat * glm.translate(pos_in_model_space)
        normal_mat = glm.transpose(glm.inverse(mvt_mat))

        light_pos_camera_space = glm.vec3(light_mat * glm.vec4(light_pos, 1.0))

        with self._model_shader.ctx(
            ModelShaderRenderingContext(
                mv_mat=mvt_mat,
                mvp_mat=mvpt_mat,
                normal_mat=normal_mat,
                light_pos_camera_space=light_pos_camera_space,
                rendering_type=float(RenderingType.SHADING_COLOR.value),
                shiness=shiness,
                ambient_intensity=ambient_intensity,
            )
        ):
            gl.glBindVertexArray(self._vao)
            gl.glDrawElements(gl.GL_TRIANGLES, self._num_indices, gl.GL_UNSIGNED_INT, None)
            gl.glBindVertexArray(0)

    def dispose(self) -> None:
        if self._vao is not None:
            gl.glDeleteVertexArrays(1, [self._vao])
            self._vao = None
        if self._vbo is not None:
            gl.glDeleteBuffers(1, [self._vbo])
            self._vbo = None
        if self._ebo is not None:
            gl.glDeleteBuffers(1, [self._ebo])
            self._ebo = None

        if self._model_shader is not None:
            self._model_shader.dispose()
            self._model_shader = None

# --------------------------------------------------------------------------------------------------------------------------------
