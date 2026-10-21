"""
CUDA-OpenGL inter-operability
=============================

This module provides a way to upload images directly from CUDA tensors to OpenGL VBOs using a custom PyTorch extension.

License
-------

Copyright (c) 2026 Shinei Arakawa
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

This code is a derivative work based on code from the harskish/pyviewer repository, whose original code is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0). The original code can be found at https://github.com/harskish/pyviewer.
"""

# Code graciously provided by Pauli Kemppinen (github.com/msqrt)

import functools
import glob
import importlib
import os
import pathlib
import platform
import sys

import torch
import torch.utils.cpp_extension as cpp
import ui.utils as ui_utils

# ------------------------------------------------------------------------------------------------------------
# Logger

logger = ui_utils.get_logger()

# --------------------------------------------------------------------------------------------------------------------------------
# Pluging loader


def get_plugin(
    plugin_name: str,
    source_files: list | tuple | str,
    source_folder: str | pathlib.Path = '.',
    ldflags: list | tuple | None = None,
    extra_cflags: list | tuple = (),
    cuda: bool = True,  # can turn off if not needed
    unsafe_load_prebuilt: bool = False,
    verbose=True
):
    if isinstance(source_files, list):
        source_files = tuple(source_files)
    if isinstance(extra_cflags, list):
        extra_cflags = tuple(extra_cflags)

    return _get_plugin_impl(
        plugin_name,
        source_files,
        source_folder,
        ldflags,
        extra_cflags,
        cuda,
        unsafe_load_prebuilt,
        verbose,
    )


@functools.cache
def _get_plugin_impl(
    plugin_name: str,
    source_files: tuple | str,
    source_folder: str | pathlib.Path = '.',
    ldflags: tuple = None,
    extra_cflags: tuple = (),
    cuda: bool = True,
    unsafe_load_prebuilt: bool = False,
    verbose=True
):
    # Make sure we can find the necessary compiler and libary binaries.
    if platform.system() == 'Windows':
        lib_dir = os.path.dirname(__file__) + r"."  # TODO: this seems incorrect

        def find_cl_path():
            for maybe_x86 in ["", " (x86)"]:
                for edition in ['Community', 'Enterprise', 'Professional', 'BuildTools']:
                    paths = sorted(glob.glob(f"C:/Program Files{maybe_x86}/Microsoft Visual Studio/*/{edition}/VC/Tools/MSVC/*/bin/Hostx64/x64"), reverse=True)
                    if paths:
                        return paths[0]

        # If cl.exe is not on path, try to find it.
        if os.system("where cl.exe >nul 2>nul") != 0:
            cl_path = find_cl_path()
            if cl_path is None:
                raise RuntimeError("Could not locate a supported Microsoft Visual C++ installation")
            os.environ['PATH'] += ';' + cl_path

        cuda_path = os.getenv('CUDA_PATH')
        if cuda:
            if cuda_path is None:
                logger.info('Please set environment variable CUDA_PATH')
            else:
                # As of Python 3.8, cwd and $PATH are no longer searched for DLLs
                os.add_dll_directory(os.path.join(cuda_path, 'bin'))

    # Linker options.
    if platform.system() == 'Linux':
        cflags = ['-O3', '-std=c++17', *extra_cflags]
        ldflags = ['-lGL', '-lGLEW', '-lEGL'] if ldflags is None else ldflags
    elif platform.system() == 'Darwin':
        cflags = ['-O3', '-std=c++17', '-DGL_SILENCE_DEPRECATION', *extra_cflags]
        ldflags = ['-framework OpenGL', '-framework Cocoa'] if ldflags is None else ldflags
    elif platform.system() == 'Windows':
        libs = ['user32', 'opengl32']
        ldflags = (['/LIBPATH:' + lib_dir] + ['/DEFAULTLIB:' + x for x in libs]) if ldflags is None else ldflags
        cflags = ['/O2', '/DWIN32', '/std:c++17', '/permissive-', '/w', *extra_cflags]

    # Some containers set this to contain old architectures that won't compile. We only need the one installed in the machine.
    os.environ['TORCH_CUDA_ARCH_LIST'] = ''

    # Custom build dir, avoids expensive torch.utils.cpp_extension import (for _get_build_directory())
    # Enables quicker loads if compiled module exists.
    cu_str = ('cpu' if torch.version.cuda is None else f'cu{torch.version.cuda.replace(".", "")}')
    python_version = f'py{sys.version_info.major}{sys.version_info.minor}{getattr(sys, "abiflags", "")}'
    ext_dir = os.path.expanduser(f'~/.cache/torch_extensions/{python_version}_{cu_str}')
    build_dir = os.path.join(ext_dir, plugin_name)
    os.environ['TORCH_EXTENSIONS_DIR'] = ext_dir

    # Try to load prebuilt, continue to build on failure
    # Might load an old version if the sources have changed
    if unsafe_load_prebuilt:
        prev_path = [*sys.path]
        try:
            sys.path.append(build_dir)
            return importlib.import_module(plugin_name)  # .pyd on Windows
        except ImportError as e:
            if e.msg.startswith('DLL load failed') and cuda:
                logger.error('DLL load failed, make sure all DLLs required by module are available (NB: $PATH and cwd are not searched)')
        except Exception as e:
            logger.error('Failed to load prebuilt plugin: %s', e)
            pass  # could not load cached, proceed to compilation step
        finally:
            sys.path = prev_path

    # Make sure build dir is honored
    _build_dir = cpp._get_build_directory(plugin_name, False)
    assert _build_dir == build_dir, 'Torch extension build dir mismatch'

    # if system() == 'Windows':
    try:
        # Try to detect if a stray lock file is left in cache directory and show a warning.
        # This sometimes happens on Windows if the build is interrupted at just the right moment.
        lock_fn = os.path.join(build_dir, 'lock')
        if os.path.exists(lock_fn):
            logger.warning("Lock file exists in build directory: '%s', removing" % lock_fn)
            os.remove(lock_fn)
    except:
        pass

    # Compile and load.
    original = os.getcwd()
    os.chdir(source_folder)

    try:
        cpp.load(verbose=verbose, name=plugin_name, extra_cflags=cflags, extra_cuda_cflags=['-O2'], sources=source_files, extra_ldflags=ldflags, with_cuda=cuda)
    except ImportError as e:
        if e.msg.startswith('DLL load failed') and cuda:
            # Debug with:
            # dumpbin.exe /dependents cuda_gl_interop.pyd
            # python -m dlldiag deps cuda_gl_interop.pyd
            # Dependencies.exe -chain cuda_gl_interop.pyd -depth 1
            logger.error('DLL load failed, make sure all DLLs required by module are available (NB: $PATH and cwd are not searched)')
    except Exception as e:
        logger.error('Custom op compilation failed: %s', e)
    finally:
        os.chdir(original)
    return importlib.import_module(plugin_name)


@functools.lru_cache
def get_cuda_plugin():
    try:
        logger.info('Compiling CUDA-PyTorch interop plugin. This may take a while ...')
        pt_plugin = get_plugin('cuda_gl_interop', 'cuda_gl_interop.cpp', pathlib.Path(__file__).parent / 'custom_ops', unsafe_load_prebuilt=True)
        logger.info('done.')
        return pt_plugin
    except Exception as e:
        logger.error('Failed to build CUDA-GL plugin: %s', e)
        logger.info('Images will be uploaded from RAM.')
        return None

# --------------------------------------------------------------------------------------------------------------------------------
# Resouurce mapper


class VBOMapper:
    def __init__(self, vbo: int) -> None:
        self._plugin = get_cuda_plugin()
        if self._plugin is None:
            raise RuntimeError('CUDA-GL interop plugin is not available.')
        self._vbo = vbo
        self._resource: int | None = self._plugin.register_vbo(vbo)

    def unregister(self) -> None:
        if self._resource is None:
            return
        self._plugin.unregister_vbo(self._resource)
        self._resource = None

    def upload(self, tensor: torch.Tensor) -> None:
        if self._resource is None:
            raise RuntimeError('VBO resource is not registered.')
        if not tensor.is_cuda:
            raise ValueError('Expected a CUDA tensor for VBO upload.')
        src_tensor = tensor if tensor.is_contiguous() else tensor.contiguous()
        num_bytes = src_tensor.element_size() * src_tensor.nelement()
        self._plugin.upload_vbo(src_tensor.data_ptr(), num_bytes, self._resource)

    def __del__(self) -> None:
        try:
            self.unregister()
        except Exception:
            pass

# --------------------------------------------------------------------------------------------------------------------------------
