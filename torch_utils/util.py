"""Miscellaneous utility classes and functions.

License
-------

Copyright (c) 2026 Shinei Arakawa
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

This code is a derivative work based on code from the NVlabs/edm2 repository, whose original code is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0). The original code can be found at https://github.com/NVlabs/edm2.
"""

# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/


import contextlib
import ctypes
import enum
import fnmatch
import functools
import glob
import hashlib
import html
import importlib
import inspect
import io
import os
import pickle
import random
import re
import shlex
import shutil
import sys
import tempfile
import types
import typing
import urllib
import urllib.parse
import uuid
from typing import Any, BinaryIO, Callable, List, Optional, Tuple, Union

import gdown
import gdown.parse_url
import numpy as np
import paramiko
import requests
import torch
import tqdm

# Seeding function
# ------------------------------------------------------------------------------------------


def seed_everything(base_seed: int, rank: int = 0, offset: int = -1) -> int:
    """Set random seeds for reproducibility. This function sets seeds for Python's `random` module, NumPy, and PyTorch.

    Parameters
    ----------
    base_seed : int
        The base seed to use.
    rank : int, optional
        The rank of the current process (for distributed training). Default is 0.
    offset : int, optional
        An additional offset to ensure different seeds for different calls. Default is -1.

    Returns
    -------
    int
        The computed seed value.
    """

    seed_str = hashlib.sha256(f'base_seed={base_seed}_rank={rank}_offset={offset}'.encode('utf-8')).hexdigest()
    seed = int(seed_str, 16) % (1 << 31)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


# Util classes
# ------------------------------------------------------------------------------------------


class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]

    def to_stddict(self) -> dict:
        d = dict()

        for k, v in self.items():
            if isinstance(v, EasyDict):
                d[k] = v.to_stddict()
            elif isinstance(v, list) or isinstance(v, tuple):
                d[k] = [e.to_stddict() if isinstance(e, EasyDict) else e for e in v]
            else:
                d[k] = v

        return d


class StrEnum(str, enum.Enum):
    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return self.value

    def __list__(self) -> list[str]:
        return [e.value for e in self.__class__]

    @classmethod
    def _items(cls) -> list[str]:
        return [e.value for e in cls]


class Logger(object):
    """Redirect stderr to stdout, optionally print stdout to a file, and optionally force flushing on both stdout and the file."""

    def __init__(self, file_name: Optional[str] = None, file_mode: str = "w", should_flush: bool = True):
        self.file = None

        if file_name is not None:
            self.file = open(file_name, file_mode, encoding="utf-8")

        self.should_flush = should_flush
        self.stdout = sys.stdout
        self.stderr = sys.stderr

        sys.stdout = self
        sys.stderr = self

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def isatty(self) -> bool:
        # This function is needed for some libraries (e.g. wandb) to detect whether the output stream is a terminal.
        return bool(getattr(self.stdout, "isatty", lambda: False)())

    def write(self, text: Union[str, bytes]) -> None:
        """Write text to stdout (and a file) and optionally flush."""
        if isinstance(text, bytes):
            text = text.decode()
        if len(text) == 0:  # workaround for a bug in VSCode debugger: sys.stdout.write(''); sys.stdout.flush() => crash
            return

        if self.file is not None:
            self.file.write(text)

        self.stdout.write(text)

        if self.should_flush:
            self.flush()

    def flush(self) -> None:
        """Flush written text to both stdout and a file, if open."""
        if self.file is not None:
            self.file.flush()

        self.stdout.flush()

    def close(self) -> None:
        """Flush, close possible files, and remove stdout/stderr mirroring."""
        self.flush()

        # if using multiple loggers, prevent closing in wrong order
        if sys.stdout is self:
            sys.stdout = self.stdout
        if sys.stderr is self:
            sys.stderr = self.stderr

        if self.file is not None:
            self.file.close()
            self.file = None


# Cache directories
# ------------------------------------------------------------------------------------------

_dnnlib_cache_dir = None


def set_cache_dir(path: str) -> None:
    global _dnnlib_cache_dir
    _dnnlib_cache_dir = path


def make_cache_dir_path(*paths: str) -> str:
    if _dnnlib_cache_dir is not None:
        return os.path.join(_dnnlib_cache_dir, *paths)
    if 'DNNLIB_CACHE_DIR' in os.environ:
        return os.path.join(os.environ['DNNLIB_CACHE_DIR'], *paths)
    if 'HOME' in os.environ:
        return os.path.join(os.environ['HOME'], '.cache', 'dnnlib', *paths)
    if 'USERPROFILE' in os.environ:
        return os.path.join(os.environ['USERPROFILE'], '.cache', 'dnnlib', *paths)
    return os.path.join(tempfile.gettempdir(), '.cache', 'dnnlib', *paths)

# Small util functions
# ------------------------------------------------------------------------------------------


def format_time(seconds: Union[int, float]) -> str:
    """Convert the seconds to human readable string with days, hours, minutes and seconds."""
    s = int(np.rint(seconds))

    if s < 60:
        return "{0}s".format(s)
    elif s < 60 * 60:
        return "{0}m {1:02}s".format(s // 60, s % 60)
    elif s < 24 * 60 * 60:
        return "{0}h {1:02}m {2:02}s".format(s // (60 * 60), (s // 60) % 60, s % 60)
    else:
        return "{0}d {1:02}h {2:02}m".format(s // (24 * 60 * 60), (s // (60 * 60)) % 24, (s // 60) % 60)


def format_time_brief(seconds: Union[int, float]) -> str:
    """Convert the seconds to human readable string with days, hours, minutes and seconds."""
    s = int(np.rint(seconds))

    if s < 60:
        return "{0}s".format(s)
    elif s < 60 * 60:
        return "{0}m {1:02}s".format(s // 60, s % 60)
    elif s < 24 * 60 * 60:
        return "{0}h {1:02}m".format(s // (60 * 60), (s // 60) % 60)
    else:
        return "{0}d {1:02}h".format(s // (24 * 60 * 60), (s // (60 * 60)) % 24)


def tuple_product(t: Tuple) -> Any:
    """Calculate the product of the tuple elements."""
    result = 1

    for v in t:
        result *= v

    return result


_str_to_ctype = {
    "uint8": ctypes.c_ubyte,
    "uint16": ctypes.c_uint16,
    "uint32": ctypes.c_uint32,
    "uint64": ctypes.c_uint64,
    "int8": ctypes.c_byte,
    "int16": ctypes.c_int16,
    "int32": ctypes.c_int32,
    "int64": ctypes.c_int64,
    "float32": ctypes.c_float,
    "float64": ctypes.c_double
}


def get_dtype_and_ctype(type_obj: Any) -> Tuple[np.dtype, Any]:
    """Given a type name string (or an object having a __name__ attribute), return matching Numpy and ctypes types that have the same size in bytes."""
    type_str = None

    if isinstance(type_obj, str):
        type_str = type_obj
    elif hasattr(type_obj, "__name__"):
        type_str = type_obj.__name__
    elif hasattr(type_obj, "name"):
        type_str = type_obj.name
    else:
        raise RuntimeError("Cannot infer type name from input")

    assert type_str in _str_to_ctype.keys()

    my_dtype = np.dtype(type_str)
    my_ctype = _str_to_ctype[type_str]

    assert my_dtype.itemsize == ctypes.sizeof(my_ctype)

    return my_dtype, my_ctype


def is_pickleable(obj: Any) -> bool:
    try:
        with io.BytesIO() as stream:
            pickle.dump(obj, stream)
        return True
    except:
        return False


# Functionality to import modules/objects by name, and call functions by name
# ------------------------------------------------------------------------------------------

def get_module_from_obj_name(obj_name: str) -> Tuple[types.ModuleType, str]:
    """Searches for the underlying module behind the name to some python object.
    Returns the module and the object name (original name with module part removed)."""

    # allow convenience shorthands, substitute them by full names
    obj_name = re.sub("^np.", "numpy.", obj_name)
    obj_name = re.sub("^tf.", "tensorflow.", obj_name)

    # list alternatives for (module_name, local_obj_name)
    parts = obj_name.split(".")
    name_pairs = [(".".join(parts[:i]), ".".join(parts[i:])) for i in range(len(parts), 0, -1)]

    # try each alternative in turn
    for module_name, local_obj_name in name_pairs:
        try:
            module = importlib.import_module(module_name)  # may raise ImportError
            get_obj_from_module(module, local_obj_name)  # may raise AttributeError
            return module, local_obj_name
        except:
            pass

    # maybe some of the modules themselves contain errors?
    for module_name, _local_obj_name in name_pairs:
        try:
            importlib.import_module(module_name)  # may raise ImportError
        except ImportError:
            if not str(sys.exc_info()[1]).startswith("No module named '" + module_name + "'"):
                raise

    # maybe the requested attribute is missing?
    for module_name, local_obj_name in name_pairs:
        try:
            module = importlib.import_module(module_name)  # may raise ImportError
            get_obj_from_module(module, local_obj_name)  # may raise AttributeError
        except ImportError:
            pass

    # we are out of luck, but we have no idea why
    raise ImportError(obj_name)


def get_obj_from_module(module: types.ModuleType, obj_name: str) -> Any:
    """Traverses the object name and returns the last (rightmost) python object."""
    if obj_name == '':
        return module
    obj = module
    for part in obj_name.split("."):
        obj = getattr(obj, part)
    return obj


def get_obj_by_name(name: str) -> Any:
    """Finds the python object with the given name."""
    module, obj_name = get_module_from_obj_name(name)
    return get_obj_from_module(module, obj_name)


def call_func_by_name(*args, func_name: Union[str, Callable], **kwargs) -> Any:
    """Finds the python object with the given name and calls it as a function."""
    assert func_name is not None
    func_obj = get_obj_by_name(func_name) if isinstance(func_name, str) else func_name
    assert callable(func_obj)
    return func_obj(*args, **kwargs)


def construct_class_by_name(*args, class_name: Union[str, type], **kwargs) -> Any:
    """Finds the python class with the given name and constructs it with the given arguments."""
    return call_func_by_name(*args, func_name=class_name, **kwargs)


def get_module_dir_by_obj_name(obj_name: str) -> str:
    """Get the directory path of the module containing the given object name."""
    module, _ = get_module_from_obj_name(obj_name)
    return os.path.dirname(inspect.getfile(module))


def is_top_level_function(obj: Any) -> bool:
    """Determine whether the given object is a top-level function, i.e., defined at module scope using 'def'."""
    return callable(obj) and obj.__name__ in sys.modules[obj.__module__].__dict__


def get_top_level_function_name(obj: Any) -> str:
    """Return the fully-qualified name of a top-level function."""
    assert is_top_level_function(obj)
    module = obj.__module__
    if module == '__main__':
        fname = sys.modules[module].__file__
        assert fname is not None
        module = os.path.splitext(os.path.basename(fname))[0]
    return module + "." + obj.__name__


# File system helpers
# ------------------------------------------------------------------------------------------

def list_dir_recursively_with_ignore(dir_path: str, ignores: Optional[List[str]] = None, add_base_to_relative: bool = False) -> List[Tuple[str, str]]:
    """List all files recursively in a given directory while ignoring given file and directory names.
    Returns list of tuples containing both absolute and relative paths."""
    assert os.path.isdir(dir_path)
    base_name = os.path.basename(os.path.normpath(dir_path))

    if ignores is None:
        ignores = []

    result = []

    for root, dirs, files in os.walk(dir_path, topdown=True):
        for ignore_ in ignores:
            dirs_to_remove = [d for d in dirs if fnmatch.fnmatch(d, ignore_)]

            # dirs need to be edited in-place
            for d in dirs_to_remove:
                dirs.remove(d)

            files = [f for f in files if not fnmatch.fnmatch(f, ignore_)]

        absolute_paths = [os.path.join(root, f) for f in files]
        relative_paths = [os.path.relpath(p, dir_path) for p in absolute_paths]

        if add_base_to_relative:
            relative_paths = [os.path.join(base_name, p) for p in relative_paths]

        assert len(absolute_paths) == len(relative_paths)
        result += zip(absolute_paths, relative_paths)

    return result


def copy_files_and_create_dirs(files: List[Tuple[str, str]]) -> None:
    """Takes in a list of tuples of (src, dst) paths and copies files.
    Will create all necessary directories."""
    for file in files:
        target_dir_name = os.path.dirname(file[1])

        # will create all intermediate-level directories
        os.makedirs(target_dir_name, exist_ok=True)
        shutil.copyfile(file[0], file[1])


# URL helpers
# ------------------------------------------------------------------------------------------

def _is_ssh_path(url: str, ssh_config: paramiko.SSHConfig) -> bool:
    if "://" in url or ":" not in url:
        return False
    host = url.split(":", 1)[0].strip()
    return host in ssh_config.get_hostnames()


def _split_ssh_path(url: str) -> tuple[str, str]:
    host, path = url.split(":", 1)
    return host.strip(), path.strip()


def _open_cached_file_or_none(
    cache_dir: str,
    cache_key: str,
    return_filename: bool,
):
    cache_file = _get_first_cache_file(cache_dir, cache_key)
    if cache_file is None:
        return None
    return cache_file if return_filename else open(cache_file, "rb")


def _safe_cache_name(name: str, max_len: int = 128) -> str:
    name = re.sub(r"[^0-9a-zA-Z-._]", "_", name)
    return name[:max_len]


def _md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@functools.lru_cache(maxsize=None)
def _get_ssh_config(config_file: str):
    config_file = os.path.expanduser(config_file)
    ssh_config = paramiko.SSHConfig()
    with open(config_file) as f:
        ssh_config.parse(f)
    return ssh_config


def _get_first_identity_file(host_config: dict) -> str | None:
    identity_files = host_config.get("identityfile")
    if not identity_files:
        return None
    if isinstance(identity_files, str):
        return os.path.expanduser(identity_files)
    return os.path.expanduser(identity_files[0])


def _connect_ssh(
    host_alias: str,
    ssh_config: paramiko.SSHConfig,
    opened_clients: list[paramiko.SSHClient],
    sock=None,
) -> paramiko.SSHClient:
    host_config = ssh_config.lookup(host_alias)

    hostname = host_config.get("hostname", host_alias)
    port = int(host_config.get("port", 22))
    username = host_config.get("user")
    key_filename = _get_first_identity_file(host_config)

    # For ProxyCommand support
    proxy_command = host_config.get("proxycommand")
    if proxy_command and proxy_command.lower() != "none":
        sock = paramiko.ProxyCommand(proxy_command)

    # For ProxyJump support
    proxy_jump = host_config.get("proxyjump")
    if proxy_jump and proxy_jump.lower() != "none":
        jump_hosts = [h.strip() for h in proxy_jump.split(",") if h.strip()]

        current_sock = sock
        for jump_host in jump_hosts:
            jump_client = _connect_ssh(
                jump_host,
                ssh_config=ssh_config,
                opened_clients=opened_clients,
                sock=current_sock,
            )
            opened_clients.append(jump_client)

            jump_transport = jump_client.get_transport()
            if jump_transport is None:
                raise RuntimeError(f"Failed to get transport for jump host: {jump_host}")

            current_sock = jump_transport.open_channel(
                kind="direct-tcpip",
                dest_addr=(hostname, port),
                src_addr=("127.0.0.1", 0),
            )

        sock = current_sock

    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    client.connect(
        hostname=hostname,
        port=port,
        username=username,
        key_filename=key_filename,
        sock=sock,
    )

    return client


@contextlib.contextmanager
def _open_ssh_with_config(host_alias: str, ssh_config: paramiko.SSHConfig,) -> typing.Iterator[paramiko.SSHClient]:
    opened_clients: list[paramiko.SSHClient] = []
    main_client: paramiko.SSHClient | None = None

    try:
        main_client = _connect_ssh(host_alias=host_alias, ssh_config=ssh_config, opened_clients=opened_clients)
        yield main_client
    finally:
        if main_client is not None:
            main_client.close()

        for client in reversed(opened_clients):
            client.close()


def _is_url(obj: Any, allow_file_urls: bool = False) -> bool:
    """Determine whether the given object is a valid URL string."""
    if not isinstance(obj, str) or not "://" in obj:
        return False
    if allow_file_urls and obj.startswith('file://'):
        return True
    try:
        res = urllib.parse.urlparse(obj)
        if not res.scheme or not res.netloc or not "." in res.netloc:
            return False
        res = urllib.parse.urlparse(urllib.parse.urljoin(obj, "/"))
        if not res.scheme or not res.netloc or not "." in res.netloc:
            return False
    except:
        return False
    return True


def _get_first_cache_file(cache_dir: str, prefix: str) -> Optional[str]:
    cache_files = glob.glob(os.path.join(cache_dir, prefix + "_*"))
    if len(cache_files) == 0:
        return None
    if len(cache_files) > 1:
        print(f"Warning: multiple cache files found for prefix '{prefix}': {cache_files}. Using the first one.")
    return cache_files[0]


def _open_ssh_file(
    url: str,
    *,
    ssh_config: paramiko.SSHConfig,
    verbose: bool,
    cache_dir: str,
    return_filename: bool,
    cache: bool,
):
    host_name, remote_path = _split_ssh_path(url)
    remote_path_quoted = shlex.quote(remote_path)

    with _open_ssh_with_config(host_name, ssh_config) as ssh:
        _, stdout, _ = ssh.exec_command(f"test -f {remote_path_quoted} && echo exists || echo 'not found'")
        if stdout.read().decode().strip() != "exists":
            raise FileNotFoundError(f"File not found on remote host '{host_name}': {remote_path}")

        _, stdout, _ = ssh.exec_command(f"md5sum {remote_path_quoted} | awk '{{print $1}}'")
        remote_hash = stdout.read().decode().strip()

        if not remote_hash:
            raise RuntimeError(f"Failed to retrieve hash of remote file: {remote_path}")

        if cache:
            cached = _open_cached_file_or_none(
                cache_dir=cache_dir,
                cache_key=remote_hash,
                return_filename=return_filename,
            )
            if cached is not None:
                return cached

        basename = os.path.basename(remote_path)
        cache_file = os.path.join(cache_dir, f"{remote_hash}_{basename}")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_file = os.path.join(tmpdir, f"{remote_hash}_{basename}")

            with tqdm.tqdm(
                ascii=True,
                unit="b",
                unit_scale=True,
                desc=f"Downloading {basename}",
                disable=not verbose,
            ) as pbar:
                def progress_callback(transferred, total):
                    pbar.total = total
                    pbar.update(transferred - pbar.n)

                with ssh.open_sftp() as sftp:
                    sftp.get(remote_path, tmp_file, callback=progress_callback)

            local_hash = _md5_file(tmp_file)
            if local_hash != remote_hash:
                raise ValueError(f"Hash mismatch: expected {remote_hash}, got {local_hash}")

            if cache:
                os.makedirs(cache_dir, exist_ok=True)
                os.replace(tmp_file, cache_file)
                return cache_file if return_filename else open(cache_file, "rb")

            return tmp_file if return_filename else open(tmp_file, "rb")


def _open_http_url_direct(
    url: str,
    *,
    cache_dir: str,
    num_attempts: int,
    verbose: bool,
    return_filename: bool,
    cache: bool,
):
    url_md5 = hashlib.md5(url.encode("utf-8")).hexdigest()

    if cache:
        cached = _open_cached_file_or_none(cache_dir=cache_dir, cache_key=url_md5, return_filename=return_filename)
        if cached is not None:
            return cached

    url_name = None
    url_data = None

    with requests.Session() as session:
        if verbose:
            print(f"Downloading {url} ...", end="", flush=True)

        for attempts_left in reversed(range(num_attempts)):
            try:
                with session.get(url) as res:
                    res.raise_for_status()

                    if len(res.content) == 0:
                        raise IOError("No data received")

                    if len(res.content) < 8192:
                        content_str = res.content.decode("utf-8", errors="replace")

                        if "download_warning" in res.headers.get("Set-Cookie", ""):
                            links = [html.unescape(link) for link in content_str.split('"') if "export=download" in link]
                            if len(links) == 1:
                                url = urllib.parse.urljoin(url, links[0])
                                raise IOError("Google Drive virus checker nag")

                        if "Google Drive - Quota exceeded" in content_str:
                            raise IOError("Google Drive download quota exceeded -- please try again later")

                    match = re.search(r'filename="([^"]*)"', res.headers.get("Content-Disposition", ""))
                    url_name = match[1] if match else url
                    url_data = res.content

                    if verbose:
                        print(" done")
                    break

            except KeyboardInterrupt:
                raise
            except Exception:
                if not attempts_left:
                    if verbose:
                        print(" failed")
                    raise
                if verbose:
                    print(".", end="", flush=True)

    assert url_data is not None

    if cache:
        assert url_name is not None
        safe_name = _safe_cache_name(url_name)
        cache_file = os.path.join(cache_dir, f"{url_md5}_{safe_name}")
        temp_file = os.path.join(cache_dir, f"tmp_{uuid.uuid4().hex}_{url_md5}_{safe_name}",)

        os.makedirs(cache_dir, exist_ok=True)
        with open(temp_file, "wb") as f:
            f.write(url_data)
        os.replace(temp_file, cache_file)

        if return_filename:
            return cache_file

    assert not return_filename
    return io.BytesIO(url_data)


def _open_http_url_gdrive(
    url: str,
    *,
    cache_dir: str,
    verbose: bool,
    return_filename: bool,
    cache: bool,
):
    url_md5 = hashlib.md5(url.encode("utf-8")).hexdigest()

    if cache:
        cached = _open_cached_file_or_none(cache_dir=cache_dir, cache_key=url_md5, return_filename=return_filename)
        if cached is not None:
            return cached

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_file = os.path.join(tmpdir, f"{uuid.uuid4().hex}_{url_md5}.tmp")

        try:
            with tqdm.tqdm(
                ascii=True,
                unit="b",
                unit_scale=True,
                desc=f"Downloading from Google Drive",
                disable=not verbose,
            ) as pbar:
                def progress_callback(transferred, total):
                    pbar.total = total
                    pbar.update(transferred - pbar.n)
                gdown.download(url, temp_file, quiet=True, use_cookies=True, progress=progress_callback)
        except Exception as e:
            raise IOError(f"Failed to download Google Drive URL: {url}") from e

        if not os.path.isfile(temp_file):
            raise IOError(f"Failed to download Google Drive URL: {url}")

        with open(temp_file, "rb") as f:
            url_data = f.read()

        if cache:
            url_name = os.path.basename(urllib.parse.urlparse(url).path)
            safe_name = _safe_cache_name(url_name)
            cache_file = os.path.join(cache_dir, f"{url_md5}_{safe_name}")

            os.makedirs(cache_dir, exist_ok=True)
            os.replace(temp_file, cache_file)

            if return_filename:
                return cache_file

        assert not return_filename
        return io.BytesIO(url_data)


def _open_http_url(
    url: str,
    *,
    cache_dir: str,
    num_attempts: int,
    verbose: bool,
    return_filename: bool,
    cache: bool,
):
    if gdown.parse_url.is_google_drive_url(url):
        return _open_http_url_gdrive(
            url,
            cache_dir=cache_dir,
            verbose=verbose,
            return_filename=return_filename,
            cache=cache,
        )

    return _open_http_url_direct(
        url,
        cache_dir=cache_dir,
        num_attempts=num_attempts,
        verbose=verbose,
        return_filename=return_filename,
        cache=cache,
    )

# Note on static typing: a better API would be to split 'open_url' to 'openl_url' and
# 'download_url' with separate return types (BinaryIO, str).  As the `return_filename=True`
# case is somewhat uncommon, we just pretend like this function never returns a string
# and type ignore return value for those cases.


def open_url(
    url: str,
    cache_dir: Optional[str] = None,
    num_attempts: int = 10,
    verbose: bool = True,
    return_filename: bool = False,
    cache: bool = True,
    ssh_conf_file: str = "~/.ssh/config",
) -> BinaryIO:
    """Download the given URL and return a binary-mode file object to access the data. Supports SSH paths and HTTP(S) URLs including Google Drive links. If the URL is a local file path or a file:// URL, it will be opened directly without downloading.

    Parameters
    ----------
    url : str
        The URL to download. Can be an HTTP(S) URL, an SSH path (e.g. 'myserver:/path/to/file'), a file:// URL, or a local file path.
    cache_dir : str, optional
        Directory to use for caching downloaded files. Default is None, which means using a default cache directory.
    num_attempts : int, optional
        Number of attempts to try downloading the file in case of failure. Default is 10.
    verbose : bool, optional
        Whether to print verbose output during downloading. Default is True.
    return_filename : bool, optional
        Whether to return the filename of the downloaded file instead of a file object. Default is False. Note that this option is not compatible with caching, as the cache file is managed internally and may be deleted or replaced.
    cache : bool, optional
        Whether to cache the downloaded file. Default is True. If True, the file will be saved in the cache directory and reused for future calls with the same URL. If False, the file will not be saved and will be downloaded every time. Note that caching is not compatible with return_filename, as the cache file is managed internally and may be deleted or replaced.
    ssh_conf_file : str, optional
        Path to the SSH config file to use when connecting to SSH hosts. Default is '~/.ssh/config'.

    Returns
    -------
    BinaryIO
        A binary-mode file object containing the downloaded data. Note that if `return_filename=True`, the return type is actually `str` representing the filename of the downloaded file, but this is not reflected in the type signature for simplicity.
    """

    assert num_attempts >= 1
    assert not (return_filename and not cache)

    if cache_dir is None:
        cache_dir = make_cache_dir_path("downloads")

    ssh_config = _get_ssh_config(ssh_conf_file)

    if url.startswith("file://"):
        filename = urllib.parse.urlparse(url).path
        if re.match(r"^/[a-zA-Z]:", filename):
            filename = filename[1:]
        return filename if return_filename else open(filename, "rb")  # type: ignore

    if _is_ssh_path(url, ssh_config):
        return _open_ssh_file(
            url,
            ssh_config=ssh_config,
            verbose=verbose,
            cache_dir=cache_dir,
            return_filename=return_filename,
            cache=cache,
        )

    if not re.match(r"^[a-z]+://", url):
        return url if return_filename else open(url, "rb")  # type: ignore

    assert _is_url(url)

    return _open_http_url(
        url,
        cache_dir=cache_dir,
        num_attempts=num_attempts,
        verbose=verbose,
        return_filename=return_filename,
        cache=cache,
    )

# ------------------------------------------------------------------------------------------
