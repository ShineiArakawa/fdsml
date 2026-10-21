"""Miscellaneous utilities for PyTorch.

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
import functools
import re
import time
import warnings

import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import torch

import torch_utils.font as font_util
import torch_utils.util as util

# ----------------------------------------------------------------------------
# Re-seed torch & numpy random generators based on the given arguments.


def set_random_seed(*args):
    seed = hash(args) % (1 << 31)
    torch.manual_seed(seed)
    np.random.seed(seed)

# ----------------------------------------------------------------------------
# Cached construction of constant tensors. Avoids CPU=>GPU copy when the
# same constant is used multiple times.


_constant_cache = dict()


def constant(value, shape=None, dtype=None, device=None, memory_format=None):
    value = np.asarray(value)
    if shape is not None:
        shape = tuple(shape)
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device('cpu')
    if memory_format is None:
        memory_format = torch.contiguous_format

    key = (value.shape, value.dtype, value.tobytes(), shape, dtype, device, memory_format)
    tensor = _constant_cache.get(key, None)
    if tensor is None:
        tensor = torch.as_tensor(value.copy(), dtype=dtype, device=device)
        if shape is not None:
            tensor, _ = torch.broadcast_tensors(tensor, torch.empty(shape))
        tensor = tensor.contiguous(memory_format=memory_format)
        _constant_cache[key] = tensor
    return tensor

# ----------------------------------------------------------------------------
# Variant of constant() that inherits dtype and device from the given
# reference tensor by default.


def const_like(ref, value, shape=None, dtype=None, device=None, memory_format=None):
    if dtype is None:
        dtype = ref.dtype
    if device is None:
        device = ref.device
    return constant(value, shape=shape, dtype=dtype, device=device, memory_format=memory_format)

# ----------------------------------------------------------------------------
# Cached construction of temporary tensors in pinned CPU memory.


@functools.lru_cache(None)
def pinned_buf(shape, dtype):
    return torch.empty(shape, dtype=dtype).pin_memory()

# ----------------------------------------------------------------------------
# Symbolic assert.


try:
    symbolic_assert = torch._assert  # 1.8.0a0 # pylint: disable=protected-access
except AttributeError:
    symbolic_assert = torch.Assert  # 1.7.0

# ----------------------------------------------------------------------------
# Context manager to temporarily suppress known warnings in torch.jit.trace().
# Note: Cannot use catch_warnings because of https://bugs.python.org/issue29672


@contextlib.contextmanager
def suppress_tracer_warnings():
    flt = ('ignore', None, torch.jit.TracerWarning, None, 0)
    warnings.filters.insert(0, flt)
    yield
    warnings.filters.remove(flt)

# ----------------------------------------------------------------------------
# Assert that the shape of a tensor matches the given list of integers.
# None indicates that the size of a dimension is allowed to vary.
# Performs symbolic assertion when used in torch.jit.trace().


def assert_shape(tensor, ref_shape):
    if tensor.ndim != len(ref_shape):
        raise AssertionError(f'Wrong number of dimensions: got {tensor.ndim}, expected {len(ref_shape)}')
    for idx, (size, ref_size) in enumerate(zip(tensor.shape, ref_shape)):
        if ref_size is None:
            pass
        elif isinstance(ref_size, torch.Tensor):
            with suppress_tracer_warnings():  # as_tensor results are registered as constants
                symbolic_assert(torch.equal(torch.as_tensor(size), ref_size), f'Wrong size for dimension {idx}')
        elif isinstance(size, torch.Tensor):
            with suppress_tracer_warnings():  # as_tensor results are registered as constants
                symbolic_assert(torch.equal(size, torch.as_tensor(ref_size)), f'Wrong size for dimension {idx}: expected {ref_size}')
        elif size != ref_size:
            raise AssertionError(f'Wrong size for dimension {idx}: got {size}, expected {ref_size}')

# ----------------------------------------------------------------------------
# Function decorator that calls torch.autograd.profiler.record_function().


def profiled_function(fn):
    def decorator(*args, **kwargs):
        with torch.autograd.profiler.record_function(fn.__name__):
            return fn(*args, **kwargs)
    decorator.__name__ = fn.__name__
    return decorator

# ----------------------------------------------------------------------------
# Sampler for torch.utils.data.DataLoader that loops over the dataset
# indefinitely, shuffling items as it goes.


class InfiniteSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, rank=0, num_replicas=1, shuffle=True, seed=0, start_idx=0):
        assert len(dataset) > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        warnings.filterwarnings('ignore', '`data_source` argument is not used and will be removed')
        super().__init__()
        self.dataset_size = len(dataset)
        self.start_idx = start_idx + rank
        self.stride = num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        idx = self.start_idx
        epoch = None
        while True:
            if epoch != idx // self.dataset_size:
                epoch = idx // self.dataset_size
                order = np.arange(self.dataset_size)
                if self.shuffle:
                    np.random.RandomState(hash((self.seed, epoch)) % (1 << 31)).shuffle(order)
            yield int(order[idx % self.dataset_size])
            idx += self.stride


# ----------------------------------------------------------------------------
# Utilities for operating with torch.nn.Module parameters and buffers.


def params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.parameters()) + list(module.buffers())


def named_params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.named_parameters()) + list(module.named_buffers())


@torch.no_grad()
def copy_params_and_buffers(src_module, dst_module, require_all=False):
    assert isinstance(src_module, torch.nn.Module)
    assert isinstance(dst_module, torch.nn.Module)
    src_tensors = dict(named_params_and_buffers(src_module))
    for name, tensor in named_params_and_buffers(dst_module):
        assert (name in src_tensors) or (not require_all)
        if name in src_tensors:
            tensor.copy_(src_tensors[name])

# ----------------------------------------------------------------------------
# Context manager for easily enabling/disabling DistributedDataParallel
# synchronization.


@contextlib.contextmanager
def ddp_sync(module, sync):
    assert isinstance(module, torch.nn.Module)
    if sync or not isinstance(module, torch.nn.parallel.DistributedDataParallel):
        yield
    else:
        with module.no_sync():
            yield

# ----------------------------------------------------------------------------
# Check DistributedDataParallel consistency across processes.


def check_ddp_consistency(module, ignore_regex=None):
    assert isinstance(module, torch.nn.Module)

    if torch.distributed.is_initialized() == False:
        # Distributed training is not initialized, so consistency check is not applicable.
        return

    for name, tensor in named_params_and_buffers(module):
        fullname = type(module).__name__ + '.' + name
        if ignore_regex is not None and re.fullmatch(ignore_regex, fullname):
            continue
        tensor = tensor.detach()
        if tensor.is_floating_point():
            tensor = torch.nan_to_num(tensor)
        other = tensor.clone()
        torch.distributed.broadcast(tensor=other, src=0)
        assert (tensor == other).all(), fullname

# ----------------------------------------------------------------------------
# Print summary table of module hierarchy.


@torch.no_grad()
def print_module_summary(module, inputs, max_nesting=3, skip_redundant=True):
    assert isinstance(module, torch.nn.Module)
    assert not isinstance(module, torch.jit.ScriptModule)
    assert isinstance(inputs, (tuple, list))

    # Register hooks.
    entries = []
    nesting = [0]

    def pre_hook(_mod, _inputs):
        nesting[0] += 1

    def post_hook(mod, _inputs, outputs):
        nesting[0] -= 1
        if nesting[0] <= max_nesting:
            outputs = list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
            outputs = [t for t in outputs if isinstance(t, torch.Tensor)]
            entries.append(util.EasyDict(mod=mod, outputs=outputs))
    hooks = [mod.register_forward_pre_hook(pre_hook) for mod in module.modules()]
    hooks += [mod.register_forward_hook(post_hook) for mod in module.modules()]

    # Run module.
    outputs = module(*inputs)
    for hook in hooks:
        hook.remove()

    # Identify unique outputs, parameters, and buffers.
    tensors_seen = set()
    for e in entries:
        e.unique_params = [t for t in e.mod.parameters() if id(t) not in tensors_seen]
        e.unique_buffers = [t for t in e.mod.buffers() if id(t) not in tensors_seen]
        e.unique_outputs = [t for t in e.outputs if id(t) not in tensors_seen]
        tensors_seen |= {id(t) for t in e.unique_params + e.unique_buffers + e.unique_outputs}

    # Filter out redundant entries.
    if skip_redundant:
        entries = [e for e in entries if len(e.unique_params) or len(e.unique_buffers) or len(e.unique_outputs)]

    # Construct table.
    rows = [[type(module).__name__, 'Parameters', 'Buffers', 'Output shape', 'Datatype']]
    rows += [['---'] * len(rows[0])]
    param_total = 0
    buffer_total = 0
    submodule_names = {mod: name for name, mod in module.named_modules()}
    for e in entries:
        name = '<top-level>' if e.mod is module else submodule_names[e.mod]
        param_size = sum(t.numel() for t in e.unique_params)
        buffer_size = sum(t.numel() for t in e.unique_buffers)
        output_shapes = [str(list(t.shape)) for t in e.outputs]
        output_dtypes = [str(t.dtype).split('.')[-1] for t in e.outputs]
        rows += [[
            name + (':0' if len(e.outputs) >= 2 else ''),
            str(param_size) if param_size else '-',
            str(buffer_size) if buffer_size else '-',
            (output_shapes + ['-'])[0],
            (output_dtypes + ['-'])[0],
        ]]
        for idx in range(1, len(e.outputs)):
            rows += [[name + f':{idx}', '-', '-', output_shapes[idx], output_dtypes[idx]]]
        param_total += param_size
        buffer_total += buffer_size
    rows += [['---'] * len(rows[0])]
    rows += [['Total', str(param_total), str(buffer_total), '-', '-']]

    # Print table.
    widths = [max(len(cell) for cell in column) for column in zip(*rows)]
    print()
    for row in rows:
        print('  '.join(cell + ' ' * (width - len(cell)) for cell, width in zip(row, widths)))
    print()

# ----------------------------------------------------------------------------
# Tile a batch of images into a 2D grid.


def tile_images(x, w, h):
    assert x.ndim == 4  # NCHW => CHW
    return x.reshape(h, w, *x.shape[1:]).permute(2, 0, 3, 1, 4).reshape(x.shape[1], h * x.shape[2], w * x.shape[3])

# ----------------------------------------------------------------------------
# Create a grid image for visualization.


def create_grid(imgs_A: torch.Tensor, imgs_B: torch.Tensor, label: torch.Tensor, pred: torch.Tensor | None = None, color: tuple[int, int, int] = (230, 15, 115), mag_factor: int = 2, padding: int = 4, txt_prefix: str = '') -> np.ndarray:
    """Create a grid image for visualization.

    Parameters
    ----------
    imgs_A : torch.Tensor
        A batch of images A. Shape: (B, C, H, W), values in [0, 1]
    imgs_B : torch.Tensor
        A batch of images B. Shape: (B, C, H, W), values in [0, 1]
    label : torch.Tensor–
        A batch of labels. Shape: (B,)
    pred : torch.Tensor | None
        A batch of predictions. Shape: (B,). If provided, will be ignored in this function.
    color : tuple[int, int, int], optional
        Text color in RGB format, by default (230, 15, 115)
    mag_factor : int, optional
        Nearest neighbor upscaling factor for each image, by default 2
    padding : int, optional
        Padding size between images in the grid, by default 4
    txt_prefix : str, optional
        Prefix string for the label text, by default ''

    Returns
    -------
    grid_img : np.ndarray
        The grid image as a numpy array. Shape: (H_grid, W_grid, C)
    """

    assert imgs_A.shape == imgs_B.shape, f'imgs_A and imgs_B must have the same shape, but got {imgs_A.shape} and {imgs_B.shape}'

    batch_size, channels, height, width = imgs_A.size()
    mag_height = height * mag_factor
    mag_width = width * mag_factor

    grid_cell_width = mag_width * 2 + padding * 3
    grid_cell_height = mag_height + padding * 2

    img_A_bhwc = imgs_A.permute(0, 2, 3, 1)  # (B, H, W, C)
    img_B_bhwc = imgs_B.permute(0, 2, 3, 1)  # (B, H, W, C)

    imgs: list[PIL.Image.Image] = []
    for i_img in range(batch_size):
        img_A_np = img_A_bhwc[i_img].cpu().detach().numpy()
        img_B_np = img_B_bhwc[i_img].cpu().detach().numpy()

        img_A_uint8 = (img_A_np * 255).astype(np.uint8)
        img_B_uint8 = (img_B_np * 255).astype(np.uint8)

        img_A_pil = PIL.Image.fromarray(img_A_uint8)
        img_B_pil = PIL.Image.fromarray(img_B_uint8)

        img_A_pil = img_A_pil.resize((mag_width, mag_height), resample=PIL.Image.Resampling.NEAREST)
        img_B_pil = img_B_pil.resize((mag_width, mag_height), resample=PIL.Image.Resampling.NEAREST)

        img_pil = PIL.Image.new('RGB', (grid_cell_width, grid_cell_height), (0, 0, 0))

        img_pil.paste(img_A_pil, (padding, padding))
        img_pil.paste(img_B_pil, (mag_width + padding * 2, padding))

        imgs.append(img_pil)

    # Paint label values
    for idx in range(batch_size):
        img = imgs[idx]
        label_value = label[idx].item()
        pred_value = pred[idx].item() if pred is not None else None

        img_width, _ = img.size

        draw = PIL.ImageDraw.Draw(img)
        font_size = int(height * mag_factor * 0.08)
        # cur_font = PIL.ImageFont.truetype(font_util.DEFAULT_FONT_GOTHIC_PATH, size=font_size)
        try:
            cur_font = PIL.ImageFont.truetype(font_util.DEFAULT_FONT_GOTHIC_PATH, size=font_size)
        except:
            cur_font = PIL.ImageFont.load_default()

        if pred_value is not None:
            txt = f'{txt_prefix}GT:{label_value:.4f} / Pred:{pred_value:.4f}'
        else:
            txt = f'{txt_prefix}{label_value:.4f}'

        txt_width = draw.textlength(txt, font=cur_font)

        anchor_x = max((img_width - txt_width) // 2, 0.0)
        anchor_y = height * 0.05

        draw.text((anchor_x, anchor_y), txt, fill=color, font=cur_font)

    # Create grid
    grid_W = int(np.ceil(np.sqrt(batch_size)))
    grid_H = int(np.ceil(batch_size / grid_W))

    grid_width = grid_W * grid_cell_width + padding * (grid_W + 1)
    grid_height = grid_H * grid_cell_height + padding * (grid_H + 1)

    grid_img = PIL.Image.new('RGB', (grid_width, grid_height))
    for idx, img in enumerate(imgs):
        w = idx % grid_W
        h = idx // grid_W

        grid_img.paste(
            img,
            (
                w * grid_cell_width + padding * (w + 1),
                h * grid_cell_height + padding * (h + 1),
            )
        )

    return np.array(grid_img)


def create_triplet_grid(imgs: torch.Tensor, meta: dict | None = None, bg_color: tuple[int, int, int] = (255, 255, 255), txt_color: tuple[int, int, int] = (230, 15, 115), mag_factor: int = 2, padding: int = 4) -> np.ndarray:
    """Create a grid image for visualization.

    Parameters
    ----------
    imgs: torch.Tensor
        A batch of interleaved triplet images. Shape: (B*3, C, H, W), values in [0, 1]
    bg_color : tuple[int, int, int], optional
        Background color in RGB format, by default (255, 255, 255)
    txt_color : tuple[int, int, int], optional
        Text color in RGB format, by default (230, 15, 115)
    mag_factor : int, optional
        Nearest neighbor upscaling factor for each image, by default 2
    padding : int, optional
        Padding size between images in the grid, by default 4

    Returns
    -------
    grid_img : np.ndarray
        The grid image as a numpy array. Shape: (H_grid, W_grid, C)
    """

    batch_size, channels, height, width = imgs.size()
    assert batch_size % 3 == 0, f'Batch size must be a multiple of 3, but got {batch_size}'
    batch_size = batch_size // 3
    mag_height = height * mag_factor
    mag_width = width * mag_factor

    grid_cell_width = mag_width * 3 + padding * 4
    grid_cell_height = mag_height + padding * 2

    imgs_bhwc = imgs.permute(0, 2, 3, 1)  # (B*3, H, W, C)

    imgs: list[PIL.Image.Image] = []
    for i_img in range(batch_size):
        img_anc = imgs_bhwc[i_img * 3 + 0].detach().cpu().numpy()
        img_pos = imgs_bhwc[i_img * 3 + 1].detach().cpu().numpy()
        img_neg = imgs_bhwc[i_img * 3 + 2].detach().cpu().numpy()

        img_anc = (img_anc * 255).astype(np.uint8)
        img_pos = (img_pos * 255).astype(np.uint8)
        img_neg = (img_neg * 255).astype(np.uint8)

        img_anc = PIL.Image.fromarray(img_anc)
        img_pos = PIL.Image.fromarray(img_pos)
        img_neg = PIL.Image.fromarray(img_neg)

        img_anc = img_anc.resize((mag_width, mag_height), resample=PIL.Image.Resampling.NEAREST)
        img_pos = img_pos.resize((mag_width, mag_height), resample=PIL.Image.Resampling.NEAREST)
        img_neg = img_neg.resize((mag_width, mag_height), resample=PIL.Image.Resampling.NEAREST)

        img_pil = PIL.Image.new('RGB', (grid_cell_width, grid_cell_height), (0, 0, 0))

        img_pil.paste(img_anc, (padding, padding))
        img_pil.paste(img_pos, (mag_width + padding * 2, padding))
        img_pil.paste(img_neg, (mag_width * 2 + padding * 3, padding))

        imgs.append(img_pil)

    # Paint label values
    for idx in range(batch_size):
        img = imgs[idx]

        draw = PIL.ImageDraw.Draw(img)
        font_size = int(mag_height * 0.08)
        try:
            cur_font = PIL.ImageFont.truetype(font_util.DEFAULT_FONT_GOTHIC_PATH, size=font_size)
        except:
            cur_font = PIL.ImageFont.load_default()

        csv_idx = meta["csv_idx"][idx].item() if "csv_idx" in meta else None
        dim = meta["dim"][idx].item()

        signed_pos = meta["signed_pos"][idx].item()
        signed_neg = meta["signed_neg"][idx].item()

        abs_pos = abs(signed_pos)
        abs_neg = abs(signed_neg)

        dim_name = meta["dim_names"][dim]

        pos_color = (0, 180, 0) if abs_pos <= abs_neg else (200, 50, 50)
        neg_color = (200, 50, 50) if abs_pos <= abs_neg else (0, 180, 0)

        # Anchor
        txt = f'Anchor (CSV {csv_idx})' if csv_idx is not None else 'Anchor'
        txt_width = draw.textlength(txt, font=cur_font)
        anchor_x = max(padding, (mag_width - txt_width) // 2 + padding)
        anchor_y = mag_height * 0.02
        draw.text((anchor_x, anchor_y), txt, fill=txt_color, font=cur_font)

        # Positive
        txt = f'Positive ({dim_name}, {signed_pos:+.3f})'
        txt_width = draw.textlength(txt, font=cur_font)
        pos_x = max(mag_width + padding * 2,
                    (mag_width - txt_width) // 2 + mag_width + padding * 2)
        pos_y = mag_height * 0.02
        draw.text((pos_x, pos_y), txt, fill=pos_color, font=cur_font)

        # Negative
        txt = f'Negative ({dim_name}, {signed_neg:+.3f})'
        txt_width = draw.textlength(txt, font=cur_font)
        neg_x = max(mag_width * 2 + padding * 3,
                    (mag_width - txt_width) // 2 + mag_width * 2 + padding * 3)
        neg_y = mag_height * 0.02
        draw.text((neg_x, neg_y), txt, fill=neg_color, font=cur_font)

    # Create grid
    grid_W = int(np.ceil(np.sqrt(batch_size)))
    grid_H = int(np.ceil(batch_size / grid_W))

    grid_width = grid_W * grid_cell_width + padding * (grid_W + 1)
    grid_height = grid_H * grid_cell_height + padding * (grid_H + 1)

    grid_img = PIL.Image.new('RGB', (grid_width, grid_height), color=bg_color)
    for idx, img in enumerate(imgs):
        w = idx % grid_W
        h = idx // grid_W

        grid_img.paste(
            img,
            (
                w * grid_cell_width + padding * (w + 1),
                h * grid_cell_height + padding * (h + 1),
            )
        )

    return np.array(grid_img)


# ----------------------------------------------------------------------------
# Replace batch normalization layers with sync batch normalization layers.


def convert_sync_batchnorm_module(module: torch.nn.Module) -> torch.nn.Module:
    assert isinstance(module, torch.nn.Module)

    if not torch.distributed.is_initialized():
        # Distributed training is not initialized, so SyncBatchNorm conversion is not applicable.
        print('Warning: torch.distributed is not initialized. Skipping SyncBatchNorm conversion.')
        return module

    return torch.nn.SyncBatchNorm.convert_sync_batchnorm(module)

# ----------------------------------------------------------------------------
# Perlin noise implementation (NumPy version)


PERM = np.array([
    151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140,
    36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148, 247, 120,
    234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32, 57, 177, 33,
    88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175, 74, 165, 71,
    134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122, 60, 211, 133,
    230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54, 65, 25, 63, 161,
    1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169, 200, 196, 135, 130,
    116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64, 52, 217, 226, 250,
    124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212, 207, 206, 59, 227,
    47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213, 119, 248, 152, 2, 44,
    154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9, 129, 22, 39, 253, 19, 98,
    108, 110, 79, 113, 224, 232, 178, 185, 112, 104, 218, 246, 97, 228, 251, 34,
    242, 193, 238, 210, 144, 12, 191, 179, 162, 241, 81, 51, 145, 235, 249, 14,
    239, 107, 49, 192, 214, 31, 181, 199, 106, 157, 184, 84, 204, 176, 115, 121,
    50, 45, 127, 4, 150, 254, 138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243,
    141, 128, 195, 78, 66, 215, 61, 156, 180, 151, 160, 137, 91, 90, 15, 131,
    13, 201, 95, 96, 53, 194, 233, 7, 225, 140, 36, 103, 30, 69, 142, 8, 99, 37,
    240, 21, 10, 23, 190, 6, 148, 247, 120, 234, 75, 0, 26, 197, 62, 94, 252,
    219, 203, 117, 35, 11, 32, 57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125,
    136, 171, 168, 68, 175, 74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158,
    231, 83, 111, 229, 122, 60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245,
    40, 244, 102, 143, 54, 65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187,
    208, 89, 18, 169, 200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198,
    173, 186, 3, 64, 52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126,
    255, 82, 85, 212, 207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223,
    183, 170, 213, 119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167,
    43, 172, 9, 129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185,
    112, 104, 218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179,
    162, 241, 81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199,
    106, 157, 184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236,
    205, 93, 222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156,
    180
], dtype=np.uint8)


def grad1(hash: np.ndarray, x: np.ndarray) -> np.ndarray:
    g = (hash & 7).astype(np.float32) + np.float32(1.0)
    g = np.where((hash & 8) != 0, np.float32(-1.0), g)  # I think there is a mistake here, but keeping it for compatibility
    return g * x


def noise1_np(x: np.ndarray, repeat: int = 1024, base: int = 0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)

    i = np.floor(x).astype(np.int32) % repeat
    ii = (i + 1) % repeat
    i = (i & 255) + base
    ii = (ii & 255) + base

    xf = x - np.floor(x)  # frac
    fx = xf * xf * xf * (xf * (xf * np.float32(6.0) - np.float32(15.0)) + np.float32(10.0))

    a = grad1(PERM[i].astype(np.int32), xf)
    b = grad1(PERM[ii].astype(np.int32), xf - np.float32(1.0))

    return (a + fx * (b - a)) * np.float32(0.4)


# ----------------------------------------------------------------------------
# Timer

class Timer:
    def __init__(self, sync: bool = True):
        """Context manager for timing code execution, with optional synchronization for CUDA operations.

        Parameters
        ----------
        sync : bool, optional
            Whether to synchronize CUDA operations, by default True
        """

        self._sync = sync
        self._elapsed_times: dict[str, list[float]] = dict()

    def _sync_if_needed(self):
        if self._sync and torch.cuda.is_available():
            torch.cuda.synchronize()

    def clear(self, name: str | None = None) -> None:
        """Clear recorded times.

        Parameters
        ----------
        name : str | None, optional
            If provided, only clear the time for the specified name. If None, clear all times. By default None.
        """

        if name is None:
            self._elapsed_times.clear()
        elif name in self._elapsed_times:
            del self._elapsed_times[name]
        else:
            raise KeyError(f'Timer name not found: {name}')

    def reset(self, name: str | None = None) -> None:
        """Reset the time for the specified name or all names.

        Parameters
        ----------
        name : str | None, optional
            If provided, only reset the time for the specified name. If None, reset all times. By default None.
        """

        if name is None:
            for key in self._elapsed_times.keys():
                self._elapsed_times[key] = []
        elif name in self._elapsed_times:
            self._elapsed_times[name] = []
        else:
            raise KeyError(f'Timer name not found: {name}')

    def get_time_sec(self, name: str) -> float:
        """Get the elapsed time for the specified name.

        Parameters
        ----------
        name : str
            The name of the timer.

        Returns
        -------
        float
            The elapsed time in seconds.
        """

        if name in self._elapsed_times:
            return sum(self._elapsed_times[name])
        else:
            raise KeyError(f'Timer name not found: {name}')

    def get_times_sec(self) -> dict[str, float]:
        """Get all recorded times.

        Returns
        -------
        dict[str, float]
            A dictionary mapping timer names to their elapsed times in seconds.
        """

        return {name: sum(times) for name, times in self._elapsed_times.items()}

    def print_times(self) -> None:
        """Print all recorded times."""

        for name, time_sec in self._elapsed_times.items():
            print(f"[Timer] {name} | total: {sum(time_sec):.6f} [sec], avg: {sum(time_sec) / len(time_sec):.6f} [sec] over {len(time_sec)} runs")

    @contextlib.contextmanager
    def timeit(self, name: str):
        """Context manager for timing a block of code.

        Parameters
        ----------
        name : str
            The name of the timer to record the elapsed time under.
        """

        self._sync_if_needed()

        start_time = time.perf_counter()
        yield

        self._sync_if_needed()
        elapsed_time = time.perf_counter() - start_time

        if name not in self._elapsed_times:
            self._elapsed_times[name] = []

        self._elapsed_times[name].append(elapsed_time)


# ----------------------------------------------------------------------------
