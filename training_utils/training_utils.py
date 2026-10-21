"""Utility functions for training metric learning models.

License
-------

Copyright (c) 2026 Shinei Arakawa
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
"""

import copy
import hashlib
import json
import os
import pathlib
import random
import re
import shutil
import typing

import asciichartpy
import numpy as np
import timm.models.vision_transformer
import torch

import models.resnet as resnet
import models.vit as vit
import torch_utils.distributed as dist
import torch_utils.util as util

# ---------------------------------------------------------------------------------------------------------------------------
# Model selection


model_catalog: typing.Final[list[str]] = [
    # ResNet
    'resnet18',
    'resnet18_imagenet_pretrained',
    'resnet18_fdsl_pretrained',
    'resnet34',
    'resnet34_imagenet_pretrained',
    'resnet34_fdsl_pretrained',
    'resnet50',
    'resnet50_imagenet_pretrained',
    'resnet50_fdsl_pretrained',
    'resnet50_fdsml_all_pretrained',
    'resnet50_fdsml_each_pretrained',
    'resnet50_fdsml_same_pretrained',
    'resnet50_fdsml_diff_pretrained',
    # DeiT-Tiny
    'deit_tiny_patch16_224',
    'deit_tiny_patch16_224_imagenet_pretrained',
    'deit_tiny_patch16_224_fdsl_pretrained',
    'deit_tiny_patch16_224_mocov3',
    'deit_tiny_patch16_224_mocov3_fdsml_all_pretrained',
    'deit_tiny_patch16_224_mocov3_fdsml_each_pretrained',
    'deit_tiny_patch16_224_mocov3_fdsml_same_pretrained',
    'deit_tiny_patch16_224_mocov3_fdsml_diff_pretrained',
    # DeiT-Base
    'deit_base_patch16_224',
    'deit_base_patch16_224_imagenet_pretrained',
    'deit_base_patch16_224_fdsl_pretrained',
    'deit_base_patch16_224_mocov3',
    'deit_base_patch16_224_mocov3_fdsml_all_pretrained',
    'deit_base_patch16_224_mocov3_fdsml_each_pretrained',
    'deit_base_patch16_224_mocov3_fdsml_same_pretrained',
    'deit_base_patch16_224_mocov3_fdsml_diff_pretrained',
]

_HF_ORG: typing.Final[str] = 'fdsml'


def _hf_url(repo: str, filename: str, revision: str = 'main') -> str:
    """Build a direct download URL for a file in a Hugging Face model repository.

    Parameters
    ----------
    repo : str
        Repository name under the ``fdsml`` organization (e.g., ``'fdsml-finefractaldb-resnet50-same'``).
    filename : str
        Path of the file inside the repository.
    revision : str, optional
        Git revision (branch, tag, or commit hash). Defaults to ``'main'``.

    Returns
    -------
    str
        URL that can be passed to :func:`torch_utils.util.open_url`.
    """

    return f'https://huggingface.co/{_HF_ORG}/{repo}/resolve/{revision}/{filename}'


_pretrained_weights: typing.Final[dict[str, (int, str, str)]] = {
    # autopep8: off
    # ResNet
    'resnet18_fdsl_pretrained'                              : util.EasyDict(proj_dim=1000, url='https://drive.google.com/file/d/1YP64WPXiqhyft8uuexwhNetC5ObL3hxY/view?usp=drive_link'              , sub_name=None   ),
    'resnet34_fdsl_pretrained'                              : util.EasyDict(proj_dim=1000, url='https://drive.google.com/file/d/1rxMBvuH-JpxLdW3g25fFVcz_ZhPGZYhw/view?usp=drive_link'              , sub_name=None   ),
    'resnet50_fdsl_pretrained'                              : util.EasyDict(proj_dim=1000, url='https://drive.google.com/file/d/1BBayI7aedcxlgzCPtZtJOsu-GajB0zUx/view?usp=drive_link'              , sub_name=None   ),
    'resnet50_fdsml_all_pretrained'                         : util.EasyDict(proj_dim=512 , url=_hf_url('fdsml-finefractaldb-resnet50-all' , 'checkpoint-134217kimg-134.22ep_fdsml_resnet50_all.pt' ), sub_name=None   ),
    'resnet50_fdsml_each_pretrained'                        : util.EasyDict(proj_dim=512 , url=_hf_url('fdsml-finefractaldb-resnet50-each', 'checkpoint-134217kimg-134.22ep_fdsml_resnet50_each.pt'), sub_name=None   ),
    'resnet50_fdsml_same_pretrained'                        : util.EasyDict(proj_dim=512 , url=_hf_url('fdsml-finefractaldb-resnet50-same', 'checkpoint-134217kimg-134.22ep_fdsml_resnet50_same.pt'), sub_name=None   ),
    'resnet50_fdsml_diff_pretrained'                        : util.EasyDict(proj_dim=512 , url=_hf_url('fdsml-finefractaldb-resnet50-diff', 'checkpoint-134217kimg-134.22ep_fdsml_resnet50_diff.pt'), sub_name=None   ),
    # DeiT-Tiny
    'deit_tiny_patch16_224_fdsl_pretrained'                 : util.EasyDict(proj_dim=1000, url='https://drive.google.com/file/d/1lzysNlVUQiH5sl1VZ45EDnGu3AbsuNS5/view?usp=drive_link'              , sub_name='model'),
    'deit_tiny_patch16_224_mocov3_fdsml_all_pretrained'     : util.EasyDict(proj_dim=128 , url=_hf_url('fdsml-finefractaldb-vittiny-all' , 'checkpoint-536870kimg-536.87ep_fdsml_deittiny_all.pt' ) , sub_name=None   ),
    'deit_tiny_patch16_224_mocov3_fdsml_each_pretrained'    : util.EasyDict(proj_dim=128 , url=_hf_url('fdsml-finefractaldb-vittiny-each', 'checkpoint-536870kimg-536.87ep_fdsml_deittiny_each.pt') , sub_name=None   ),
    'deit_tiny_patch16_224_mocov3_fdsml_same_pretrained'    : util.EasyDict(proj_dim=128 , url=_hf_url('fdsml-finefractaldb-vittiny-same', 'checkpoint-536870kimg-536.87ep_fdsml_deittiny_same.pt') , sub_name=None   ),
    'deit_tiny_patch16_224_mocov3_fdsml_diff_pretrained'    : util.EasyDict(proj_dim=128 , url=_hf_url('fdsml-finefractaldb-vittiny-diff', 'checkpoint-536870kimg-536.87ep_fdsml_deittiny_diff.pt') , sub_name=None   ),
    # DeiT-Base
    'deit_base_patch16_224_fdsl_pretrained'                 : util.EasyDict(proj_dim=1000, url='https://drive.google.com/file/d/1skH8A9vVdOwgENkbSRoslFwRFcrHdBah/view?usp=drive_link'              , sub_name='model'),
    'deit_base_patch16_224_mocov3_fdsml_all_pretrained'     : util.EasyDict(proj_dim=256 , url=_hf_url('fdsml-finefractaldb-vitbase-all' , 'checkpoint-536870kimg-536.87ep_fdsml_deitbase_all.pt' ) , sub_name=None   ),
    'deit_base_patch16_224_mocov3_fdsml_each_pretrained'    : util.EasyDict(proj_dim=256 , url=_hf_url('fdsml-finefractaldb-vitbase-each', 'checkpoint-536870kimg-536.87ep_fdsml_deitbase_each.pt') , sub_name=None   ),
    'deit_base_patch16_224_mocov3_fdsml_same_pretrained'    : util.EasyDict(proj_dim=256 , url=_hf_url('fdsml-finefractaldb-vitbase-same', 'checkpoint-536870kimg-536.87ep_fdsml_deitbase_same.pt') , sub_name=None   ),
    'deit_base_patch16_224_mocov3_fdsml_diff_pretrained'    : util.EasyDict(proj_dim=256 , url=_hf_url('fdsml-finefractaldb-vitbase-diff', 'checkpoint-524288kimg-524.29ep_fdsml_deitbase_diff.pt') , sub_name=None   ),
    # autopep8: on
}


def is_vit(net: str) -> bool:
    """Check if the given model name corresponds to a vision transformer (ViT) architecture.

    Parameters
    ----------
    net : str
        The name of the neural network architecture.

    Returns
    -------
    bool
        True if the model is a vision transformer, False otherwise.
    """

    return net.startswith("deit_tiny_patch16_224") or net.startswith("deit_small_patch16_224") or net.startswith("deit_base_patch16_224")


def _to_mocov3_vit(net: timm.models.vision_transformer.VisionTransformer, proj_dim: int) -> timm.models.vision_transformer.VisionTransformer:
    """Convert a standard ViT model to a MoCo v3 style ViT by replacing the head with a projection MLP.

    Parameters
    ----------
    net : timm.models.vision_transformer.VisionTransformer
        The original ViT model to convert.
    proj_dim : int
        The dimension of the projection head output.

    Returns
    -------
    timm.models.vision_transformer.VisionTransformer
        The modified ViT model with a MoCo v3 style projection head.
    """

    in_dim = net.head.in_features
    net.head = torch.nn.Sequential(
        torch.nn.Linear(in_dim, in_dim, bias=False),
        torch.nn.BatchNorm1d(in_dim),
        torch.nn.ReLU(inplace=True),
        torch.nn.Linear(in_dim, proj_dim, bias=False)
    )

    return net


def select_model(net: str, is_feature_extractor: bool = False, pool: str = 'token', drop_rate: float = 0., drop_path_rate: float = 0.1) -> torch.nn.Module:
    """Select and return a model based on the specified architecture.

    Parameters
    ----------
    net : str
        The name of the neural network architecture.
    is_feature_extractor : bool
        If True, return a feature extractor model.
    pool : str
        The pooling method to use for vision transformers.
    drop_rate : float
        The dropout rate to use for vision transformers.
    drop_path_rate : float
        The drop path rate to use for vision transformers.

    Returns
    -------
    torch.nn.Module
        The selected model with the appropriate projection head if specified.
    """

    assert is_feature_extractor is False or 'resnet' in net, 'Feature extractor is currently only supported for ResNet models.'

    is_imagenet_pretrained = 'imagenet_pretrained' in net
    is_fdsl_pretrained = 'fdsl' in net
    is_fdsml_pretrained = 'fdsml' in net

    if net.startswith('resnet18'):
        resnet_cls = resnet.ResNetFeatureExtractor if is_feature_extractor else resnet.ResNet

        base_out_dim = 1
        if is_imagenet_pretrained or is_fdsl_pretrained:
            base_out_dim = 1000
        elif is_fdsml_pretrained:
            base_out_dim = 128

        model = resnet.resnet18(pretrained=is_imagenet_pretrained, resnet_cls=resnet_cls, num_classes=base_out_dim)
    elif net.startswith('resnet34'):
        resnet_cls = resnet.ResNetFeatureExtractor if is_feature_extractor else resnet.ResNet

        base_out_dim = 1
        if is_imagenet_pretrained or is_fdsl_pretrained:
            base_out_dim = 1000
        elif is_fdsml_pretrained:
            base_out_dim = 128

        model = resnet.resnet34(pretrained=is_imagenet_pretrained, resnet_cls=resnet_cls, num_classes=base_out_dim)
    elif net.startswith('resnet50'):
        resnet_cls = resnet.ResNetFeatureExtractor if is_feature_extractor else resnet.ResNet

        base_out_dim = 1
        if is_imagenet_pretrained or is_fdsl_pretrained:
            base_out_dim = 1000
        elif is_fdsml_pretrained:
            base_out_dim = 128

        model = resnet.resnet50(pretrained=is_imagenet_pretrained, resnet_cls=resnet_cls, num_classes=base_out_dim)
    elif net.startswith('deit_tiny_patch16_224'):
        base_out_dim = 1000
        if is_imagenet_pretrained or is_fdsl_pretrained:
            base_out_dim = 1000
        elif is_fdsml_pretrained:
            base_out_dim = 128

        model = vit.deit_tiny_patch16_224(pretrained=is_imagenet_pretrained, num_classes=base_out_dim, global_pool=pool, drop_rate=drop_rate, drop_path_rate=drop_path_rate)
    elif net.startswith('deit_base_patch16_224'):
        base_out_dim = 1000
        if is_imagenet_pretrained or is_fdsl_pretrained:
            base_out_dim = 1000
        elif is_fdsml_pretrained:
            base_out_dim = 256

        model = vit.deit_base_patch16_224(pretrained=is_imagenet_pretrained, num_classes=base_out_dim, global_pool=pool, drop_rate=drop_rate, drop_path_rate=drop_path_rate)

    # Load pre-trained weights if specified
    if net in _pretrained_weights:
        opts = _pretrained_weights[net]

        model = replace_projection_head(net, model, opts.proj_dim, verbose=False)  # Replace the head with a new projection head if specified by the user (e.g., for metric learning)

        with util.open_url(opts.url) as file:
            state_dict = torch.load(file, map_location='cpu')
            if opts.sub_name is not None:
                state_dict = state_dict[opts.sub_name]
            model.load_state_dict(state_dict, strict=True)

        dist.print0(f'Loaded pre-trained weights for {net} from {opts.url}.')

    return model


def replace_projection_head(net_name: str, net: torch.nn.Module, proj_dim: int | None, verbose: bool = True) -> torch.nn.Module:
    """Replace the projection head of the given model with a new one that has the specified output dimension.

    Parameters
    ----------
    net_name : str
        The name of the neural network architecture.
    net : torch.nn.Module
        The model whose projection head is to be replaced.
    proj_dim : int | None
        The output dimension of the new projection head. If None, the projection head will be replaced with an identity layer (i.e., no projection head).
    verbose : bool
        If True, print information about the replacement process.

    Returns
    -------
    torch.nn.Module
        The modified model with the new projection head.
    """

    def print_if_verbose(message: str) -> None:
        if verbose:
            dist.print0(message)

    if proj_dim is None and is_vit(net_name):
        net.head = torch.nn.Identity()
        print_if_verbose(f'Replaced head of {net_name} with identity layer (no projection head).')
    elif proj_dim is None and not is_vit(net_name):
        net.fc = torch.nn.Identity()
        print_if_verbose(f'Replaced fc layer of {net_name} with identity layer (no projection head).')
    elif 'mocov3' in net_name:
        assert is_vit(net_name), 'MoCo v3 style projection head is only supported for ViT models.'
        net = _to_mocov3_vit(net, proj_dim)
        print_if_verbose(f"Added MoCo v3 style projection head to DeiT-Base with output dim {proj_dim}.")
    elif is_vit(net_name):
        in_dim = net.head.in_features
        net.head = torch.nn.Linear(in_dim, proj_dim)
        print_if_verbose(f'Replaced head of {net_name} with new projection head with output dim {proj_dim}.')
    else:
        in_dim = net.fc.in_features
        net.fc = torch.nn.Linear(in_dim, proj_dim)
        print_if_verbose(f'Replaced fc layer of {net_name} with new projection head with output dim {proj_dim}.')

    return net


def remove_projection_head(model: torch.nn.Module) -> torch.nn.Module:
    """
    Remove the fully connected (fc) layer from the given model by replacing it with an identity layer.

    Parameters
    ----------
    model : torch.nn.Module
        The model from which to remove the fc layer.

    Returns
    -------
    torch.nn.Module
        The modified model with the fc layer replaced by an identity layer.
    """

    if hasattr(model, 'fc'):
        model.fc = torch.nn.Identity()
    else:
        raise AttributeError('The model does not have an fc layer to remove.')

    return model

# ---------------------------------------------------------------------------------------------------------------------------
# Seed setting


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

# ---------------------------------------------------------------------------------------------------------------------------
# Run directory utilities


def get_next_run_prefix(run_root_dir: pathlib.Path | str) -> str:
    """Get the next run prefix based on existing run directories.
    Directories are expected to be named as '00000-description', '00001-description', ...

    Parameters
    ----------
    run_root_dir : pathlib.Path | str
        Root directory containing previous run directories.

    Returns
    -------
    str
        Next run prefix as a zero-padded 5-digit string.
    """

    if isinstance(run_root_dir, str):
        run_root_dir = pathlib.Path(run_root_dir)
    run_root_dir = run_root_dir.resolve()

    prev_run_dirs = []

    if run_root_dir.is_dir():
        prev_run_dirs = [x for x in run_root_dir.iterdir() if x.is_dir()]

    prev_run_ids = [re.match(r'^\d+', x.name) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]

    cur_run_id = max(prev_run_ids, default=-1) + 1

    return f'{cur_run_id:05d}'


# ---------------------------------------------------------------------------------------------------------------------------
# Module compiler

def compile_module(module: torch.nn.Module, mode: str = 'max-autotune') -> torch.nn.Module:
    """Compile the given module using torch.compile() if available.

    Parameters
    ----------
    module : torch.nn.Module
        The module to compile.
    mode : str, optional
        The compilation mode to use ("default", "reduce-overhead", "max-autotune" or "max-autotune-no-cudagraphs"). Default is "max-autotune".

    Returns
    -------
    torch.nn.Module
        The compiled module.
    """

    if hasattr(torch, 'compile'):
        compiled_module = torch.compile(module, mode=mode)
        return compiled_module
    else:
        return module

# ---------------------------------------------------------------------------------------------------------------------------
# Reproducibility utilities


def copy_source_files(
    source_dir: pathlib.Path,
    dest_dir: pathlib.Path,
    max_source_size: int = 10 << 20,  # 10 MiB
    exts: list[str] = ['.py', '.sh'],
    exclude_sub_dirs: list[str] = ['__pycache__', 'data', 'datasets', 'out', '.venv'],
) -> None:
    """Copy source files from source_dir to dest_dir for reproducibility.

    Parameters
    ----------
    source_dir : pathlib.Path
        Directory containing source files to copy.
    dest_dir : pathlib.Path
        Destination directory to copy source files to.
    max_source_size : int, optional
        Maximum allowed total size of source files in bytes. Default is 10 MiB.
    exts : list[str], optional
        List of file extensions to include. Default is ['.py', '.sh'].
    """

    # Globe source files
    source_files: list[pathlib.Path] = []
    for ext in exts:
        for file in source_dir.glob(f'**/*{ext}'):
            # Exclude files in certain sub-directories
            rel_path = file.relative_to(source_dir)
            if any(part in exclude_sub_dirs for part in rel_path.parts):
                continue

            # Include the file
            if file.is_file():
                source_files.append(file)

    # Estimate total size of source files
    total_size = sum(os.path.getsize(str(f)) for f in source_files)
    dist.print0(f'Estimated total size of source files: {total_size / (1 << 20):.2f} MiB', flush=True)
    assert total_size < max_source_size, f'Total size of source files ({total_size / (1 << 20):.2f} MiB) exceeds the limit ({max_source_size / (1 << 20):.2f} MiB).'

    # Copy source files
    dest_dir.mkdir(parents=True, exist_ok=True)
    for file in source_files:
        rel_path = file.relative_to(source_dir)
        dst_file = (dest_dir / rel_path).resolve()
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, dst_file)


def save_args(args: dict, filepath: pathlib.Path) -> None:
    """Save command line arguments to a JSON file."""

    with open(filepath, 'w') as f:
        json.dump(args, f, indent=4)

# ---------------------------------------------------------------------------------------------------------------------------
# Argument parsing utilities


def str_to_tuple_float(s: str) -> tuple[float, ...]:
    """Convert a comma-separated string to a tuple of floats.

    Parameters
    ----------
    s : str
        Comma-separated string of floats.

    Returns
    -------
    tuple[float, ...]
        Tuple of floats.
    """

    s = s.lstrip('(').rstrip(')')
    return tuple(float(x.strip()) for x in s.split(','))

# ---------------------------------------------------------------------------------------------------------------------------
# Label normalization transform


class ParameterNormalize(object):
    """Normalize labels using specified mean and standard deviation."""

    def __init__(self, mean: tuple[float, ...], std: tuple[float, ...]):
        """Initialize the LabelNormalize transform.

        Parameters
        ----------
        mean : tuple[float, ...]
            Mean values for normalization.
        std : tuple[float, ...]
            Standard deviation values for normalization.
        """

        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def __call__(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Apply normalization to the input labels.

        Parameters
        ----------
        x : torch.Tensor | np.ndarray
            Input labels to normalize.

        Returns
        -------
        torch.Tensor
            Normalized labels.
        """

        if isinstance(x, torch.Tensor):
            x = x.detach().clone()
        else:
            x = torch.tensor(x, dtype=torch.float32)

        return (x - self.mean) / self.std

    def __repr__(self) -> str:
        return f'LabelNormalize(mean={self.mean.tolist()}, std={self.std.tolist()})'

# ---------------------------------------------------------------------------------------------------------------------------
# Learning rate scheduler


class CosineWithConstantStartsLR(torch.optim.lr_scheduler._LRScheduler):
    """Cosine decay learning rate scheduler with an initial constant learning rate phase.

    References
    ----------
    - https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html#cosineannealinglr
    """

    def __init__(self, optimizer: torch.optim.Optimizer, constant_steps: int, total_steps: int, eta_min: float = 0.0, last_epoch: int = -1):
        self.constant_steps = constant_steps
        self.total_steps = total_steps
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        if self.last_epoch < self.constant_steps:
            # Constant learning rate phase
            return [base_lr for base_lr in self.base_lrs]
        else:
            # Cosine decay phase
            progress = (self.last_epoch - self.constant_steps) / (self.total_steps - self.constant_steps)
            return [self.eta_min + 0.5 * (base_lr - self.eta_min) * (1.0 + np.cos(np.pi * progress)) for base_lr in self.base_lrs]


lr_sched_catalog = [
    'cosine',
    'cosine_warmup',
    'multistep',
    'none',
]


def get_lr_scheduler(
    lr_sched: str,
    opt: torch.optim.Optimizer,
    total_steps: int,
    print_graph: bool = True,
    print_steps: int = 150,
    **opt_kwargs,
) -> torch.optim.lr_scheduler._LRScheduler:
    assert lr_sched in lr_sched_catalog, f'Unknown lr_sched: {lr_sched}'

    if lr_sched == 'cosine':
        lr = opt.param_groups[0]['lr']
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=lr * 0.01, **opt_kwargs)  # Cosine annealing to 1% of initial lr
    elif lr_sched == 'cosine_warmup':
        warmup_steps = opt_kwargs.get('warmup_steps', int(total_steps * 0.4))  # Default warmup for 40% of total steps
        scheduler = CosineWithConstantStartsLR(opt, constant_steps=warmup_steps, total_steps=total_steps, eta_min=opt.param_groups[0]['lr'] * 0.01)  # Cosine annealing to 1% of initial lr
    elif lr_sched == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[total_steps // 3, total_steps * 2 // 3], gamma=0.1)  # Step at 1/3 and 2/3 of total steps
    elif lr_sched == 'none':
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda step: 1.0)  # No learning rate scheduling
    else:
        raise NotImplementedError(f'Unknown lr_sched: {lr_sched}')

    if print_graph and dist.get_rank() == 0:
        # Print lr transition on terminal for debugging
        test_shceduler = copy.deepcopy(scheduler)

        plot_steps = np.linspace(0, total_steps - 1, num=print_steps, dtype=np.int64).tolist()

        lrs: list[float] = []
        for step in range(total_steps):
            test_shceduler.step()
            if step in plot_steps:
                lrs.append(test_shceduler.get_last_lr()[0])

        graph_str = asciichartpy.plot(lrs, cfg={'height': 40, 'format': '{:.1e}'})
        width = 0
        for line in graph_str.split('\n'):
            width = max(width, len(line))

        dist.print0(f'-' * width)
        dist.print0(f'[Learning Rate Schedule: {test_shceduler.__class__.__name__}]\n')
        dist.print0(graph_str + '\n')
        dist.print0(f'-' * width)

    return scheduler

# ---------------------------------------------------------------------------------------------------------------------------
