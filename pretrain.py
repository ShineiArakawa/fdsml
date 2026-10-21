"""Main script to launch distributed metric learning training jobs.

License
-------

Copyright (c) 2026 Shinei Arakawa
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
"""

import os
import pathlib
import pprint
import textwrap

import click
import torch

import metric_learning.metric_loss as metric_loss
import metric_learning.training_loop as training_loop
import torch_utils.distributed as dist
import torch_utils.util as util
import training_utils

# ---------------------------------------------------------------------------------------------------------------------------
# Utilities


def compile_description(args: util.EasyDict) -> str:
    """Compile a description string based on the given arguments.

    Parameters
    ----------
    args : util.EasyDict
        Arguments containing training configuration.

    Returns
    -------
    str
        Compiled description string.
    """

    desc = 'fdsml'

    # model name
    desc += f'-{args.net}'
    # projection head
    if args.projection_dim is not None:
        desc += f'-proj{args.projection_dim}'
    else:
        desc += f'-noproj'
    # loss
    desc += f'-{args.loss}'
    # miner
    if args.loss == 'patriplet':
        desc += f'-{args.miner}_miner'
    # margin
    desc += f'-m{args.margin}'
    # dataset name
    desc += f'-{os.path.basename(args.dset).replace("-", "_")}'
    # sampler
    desc += f'-{args.sampler}'
    # parameter dim
    if args.param_dim is not None:
        desc += f'-pdim{args.param_dim}'
    # instances per class
    if args.insts_per_class is not None:
        desc += f'-ipc{args.insts_per_class}'
    # batch size
    desc += f'-b{args.batch_size}'
    # learning rate
    desc += f'-lr{args.lr:.6f}'
    # world size
    desc += f'-{dist.get_world_size()}GPUS'

    return desc


# ---------------------------------------------------------------------------------------------------------------------------
# Main function


def main(**kwargs) -> None:
    """Launch a distributed metric learning training job. Please use `torchrun` to launch this script for multi-GPU training."""

    def is_torchrun() -> bool:
        return not kwargs['not_ddp']

    # ------------------------------------------------------------------------------------------
    # Setup for distributed training

    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError as e:
        print(e, flush=True)

    if is_torchrun():
        dist.init()

    # ------------------------------------------------------------------------------------------
    # Option

    args = util.EasyDict(kwargs)

    # Description
    desc = compile_description(args)
    if args.desc_suffix is not None:
        desc += f'-{args.desc_suffix}'
    args.desc = desc
    dist.print0(f'Run description: {desc}', flush=True)

    # Set run dir
    prefix = training_utils.get_next_run_prefix(args.run_dir_root)
    args.run_dir = os.path.join(os.path.abspath(args.run_dir_root), f'{prefix}-{desc}')

    dist.print0(f'Argsuments: {pprint.pformat(args, indent=4)}\n', flush=True)

    # ------------------------------------------------------------------------------------------
    # Save source files and command line arguments for reproducibility

    dist.barrier()  # wait for all processes to reach this point

    if dist.get_rank() == 0:  # to avoid increasing run_dir number
        # Copy source files for reproducibility
        source_dir = pathlib.Path(__file__).resolve().parent
        training_utils.copy_source_files(source_dir, pathlib.Path(args.run_dir) / 'source')

        # Save command line arguments
        training_utils.save_args(args.to_stddict(), pathlib.Path(args.run_dir) / 'args.json')

    # ------------------------------------------------------------------------------------------
    # Setup logging

    dist.barrier()  # wait for all processes to reach this point

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    util.Logger(file_name=os.path.join(args.run_dir, f'log_rank={rank}.log'), file_mode='a', should_flush=True)

    # ------------------------------------------------------------------------------------------
    # Run the training

    dist.barrier()  # wait for all processes to reach this point

    print(f'[Rank {rank}/{world_size - 1}] Starting training ...\n', flush=True)

    if not args.dry_run:
        training_loop.training_loop(args)

    # ------------------------------------------------------------------------------------------
    # Graceful termination

    print(f'[Rank {rank}/{world_size - 1}] Terminating ...\n', flush=True)

    dist.barrier()  # wait for all processes to reach this point

    torch.distributed.destroy_process_group()

    print(f'[Rank {rank}/{world_size - 1}] done.\n', flush=True)

    # ------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------------------
# Command line interface


@click.command(context_settings=dict(show_default=True, max_content_width=200))
# Run settings
@click.option('--run_dir_root', type=click.Path(file_okay=False, dir_okay=True), required=True, help='Root directory to save the training results.')
@click.option('--desc_suffix', type=str, default=None, help='Suffix to add to the run description.')
@click.option('--seed', type=int, default=0, help='Global random seed.')
@click.option('--resume_snapshot', type=click.Path(exists=True, file_okay=True), default=None, help='Path to a training snapshot to resume from.')
# Network settings
@click.option('--net', type=str, required=True, help='Type of the neural network architecture.')
@click.option('--projection_dim', type=int, default=None, help='Dimension of the projection head output. If not set, no projection head is used.')
@click.option('--compile', type=click.Choice(['none', 'default', 'reduce-overhead', 'max-autotune', 'max-autotune-no-cudagraphs']), default='none', help='Compilation mode for the model and loss function.')
@click.option('--freeze_patch_proj', is_flag=True, default=False, help='If set, freeze the patch projection layer in ViT-based models.')
@click.option('--pool', type=click.Choice(['avg', 'avgmax', 'max', 'token', 'tokenavgmax', 'map']), default='token', help='Pooling method for ViT-based models.')
# Loss settings
@click.option('--loss', type=click.Choice(list(metric_loss.loss_catalog.keys())), default='patriplet', help='Type of the loss function.')
@click.option('--margin', type=float, default=0.05, help='Margin parameter for the loss function.')
@click.option('--miner', type=click.Choice(list(metric_loss.MinerType)), default=None, help='Type of the miner to select triplets (if applicable).')
# Dataset settings
@click.option('--dset', type=click.Path(exists=True, file_okay=False), required=True, help='Path to the dataset root directory containing image files.')
@click.option('--dset_cls', type=click.Choice(['imagefolder', 'visual_atoms']), default='imagefolder', help='Type of the dataset class to use.')
@click.option('--param_dir', type=click.Path(exists=True, file_okay=False), default=None, help='Path to the directory containing param files.')
@click.option('--dset_split', type=click.Choice(['train', 'val', 'test', 'none']), default='none', help='Dataset split to use for training.')
@click.option('--sampler', type=click.Choice(['random', 'pk']), default='pk', help='Type of the data sampler.')
@click.option('--insts_per_class', type=int, default=16, help='Number of instances to sample per class in each batch.')
@click.option('--img_mean', type=click.Tuple([float, float, float]), default=(0.2, 0.2, 0.2), help='Dataset mean for normalization.')
@click.option('--img_std', type=click.Tuple([float, float, float]), default=(0.5, 0.5, 0.5), help='Dataset standard deviation for normalization.')
@click.option('--param_dim', type=int, default=6, help='Dimensionality of the param vectors (if applicable).')
@click.option('--param_mean', type=click.Tuple([float, float, float, float, float, float]), default=(-0.00020500538802303397, -0.01737613659944719, 0.01417852385124239, -0.00803841852193487, 0.01674297180468388, 0.023242451435772638), help='Mean for parameter normalization (if applicable). The default values are computed from the training set of FineFractalDB-1k. Please compute and set appropriate values for other datasets.')
@click.option('--param_std', type=click.Tuple([float, float, float, float, float, float]), default=(0.4420339894511067, 0.45483485794689393, 0.4547010185027264, 0.4432377997619619, 0.4441378708547926, 0.42966210711089475), help='Standard deviation for parameter normalization (if applicable) The default values are computed from the training set of FineFractalDB-1k. Please compute and set appropriate values for other datasets.')
@click.option('--aug', type=click.Choice(['default', 'simple', 'autoaug']), default='default', help='Data augmentation strategy to use during training.')
@click.option('--img_size', type=int, default=256, help='Input image size.')
@click.option('--crop_size', type=int, default=224, help='Input image crop size.')
@click.option('--batch_size', type=int, default=1 << 12, help='Total batch size for all GPUs.')
@click.option('--batch_gpu', type=int, required=True, help='Batch size per GPU.')
@click.option('--num_workers', type=int, default=16, help='Number of worker processes for data loading.')
# Optimization settings
@click.option('--lr', type=float, default=1e-5, help='Learning rate.')
@click.option('--momentum', type=float, default=0.9, help='SGD momentum.')
@click.option('--weight_decay', type=float, default=0.05, help='Weight decay.')
# Training settings
@click.option('--total_nimg', type=int, default=1 << 29, help='Total number of images to process during training.')
@click.option('--status_nimg', type=int, default=1 << 18, help='Interval (in number of images) to report training status.')
@click.option('--checkpoint_nimg', type=int, default=1 << 22, help='Interval (in number of images) to save checkpoints.')
@click.option('--snapshot_nimg', type=int, default=1 << 25, help='Interval (in number of images) to take training snapshots.')
@click.option('--precision', type=click.Choice(['fp32', 'tf32', 'bf16']), default='bf16', help='Numerical precision to use for training.')
#  Misc settings
@click.option('--dry_run', is_flag=True, default=False, help='If set, do not run the training loop.')
@click.option('--disable_wandb', is_flag=True, default=False, help='Disable Weights & Biases logging.')
@click.option('--disable_report_triplet_stats', is_flag=True, default=False, help='Disable logging of triplet statistics such as the number of triplets and average distances.')
@click.option('--not_ddp', is_flag=True, default=False, help='This run is launched not with torchrun?')
def cmdline(**kwargs) -> None:
    """Launch a distributed metric learning training job. Please use `torchrun` to launch this script for multi-GPU training."""

    main(**kwargs)


def create_help_message() -> str:
    return textwrap.dedent(f"""Launch a distributed metric learning training job. Please use `torchrun` to launch this script for multi-GPU training.\n\n
    
You can select network architectures from:

\b
{pprint.pformat(training_utils.model_catalog, width=1, compact=False)}

Examples:

\b
# Train DeiT-Tiny model for FineFractalDB using 2 GPUs
torchrun \\
    --standalone \\
    --nproc_per_node=2 \\
    pretrain.py \\
    --run_dir_root="out/fdsml" \\
    --net=deit_tiny_patch16_224_mocov3 \\
    --projection_dim=128 \\
    --freeze_patch_proj \\
    --loss=patriplet \\
    --margin=0.05 \\
    --miner=same_class \\
    --dset="datasets/finefractaldb/images" \\
    --param_dir="datasets/finefractaldb/params" \\
    --dset_split="none" \\
    --sampler="pk" \\
    --insts_per_class=16 \\
    --batch_size=4096 \\
    --batch_gpu=256 \\
    --lr=1e-4 \\
    --weight_decay=0.05 \\
    --total_nimg=$(( 256 << 21 )) \\
    --status_nimg=$(( 256 << 9 )) \\
    --checkpoint_nimg=$(( 256 << 12 )) \\
    --snapshot_nimg=$(( 256 << 14 ))
"""
    )


cmdline.help = create_help_message()

# ---------------------------------------------------------------------------------------------------------------------------
# Entry point


if __name__ == "__main__":
    cmdline()

# ---------------------------------------------------------------------------------------------------------------------------
