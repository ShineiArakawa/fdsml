"""Main script to launch distributed fine-tuning jobs.

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
import timm.data.constants
import torch

import finetuning.training_loop as training_loop
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

    desc = 'fdsml_ft'

    # model name
    desc += f'-{args.net}'
    # dataset name
    desc += f'-{os.path.basename(args.dset).replace("-", "_")}'
    # freeze patch projection layer?
    if args.freeze_patch_proj:
        desc += '-fzproj'
    # mixup
    desc += f'-mixup{args.mixup_alpha}'
    # learning rate
    desc += f'-lr{args.lr:.6f}'
    # scheduler
    desc += f'-{args.sched}_sched'
    # precision
    desc += f'-{args.precision}'
    # batch size
    desc += f'-b{args.batch_size}'
    # world size
    desc += f'-{dist.get_world_size()}GPUS'

    return desc


# ---------------------------------------------------------------------------------------------------------------------------
# Main function


def main(**kwargs) -> None:
    """Launch a distributed fine-tuning training job. Please use `torchrun` to launch this script for multi-GPU training."""

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
# Network settings
@click.option('--net', type=str, required=True, help='Type of the neural network architecture.')
@click.option('--pretrained_dim', type=int, default=False, help='The dimension of the projection head for the pretrained model.')
@click.option('--ckpt', type=str, default=None, help='Path to a checkpoint file to initialize the model from. HTTP(S) URLs, Google Drive URLs, and SSH paths (e.g., host:/path/to/checkpoint.pt) are also accepted.')
@click.option('--drop_rate', type=float, default=0.0, help='Dropout rate for the projection head (if applicable).')
@click.option('--drop_path_rate', type=float, default=0.1, help='Drop path rate for the backbone model (if applicable).')
@click.option('--compile', type=click.Choice(['none', 'default', 'reduce-overhead', 'max-autotune', 'max-autotune-no-cudagraphs']), default='none', help='Compilation mode for the model and loss function.')
@click.option('--freeze_patch_proj', is_flag=True, default=False, help='If set, freeze the patch projection layer in ViT-based models.')
# Dataset settings
@click.option('--dset', type=click.Path(exists=True, file_okay=False), required=True, help='Path to the dataset root directory containing csv files.')
@click.option('--val_split', type=str, default='val', help='Name of the subdirectory under --dset to use as the validation set.')
@click.option('--img_mean', type=click.Tuple([float, float, float]), default=timm.data.constants.IMAGENET_DEFAULT_MEAN, help='Dataset mean for normalization.')
@click.option('--img_std', type=click.Tuple([float, float, float]), default=timm.data.constants.IMAGENET_DEFAULT_STD, help='Dataset standard deviation for normalization.')
@click.option('--input_size', type=int, default=224, help='Input image size.')
@click.option('--batch_size', type=int, default=2048, help='Total batch size for all GPUs.')
@click.option('--batch_gpu', type=int, required=True, help='Batch size per GPU.')
@click.option('--num_workers', type=int, default=16, help='Number of worker processes for data loading.')
# Mixup settings
@click.option('--mixup_alpha', type=float, default=0.8, help='Alpha parameter for mixup.')
@click.option('--cutmix_alpha', type=float, default=1.0, help='Alpha parameter for cutmix.')
@click.option('--label_smoothing', type=float, default=0.1, help='Label smoothing factor.')
# Optimization settings
@click.option('--opt', type=str, default='adamw', help='Optimizer type.')
@click.option('--lr', type=float, default=5e-4, help='Learning rate.')
@click.option('--momentum', type=float, default=0.9, help='SGD momentum.')
@click.option('--weight_decay', type=float, default=0.05, help='Weight decay.')
# Scheduler settings
@click.option('--sched', type=str, default='cosine', help='Learning rate scheduler type.')
@click.option('--warmup_lr', type=float, default=1e-6, help='Initial learning rate for warmup.')
@click.option('--min_lr', type=float, default=1e-5, help='Minimum learning rate after decay.')
# Training settings
@click.option('--eval_intervals', type=str, default='1-799:100,800-:1', help='Evaluation intervals during training.')
@click.option('--total_nimg', type=int, default=625570 << 11, help='Total number of images to process during training.')
@click.option('--status_nimg', type=int, default=1 << 19, help='Interval (in number of images) to report training status.')
@click.option('--checkpoint_nimg', type=int, default=1 << 23, help='Interval (in number of images) to save checkpoints.')
@click.option('--snapshot_nimg', type=int, default=1 << 25, help='Interval (in number of images) to take training snapshots.')
@click.option('--precision', type=click.Choice(['fp32', 'tf32', 'bf16']), default='bf16', help='Numerical precision to use for training.')
#  Misc settings
@click.option('--dry_run', is_flag=True, default=False, help='If set, do not run the training loop.')
@click.option('--disable_wandb', is_flag=True, default=False, help='Disable Weights & Biases logging.')
@click.option('--not_ddp', is_flag=True, default=False, help='This run is launched not with torchrun?')
def cmdline(**kwargs) -> None:
    """Launch a distributed fine-tuning training job. Please use `torchrun` to launch this script for multi-GPU training."""

    main(**kwargs)


def create_help_message() -> str:
    return textwrap.dedent(f"""Launch a distributed fine-tuning training job. Please use `torchrun` to launch this script for multi-GPU training.\n\n
    
You can select network architectures from:

\b
{pprint.pformat(training_utils.model_catalog, width=1, compact=False)}

Examples:

\b
# Fine-tune DeiT-Tiny pre-trained model for FineFractalDB using 2 GPUs
torchrun \\
    --nproc_per_node=2 \\
    --standalone \\
    finetune.py \\
    --run_dir_root="out/fdsml" \\
    --seed=0 \\
    --net=deit_tiny_patch16_224_mocov3_fdsml_each_pretrained \\
    --freeze_patch_proj \\
    --dset=datasets/imagenet100 \\
    --batch_size=1024 \\
    --batch_gpu=512 \\
    --eval_intervals="-799:100,800-:1" \\
    --total_nimg=$(( 256 << 21 )) \\
    --status_nimg=$(( 256 << 9 )) \\
    --checkpoint_nimg=$(( 256 << 14 )) \\
    --snapshot_nimg=$(( 256 << 15 ))
"""
    )


cmdline.help = create_help_message()

# ---------------------------------------------------------------------------------------------------------------------------
# Entry point


if __name__ == "__main__":
    cmdline()

# ---------------------------------------------------------------------------------------------------------------------------
