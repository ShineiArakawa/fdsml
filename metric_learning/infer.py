"""Main script to launch a metric learning inference job.

License
-------

Copyright (c) 2026 Shinei Arakawa
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
"""

import functools
import os
import pathlib
import pprint

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchinfo
import torchvision
import tqdm

import metric_learning.dataset as dataset
import torch_utils.distributed as dist
import torch_utils.util as util
import training_utils

# ---------------------------------------------------------------------------------------------------------------------------
# Inference loop


def inference_loop(args: util.EasyDict) -> None:
    device = torch.device('cuda')

    # ------------------------------------------------------------------------------------------
    # Set seeds based on rank

    global_seed = args.seed
    training_utils.seed_everything(global_seed)

    # ------------------------------------------------------------------------------------------
    # Setup internal states

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.set_float32_matmul_precision('highest')

    # ------------------------------------------------------------------------------------------
    # Setup dataset

    assert args.img_size >= args.crop_size, f"img_size ({args.img_size}) must be greater than or equal to crop_size ({args.crop_size})"

    img_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((args.img_size, args.img_size), interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
        torchvision.transforms.RandomCrop((args.crop_size, args.crop_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=args.dset_mean, std=args.dset_std)
    ])

    dset_path = os.path.join(args.dset, args.dset_split) if args.dset_split != 'none' else args.dset
    train_dataset = dataset.ImageFolderDataset(root=dset_path, transform=img_transform, num_classes=args.num_classes, insts_per_class=args.insts_per_class, shuffle_classes=args.shuffle, shuffle_instances=args.shuffle, return_img_path=True)

    # ------------------------------------------------------------------------------------------
    # Batch size settings

    dist.print0(f'Found {len(train_dataset)} images for inference.\n')

    # ------------------------------------------------------------------------------------------
    # Model

    net = training_utils.select_model(args.net, is_feature_extractor=True)
    dist.print0(f'Model selected: {args.net}')

    # Load checkpoint
    if args.ckpt is not None:
        net = training_utils.replace_projection_head(args.net, net, args.projection_dim)
        checkpoint = torch.load(args.ckpt, map_location=device)
        state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))  # Try some keys: ['model', 'state_dict']
        net.load_state_dict(state_dict, strict=True)
        dist.print0(f'Loaded checkpoint from: {args.ckpt}\n')
    if args.remove_projection_head:
        training_utils.remove_projection_head(net)
        dist.print0('Removed projection head from the loaded model.\n')
    net.eval().requires_grad_(False).to(device)

    net_info = torchinfo.summary(net, input_size=(args.batch_size, 3, args.crop_size, args.crop_size), col_names=["input_size", "output_size", "num_params", "trainable"], device=str(device), verbose=0)
    dist.print0(f'\n{net_info}\n')

    # ------------------------------------------------------------------------------------------
    # Setup dataloader with DistributedSampler

    sampler = torch.utils.data.DistributedSampler(
        dataset=train_dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=False,
        seed=global_seed,  # Use the same seed to ensure the same order across runs
        drop_last=False,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,                                                                         # NOTE: Keep workers alive to avoid overhead of respawning them
        prefetch_factor=2,                                                                               # NOTE: Number of samples loaded in advance by each worker
        worker_init_fn=functools.partial(training_utils.seed_everything, global_seed),                   # NOTE: Use different seeds for each worker, otherwise they generate the same images
    )

    # ------------------------------------------------------------------------------------------
    # Inference loop

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    to_save: dict[str, list[torch.Tensor]] = {}
    out_emb_sum: torch.Tensor = None
    local_img_paths: list[str] = []

    with torch.no_grad():
        sampler.set_epoch(0)  # For DistributedSampler, set epoch to get deterministic ordering

        for imgs, labels, img_paths in tqdm.tqdm(dataloader, total=len(dataloader), desc=f'[Rank {rank}/{world_size - 1}] Inference ... ', unit='batch', position=rank, leave=False, dynamic_ncols=True):
            imgs_cuda = imgs.to(device, non_blocking=True)

            feats: dict[str, torch.Tensor] = net(imgs_cuda)

            for key, feat in feats.items():
                to_save.setdefault(key, []).append(feat.cpu())
            to_save.setdefault('labels', []).append(labels.cpu())

            # Accumulate image paths
            local_img_paths.extend(img_paths)

            # Accumulate sum of embeddings for plotting
            embs = feats['logits']
            if out_emb_sum is None:
                out_emb_sum = embs.sum(dim=0).cpu()  # Sum over batch dimension
            else:
                out_emb_sum += embs.sum(dim=0).cpu()  # Sum over batch dimension

    # ------------------------------------------------------------------------------------------
    # Concatenate all tensors

    enough_size = len(train_dataset) // world_size + 1

    to_save_tensor: dict[str, torch.Tensor] = {}
    for key in to_save.keys():
        data = torch.cat(to_save[key], dim=0)
        assert data.size(0) <= enough_size, f"Data size ({data.size(0)}) exceeds enough size ({enough_size}) for key: {key}"

        if world_size > 1 and data.size(0) < enough_size:
            # Pad data to have enough size for all ranks
            pad_size = enough_size - data.size(0)
            pad_tensor = torch.zeros((pad_size, *data.size()[1:]), dtype=data.dtype, device=data.device)
            data = torch.cat([data, pad_tensor], dim=0)

        to_save_tensor[key] = data

    # ------------------------------------------------------------------------------------------
    # Gather

    # Gather to_save from all ranks to rank 0
    to_save_np: dict[str, np.ndarray] = {}
    if world_size > 1:
        for key in to_save_tensor.keys():
            dist.barrier()  # wait for all processes to reach this point

            if rank == 0:
                gather_list = [torch.zeros_like(to_save_tensor[key]) for _ in range(world_size)]
            else:
                gather_list = None

            torch.distributed.gather(to_save_tensor[key], gather_list=gather_list, dst=0)

            # Only rank 0 keeps the gathered data
            if rank == 0:
                # Remove padding for each rank
                for r in range(world_size):
                    num_valid_data = len(train_dataset) // world_size + (1 if r < len(train_dataset) % world_size else 0)
                    gather_list[r] = gather_list[r][:num_valid_data]

                # Combine gathered data
                base_shape = to_save_tensor[key].shape[1:]
                base_dtype = to_save_tensor[key].dtype

                all_data = torch.zeros((len(train_dataset), *base_shape), dtype=base_dtype)
                for r in range(world_size):
                    assert all_data[r::world_size].shape == gather_list[r].shape, f"Shape mismatch for rank {r}: all_data {all_data[r::world_size].shape} vs gather_list {gather_list[r].shape}"
                    all_data[r::world_size] = gather_list[r]  # See the implementation of DistributedSampler for details

                to_save_np[key] = all_data.numpy()
    else:
        for key in to_save_tensor.keys():
            to_save_np[key] = to_save_tensor[key].numpy()  # Remove padding if any

    # Gather out_emb_sum to rank 0
    dist.barrier()  # wait for all processes to reach this point
    if world_size > 1:
        torch.distributed.reduce(out_emb_sum, dst=0, op=torch.distributed.ReduceOp.SUM)

    # ------------------------------------------------------------------------------------------
    # Gather img_paths to rank 0

    all_img_paths = None

    dist.barrier()
    if world_size > 1:
        if rank == 0:
            gathered_img_paths = [None for _ in range(world_size)]
        else:
            gathered_img_paths = None

        torch.distributed.gather_object(local_img_paths, object_gather_list=gathered_img_paths, dst=0)

        if rank == 0:
            # Compute the number of valid img_paths for each rank (since some ranks may have padded data)
            per_rank_counts = [
                len(train_dataset) // world_size + (1 if r < len(train_dataset) % world_size else 0)
                for r in range(world_size)
            ]

            # trim
            for r in range(world_size):
                gathered_img_paths[r] = gathered_img_paths[r][:per_rank_counts[r]]

            # Combine gathered img_paths
            all_img_paths = [None] * len(train_dataset)
            for r in range(world_size):
                assert len(all_img_paths[r::world_size]) == len(gathered_img_paths[r]), f"img_paths length mismatch for rank {r}: {len(all_img_paths[r::world_size])} vs {len(gathered_img_paths[r])}"
                all_img_paths[r::world_size] = gathered_img_paths[r]  # See the implementation of DistributedSampler for details
    else:
        all_img_paths = local_img_paths

    assert len(all_img_paths) == len(train_dataset), f"Total img_paths length mismatch: {len(all_img_paths)} vs {len(train_dataset)}"

    # ------------------------------------------------------------------------------------------
    # Save outputs

    if rank == 0:
        # Prepare data to save
        to_save_contents = {}
        for key, value in to_save_np.items():
            if key == 'labels':
                to_save_contents[key] = value
            elif key == 'logits':
                # Change 'logits' key to 'embs' for backward compatibility
                to_save_contents['embs'] = value
            elif args.save_internal_feats:
                to_save_contents[key] = value

        # Add img_paths to the saved contents
        to_save_contents['img_paths'] = np.asarray(all_img_paths, dtype=np.str_)

        # Save to .npz file
        out_npz = pathlib.Path(args.out).resolve()
        out_npz.parent.mkdir(parents=True, exist_ok=True)

        np.savez(out_npz, **to_save_contents)
        dist.print0(f'Saved embeddings to: {out_npz}\n')

        if args.out_emb_plot is not None:
            out_emb_sum_np = (out_emb_sum / len(train_dataset)).numpy()  # Compute mean embedding

            fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
            ax.hist(out_emb_sum_np, bins=50, color='blue', alpha=0.7)
            ax.set_title('Histogram of Mean Embedding Values')
            ax.set_xlabel('Embedding Value')
            ax.set_ylabel('Frequency')
            out_emb_plot_path = pathlib.Path(args.out_emb_plot).resolve()
            out_emb_plot_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_emb_plot_path)
            plt.close(fig)
            dist.print0(f'Saved embedding plot to: {out_emb_plot_path}\n')


# ---------------------------------------------------------------------------------------------------------------------------
# Main function


@click.command(context_settings=dict(show_default=True, max_content_width=200))
# Run settings
@click.option('--seed', type=int, default=0, help='Global random seed.')
@click.option('--out', type=click.Path(dir_okay=False, file_okay=True), required=True, help='Path to the output .npz file to save the embeddings.')
@click.option('--out_emb_plot', type=click.Path(dir_okay=False, file_okay=True), default=None, help='Path to the output embedding plot .png file. If not set, no plot is saved.')
@click.option('--save_internal_feats', is_flag=True, default=False, help='If specified, save internal features as well.')
# Network settings
@click.option('--net', type=click.Choice(training_utils.model_catalog), required=True, help='Type of the neural network architecture.')
@click.option('--projection_dim', type=int, default=None, help='Dimension of the projection head output. If not set, no projection head is used.')
@click.option('--ckpt', type=str, default=None, help='Path to the model checkpoint file. HTTP(S) URLs, Google Drive URLs, and SSH paths (e.g., host:/path/to/checkpoint.pt) are also accepted.')
@click.option('--remove_projection_head', is_flag=True, default=False, help='If specified, remove the projection head from the loaded model.')
# Dataset settings
@click.option('--dset', type=click.Path(exists=True, file_okay=False), required=True, help='Path to the dataset root directory containing csv files.')
@click.option('--dset_split', type=click.Choice(['train', 'val', 'test', 'none']), default='test', help='Dataset split to use for inference.')
@click.option('--num_classes', type=int, default=None, help='Number of classes to use. If not set, all classes are used.')
@click.option('--insts_per_class', type=int, default=None, help='Number of instances per class to use. If not set, all instances are used.')
@click.option('--shuffle', is_flag=True, default=False, help='Whether to shuffle the dataset before inference. Note that shuffling is done after selecting num_classes and insts_per_class, so it does not affect which samples are selected but only their order.')
@click.option('--dset_mean', type=click.Tuple([float, float, float]), default=(0.2, 0.2, 0.2), help='Dataset mean for normalization.')
@click.option('--dset_std', type=click.Tuple([float, float, float]), default=(0.5, 0.5, 0.5), help='Dataset standard deviation for normalization.')
@click.option('--img_size', type=int, default=362, help='Input image size.')
@click.option('--crop_size', type=int, default=224, help='Input image crop size.')
@click.option('--batch_size', type=int, required=True, help='Batch size.')
@click.option('--num_workers', type=int, default=16, help='Number of worker processes for data loading.')
# Misc settings
@click.option('--dry_run', is_flag=True, default=False, help='If specified, do not run the actual inference.')
def main(**kwargs) -> None:
    """Launch a inference job for metric learning. Please use `torchrun` to launch this script for distributed inference.
    """

    # ------------------------------------------------------------------------------------------
    # Setup for distributed inference

    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError as e:
        print(e, flush=True)

    dist.init(backend='gloo')  # In inference, we select 'gloo' backend because we do not use GPU reducer but CPU-based communication.

    # ------------------------------------------------------------------------------------------
    # Option

    args = util.EasyDict(kwargs)

    dist.print0(f'Argsuments: {pprint.pformat(args, indent=4)}\n', flush=True)

    # ------------------------------------------------------------------------------------------
    # Run the inference

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    dist.barrier()  # wait for all processes to reach this point

    print(f'[Rank {rank}/{world_size - 1}] Starting inference ...\n', flush=True)

    if not args.dry_run:
        inference_loop(args)

    # ------------------------------------------------------------------------------------------
    # Graceful termination

    print(f'[Rank {rank}/{world_size - 1}] Terminating ...\n', flush=True)

    dist.barrier()
    torch.distributed.destroy_process_group()

    print(f'[Rank {rank}/{world_size - 1}] done.\n', flush=True)

    # ------------------------------------------------------------------------------------------


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------------------------------------------------------
