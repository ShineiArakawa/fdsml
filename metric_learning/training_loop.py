"""Training loop for metric learning.

License
-------

Copyright (c) 2026 Shinei Arakawa
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

This code is a derivative work based on code from the NVlabs/edm2 repository, whose original code is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0). The original code can be found at https://github.com/NVlabs/edm2.
"""

import functools
import math
import os
import time

import numpy as np
import psutil
import timm.data
import timm.optim
import timm.scheduler
import timm.scheduler.scheduler
import torch
import torchinfo
import torchvision
import torchvision.transforms.functional
import torchvision.transforms.v2 as v2
import wandb

import metric_learning.dataset as dataset
import metric_learning.metric_loss as metric_loss
import torch_utils.distributed as dist
import torch_utils.misc as misc
import torch_utils.training_stats as training_stats
import torch_utils.util as util
import training_utils

# ---------------------------------------------------------------------------------------------------------------------------
# Training loop


def training_loop(args: util.EasyDict) -> None:
    """Main training loop for metric learning. This function is called by each process in the distributed setup. Each process will run this function independently, and they will coordinate with each other using torch.distributed."""

    device = torch.device('cuda')

    # ------------------------------------------------------------------------------------------
    # Processing time

    prev_status_time = time.time()

    # ------------------------------------------------------------------------------------------
    # Set seeds based on rank

    global_seed = args.seed
    training_utils.seed_everything(global_seed, rank=dist.get_rank())

    # ------------------------------------------------------------------------------------------
    # Setup internal states

    compute_dtype = None
    is_amp_enabled = False

    if args.precision == 'fp32':
        # Full precision and deterministic mode
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.set_float32_matmul_precision('highest')

        compute_dtype = torch.float32
        is_amp_enabled = False
    elif args.precision == 'tf32':
        # Reduced precision and non-deterministic mode for faster training on Ampere and newer GPUs with TF32 support
        is_tf32_available = torch.cuda.is_available() and torch.cuda.is_tf32_supported()
        assert is_tf32_available, "TF32 support is required for training with tf32 precision. Please use a GPU with Ampere architecture or newer, and ensure that you have the latest CUDA and cuDNN versions installed."

        torch.backends.cudnn.benchmark = True           # Enable cudnn auto-tuner to find the best algorithm for the hardware.
        torch.backends.cudnn.allow_tf32 = True          # Allow TF32 for cudnn operations including convolutions.
        torch.backends.cudnn.deterministic = False

        torch.backends.cuda.matmul.allow_tf32 = True    # Allow TF32 for cuda matmul operations.
        torch.set_float32_matmul_precision('high')      # Allow TF32 for matmul operations.

        dist.print0('Enabled cudnn.benchmark and set float32 matmul precision to high. Notice that this may lead to non-deterministic behavior.\n')

        compute_dtype = torch.float32
        is_amp_enabled = False
    elif args.precision == 'bf16':
        # Reduced precision and non-deterministic mode for faster training on Ampere and newer GPUs with native BF16 support
        is_bf16_available = torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False)
        assert is_bf16_available, "BF16 support is required for training with bf16 precision. Please use a GPU with Ampere architecture or newer, and ensure that you have the latest CUDA and cuDNN versions installed."

        torch.backends.cudnn.benchmark = True       # Enable cudnn auto-tuner to find the best algorithm for the hardware.
        torch.backends.cudnn.allow_tf32 = True      # Allow TF32 for cudnn operations including convolutions.
        torch.backends.cudnn.deterministic = False

        torch.backends.cuda.matmul.allow_tf32 = True                              # Allow TF32 for cuda matmul operations.
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True  # Allow BF16 for reduction in matmul operations.
        torch.set_float32_matmul_precision('high')                                # Allow TF32 for matmul operations.

        dist.print0('Enabled cudnn.benchmark and set float32 matmul precision to high. Notice that this may lead to non-deterministic behavior.\n')

        compute_dtype = torch.bfloat16
        is_amp_enabled = True
    else:
        raise ValueError(f'Unknown precision: {args.precision}')

    dist.print0(f"Using precision: {args.precision}, compute_dtype: {compute_dtype}, is_amp_enabled: {is_amp_enabled}")
    dist.print0(f"    - torch.backends.cudnn.benchmark                                    : {torch.backends.cudnn.benchmark}")
    dist.print0(f"    - torch.backends.cudnn.allow_tf32                                   : {torch.backends.cudnn.allow_tf32}")
    dist.print0(f"    - torch.backends.cudnn.deterministic                                : {torch.backends.cudnn.deterministic}")
    dist.print0(f"    - torch.backends.cuda.matmul.allow_tf32                             : {torch.backends.cuda.matmul.allow_tf32}")
    dist.print0(f"    - torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction : {torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction}")
    dist.print0(f"    - torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction : {torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}")
    dist.print0(f"    - torch.set_float32_matmul_precision                                : {torch.get_float32_matmul_precision()}\n")

    # ------------------------------------------------------------------------------------------
    # Setup wandb

    def wandb_enabled() -> bool:
        return not args.disable_wandb and dist.get_rank() == 0

    if wandb_enabled():
        wandb.init(
            project="fdsml-pretraining",
            name=args.desc,
            config=args.to_stddict(),
            save_code=True,
            dir=args.run_dir,
            tags=[tag.strip() for tag in args.desc.split('-')],
        )

        dist.print0(f'Weights & Biases logging enabled, run directory: {args.run_dir}\n')

    # ------------------------------------------------------------------------------------------
    # Setup data augmentation

    assert args.img_size >= args.crop_size, f"img_size ({args.img_size}) must be greater than or equal to crop_size ({args.crop_size})"

    is_vit = training_utils.is_vit(args.net)

    if args.aug == 'default' and is_vit:
        img_transform = v2.Compose([
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomAffine(degrees=45, translate=(0.2, 0.2), scale=(0.4, 1.0), shear=0, interpolation=v2.InterpolationMode.BILINEAR, fill=0),
            v2.Resize((args.img_size, args.img_size), interpolation=v2.InterpolationMode.BILINEAR),
            v2.RandomCrop((args.crop_size, args.crop_size)),
            v2.RandomApply([v2.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.7),
            v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            v2.RandomErasing(p=0.5, value='random'),
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float32, scale=True),
            v2.Normalize(mean=args.img_mean, std=args.img_std)
        ])
        dist.print0(f'Using data augmentation pipeline for ViT-based model (DeiT):\n{img_transform}\n')
    elif (args.aug == 'default' and not is_vit) or args.aug == 'simple':
        img_transform = v2.Compose([
            v2.Resize((args.img_size, args.img_size), interpolation=v2.InterpolationMode.BILINEAR),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomCrop((args.crop_size, args.crop_size)),
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float32, scale=True),
            v2.Normalize(mean=args.img_mean, std=args.img_std)
        ])
        dist.print0(f'Using data augmentation pipeline for CNN-based model (ResNet):\n{img_transform}\n')
    elif args.aug == 'autoaug':
        img_transform = timm.data.create_transform(
            input_size=args.crop_size,
            is_training=True,
            no_aug=False,
            color_jitter=0.4,
            hflip=0.5,  # Enable horizontal flip with 50% probability
            vflip=0.5,  # Enable vertical flip with 50% probability
            auto_augment="rand-m9-mstd0.5-inc1",
            interpolation="bicubic",
            mean=args.img_mean,
            std=args.img_std,
            re_prob=0.25,
            re_mode="pixel",
            re_count=1,
        )
        dist.print0(f'Using AutoAugment data augmentation pipeline:\n{img_transform}\n')
    else:
        raise ValueError(f'Unknown augmentation strategy: {args.aug}')

    # Parameter transform (normalization)
    param_transform = None
    if args.param_mean is not None and args.param_std is not None:
        param_transform = training_utils.ParameterNormalize(mean=args.param_mean[:args.param_dim], std=args.param_std[:args.param_dim])

    dist.print0(f'Parameter transform pipeline:\n{param_transform}\n')

    # ------------------------------------------------------------------------------------------
    # Setup dataset

    dset_path = os.path.join(args.dset, args.dset_split) if args.dset_split != 'none' else args.dset
    if args.dset_cls == 'imagefolder':
        train_dataset = dataset.ImageFolderSequentialDataset(
            dset=dset_path,
            param_dir=args.param_dir,
            param_dim=args.param_dim,
            transform=img_transform,
            target_transform=param_transform,
        )
    elif args.dset_cls == 'visual_atoms':
        train_dataset = dataset.VisualAtomsDataset(
            dset=dset_path,
            transform=img_transform,
            target_transform=param_transform,
        )
    else:
        raise NotImplementedError(f'Unknown dataset class: {args.dset_cls}')

    dist.print0(f'Found {len(train_dataset)} images for training in {dset_path}.\n')

    # ------------------------------------------------------------------------------------------
    # Batch size settings

    batch_size_total = args.batch_size  # total batch size for all gpus
    world_size = dist.get_world_size()
    batch_gpu_total = batch_size_total // world_size
    batch_gpu = args.batch_gpu
    if batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total  # limit batch_gpu to batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu

    assert batch_size_total == batch_gpu * num_accumulation_rounds * world_size, f"batch_size ({batch_size_total}) must be equal to batch_gpu ({batch_gpu}) * num_accumulation_rounds ({num_accumulation_rounds}) * world_size ({world_size})"
    assert args.total_nimg % batch_size_total == 0, f"total_nimg ({args.total_nimg}) must be divisible by batch_size ({batch_size_total})"
    assert args.status_nimg is None or args.status_nimg % batch_size_total == 0, f"status_nimg ({args.status_nimg}) must be divisible by batch_size ({batch_size_total})"
    assert args.checkpoint_nimg is None or args.checkpoint_nimg % batch_size_total == 0, f"checkpoint_nimg ({args.checkpoint_nimg}) must be divisible by batch_size ({batch_size_total})"
    assert args.snapshot_nimg is None or args.snapshot_nimg % batch_size_total == 0, f"snapshot_nimg ({args.snapshot_nimg}) must be divisible by batch_size ({batch_size_total})"

    dist.print0(f'Batch size settings: batch_size_total={batch_size_total}, world_size={world_size}, batch_gpu={batch_gpu}, num_accumulation_rounds={num_accumulation_rounds}\n')

    # ------------------------------------------------------------------------------------------
    # Model

    net = training_utils.select_model(args.net, pool=args.pool)
    net = training_utils.replace_projection_head(args.net, net, args.projection_dim)  # Replace the head with a new projection head if specified by the user (e.g., for metric learning)
    net = misc.convert_sync_batchnorm_module(net)  # Convert BatchNorm layers to SyncBatchNorm to share running stats across multiple GPUs
    net.train().requires_grad_(True).to(device)

    if args.freeze_patch_proj:
        if not is_vit:
            raise ValueError(f'freeze_patch_proj is only applicable to ViT-based models like DeiT, but got {args.net}')
        net.patch_embed.requires_grad_(False)
        dist.print0('Froze the patch projection layer in the DeiT model.')

    net_info = torchinfo.summary(net, input_size=(batch_gpu, 3, args.crop_size, args.crop_size), col_names=["input_size", "output_size", "num_params", "trainable"], device=str(device), verbose=0)
    dist.print0(f'Model architecture ({args.net}):\n{net_info}\n')

    # ------------------------------------------------------------------------------------------
    # Optimizer

    if is_vit:
        scaled_lr = args.lr * batch_size_total / 256.0  # Scale the base learning rate by the effective batch size (Reference: https://arxiv.org/abs/2104.02057)
        dist.print0(f'Applying learning rate scaling for ViT-based model (DeiT): {batch_size_total}: {args.lr} --> {scaled_lr}')

        optimizer = timm.optim.create_optimizer_v2(model_or_params=net, opt='adamw', lr=scaled_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        scaled_lr = args.lr * num_accumulation_rounds * world_size
        dist.print0(f'Applying learning rate scaling for CNN-based model (ResNet): {args.lr} --> {scaled_lr}')

        # Use SGD momentum optimizer for ResNet, which is used in the parper "Pre-training without Natural Images" (ACCV 2020).
        optimizer = torch.optim.SGD(net.parameters(), lr=scaled_lr, momentum=args.momentum, weight_decay=args.weight_decay)

    dist.print0(f'Optimizer:\n{optimizer}\n')

    # ------------------------------------------------------------------------------------------
    # Loss function

    loss_cls = metric_loss.loss_catalog[args.loss]
    loss_func = loss_cls(margin=args.margin, miner=args.miner, log_stats=not args.disable_report_triplet_stats)

    # ------------------------------------------------------------------------------------------
    # Scheduler

    total_steps = args.total_nimg // batch_size_total

    if is_vit:
        # NOTE: The scheduler settings are based on Moco v3 pre-training recipe
        # Please translate 'epochs' to 'steps'
        scheduler_args = {
            'num_epochs': total_steps,
            # Warm up
            'warmup_lr': 0.0,
            'warmup_epochs': 10_000,
            # Cosine
            'sched': 'cosine',
            'min_lr': 0.0,
            # Misc
            'noise_seed': global_seed,
        }

        scheduler, _ = timm.scheduler.create_scheduler_v2(optimizer=optimizer, **scheduler_args)
    else:
        scheduler = training_utils.CosineWithConstantStartsLR(
            optimizer=optimizer,
            constant_steps=int(total_steps * 0.4),  # Start with 40% of the total steps as a constant learning rate phase, then switch to cosine decay for the remaining 60% of the training.
            total_steps=total_steps,
            eta_min=optimizer.param_groups[0]['lr'] * 0.01,  # Set the minimum learning rate for cosine decay to be 1% of the initial learning rate.
        )

    dist.print0(f'LR Scheduler:\n{scheduler}\n')

    # ------------------------------------------------------------------------------------------
    # Training state

    state = util.EasyDict(total_elapsed_time=0.0, cumulative_training_time=0.0, cur_nimg=0, cur_step=0)

    # ------------------------------------------------------------------------------------------
    # Create DistributedDataParallel model

    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device]) if torch.distributed.is_initialized() else net

    if args.compile != 'none':
        ddp = training_utils.compile_module(ddp, mode=args.compile)
        dist.print0('Model will be compiled with torch.compile() in the training loop. Notice that this may lead to non-deterministic behavior.\n')

    # ------------------------------------------------------------------------------------------
    # Setup dataloader with DistributedSampler

    sampler_kwargs = dict(
        rank=dist.get_rank(),
        num_replicas=dist.get_world_size(),
        shuffle=True,
        seed=global_seed,
    )

    loader_kwargs = util.EasyDict(
        dataset=train_dataset,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,                                                                         # NOTE: Keep workers alive to avoid overhead of respawning them
        prefetch_factor=2,                                                                               # NOTE: Number of samples loaded in advance by each worker
        worker_init_fn=functools.partial(training_utils.seed_everything, global_seed, dist.get_rank()),  # NOTE: Use different seeds for each worker, otherwise they generate the same images
    )

    if args.sampler == 'random':
        loader_kwargs.batch_size = batch_gpu
        loader_kwargs.sampler = misc.InfiniteSampler(dataset=train_dataset, **sampler_kwargs)
    elif args.sampler == 'pk':
        assert args.insts_per_class is not None, 'insts_per_class must be specified when using pk sampler'
        assert batch_gpu % args.insts_per_class == 0, f"batch_gpu ({batch_gpu}) must be divisible by insts_per_class ({args.insts_per_class})"
        dist.print0(f'Batch size {batch_gpu} = {batch_gpu // args.insts_per_class} classes x {args.insts_per_class} instances per class\n')
        loader_kwargs.batch_sampler = dataset.PKBatchSampler(labels=train_dataset._cls_labels, P=batch_gpu // args.insts_per_class, K=args.insts_per_class, **sampler_kwargs)
    else:
        raise NotImplementedError(f'Unknown sampler: {args.sampler}')

    dataset_iterator = iter(torch.utils.data.DataLoader(**loader_kwargs))

    # ------------------------------------------------------------------------------------------
    # Resume from snapshot if specified

    if args.resume_snapshot is not None:
        with util.open_url(args.resume_snapshot) as file:
            snapshot = torch.load(file, map_location='cpu')

            net.load_state_dict(snapshot['state_dict'])
            optimizer.load_state_dict(snapshot['optimizer'])
            scheduler.load_state_dict(snapshot['scheduler'])
            state.cur_step = snapshot['cur_step']
            state.cur_nimg = snapshot['cur_nimg']

            # Check if DDP models are consistent after loading
            misc.check_ddp_consistency(net)

            dist.print0(f'Resumed training from snapshot: step: {state.cur_step}, nimg: {state.cur_nimg}, snapshot: {args.resume_snapshot}\n', flush=True)

    # ------------------------------------------------------------------------------------------
    # Start the training loop

    stop_at_nimg = args.total_nimg
    prev_status_nimg = state.cur_nimg

    assert stop_at_nimg > state.cur_nimg, f'total_nimg ({stop_at_nimg}) must be greater than the number of images already trained on (cur_nimg: {state.cur_nimg})'

    dist.print0(f'Training from {state.cur_nimg // 1000} kimg to {stop_at_nimg // 1000} kimg:\n')

    start_nimg = state.cur_nimg
    stats_jsonl = None

    batch_data_gpu = None

    while True:
        done = (state.cur_nimg >= stop_at_nimg)

        # Report status.
        if args.status_nimg is not None and (done or state.cur_nimg % args.status_nimg == 0) and (state.cur_nimg != start_nimg or start_nimg == 0):
            cur_time = time.time()
            state.total_elapsed_time += cur_time - prev_status_time
            cur_process = psutil.Process(os.getpid())
            cpu_memory_usage = sum(p.memory_info().rss for p in [cur_process] + cur_process.children(recursive=True))
            cur_epoch = state.cur_nimg / len(train_dataset)
            dist.print0(' '.join(
                [
                    'Status:',
                    'kimg',         f"{training_stats.report0('Progress/kimg',                              state.cur_nimg / 1e3):<9.1f}",
                    'step',         f"{training_stats.report0('Progress/step',                              state.cur_step):<7.0f}",
                    'epoch',        f"{training_stats.report0('Progress/epoch',                             cur_epoch):<7.2f}",
                    'time',         f"{util.format_time(training_stats.report0('Timing/total_sec',          state.total_elapsed_time)):<12s}",
                    'sec/tick',     f"{training_stats.report0('Timing/sec_per_tick',                        cur_time - prev_status_time):<8.2f}",
                    'sec/kimg',     f"{training_stats.report0('Timing/sec_per_kimg',                        state.cumulative_training_time / max(state.cur_nimg - prev_status_nimg, 1) * 1e3):<7.3f}",
                    'maintenance',  f"{training_stats.report0('Timing/maintenance_sec',                     cur_time - prev_status_time - state.cumulative_training_time):<7.2f}",
                    'cpumem',       f"{training_stats.report0('Resources/cpu_mem_gb',                       cpu_memory_usage / 2**30):<6.2f}",
                    'gpumem',       f"{training_stats.report0('Resources/peak_gpu_mem_gb',                  torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}",
                    'reserved',     f"{training_stats.report0('Resources/peak_gpu_mem_reserved_gb',         torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}",
                ]))
            state.cumulative_training_time = 0.0
            prev_status_nimg = state.cur_nimg
            prev_status_time = cur_time
            torch.cuda.reset_peak_memory_stats()

            # Flush training stats.
            training_stats.default_collector.update()
            if dist.get_rank() == 0:

                if stats_jsonl is None:
                    stats_jsonl = open(os.path.join(args.run_dir, 'stats.jsonl'), 'at')
                fmt = {'Progress/tick': '%.0f', 'Progress/kimg': '%.3f', 'Progress/step': '%.0f', 'Progress/epoch': '%.2f', 'timestamp': '%.3f'}
                items = [(name, value.mean) for name, value in training_stats.default_collector.as_dict().items()] + [('timestamp', time.time())]
                items = [f'"{name}": ' + (fmt.get(name, '%g') % value if np.isfinite(value) else 'NaN') for name, value in items]
                stats_jsonl.write('{' + ', '.join(items) + '}\n')
                stats_jsonl.flush()

                if wandb_enabled():
                    # Log to the wandb.
                    value_keys = list(training_stats.default_collector.as_dict().keys())
                    values = training_stats.default_collector.as_dict()
                    means, stds = {}, {}
                    for key in value_keys:
                        if key in [
                            'Progress/kimg', 'Progress/step', 'Progress/epoch',
                            'Timing/total_sec', 'Timing/sec_per_tick', 'Timing/sec_per_kimg', 'Timing/maintenance_sec',
                            'Resources/cpu_mem_gb', 'Resources/peak_gpu_mem_gb', 'Resources/peak_gpu_mem_reserved_gb',
                            'metric_stats/num_triplets', 'metric_stats/num_unique_anchors', 'metric_stats/hard_ratio', 'metric_stats/semi_hard_ratio', 'metric_stats/easy_ratio', 'metric_stats/nonzero_ratio', 'metric_stats/ap_dist_mean', 'metric_stats/ap_dist_std', 'metric_stats/an_dist_mean', 'metric_stats/an_dist_std', 'metric_stats/dist_gap_mean',
                        ]:
                            means[key] = values[key].mean
                        else:
                            means[f'{key}_mean'] = values[key].mean
                            if values[key].std != 0:
                                stds[f'{key}_std'] = values[key].std

                    to_log = {**means, **stds}
                    wandb.log(to_log, step=state.cur_step)

            # Create a grid and report it to wandb
            if dist.get_rank() == 0 and batch_data_gpu is not None:
                num_imgs = min(batch_gpu, 8 * 8)  # limit to 64 images
                imgs: torch.Tensor = batch_data_gpu['imgs'][:num_imgs].clone()

                # Undo normalization
                img_mean = torch.tensor(args.img_mean).reshape(1, 3, 1, 1).to(device=imgs.device)
                img_std = torch.tensor(args.img_std).reshape(1, 3, 1, 1).to(device=imgs.device)
                imgs = (imgs * img_std + img_mean).clamp(0.0, 1.0)

                # Make grid
                nrows = math.ceil(math.sqrt(num_imgs))
                grid_img = torchvision.utils.make_grid(imgs, nrow=nrows, padding=5, pad_value=1.0)  # add white padding
                grid_img_pil = torchvision.transforms.functional.to_pil_image(grid_img)

                # Save to disk
                grid_fname_stem = f'grid-{state.cur_step}step-{state.cur_nimg // 1000}kimg'
                grid_path = os.path.join(args.run_dir, 'training_imgs', grid_fname_stem + '.png')
                os.makedirs(os.path.dirname(grid_path), exist_ok=True)
                grid_img_pil.save(grid_path)

                # Report to wandb
                if wandb_enabled():
                    wandb_grid_path = os.path.join(args.run_dir, 'training_imgs', grid_fname_stem + '.avif')
                    grid_img_pil.save(wandb_grid_path, format='AVIF')  # Save in AVIF format for smaller file size
                    wandb.log({'train/images': wandb.Image(wandb_grid_path, caption=f'step: {state.cur_step}, kimg: {state.cur_nimg // 1000}')}, step=state.cur_step)

            # Update progress and check for abort.
            dist.update_progress(state.cur_nimg // 1000, stop_at_nimg // 1000)
            if state.cur_nimg == stop_at_nimg and state.cur_nimg < args.total_nimg:
                dist.request_suspend()
            if dist.should_stop() or dist.should_suspend():
                done = True

        # Save state checkpoint.
        if args.checkpoint_nimg is not None and (done or state.cur_nimg % args.checkpoint_nimg == 0) and state.cur_nimg != start_nimg:
            # Check if DDP models are consistent before saving
            misc.check_ddp_consistency(net)

            if dist.get_rank() == 0:
                checkpoint_path = os.path.join(args.run_dir, 'checkpoints', f'checkpoint-{state.cur_nimg // 1000}kimg-{cur_epoch:.2f}ep.pt')
                state_dict = net.state_dict()
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save(state_dict, checkpoint_path)
                dist.print0(f'Saved checkpoint: {checkpoint_path}')

        # Save training snapshot.
        if args.snapshot_nimg is not None and (done or state.cur_nimg % args.snapshot_nimg == 0) and state.cur_nimg != start_nimg:
            # Check if DDP models are consistent before saving
            misc.check_ddp_consistency(net)

            if dist.get_rank() == 0:
                checkpoint_path = os.path.join(args.run_dir, 'snapshots', f'{state.cur_step}step-{state.cur_nimg // 1000}kimg-{cur_epoch:.2f}ep.pt')
                state_dict = {
                    'cur_step': state.cur_step,
                    'cur_nimg': state.cur_nimg,
                    'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save(state_dict, checkpoint_path)
                dist.print0(f'Saved snapshot: {checkpoint_path}')

        # Done?
        if done:
            break

        net.train()

        batch_start_time = time.time()
        training_utils.seed_everything(global_seed, dist.get_rank(), state.cur_nimg)
        optimizer.zero_grad(set_to_none=True)

        # compute loss
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                batch_data = next(dataset_iterator)

                # ------------------------------------------------------------
                # Copy to device

                batch_data_gpu = {k: v.to(device=device, non_blocking=True) for k, v in batch_data.items()}

                # ------------------------------------------------------------
                # Compute loss

                with torch.amp.autocast(device_type='cuda', dtype=compute_dtype, enabled=is_amp_enabled):
                    loss = loss_func(net=ddp, **batch_data_gpu)

                training_stats.report('train/loss', loss)

                loss.div(num_accumulation_rounds).backward()  # NOTE: Normalize the loss to account for gradient accumulation. this assumes the number of triplets are roughly the same in each accumulation round.

        # Report lr
        training_stats.report('train/lr', optimizer.param_groups[0]['lr'])

        # update weights
        optimizer.step()

        # update lr scheduler
        if isinstance(scheduler, timm.scheduler.scheduler.Scheduler):
            scheduler.step(epoch=state.cur_step)  # timm schedulers require an explicit count (interpreted as step count here)
        else:
            scheduler.step()

        # Update training state.
        state.cur_nimg += batch_size_total
        state.cumulative_training_time += time.time() - batch_start_time
        state.cur_step += 1  # Advance step counter.

        if done:
            break

# ---------------------------------------------------------------------------------------------------------------------------
