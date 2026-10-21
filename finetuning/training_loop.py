"""Training loop for fine-tuning.

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
import re
import time

import numpy as np
import psutil
import timm.data
import timm.loss
import timm.optim
import timm.scheduler
import torch
import torchinfo
import torchvision
import torchvision.transforms.v2 as v2
import wandb

import torch_utils.distributed as dist
import torch_utils.misc as misc
import torch_utils.training_stats as training_stats
import torch_utils.util as util
import training_utils

# ---------------------------------------------------------------------------------------------------------------------------
# Utilities


def parse_eval_intervals(eval_intervals_str: str, total_epochs: int):
    eval_epochs = list()

    for part in eval_intervals_str.split(','):
        target_epoch_str, interval_str = part.split(':')

        match = re.match(r'(\d*)-(\d*|)', target_epoch_str)
        if not match:
            raise ValueError(f'Invalid eval_intervals format: {part}')

        start_epoch_str, end_epoch_str = match.groups()
        start_epoch = int(start_epoch_str) if start_epoch_str else 0
        end_epoch = min(total_epochs, int(end_epoch_str) if end_epoch_str else total_epochs) + 1
        if start_epoch >= end_epoch:
            continue  # skip invalid interval

        interval = max(1, int(interval_str))

        eval_epochs.extend(list(range(start_epoch, end_epoch, interval)))

    eval_epochs = sorted(set(eval_epochs))  # remove duplicates and sort

    return eval_epochs

# ---------------------------------------------------------------------------------------------------------------------------
# Training loop


def training_loop(args: util.EasyDict) -> None:
    """Main training loop for fine-tuning.  This function is called by each process in the distributed setup. Each process will run this function independently, and they will coordinate with each other using torch.distributed."""

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
            project="fdsml-finetuning",
            name=args.desc,
            config=args.to_stddict(),
            save_code=True,
            dir=args.run_dir,
            tags=[tag.strip() for tag in args.desc.split('-')],
        )

        dist.print0(f'Weights & Biases logging enabled, run directory: {args.run_dir}\n')

    # ------------------------------------------------------------------------------------------
    # Setup data augmentation

    is_vit = training_utils.is_vit(args.net)

    if is_vit:
        transform_args = util.EasyDict(
            input_size=args.input_size,
            scale=(0.08, 1.0),
            ratio=(0.75, 1.3333),
            no_aug=False,
            hflip=0.5,
            vflip=0.0,
            color_jitter=0.4,
            auto_augment='rand-m9-mstd0.5-inc1',
            interpolation='bicubic',
            mean=args.img_mean,
            std=args.img_std,
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
        )

        train_transform = timm.data.create_transform(**transform_args, is_training=True)
        val_transform = timm.data.create_transform(**transform_args, is_training=False)
    else:
        train_transform = v2.Compose([
            v2.Resize((256, 256), interpolation=v2.InterpolationMode.BILINEAR),
            v2.RandomCrop((args.input_size, args.input_size)),
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float32, scale=True),
            v2.Normalize(mean=args.img_mean, std=args.img_std),
        ])
        val_transform = v2.Compose([
            v2.Resize((256, 256), interpolation=v2.InterpolationMode.BILINEAR),
            v2.CenterCrop((args.input_size, args.input_size)),
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float32, scale=True),
            v2.Normalize(mean=args.img_mean, std=args.img_std),
        ])

    dist.print0(f'Training data augmentation:\n{train_transform}\n')
    dist.print0(f'Validation data augmentation:\n{val_transform}\n')

    # ------------------------------------------------------------------------------------------
    # Setup dataset

    train_dset_path = os.path.join(args.dset, 'train')
    train_dset = torchvision.datasets.ImageFolder(train_dset_path, transform=train_transform)

    val_dset_path = os.path.join(args.dset, args.val_split)
    val_dset = torchvision.datasets.ImageFolder(val_dset_path, transform=val_transform)

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
    # Setup dataloaders

    train_dset_iterator = iter(torch.utils.data.DataLoader(
        dataset=train_dset,
        batch_size=batch_gpu,
        sampler=misc.InfiniteSampler(dataset=train_dset, rank=dist.get_rank(), num_replicas=world_size, shuffle=True, seed=global_seed),
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,                                                                         # NOTE: Keep workers alive to avoid overhead of respawning them
        prefetch_factor=2,                                                                               # NOTE: Number of samples loaded in advance by each worker
        worker_init_fn=functools.partial(training_utils.seed_everything, global_seed, dist.get_rank()),  # NOTE: Use different seeds for each worker, otherwise they generate the same images
    ))

    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dset,
        batch_size=batch_gpu,
        sampler=torch.utils.data.DistributedSampler(dataset=val_dset, rank=dist.get_rank(), num_replicas=world_size, shuffle=False, seed=global_seed, drop_last=False),
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=functools.partial(training_utils.seed_everything, global_seed, dist.get_rank() + 1024),  # NOTE: Use different seeds for each worker, otherwise they generate the same images
    )

    # ------------------------------------------------------------------------------------------
    # Setup mixup

    if is_vit and (args.mixup_alpha > 0.0 or args.cutmix_alpha > 0.0):
        mixup = timm.data.mixup.Mixup(mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha, prob=1.0, switch_prob=0.5, mode='batch', label_smoothing=args.label_smoothing, num_classes=len(train_dset.classes))
    else:
        mixup = None

    dist.print0(f'Mixup settings: {mixup}\n')

    # ------------------------------------------------------------------------------------------
    # Model

    net_kwargs = util.EasyDict(net=args.net)
    if is_vit:
        net_kwargs.drop_rate = args.drop_rate
        net_kwargs.drop_path_rate = args.drop_path_rate

    net = training_utils.select_model(**net_kwargs)
    net = training_utils.replace_projection_head(args.net, net, len(train_dset.classes))  # Replace the head with a new projection head if specified by the user (e.g., for metric learning)

    if args.ckpt is not None:
        with util.open_url(args.ckpt) as f:
            ckpt = torch.load(f, map_location='cpu')

        # Unpack the checkpoint if it contains a 'model' key
        if 'model' in ckpt:
            ckpt = ckpt['model']

        ignore_keys = [
            # Remove keys from Moco v3 style projection head if they exist in the checkpoint.
            "head.0.weight", "head.1.weight", "head.1.bias", "head.1.running_mean", "head.1.running_var", "head.1.num_batches_tracked", "head.3.weight",
            # Remove keys from the original projection head if they exist in the checkpoint.
            "head.weight", "head.bias",
            # Remove keys from the ResNet projection head (fc layer) if they exist in the checkpoint.
            "fc.weight", "fc.bias",
        ]
        for k in ignore_keys:
            if k in ckpt:
                dist.print0(f'Remove key [{k}] from pretrained checkpoint')
                del ckpt[k]

        imcompatibles = net.load_state_dict(ckpt, strict=False)  # Load the checkpoint with strict=False to allow missing keys (e.g., if the checkpoint has a different head)

        dist.print0(f'Loaded checkpoint from {args.ckpt}\n')
        if imcompatibles.missing_keys:
            dist.print0(f'    Missing keys when loading checkpoint (expected if the checkpoint has a different head): {imcompatibles.missing_keys}\n')
        if imcompatibles.unexpected_keys:
            dist.print0(f'    Unexpected keys when loading checkpoint: {imcompatibles.unexpected_keys}\n')

    net = misc.convert_sync_batchnorm_module(net)  # Convert BatchNorm layers to SyncBatchNorm to share running stats across multiple GPUs
    net.train().requires_grad_(True).to(device)

    if is_vit and args.freeze_patch_proj:
        net.patch_embed.requires_grad_(False)
        dist.print0('Froze the patch projection layer in the DeiT model.')

    net_info = torchinfo.summary(net, input_size=(batch_gpu, 3, args.input_size, args.input_size), col_names=["input_size", "output_size", "num_params", "trainable"], device=str(device), verbose=0)
    dist.print0(f'Model architecture ({args.net}):\n{net_info}\n')

    # ------------------------------------------------------------------------------------------
    # Create DistributedDataParallel model

    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device]) if torch.distributed.is_initialized() else net

    if args.compile != 'none':
        ddp = training_utils.compile_module(ddp, mode=args.compile)
        dist.print0('Model will be compiled with torch.compile() in the training loop. Notice that this may lead to non-deterministic behavior.\n')

    # ------------------------------------------------------------------------------------------
    # Optimizer

    if is_vit:
        scaled_lr = args.lr * batch_size_total / 512.0
        scaled_warmup_lr = args.warmup_lr * batch_size_total / 512.0
        scaled_min_lr = args.min_lr * batch_size_total / 512.0
        optimizer = timm.optim.create_optimizer_v2(model_or_params=net, opt=args.opt, lr=scaled_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    dist.print0(f'Optimizer: {optimizer}\n')

    # ------------------------------------------------------------------------------------------
    # Scheduler

    num_train_imgs = len(train_dset)
    total_epochs = math.ceil(args.total_nimg / num_train_imgs)
    dist.print0(f'Total training images: {args.total_nimg}, Total training image data: {num_train_imgs}, Total epochs: {total_epochs}, \n')

    if is_vit:
        scheduler, _ = timm.scheduler.create_scheduler_v2(
            optimizer=optimizer,
            sched=args.sched,
            warmup_lr=scaled_warmup_lr,
            min_lr=scaled_min_lr,
            decay_epochs=30,
            cooldown_epochs=10,
            patience_epochs=10,
            decay_rate=0.1,
            warmup_epochs=5,
            step_on_epochs=True,
            num_epochs=total_epochs,
            noise_seed=global_seed,
        )
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)

    dist.print0(f'LR Scheduler:\n{scheduler}\n')

    # ------------------------------------------------------------------------------------------
    # Loss function

    if mixup is not None:
        loss_fn = timm.loss.SoftTargetCrossEntropy()  # Mixup/CutMix produces soft (one-hot) targets
    elif args.label_smoothing > 0.0:
        loss_fn = timm.loss.LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    loss_fn.to(device)

    dist.print0(f'Loss function: {loss_fn}\n')

    # ------------------------------------------------------------------------------------------
    # Training state

    state = util.EasyDict(total_elapsed_time=0.0, cumulative_training_time=0.0, cur_nimg=0, cur_step=0, cur_epoch=0.0)

    # ------------------------------------------------------------------------------------------
    # Start the training loop

    stop_at_nimg = args.total_nimg
    prev_status_nimg = state.cur_nimg
    prev_eval_epoch = -1  # To track when to run evaluation based on epoch intervals
    best_val_acc = 0.0  # To track the best validation accuracy achieved during training
    eval_epochs = parse_eval_intervals(args.eval_intervals, total_epochs)

    assert stop_at_nimg > state.cur_nimg, f'total_nimg ({stop_at_nimg}) must be greater than the number of images already trained on (cur_nimg: {state.cur_nimg})'

    dist.print0(f'Training from {state.cur_nimg // 1000} kimg to {stop_at_nimg // 1000} kimg, equivalent to {total_epochs} epochs.\n')
    dist.print0(f'Run evaluation at epochs: {eval_epochs}\n')

    start_nimg = state.cur_nimg
    stats_jsonl = None

    while True:
        done = (state.cur_nimg >= stop_at_nimg)

        # Report status.
        if args.status_nimg is not None and (done or state.cur_nimg % args.status_nimg == 0) and (state.cur_nimg != start_nimg or start_nimg == 0):
            cur_time = time.time()
            state.total_elapsed_time += cur_time - prev_status_time
            cur_process = psutil.Process(os.getpid())
            cpu_memory_usage = sum(p.memory_info().rss for p in [cur_process] + cur_process.children(recursive=True))
            dist.print0(' '.join(
                [
                    'Status:',
                    'kimg',         f"{training_stats.report0('Progress/kimg',                              state.cur_nimg / 1e3):<9.1f}",
                    'step',         f"{training_stats.report0('Progress/step',                              state.cur_step):<7.0f}",
                    'epoch',        f"{training_stats.report0('Progress/epoch',                             state.cur_epoch):<7.2f}",
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
                            'val/acc', 'val/best_acc',
                        ]:
                            means[key] = values[key].mean
                        else:
                            means[f'{key}_mean'] = values[key].mean
                            if values[key].std != 0:
                                stds[f'{key}_std'] = values[key].std

                    to_log = {**means, **stds}
                    wandb.log(to_log, step=state.cur_step)

            # Update progress and check for abort.
            dist.update_progress(state.cur_nimg // 1000, stop_at_nimg // 1000)
            if state.cur_nimg == stop_at_nimg and state.cur_nimg < args.total_nimg:
                dist.request_suspend()
            if dist.should_stop() or dist.should_suspend():
                done = True

        # Evaluate on the validation set.
        if (int(state.cur_epoch) in eval_epochs and int(state.cur_epoch) > prev_eval_epoch) or done:
            prev_eval_epoch = int(state.cur_epoch)
            dist.print0(f'Running evaluation at epoch {state.cur_epoch:.2f} ({state.cur_nimg // 1000} kimg) ... ', end='', flush=True)

            dist.barrier()  # Ensure all processes have finished the current epoch before starting evaluation

            net.eval()

            loss, corrects, total_num = torch.tensor(0.0, device=device), torch.tensor(0, device=device), torch.tensor(0, device=device)
            with torch.no_grad():
                for val_imgs, val_labels in val_dataloader:
                    val_imgs = val_imgs.to(device=device, non_blocking=True)
                    val_labels = val_labels.to(device=device, non_blocking=True)

                    with torch.amp.autocast(device_type='cuda', dtype=compute_dtype, enabled=is_amp_enabled):
                        val_out = ddp(val_imgs)

                        val_loss = torch.nn.functional.cross_entropy(val_out, val_labels, reduction='sum')

                        pred_cls = val_out.argmax(dim=1)
                        correct = (pred_cls == val_labels).sum()

                        loss += val_loss
                        corrects += correct
                        total_num += val_labels.size(0)

            # Synchronize and aggregate metrics across processes
            dist.barrier()
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(corrects, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(total_num, op=torch.distributed.ReduceOp.SUM)

            # Report.
            val_loss = (loss / total_num).item()
            val_acc = (corrects / total_num).item()

            dist.print0(f'Validation loss: {val_loss:.4f}, accuracy: {val_acc:.4f} ({corrects.item()}/{total_num.item()}).')

            best_recorded = False
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_recorded = True
                dist.print0(f'New best validation accuracy: {best_val_acc:.4f} at epoch {state.cur_epoch:.2f} ({state.cur_nimg // 1000} kimg)')

            training_stats.report0('val/loss', val_loss)
            training_stats.report0('val/acc', val_acc)
            training_stats.report0('val/best_acc', best_val_acc)

            # Save best checkpoint if this is the best validation accuracy so far
            if best_recorded:
                # Check if DDP models are consistent before saving
                misc.check_ddp_consistency(net)

                if dist.get_rank() == 0:
                    checkpoint_path = os.path.join(args.run_dir, f'best_checkpoint-{state.cur_nimg // 1000}kimg-{state.cur_epoch:.2f}ep.pt')
                    state_dict = net.state_dict()
                    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                    torch.save(state_dict, checkpoint_path)
                    dist.print0(f'Saved new best checkpoint: {checkpoint_path}')

            dist.print0('done.')

        # Save state checkpoint.
        if args.checkpoint_nimg is not None and (done or state.cur_nimg % args.checkpoint_nimg == 0) and state.cur_nimg != start_nimg:
            # Check if DDP models are consistent before saving
            misc.check_ddp_consistency(net)

            if dist.get_rank() == 0:
                checkpoint_path = os.path.join(args.run_dir, 'checkpoints', f'checkpoint-{state.cur_nimg // 1000}kimg-{state.cur_epoch:.2f}ep.pt')
                state_dict = net.state_dict()
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save(state_dict, checkpoint_path)
                dist.print0(f'Saved checkpoint: {checkpoint_path}')

        # Save training snapshot.
        if args.snapshot_nimg is not None and (done or state.cur_nimg % args.snapshot_nimg == 0) and state.cur_nimg != start_nimg:
            # Check if DDP models are consistent before saving
            misc.check_ddp_consistency(net)

            if dist.get_rank() == 0:
                checkpoint_path = os.path.join(args.run_dir, 'snapshots', f'{state.cur_step}step-{state.cur_nimg // 1000}kimg-{state.cur_epoch:.2f}ep.pt')
                state_dict = {
                    'cur_step': state.cur_step,
                    'cur_nimg': state.cur_nimg,
                    'cur_epoch': state.cur_epoch,
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
                batch_data = next(train_dset_iterator)

                # ------------------------------------------------------------
                # Copy to device

                imgs_gpu = batch_data[0].to(device=device, non_blocking=True)
                labels_gpu = batch_data[1].to(device=device, non_blocking=True)

                # ------------------------------------------------------------
                # Apply mixup

                if mixup is not None:
                    imgs_gpu, labels_gpu = mixup(imgs_gpu, labels_gpu)

                # ------------------------------------------------------------
                # Compute loss

                with torch.amp.autocast(device_type='cuda', dtype=compute_dtype, enabled=is_amp_enabled):
                    out = ddp(imgs_gpu)
                    loss = loss_fn(out, labels_gpu)

                training_stats.report('train/loss', loss)

                loss.div(num_accumulation_rounds).backward()  # Scale the loss by num_accumulation_rounds to get the correct gradient magnitude

        # Report lr
        training_stats.report('train/lr', optimizer.param_groups[0]['lr'])

        # update weights
        optimizer.step()

        # update lr scheduler
        scheduler.step(epoch=int(state.cur_epoch))

        # Update training state.
        state.cur_nimg += batch_size_total
        state.cumulative_training_time += time.time() - batch_start_time
        state.cur_step += 1  # Advance step counter.
        state.cur_epoch = state.cur_nimg / num_train_imgs

        if done:
            break

# ---------------------------------------------------------------------------------------------------------------------------
