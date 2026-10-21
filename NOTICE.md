# NOTICE

---

## 1. EDM2

This project is partially based on the source code and README structure of NVlabs/edm2:
https://github.com/NVlabs/edm2

Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.

The original code is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0).

See the original [edm2 license](./docs/LICENSE_edm2.txt) for more details.

### 1.1 Modifications

The EDM2-derived code in this repository has been substantially modified and repurposed for FDSML-specific supervised and metric-learning workflows. In particular, the following files contain code derived from or inspired by `NVlabs/edm2`:

- `torch_utils/__init__.py`
- `torch_utils/distributed.py`
- `torch_utils/persistence.py`
- `torch_utils/training_stats.py`
- `torch_utils/util.py`
- `torch_utils/misc.py`
- `finetuning/training_loop.py`
- `metric_learning/training_loop.py`

The main modifications include:

- Reorganized EDM2's general PyTorch utility modules under `torch_utils` for use in FDSML training, fine-tuning, checkpointing, logging, and distributed execution.
- Added FDSML-specific license headers and attribution notices while retaining the original NVIDIA copyright and CC BY-NC-SA 4.0 notice where applicable.
- Adapted distributed initialization utilities for FDSML's `torchrun`-based workflows, including explicit backend selection, CUDA device assignment by local rank, and a reusable distributed barrier helper.
- Reused and adapted EDM2's checkpoint I/O, rank-aware printing, distributed rank/world-size helpers, and multiprocessing-aware training-statistics synchronization.
- Retained and adapted EDM2's persistence utilities for pickling Python class definitions together with model state, enabling compatibility with serialized training artifacts and checkpoints.
- Reused and adapted EDM2's training-statistics collection and reporting utilities for FDSML metrics, including progress, timing, resource usage, training loss, validation metrics, and Weights & Biases logging.
- Reused and adapted EDM2's miscellaneous PyTorch helpers, including random seeding, infinite distributed sampling, DDP synchronization control, DDP consistency checks, parameter/buffer utilities, and module-summary helpers.
- Added or reorganized general utility functions in `torch_utils/util.py`, including dictionary-style configuration objects, cache-directory handling, URL/file opening helpers, time formatting, seeding helpers, and related FDSML infrastructure utilities.
- Reworked EDM2's diffusion-model training-loop structure into FDSML training loops for image classification fine-tuning and supervised metric-learning pre-training.
- Removed EDM2-specific diffusion components such as latent encoders, denoising losses, uncertainty-weighted EDM2 loss computation, diffusion network construction, EMA snapshot export, and EDM2-specific learning-rate schedule logic where they are not used by FDSML.
- Added FDSML-specific dataset handling based on image-folder datasets, FractalDB/FineFractalDB parameter files, and downstream classification datasets.
- Added FDSML-specific model construction through the project model-selection utilities, including ResNet/DeiT/timm model selection, projection-head replacement, optional patch-projection freezing, SyncBatchNorm conversion, and optional `torch.compile` integration.
- Added supervised fine-tuning support, including classification losses, timm data augmentation, mixup/cutmix, label smoothing, timm optimizers and schedulers, validation loops, best-checkpoint saving, and validation metric reporting.
- Added metric-learning pre-training support, including parameter-aware triplet-learning workflows, miner strategies, projection embeddings, and metric-learning loss/statistics reporting.
- Added FDSML-specific checkpoint and snapshot formats, including PyTorch state-dict checkpoints, optimizer/scheduler snapshots, best-validation checkpoints, and run-directory-based logging.
- Added experiment tracking and reporting integrations such as Weights & Biases logging and richer status output including step, epoch, validation loss, validation accuracy, best accuracy, learning rate, timing, CPU memory, and GPU memory statistics.
- Updated README-derived documentation and command examples to describe FDSML installation, dataset generation, FineFractalDB creation, pre-training, fine-tuning, model zoo entries, licensing, citation, and acknowledgements instead of EDM2 diffusion-model training.

Unless otherwise stated, all modifications are also distributed under the same CC BY-NC-SA 4.0 license.

## 2. Improved Fractal Pretraining

Some parts of this repository are derived from:
https://github.com/catalys1/fractal-pretraining

Copyright (c) 2023, Connor Anderson.

The original repository is licensed under the MIT License.

See the original [fractal-pretraining license](./docs/LICENSE_fractal-pretraining.txt) for more details.

---

## 3. PyViewer

Some parts of this repository are derived from:
https://github.com/harskish/pyviewer

Copyright (c) 2022, Erik Härkönen.

The original code is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0).

See the original [pyviewer license](./docs/LICENSE_pyviewer.txt) for more details.

### 3.1 Modifications

The PyViewer-derived code in this repository has been substantially modified for FDSML-specific visualization workflows. In particular, the following files contain code derived from or inspired by `harskish/pyviewer`:

- `viz/ui/docking_viewer.py`
- `viz/ui/drawable_window.py`
- `viz/ui/drawable_area.py`
- `viz/ui/utils.py`
- `viz/ui/cuda_gl_interop.py`
- `viz/ui/custom_ops/cuda_gl_interop.cpp`

The main modifications include:

- Reworked PyViewer's original single-output `DockingViewer` design into a multi-texture docking viewer for displaying and updating multiple image/texture panels independently.
- Replaced the original single `PannableArea`/main-output image path with a `DrawableTextureWindows` abstraction that manages multiple dockable texture windows.
- Added support for drawable image regions and interactive mask editing, including brush-based drawing, brush-size control, mask inversion, draw-to-show/draw-to-hide modes, and optional origin-symmetric drawing.
- Added per-texture interaction features such as hover highlighting, grid-cell highlighting, per-cell attribute text overlays, and configurable highlight behavior.
- Extended the original panning and zooming behavior by deriving `DrawableArea` from PyViewer's `PannableArea`, adding drawing-mode handling while preserving the original pan/zoom interactions.
- Reorganized PyViewer-related UI utilities into FDSML-specific helper modules, including dockable-window registration, shared UI locking, logging utilities, parameter widgets, string enums, and lightweight dictionary-style state containers.
- Modified the viewer lifecycle and layout logic to support multiple generated dockable windows, texture-window callback registration, texture creation/disposal, optional fullscreen mode, optional vsync, and viewport-enabled docking behavior.
- Adapted font loading and theme setup to the FDSML viewer structure while still using PyViewer-provided font resources where applicable.
- Reworked image upload and texture handling to support both NumPy arrays and PyTorch tensors, including CUDA tensor upload through OpenGL-CUDA interoperability.
- Split and adapted PyViewer's custom operation loader into `viz/ui/cuda_gl_interop.py`, with FDSML-specific logging, module-loading behavior, and helper wrappers.
- Extended the original CUDA/OpenGL interoperability custom operation from texture-only upload to also support VBO registration, mapping, unmapping, and CUDA-to-VBO data upload for point/geometry visualization use cases.
- Renamed and separated CUDA/OpenGL binding functions for clarity, such as `register_texture`, `unregister_texture`, `upload_texture`, `register_vbo`, `unregister_vbo`, `map_vbo`, `unmap_vbo`, and `upload_vbo`.
- Added additional cleanup, synchronization, and fallback behavior so that texture uploads can fall back to CPU/RAM paths when the CUDA/OpenGL plugin is unavailable.
- Removed or replaced PyViewer-specific HDR, toolbar, and single-image viewer assumptions where they did not fit the FDSML visualization workflow.
- Added explicit license notices in the modified source files indicating that they are derivative works based on `harskish/pyviewer` and are distributed under CC BY-NC-SA 4.0.

Unless otherwise stated, all modifications are also distributed under the same CC BY-NC-SA 4.0 license.

---

## 4. FractalDB-Pretrained-ResNet-PyTorch

Some parts of this repository are derived from:
https://github.com/hirokatsukataoka16/FractalDB-Pretrained-ResNet-PyTorch

Copyright (c) 2020, National Institute of Advanced Industrial Science and Technology (AIST).

The original repository is licensed under the MIT License. In particular, `models/resnet.py` contains code derived from this repository.

See the original [FractalDB-Pretrained-ResNet-PyTorch license](./docs/LICENSE_FractalDB-Pretrained-ResNet-PyTorch.txt) for more details.

---

## 5. FractalDB-Pretrained-ViT-PyTorch

Some parts of this repository are derived from:
https://github.com/nakashima-kodai/FractalDB-Pretrained-ViT-PyTorch

Copyright (c) 2021, National Institute of Advanced Industrial Science and Technology (AIST).

The original repository is licensed under the MIT License. In particular, `models/vit.py` contains code derived from this repository.

See the original [FractalDB-Pretrained-ViT-PyTorch license](./docs/LICENSE_FractalDB-Pretrained-ViT-PyTorch.txt) for more details.
