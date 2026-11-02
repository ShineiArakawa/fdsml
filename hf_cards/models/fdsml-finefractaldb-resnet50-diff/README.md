---
license: cc-by-nc-sa-4.0
library_name: pytorch
pipeline_tag: image-feature-extraction
datasets:
  - fdsml/FineFractalDB
tags:
  - fdsml
  - formula-driven-supervised-learning
  - metric-learning
  - fractal
  - synthetic-data
  - pre-training
  - resnet
  - arxiv:2610.39423
---

# FDSML ResNet-50 — `diff` strategy

ResNet-50 pre-trained on [FineFractalDB](https://huggingface.co/datasets/fdsml/FineFractalDB) with **FDSML** (Formula-Driven Supervised Metric Learning) using the Parameter-Aware Triplet Loss and the **`diff`** triplet selection strategy. No natural images are used during pre-training. The main ResNet-50 results in the paper use the [`same`](https://huggingface.co/fdsml/fdsml-finefractaldb-resnet50-same) strategy.

- **Paper:** https://arxiv.org/abs/2610.39423
- **Project page:** https://shineiarakawa.github.io/fdsml/
- **Code:** https://github.com/ShineiArakawa/fdsml
- **Dataset:** https://huggingface.co/datasets/fdsml/FineFractalDB

## Model details

| Item | Value |
|---|---|
| Architecture | ResNet-50 (25.6M parameters) |
| Pre-training data | FineFractalDB (1,000 IFS classes × 1,000 images, synthetic) |
| Objective | Parameter-Aware Triplet Loss |
| Triplet strategy | `diff` (diff_class): selects all pairs from different classes |
| Projection dim | 512 |
| Training images seen | 134,217k (134.22 epochs) |
| Checkpoint | `checkpoint-134217kimg-134.22ep_fdsml_resnet50_diff.pt` |
| Name in the official code | `resnet50_fdsml_diff_pretrained` |

## Usage

The checkpoint is a plain PyTorch file and is loaded with the [official code](https://github.com/ShineiArakawa/fdsml).

```python
import torch
from huggingface_hub import hf_hub_download

path = hf_hub_download("fdsml/fdsml-finefractaldb-resnet50-diff", "checkpoint-134217kimg-134.22ep_fdsml_resnet50_diff.pt")
ckpt = torch.load(path, map_location="cpu")
```

To fine-tune on a downstream task, pass the checkpoint to `finetune.py`:

```bash
torchrun --standalone --nproc_per_node=4 finetune.py \
    --net=resnet50 \
    --ckpt=/path/to/checkpoint-134217kimg-134.22ep_fdsml_resnet50_diff.pt \
    --dset=/path/to/imagenet
```

See the [README](https://github.com/ShineiArakawa/fdsml#fine-tuning) for the full list of options and the exact configurations used in the paper.

## Citation

```bibtex
@inproceedings{Otake2027fdsml,
  title     = {FDSML: Formula-Driven Supervised Metric Learning with Parameter-Aware Triplet Loss},
  author    = {Hina Otake and Shinei Arakawa and Keitaro Tanaka and Yoshihiro Fukuhara and Hirokatsu Kataoka and Shigeo Morishima},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2027},
}
```

## License

Released under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Non-commercial use only.
