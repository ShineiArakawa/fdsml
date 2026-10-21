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
  - vit
---

# FDSML ViT-Base (DeiT-Base/16, 224) — `same` strategy

ViT-Base (DeiT-Base/16, 224) pre-trained on [FineFractalDB](https://huggingface.co/datasets/fdsml/FineFractalDB) with **FDSML** (Formula-Driven Supervised Metric Learning) using the Parameter-Aware Triplet Loss and the **`same`** triplet selection strategy. No natural images are used during pre-training. The main ViT-Base results in the paper use the [`each`](https://huggingface.co/fdsml/fdsml-finefractaldb-vitbase-each) strategy.

- **Paper:** arXiv link coming soon
- **Project page:** https://shineiarakawa.github.io/fdsml/
- **Code:** https://github.com/ShineiArakawa/fdsml
- **Dataset:** https://huggingface.co/datasets/fdsml/FineFractalDB

## Model details

| Item | Value |
|---|---|
| Architecture | ViT-Base (DeiT-Base/16, 224) (86.6M parameters) |
| Pre-training data | FineFractalDB (1,000 IFS classes × 1,000 images, synthetic) |
| Objective | Parameter-Aware Triplet Loss |
| Triplet strategy | `same` (same_class): selects all pairs from the same class |
| Projection dim | 256 |
| Training images seen | 536,870k (536.87 epochs) |
| Checkpoint | `checkpoint-536870kimg-536.87ep_fdsml_deitbase_same.pt` |
| Name in the official code | `deit_base_patch16_224_mocov3_fdsml_same_pretrained` |

## Usage

The checkpoint is a plain PyTorch file and is loaded with the [official code](https://github.com/ShineiArakawa/fdsml).

```python
import torch
from huggingface_hub import hf_hub_download

path = hf_hub_download("fdsml/fdsml-finefractaldb-vitbase-same", "checkpoint-536870kimg-536.87ep_fdsml_deitbase_same.pt")
ckpt = torch.load(path, map_location="cpu")
```

To fine-tune on a downstream task, pass the checkpoint to `finetune.py`:

```bash
torchrun --standalone --nproc_per_node=4 finetune.py \
    --net=deit_base_patch16_224_mocov3 \
    --ckpt=/path/to/checkpoint-536870kimg-536.87ep_fdsml_deitbase_same.pt \
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
