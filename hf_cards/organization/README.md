# FDSML

Official models and data for **FDSML: Formula-Driven Supervised Metric Learning with Parameter-Aware Triplet Loss** (WACV 2027).

Hina Otake\*, Shinei Arakawa\*, Keitaro Tanaka\*, Yoshihiro Fukuhara, Hirokatsu Kataoka, Shigeo Morishima (\*equal contribution)

FDSML pre-trains vision models on synthetic fractal images only, without any natural images. Every image in FineFractalDB is paired with the parameters that generated it, and the Parameter-Aware Triplet Loss uses these parameters to shape the embedding space. With ResNet-50, FDSML raises Recall@1 over FDSL pre-training from 65.6% to 74.0% on Stanford Online Products and from 56.2% to 78.4% on Stanford Cars, while staying competitive on classification.

- **Paper:** https://arxiv.org/abs/2610.39423
- **Project page:** https://shineiarakawa.github.io/fdsml/
- **Code:** https://github.com/ShineiArakawa/fdsml

## Dataset

- [FineFractalDB](https://huggingface.co/datasets/fdsml/FineFractalDB): 1,000 IFS classes × 1,000 images with per-image rendering parameters.

## Pre-trained models

FDSML models for each architecture and triplet selection strategy (`all` / `each` / `same` / `diff`, see the paper):

| Architecture | `all` | `each` | `same` | `diff` |
|---|---|---|---|---|
| ResNet-50 | [resnet50-all](https://huggingface.co/fdsml/fdsml-finefractaldb-resnet50-all) | [resnet50-each](https://huggingface.co/fdsml/fdsml-finefractaldb-resnet50-each) | [resnet50-same](https://huggingface.co/fdsml/fdsml-finefractaldb-resnet50-same) | [resnet50-diff](https://huggingface.co/fdsml/fdsml-finefractaldb-resnet50-diff) |
| ViT-Tiny (DeiT-Tiny/16, 224) | [vittiny-all](https://huggingface.co/fdsml/fdsml-finefractaldb-vittiny-all) | [vittiny-each](https://huggingface.co/fdsml/fdsml-finefractaldb-vittiny-each) | [vittiny-same](https://huggingface.co/fdsml/fdsml-finefractaldb-vittiny-same) | [vittiny-diff](https://huggingface.co/fdsml/fdsml-finefractaldb-vittiny-diff) |
| ViT-Base (DeiT-Base/16, 224) | [vitbase-all](https://huggingface.co/fdsml/fdsml-finefractaldb-vitbase-all) | [vitbase-each](https://huggingface.co/fdsml/fdsml-finefractaldb-vitbase-each) | [vitbase-same](https://huggingface.co/fdsml/fdsml-finefractaldb-vitbase-same) | [vitbase-diff](https://huggingface.co/fdsml/fdsml-finefractaldb-vitbase-diff) |

All checkpoints are plain PyTorch files and are loaded with the [official code](https://github.com/ShineiArakawa/fdsml).

## Citation

```bibtex
@inproceedings{Otake2027fdsml,
  title     = {FDSML: Formula-Driven Supervised Metric Learning with Parameter-Aware Triplet Loss},
  author    = {Hina Otake and Shinei Arakawa and Keitaro Tanaka and Yoshihiro Fukuhara and Hirokatsu Kataoka and Shigeo Morishima},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2027},
}
```
