---
license: cc-by-nc-sa-4.0
pretty_name: FineFractalDB
task_categories:
  - image-classification
  - image-feature-extraction
size_categories:
  - 1M<n<10M
tags:
  - fdsml
  - formula-driven-supervised-learning
  - fractal
  - synthetic-data
  - pre-training
  - iterated-function-system
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train-*.parquet
dataset_info:
  features:
    - name: image
      dtype: image
    - name: label
      dtype: int64
    - name: weight_id
      dtype: int64
    - name: params
      list: float32
      length: 6
    - name: file_name
      dtype: string
---

# FineFractalDB

FineFractalDB is a synthetic image dataset for pre-training vision models **without any natural images**. It consists of **1,000 classes × 1,000 images = 1,000,000 images**. Each class corresponds to one Iterated Function System (IFS), and each image is rendered from that IFS with randomized instance-level parameters. Unlike FractalDB, every image is paired with the exact parameters used to render it, which enables parameter-aware objectives such as the Parameter-Aware Triplet Loss proposed in FDSML.

- **Paper:** arXiv link coming soon
- **Project page:** https://shineiarakawa.github.io/fdsml/
- **Code:** https://github.com/ShineiArakawa/fdsml
- **Pre-trained models:** https://huggingface.co/fdsml

Compared with FractalDB-1k, the generation process is refined for higher intra-class variation and more stable structures:

- **1,000 independent parameter perturbations per class** (instead of 25), so every instance is a distinct point in parameter space.
- **Strict contraction constraints** on every affine map, which keep the attractor stable.
- **Rejection sampling** on pixel occupancy, which removes nearly empty renderings.

## Contents

The dataset is provided in two equivalent forms:

- `finefractaldb.zip` (about 11.5 GB): the original image-folder archive used by the [official code](https://github.com/ShineiArakawa/fdsml).
- `data/train-*.parquet` (about 11.5 GB in total): the same images and parameters as Parquet shards. This is what the Dataset Viewer above shows and what `datasets.load_dataset` reads.

### Parquet shards

Each row has the following columns:

| Column | Type | Description |
|---|---|---|
| `image` | image | 362×362 RGB PNG |
| `label` | int64 | class (IFS) index, `0`–`999` |
| `weight_id` | int64 | per-class instance index, `0`–`999` |
| `params` | list of 6 float32 | instance parameters `wa, wb, wc, wd, we, wf` (last row of the corresponding CSV) |
| `file_name` | string | path of the image inside `finefractaldb.zip` |

```python
from datasets import load_dataset

ds = load_dataset("fdsml/FineFractalDB", split="train", streaming=True)
example = next(iter(ds))
example["image"].show()
print(example["label"], example["params"])
```

The shards are generated from the zip with `tools/create_hf_parquet.py` in the official code.

### Zip archive

`finefractaldb.zip` extracts to:

```text
finefractaldb/
├── images/                 # image-folder format, 1,000 class directories
│   ├── 00000/
│   │   ├── 00000_00000_00000000.png
│   │   └── ...
│   └── ...
└── params/                 # per-image instance parameters
    ├── 00000/
    │   ├── 00000_00000.csv
    │   └── ...
    ├── ...
    └── params.pt           # cache of all parameters for fast loading
```

Images are rendered with the 3×3 patch option (`--patch`) and random seed 0, exactly as used in the paper. The mean and std of the instance parameters, needed for normalization in `pretrain.py`, are:

```text
--param_mean -0.00020501 -0.01737614 0.01417852 -0.00803842 0.01674297 0.02324245
--param_std 0.44203399 0.45483486 0.45470102 0.44323780 0.44413787 0.42966211
```

## Download

To train with the official code, download and extract the zip archive:

```bash
hf download fdsml/FineFractalDB finefractaldb.zip --repo-type dataset --local-dir datasets
unzip datasets/finefractaldb.zip -d datasets
```

To download only the Parquet shards:

```bash
hf download fdsml/FineFractalDB --include "data/*.parquet" --repo-type dataset --local-dir datasets/finefractaldb_parquet
```

The dataset can also be regenerated from scratch with the [official code](https://github.com/ShineiArakawa/fdsml#generate-finefractaldb) in about 15 minutes on a 4-core CPU.

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
