"""Generate Hugging Face model, dataset, and organization cards for FDSML.

Run this script to (re)generate all cards under ``hf_cards/``:

    python hf_cards/generate_cards.py

Output layout::

    hf_cards/
    ├── models/<repo-name>/README.md    # one per model repo
    ├── datasets/FineFractalDB/README.md
    └── organization/README.md          # organization card

Update the constants below (e.g. ``ARXIV_ID``) and re-run when the paper
information changes.
"""

import pathlib
import textwrap

OUT_DIR = pathlib.Path(__file__).resolve().parent

# ----------------------------------------------------------------------------
# Shared metadata
# ----------------------------------------------------------------------------

ORG = 'fdsml'
PAPER_TITLE = 'FDSML: Formula-Driven Supervised Metric Learning with Parameter-Aware Triplet Loss'
VENUE = 'WACV 2027'
AUTHORS = 'Hina Otake\\*, Shinei Arakawa\\*, Keitaro Tanaka\\*, Yoshihiro Fukuhara, Hirokatsu Kataoka, Shigeo Morishima (\\*equal contribution)'
ARXIV_ID = None  # TODO: set to e.g. '2601.01234' once the paper is on arXiv
PROJECT_PAGE = 'https://shineiarakawa.github.io/fdsml/'
GITHUB = 'https://github.com/ShineiArakawa/fdsml'
LICENSE = 'cc-by-nc-sa-4.0'
DATASET_REPO = f'{ORG}/FineFractalDB'

BIBTEX = textwrap.dedent('''\
    @inproceedings{Otake2027fdsml,
      title     = {FDSML: Formula-Driven Supervised Metric Learning with Parameter-Aware Triplet Loss},
      author    = {Hina Otake and Shinei Arakawa and Keitaro Tanaka and Yoshihiro Fukuhara and Hirokatsu Kataoka and Shigeo Morishima},
      booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
      year      = {2027},
    }''')


def paper_link() -> str:
    if ARXIV_ID:
        return f'https://arxiv.org/abs/{ARXIV_ID}'
    return 'arXiv link coming soon'


def links_section() -> str:
    return textwrap.dedent(f'''\
        - **Paper:** {paper_link()}
        - **Project page:** {PROJECT_PAGE}
        - **Code:** {GITHUB}
        - **Dataset:** https://huggingface.co/datasets/{DATASET_REPO}''')


def arxiv_tag_lines() -> str:
    return f'  - arxiv:{ARXIV_ID}\n' if ARXIV_ID else ''


# ----------------------------------------------------------------------------
# FDSML model cards
# ----------------------------------------------------------------------------

ARCHS = {
    # key: (display name, net name for --net, projection dim, params, ckpt stem, kimg, epochs)
    'resnet50': dict(display='ResNet-50', net='resnet50', proj_dim=512, params='25.6M',
                     stem='checkpoint-134217kimg-134.22ep_fdsml_resnet50', kimg='134,217', epochs='134.22'),
    'vittiny': dict(display='ViT-Tiny (DeiT-Tiny/16, 224)', net='deit_tiny_patch16_224_mocov3', proj_dim=128, params='5.7M',
                    stem='checkpoint-536870kimg-536.87ep_fdsml_deittiny', kimg='536,870', epochs='536.87'),
    'vitbase': dict(display='ViT-Base (DeiT-Base/16, 224)', net='deit_base_patch16_224_mocov3', proj_dim=256, params='86.6M',
                    stem='checkpoint-536870kimg-536.87ep_fdsml_deitbase', kimg='536,870', epochs='536.87'),
}

# vitbase-diff was trained slightly shorter than the other ViT-Base models.
ARCH_OVERRIDES = {
    ('vitbase', 'diff'): dict(stem='checkpoint-524288kimg-524.29ep_fdsml_deitbase', kimg='524,288', epochs='524.29'),
}

STRATEGIES = {
    'all': dict(miner='all', desc='uses all possible triplets within a batch'),
    'each': dict(miner='each_class', desc='selects one positive pair from the same class and one negative from a different class'),
    'same': dict(miner='same_class', desc='selects all pairs from the same class'),
    'diff': dict(miner='diff_class', desc='selects all pairs from different classes'),
}

ARCH_TAG = {'resnet50': 'resnet', 'vittiny': 'vit', 'vitbase': 'vit'}

# Strategy used for the main results of each architecture in the paper.
MAIN_STRATEGY = {'resnet50': 'same', 'vittiny': 'each', 'vitbase': 'each'}

HEADLINE = ('With ResNet-50, FDSML raises Recall@1 over FDSL pre-training from 65.6% to 74.0% on Stanford Online Products '
            'and from 56.2% to 78.4% on Stanford Cars, while staying competitive on classification.')


def model_card(arch_key: str, strat_key: str) -> tuple[str, str]:
    arch = dict(ARCHS[arch_key])
    arch.update(ARCH_OVERRIDES.get((arch_key, strat_key), {}))
    strat = STRATEGIES[strat_key]
    repo_name = f'fdsml-finefractaldb-{arch_key}-{strat_key}'
    repo_id = f'{ORG}/{repo_name}'
    ckpt_file = f'{arch["stem"]}_{strat_key}.pt'
    net_pretrained = f'{arch["net"]}_fdsml_{strat_key}_pretrained'
    if MAIN_STRATEGY[arch_key] == strat_key:
        main_note = f'This is the main {arch["display"].split(" (")[0]} model reported in the paper.'
    else:
        main_note = (f'The main {arch["display"].split(" (")[0]} results in the paper use the '
                     f'[`{MAIN_STRATEGY[arch_key]}`](https://huggingface.co/{ORG}/fdsml-finefractaldb-{arch_key}-{MAIN_STRATEGY[arch_key]}) strategy.')

    card = f'''---
license: {LICENSE}
library_name: pytorch
pipeline_tag: image-feature-extraction
datasets:
  - {DATASET_REPO}
tags:
  - fdsml
  - formula-driven-supervised-learning
  - metric-learning
  - fractal
  - synthetic-data
  - pre-training
  - {ARCH_TAG[arch_key]}
{arxiv_tag_lines()}---

# FDSML {arch['display']} — `{strat_key}` strategy

{arch['display']} pre-trained on [FineFractalDB](https://huggingface.co/datasets/{DATASET_REPO}) with **FDSML** (Formula-Driven Supervised Metric Learning) using the Parameter-Aware Triplet Loss and the **`{strat_key}`** triplet selection strategy. No natural images are used during pre-training. {main_note}

{links_section()}

## Model details

| Item | Value |
|---|---|
| Architecture | {arch['display']} ({arch['params']} parameters) |
| Pre-training data | FineFractalDB (1,000 IFS classes × 1,000 images, synthetic) |
| Objective | Parameter-Aware Triplet Loss |
| Triplet strategy | `{strat_key}` ({strat['miner']}): {strat['desc']} |
| Projection dim | {arch['proj_dim']} |
| Training images seen | {arch['kimg']}k ({arch['epochs']} epochs) |
| Checkpoint | `{ckpt_file}` |
| Name in the official code | `{net_pretrained}` |

## Usage

The checkpoint is a plain PyTorch file and is loaded with the [official code]({GITHUB}).

```python
import torch
from huggingface_hub import hf_hub_download

path = hf_hub_download("{repo_id}", "{ckpt_file}")
ckpt = torch.load(path, map_location="cpu")
```

To fine-tune on a downstream task, pass the checkpoint to `finetune.py`:

```bash
torchrun --standalone --nproc_per_node=4 finetune.py \\
    --net={arch['net']} \\
    --ckpt=/path/to/{ckpt_file} \\
    --dset=/path/to/imagenet
```

See the [README]({GITHUB}#fine-tuning) for the full list of options and the exact configurations used in the paper.

## Citation

```bibtex
{BIBTEX}
```

## License

Released under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Non-commercial use only.
'''
    return repo_name, card


# ----------------------------------------------------------------------------
# Dataset card
# ----------------------------------------------------------------------------

def dataset_card() -> str:
    return f'''---
license: {LICENSE}
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
{arxiv_tag_lines()}configs:
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

{links_section().replace(f'- **Dataset:** https://huggingface.co/datasets/{DATASET_REPO}', f'- **Pre-trained models:** https://huggingface.co/{ORG}')}

Compared with FractalDB-1k, the generation process is refined for higher intra-class variation and more stable structures:

- **1,000 independent parameter perturbations per class** (instead of 25), so every instance is a distinct point in parameter space.
- **Strict contraction constraints** on every affine map, which keep the attractor stable.
- **Rejection sampling** on pixel occupancy, which removes nearly empty renderings.

## Contents

The dataset is provided in two equivalent forms:

- `finefractaldb.zip` (about 11.5 GB): the original image-folder archive used by the [official code]({GITHUB}).
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

ds = load_dataset("{DATASET_REPO}", split="train", streaming=True)
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
hf download {DATASET_REPO} finefractaldb.zip --repo-type dataset --local-dir datasets
unzip datasets/finefractaldb.zip -d datasets
```

To download only the Parquet shards:

```bash
hf download {DATASET_REPO} --include "data/*.parquet" --repo-type dataset --local-dir datasets/finefractaldb_parquet
```

The dataset can also be regenerated from scratch with the [official code]({GITHUB}#generate-finefractaldb) in about 15 minutes on a 4-core CPU.

## Citation

```bibtex
{BIBTEX}
```

## License

Released under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Non-commercial use only.
'''


# ----------------------------------------------------------------------------
# Organization card
# ----------------------------------------------------------------------------

def organization_card() -> str:
    rows = []
    for arch_key, arch in ARCHS.items():
        cells = ' | '.join(f'[{arch_key}-{s}](https://huggingface.co/{ORG}/fdsml-finefractaldb-{arch_key}-{s})' for s in STRATEGIES)
        rows.append(f'| {arch["display"]} | {cells} |')
    model_table = '\n'.join(rows)

    return f'''# FDSML

Official models and data for **{PAPER_TITLE}** ({VENUE}).

{AUTHORS}

FDSML pre-trains vision models on synthetic fractal images only, without any natural images. Every image in FineFractalDB is paired with the parameters that generated it, and the Parameter-Aware Triplet Loss uses these parameters to shape the embedding space. {HEADLINE}

- **Paper:** {paper_link()}
- **Project page:** {PROJECT_PAGE}
- **Code:** {GITHUB}

## Dataset

- [FineFractalDB](https://huggingface.co/datasets/{DATASET_REPO}): 1,000 IFS classes × 1,000 images with per-image rendering parameters.

## Pre-trained models

FDSML models for each architecture and triplet selection strategy (`all` / `each` / `same` / `diff`, see the paper):

| Architecture | `all` | `each` | `same` | `diff` |
|---|---|---|---|---|
{model_table}

All checkpoints are plain PyTorch files and are loaded with the [official code]({GITHUB}).

## Citation

```bibtex
{BIBTEX}
```
'''


# ----------------------------------------------------------------------------

def write(path: pathlib.Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')
    print(f'wrote {path.relative_to(OUT_DIR)}')


def main() -> None:
    for arch_key in ARCHS:
        for strat_key in STRATEGIES:
            repo_name, card = model_card(arch_key, strat_key)
            write(OUT_DIR / 'models' / repo_name / 'README.md', card)

    write(OUT_DIR / 'datasets' / 'FineFractalDB' / 'README.md', dataset_card())
    write(OUT_DIR / 'organization' / 'README.md', organization_card())


if __name__ == '__main__':
    main()
