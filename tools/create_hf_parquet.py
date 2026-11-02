"""Convert the FineFractalDB zip archive into Parquet shards for the Hugging Face Dataset Viewer.

The official distribution of FineFractalDB is a single ``finefractaldb.zip`` that contains one PNG per image and one
CSV per (class, weight) pair. The Hugging Face Dataset Viewer cannot preview that archive (mixed PNG/CSV content and a
size well above the 5 GB auto-conversion limit), so this tool re-packages the images and their instance parameters into
Parquet shards that the viewer and ``datasets.load_dataset`` understand natively. The zip itself stays untouched and
remains the file used by the training code.

Each row holds:

- ``image``: ``{"bytes": <png bytes>, "path": <file name>}`` (declared as ``dtype: image`` in the dataset card)
- ``label``: class (IFS) index, 0..999
- ``weight_id``: per-class instance index, 0..999
- ``params``: the 6 instance parameters ``wa..wf`` (same values as ``params.pt``)
- ``file_name``: path of the PNG inside the zip

Example::

    # Full dataset (1M rows, about 11.5 GB of Parquet)
    python tools/create_hf_parquet.py --zip datasets/finefractaldb.zip --out_dir datasets/finefractaldb_parquet/data

    # Preview subset with 10 images per class (10k rows)
    python tools/create_hf_parquet.py --per_class 10 --out_dir datasets/finefractaldb_preview/data

License
-------

Copyright (c) 2026 Shinei Arakawa
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
"""

import dataclasses
import io
import multiprocessing as mp
import os
import re
import zipfile

import click
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import tqdm

# ------------------------------------------------------------------------------------------------------------------------

# Image file name: {cat}_{wid}_{img_id}.png
IMG_NAME_RE = re.compile(r"^(?P<cat>\d+)_(?P<wid>\d+)_(?P<img>\d+)\.png$")
PARAM_DIM = 6

SCHEMA = pa.schema([
    pa.field("image", pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())])),
    pa.field("label", pa.int64()),
    pa.field("weight_id", pa.int64()),
    pa.field("params", pa.list_(pa.float32(), PARAM_DIM)),
    pa.field("file_name", pa.string()),
])


@dataclasses.dataclass
class Entry:
    name: str  # path inside the zip
    label: int
    weight_id: int
    params: np.ndarray  # (6,) float32


@dataclasses.dataclass
class ShardTask:
    zip_path: str
    out_path: str
    entries: list[Entry]
    row_group_size: int
    compression_level: int


# ------------------------------------------------------------------------------------------------------------------------


def list_entries(zip_path: str, per_class: int | None) -> list[Entry]:
    """Enumerate PNG entries in the zip and attach the matching instance parameters from ``params.pt``."""
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        param_names = [n for n in names if n.endswith("params.pt")]
        if len(param_names) != 1:
            raise RuntimeError(f"Expected exactly one params.pt in {zip_path}, found {param_names}")
        with zf.open(param_names[0]) as f:
            param_data: dict[tuple[int, int], torch.Tensor] = torch.load(f, map_location="cpu")

    entries: list[Entry] = []
    for name in names:
        m = IMG_NAME_RE.match(os.path.basename(name))
        if m is None or not name.endswith(".png"):
            continue
        cat, wid = int(m.group("cat")), int(m.group("wid"))
        param = param_data.get((cat, wid))
        if param is None:
            raise RuntimeError(f"No parameters found for (class={cat}, weight_id={wid}) referenced by {name}")
        param = param.numpy().astype(np.float32)
        if param.shape != (PARAM_DIM,):
            raise RuntimeError(f"Unexpected parameter shape {param.shape} for {name}")
        entries.append(Entry(name=name, label=cat, weight_id=wid, params=param))

    # Deterministic order: by class, then weight id, then name.
    entries.sort(key=lambda e: (e.label, e.weight_id, e.name))

    if per_class is not None:
        kept: list[Entry] = []
        count: dict[int, int] = {}
        for e in entries:
            if count.get(e.label, 0) < per_class:
                kept.append(e)
                count[e.label] = count.get(e.label, 0) + 1
        entries = kept

    return entries


def write_shard(task: ShardTask) -> int:
    """Write one Parquet shard. Runs in a worker process; opens its own zip handle."""
    images, labels, wids, params, names = [], [], [], [], []
    with zipfile.ZipFile(task.zip_path) as zf:
        for e in task.entries:
            data = zf.read(e.name)
            images.append({"bytes": data, "path": os.path.basename(e.name)})
            labels.append(e.label)
            wids.append(e.weight_id)
            params.append(e.params.tolist())
            names.append(e.name)

    table = pa.Table.from_pydict(
        {"image": images, "label": labels, "weight_id": wids, "params": params, "file_name": names},
        schema=SCHEMA,
    )
    tmp_path = task.out_path + ".tmp"
    pq.write_table(table, tmp_path, row_group_size=task.row_group_size, compression="zstd", compression_level=task.compression_level)
    os.replace(tmp_path, task.out_path)
    return table.num_rows


# ------------------------------------------------------------------------------------------------------------------------


@click.command(context_settings=dict(show_default=True, max_content_width=200))
@click.option("--zip", "zip_path", type=click.Path(exists=True, dir_okay=False), default="datasets/finefractaldb.zip", help="Path to finefractaldb.zip.")
@click.option("--out_dir", type=click.Path(file_okay=False), default="datasets/finefractaldb_parquet/data", help="Output directory for the Parquet shards.")
@click.option("--split", type=str, default="train", help="Split name used in the shard file names.")
@click.option("--per_class", type=int, default=None, help="Keep only the first N images of each class (preview subset). Default: all images.")
@click.option("--rows_per_shard", type=int, default=20_000, help="Number of rows per Parquet shard (about 230 MB per shard at 20k rows with zstd level 9).")
@click.option("--row_group_size", type=int, default=100, help="Parquet row group size. Hugging Face recommends 100 rows for image data.")
@click.option("--compression_level", type=int, default=9, help="zstd compression level. Level 9 brings the shards down to roughly the size of the original zip.")
@click.option("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2), help="Number of worker processes.")
def main(zip_path: str, out_dir: str, split: str, per_class: int | None, rows_per_shard: int, row_group_size: int, compression_level: int, workers: int) -> None:
    zip_path = os.path.abspath(zip_path)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Listing entries in {zip_path} ...", flush=True)
    entries = list_entries(zip_path, per_class)
    num_classes = len({e.label for e in entries})
    print(f"Found {len(entries)} images across {num_classes} classes.", flush=True)
    if len(entries) == 0:
        raise click.ClickException("No images found.")

    num_shards = (len(entries) + rows_per_shard - 1) // rows_per_shard
    tasks = [
        ShardTask(
            zip_path=zip_path,
            out_path=os.path.join(out_dir, f"{split}-{i:05d}-of-{num_shards:05d}.parquet"),
            entries=entries[i * rows_per_shard:(i + 1) * rows_per_shard],
            row_group_size=row_group_size,
            compression_level=compression_level,
        )
        for i in range(num_shards)
    ]

    total = 0
    with mp.Pool(min(workers, num_shards)) as pool:
        for n in tqdm.tqdm(pool.imap_unordered(write_shard, tasks), total=num_shards, desc="Writing shards", unit="shard"):
            total += n

    print(f"Wrote {total} rows to {num_shards} shards in {out_dir}.")
    print("done.")


if __name__ == "__main__":
    main()
