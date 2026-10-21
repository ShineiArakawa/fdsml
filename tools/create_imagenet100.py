"""Create ImageNet-100 dataset by copying images from the original ImageNet-1k dataset.

License
-------

Copyright (c) 2026 Shinei Arakawa
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
"""

import datetime
import hashlib
import json
import pathlib
import shutil

import tqdm

import torch_utils.util as util

# NOTE: This class list is from the supplementary material of the paper "Pre-training without Natural Images" (https://hirokatsukataoka16.github.io/Pretraining-without-Natural-Images/ [Accessed 2026-02-13]).
wnids_in_paper = [
    'n01560419', 'n01860187', 'n02058221', 'n02169497', 'n02655020', 'n02823750', 'n03146219', 'n03344393',
    'n03483316', 'n03891332', 'n04141975', 'n04404412', 'n07860988', 'n01630670', 'n01910747', 'n02090622',
    'n02356798', 'n02667093', 'n02835271', 'n03201208', 'n03372029', 'n03485794', 'n03938244', 'n04152593',
    'n04442312', 'n09246464', 'n01632458', 'n01985128', 'n02093859', 'n02361337', 'n02727426', 'n02894605',
    'n03207743', 'n03388549', 'n03495258', 'n03954731', 'n04254680', 'n04456115', 'n09468604', 'n01632777',
    'n02002724', 'n02096437', 'n02422699', 'n02797295', 'n02930766', 'n03208938', 'n03394916', 'n03770439',
    'n03956157', 'n04259630', 'n04525038', 'n15075141', 'n01694178', 'n02011460', 'n02099712', 'n02445715',
    'n02802426', 'n02948072', 'n03240683', 'n03404251', 'n03770679', 'n03958227', 'n04346328', 'n06874185',
    'n01728920', 'n02017213', 'n02106662', 'n02494079', 'n02804414', 'n03028079', 'n03255030', 'n03443371',
    'n03777754', 'n03980874', 'n04355933', 'n07248320', 'n01776313', 'n02018795', 'n02112137', 'n02510455',
    'n02804610', 'n03062245', 'n03272010', 'n03447447', 'n03874293', 'n04074963', 'n04366367', 'n07717410',
    'n01833805', 'n02051845', 'n02113712', 'n02607072', 'n02808304', 'n03126707', 'n03325584', 'n03481172',
    'n03876231', 'n04118538', 'n04399382', 'n07730033',
]

imagenet_root = pathlib.Path('datasets/imagenet_all').resolve()
imagenet100_root = pathlib.Path('datasets/imagenet100').resolve()


def main():
    # --------------------------------------------------------------------------------------------------------------
    # Load metadata from torchvision's online repository

    with util.open_url('https://raw.githubusercontent.com/pytorch/vision/326a11d48a97b14b7b65e11afe0f34cd7981efd4/gallery/assets/imagenet_class_index.json') as file:
        imagenet_meta = json.load(file)

    wnids_to_idx: dict[str, int] = {wnid: int(idx) for idx, (wnid, _) in imagenet_meta.items()}

    # --------------------------------------------------------------------------------------------------------------
    # Copy images to new directory structure

    print(f'Copying ImageNet-100 images to: {imagenet100_root}')

    for split in ['train', 'val']:
        print(f'Processing {split} set...')

        nimgs = 0

        src_dir = imagenet_root / split
        dst_dir = imagenet100_root / split

        for new_idx, wnid in tqdm.tqdm(enumerate(wnids_in_paper), total=len(wnids_in_paper), desc=f'Copying {split} images', unit='class'):
            cur_idx = wnids_to_idx[wnid]

            src_wnid_dir = src_dir / f'{cur_idx:05d}'
            dst_wnid_dir = dst_dir / f'{new_idx:05d}'
            dst_wnid_dir.mkdir(parents=True, exist_ok=True)

            img_files = list(src_wnid_dir.iterdir())
            for img_path in tqdm.tqdm(img_files, desc=f'Copying images for class {wnid}', unit='image', leave=False, position=1):
                dst_img_path = dst_wnid_dir / img_path.name
                shutil.copyfile(img_path, dst_img_path)
                nimgs += 1

        print(f'Copied {nimgs} images for {len(wnids_in_paper)} classes.')

    # --------------------------------------------------------------------------------------------------------------
    # Save metadata

    to_save = {}

    # Add this source code
    to_save['source_code'] = (pathlib.Path(__file__).resolve()).read_text()
    # Add class list
    to_save['wnids_in_paper'] = wnids_in_paper
    # Add ImageNet metadata
    to_save['imagenet_meta'] = imagenet_meta
    # Add timestamp
    to_save['timestamp'] = datetime.datetime.now().isoformat()

    # Add SHA256 hash of the metadata
    meta_hash = hashlib.sha256(json.dumps(to_save, indent=4, sort_keys=True).encode('utf-8')).hexdigest()
    to_save['metadata_sha256'] = meta_hash

    metadata_path = imagenet100_root / 'metadata.json'
    with metadata_path.open('w') as f:
        json.dump(to_save, f, indent=4, sort_keys=True)

    print(f'Saved metadata to: {metadata_path}')

    print('done.')


if __name__ == '__main__':
    main()
