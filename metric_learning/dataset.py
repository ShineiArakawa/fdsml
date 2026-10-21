"""Dataset and sampler classes for metric learning tasks.

License
-------

Copyright (c) 2026 Shinei Arakawa
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
"""

import glob
import json
import os
import random
import re
import typing
import zipfile

import numpy as np
import pandas as pd
import PIL.Image
import torch

import torch_utils
import torch_utils.distributed as dist

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Dataset classes


class ImageFolderSequentialDataset(torch.utils.data.Dataset):
    """A dataset class that loads images from a directory structure where each subdirectory represents a class."""

    def __init__(self, dset: str, param_dir: str | None = None, param_dim: int | None = None, transform: typing.Callable | None = None, target_transform: typing.Callable | None = None, return_path: bool = False):
        """Initialize the dataset. This class assumes that the dataset directory contains subdirectories for each class, and each subdirectory contains images belonging to that class.

        Parameters
        ----------
        dset : str
            Path to the root directory of the dataset or a single zip file containing the dataset.
        param_dir : str | None, optional
            Path to the directory containing parameter files. If None, parameters are inferred from directory names.
        param_dim : int | None, optional
            Dimension of the parameter vector. If None, parameters are assumed to be scalar.
        transform : typing.Callable | None, optional
            A function/transform that takes in an PIL image and returns a transformed version. E.g., `transforms.Compose([...])`. If None, no transformation is applied.
        target_transform : typing.Callable | None, optional
            A function/transform that takes in the target label and transforms it. If None, no transformation is applied.
        return_path : bool, optional
            Whether to return the image path along with the image and label.
        """

        super().__init__()

        self._transform = transform
        self._target_transform = target_transform
        self._return_path = return_path

        self._num_classes: int = 0
        self._img_paths: list[str] = []
        self._cls_labels: torch.Tensor | None = None
        self._unique_labels_str: list[str] = []
        self._num_instances_per_class: int = -1

        self._param_values: torch.Tensor | None = None

        self._from_zip: bool = False
        self._zipfile: zipfile.ZipFile | None = None

        if os.path.isdir(dset):
            self._init_from_image_folder(dset, param_dir)
        elif os.path.isfile(dset) and dset.lower().endswith('.zip'):
            self._from_zip = True
            self._init_from_zip(dset)
        else:
            raise ValueError(f"Invalid dataset path: {dset}. Must be a directory or a zip file.")

        # Adjust parameter dimension
        if self._param_values is not None:
            if param_dim is None:
                assert self._param_values.ndim == 1 or self._param_values.shape[1] == 1, "param_dim is None but parameters are not scalar."
            else:
                orig_dim = self._param_values.shape[1]
                assert param_dim <= orig_dim, f"param_dim {param_dim} is greater than original parameter dimension {orig_dim}."
                self._param_values = self._param_values[:, :param_dim]
                dist.print0(f'Using parameter dimension: {param_dim} (original: {orig_dim})', flush=True)

    def _init_from_image_folder(self, dset: str, param_dir: str | None = None) -> None:
        """Load images and labels from a directory structure where each subdirectory represents a class. Optionally, load parameters from a specified directory."""

        dist.print0(f'Start loading dataset from image folder: {dset}', flush=True)

        # Load dataset
        dirs = glob.glob(os.path.join(dset, '*'))
        dirs = [dir_path for dir_path in dirs if os.path.isdir(dir_path)]
        dirs = sorted(dirs)
        self._num_classes = len(dirs)

        cls_labels: list[int] = []

        for dir_path in dirs:
            assert os.path.isdir(dir_path), f'Expected directory, but got file: {dir_path}'

            img_paths = glob.glob(os.path.join(dir_path, '*'))
            img_paths = [img_path for img_path in img_paths if os.path.isfile(img_path)]
            assert len(img_paths) > 0, f'No images found in directory: {dir_path}'

            if self._num_instances_per_class == -1:
                self._num_instances_per_class = len(img_paths)
            else:
                assert self._num_instances_per_class == len(img_paths), f'Inconsistent number of instances per class. Expected {self._num_instances_per_class}, but found {len(img_paths)} in directory: {dir_path}'

            self._img_paths.extend(img_paths)
            cls_labels.extend([len(self._unique_labels_str)] * len(img_paths))

            self._unique_labels_str.append(os.path.basename(dir_path))

        assert len(self._img_paths) > 0, 'No images found in the dataset.'
        assert len(self._img_paths) == len(cls_labels), 'Mismatch between number of images and labels.'

        self._cls_labels = torch.tensor(cls_labels, dtype=torch.long)

        # Load parameters
        if param_dir is not None:
            param_data: dict[tuple[int, int], torch.Tensor] = {}

            # Search cached file
            cache_file = os.path.join(param_dir, 'params.pt')
            if os.path.isfile(cache_file):
                dist.print0(f'Loading cached parameters from: {cache_file}', flush=True)
                param_data = torch.load(cache_file)
            else:
                dist.print0(f'Loading parameters from CSV files in: {param_dir}', flush=True)

                # Read csv
                param_dir = os.path.abspath(param_dir)

                csv_file_name_template = re.compile(r"^(?P<cat>\d+)_(?P<wid>\d+)\.csv$")
                csv_files = glob.glob(os.path.join(param_dir, '**', '*.csv'), recursive=True)
                for csv_file in csv_files:
                    file_name = os.path.basename(csv_file)
                    match = csv_file_name_template.search(file_name)
                    if not match:
                        continue

                    cat = int(match.group("cat"))
                    wid = int(match.group("wid"))
                    key = (cat, wid)

                    param = ImageFolderSequentialDataset._load_csv_tail6(csv_file)

                    param_data[key] = param

            dist.print0(f'Loaded parameters for {len(param_data)} (category, weightID) pairs from {param_dir}.', flush=True)

            # Assign parameters
            self._param_values = ImageFolderSequentialDataset._get_corresponded_params(self._img_paths, param_data)

        dist.print0(f'done.', flush=True)

    def _init_from_zip(self, zip_path: str):
        """Load images and labels from a zip file. Optionally, load parameters from a specified directory."""

        dist.print0(f'Start loading dataset from zip file: {zip_path}', flush=True)

        with zipfile.ZipFile(zip_path, 'r') as zfile:
            # Load dataset state.
            #   Keys: num_classes, img_paths, cls_labels, unique_labels_str, num_instances_per_class, param_path
            with zfile.open('dataset.json') as file:
                dset_state = torch_utils.EasyDict(json.load(file))

            self._num_classes = dset_state.num_classes
            self._img_paths = dset_state.img_paths
            self._cls_labels = torch.tensor(dset_state.cls_labels, dtype=torch.long)
            self._unique_labels_str = dset_state.unique_labels_str
            self._num_instances_per_class = dset_state.num_instances_per_class

            param_path = dset_state.get('param_path', None)

            # Load parameters
            if param_path is not None:
                with zfile.open(param_path) as file:
                    param_data: dict[tuple[int, int], torch.Tensor] = torch.load(file)

                # Assign parameters
                self._param_values = ImageFolderSequentialDataset._get_corresponded_params(self._img_paths, param_data)

        dist.print0(f'done.', flush=True)

    @property
    def num_classes(self) -> int:
        """Number of unique classes in the dataset."""
        return self._num_classes

    @property
    def num_instances_per_class(self) -> int:
        """Number of instances per class in the dataset."""
        return self._num_instances_per_class

    @staticmethod
    def _load_csv_tail6(csv_path: str) -> torch.Tensor:
        df = pd.read_csv(csv_path)
        if len(df) == 0:
            raise ValueError(f"CSV has no rows: {csv_path}")

        last = df.iloc[-1]
        wa_cols = ["wa", "wb", "wc", "wd", "we", "wf"]
        if all(c in df.columns for c in wa_cols):
            vals = last[wa_cols].values.astype(np.float32)
            return torch.from_numpy(vals)

        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_cols) >= 6:
            cols = numeric_cols[-6:]
            vals = last[cols].values.astype(np.float32)
            return torch.from_numpy(vals)

        raise ValueError(f"Cannot extract 6 params from last row in {csv_path}. "
                         f"Require columns wa..wf or at least 6 numeric cols in the last row.")

    @staticmethod
    def _get_corresponded_params(img_paths: list[str], param_data: dict[tuple[int, int], torch.Tensor]) -> torch.Tensor:
        param_values = []
        img_file_name_template = re.compile(r"^(?P<cat>\d+)_(?P<wid>\d+)_")
        for img_path in img_paths:
            file_name = os.path.basename(img_path)
            match = img_file_name_template.search(file_name)
            if not match:
                raise RuntimeError(f"Filename does not match expected pattern: {file_name}")

            cat = int(match.group("cat"))
            wid = int(match.group("wid"))
            key = (cat, wid)

            if key not in param_data:
                raise RuntimeError(f"Parameter data not found for image: {img_path} (cat: {cat}, wid: {wid})")

            param_values.append(param_data[key])

        return torch.stack(param_values, dim=0).to(dtype=torch.float32)

    def __len__(self) -> int:
        """Total number of images in the dataset."""

        return len(self._img_paths)

    def __del__(self):
        """Close the zip file if it was opened."""

        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        """Exclude the zipfile object from the pickled state to avoid issues with multiprocessing."""

        return dict(super().__getstate__(), _zipfile=None)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Get the image and label at the specified index."""

        img_path = self._img_paths[index]
        cls_label = self._cls_labels[index]

        # Load image
        if self._from_zip:
            if self._zipfile is None:
                self._zipfile = zipfile.ZipFile(self._dset, 'r')
            with self._zipfile.open(img_path) as img_file:
                img = PIL.Image.open(img_file).convert('RGB')
        else:
            img = PIL.Image.open(img_path).convert('RGB')

        # Apply transforms.
        if self._transform is not None:
            img = self._transform(img)

        batch = dict(imgs=img, cls_labels=cls_label)

        if self._param_values is not None:
            param_value = self._param_values[index]
            if self._target_transform is not None:
                param_value = self._target_transform(param_value)
            batch['params'] = param_value

        if self._return_path:
            batch['img_paths'] = img_path

        return batch

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Image folder dataset


class ImageFolderDataset(torch.utils.data.Dataset):
    """A simple wrapper around torchvision.datasets.ImageFolder to return image paths along with images and labels."""

    def __init__(
        self,
        root: str,
        transform: typing.Callable | None = None,
        target_transform: typing.Callable | None = None,
        num_classes: int | None = None,
        insts_per_class: int | None = None,
        img_exts: typing.Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff'),
        shuffle_classes: bool = False,
        shuffle_instances: bool = False,
        return_img_path: bool = False,
    ):
        """Initialize the dataset.
        Parameters
        ----------
        root : str
            Path to the root directory of the dataset.
        transform : typing.Callable | None, optional
            A function/transform that takes in a PIL image and returns a transformed version. E.g., `transforms.Compose([...])`. If None, no transformation is applied.
        target_transform : typing.Callable | None, optional
            A function/transform that takes in the target label and transforms it. If None, no transformation is applied.
        num_classes : int | None, optional
            If specified, limits the number of classes to this number.
        insts_per_class : int | None, optional
            If specified, limits the number of instances per class to this number.
        img_exts : typing.Tuple[str, ...], optional
            Tuple of allowed image file extensions.
        shuffle_classes : bool, optional
            Whether to shuffle the order of classes. If True, the classes are randomly shuffled. If False, classes are taken in sorted order.
        shuffle_instances : bool, optional
            Whether to shuffle the order of instances within each class. If True, the first `insts_per_class` instances are randomly sampled from the class directory. If False, the first `insts_per_class` instances are taken in sorted order.
        return_img_path : bool, optional
            Whether to return the image path along with the image and label. If True, `__getitem__` returns a tuple of (image, label, img_path). If False, it returns (image, label).
        """

        assert num_classes is None or num_classes > 0, f'num_classes must be positive or None, but got {num_classes}'
        assert insts_per_class is None or insts_per_class > 0, f'insts_per_class must be positive or None, but got {insts_per_class}'

        super().__init__()

        self._transform = transform
        self._target_transform = target_transform
        self._return_img_path = return_img_path

        # Load dataset
        sub_dir_or_files = glob.glob(os.path.join(root, '*'))
        sub_dirs = [d for d in sub_dir_or_files if os.path.isdir(d)]
        sub_dirs = sorted(sub_dirs)

        self.shuffle_map_classes = list(range(len(sub_dirs)))
        if shuffle_classes:
            random.shuffle(self.shuffle_map_classes)
            sub_dirs = [sub_dirs[i] for i in self.shuffle_map_classes]  # Reorder sub_dirs according to the shuffled class order
            print(f'Shuffled class order. This may affect if you specify num_classes to limit the number of classes, as it will take the first num_classes after shuffling.')

        if num_classes is not None:
            cls_max_num = min(num_classes, len(sub_dirs))
            sub_dirs = sub_dirs[:cls_max_num]
            self.shuffle_map_classes = self.shuffle_map_classes[:cls_max_num]

        unique_cls_labels: list[str] = []
        img_paths: list[str] = []
        cls_ids: list[int] = []
        is_first_class = True

        for dir_path in sub_dirs:
            dir_name = os.path.basename(dir_path)
            unique_cls_labels.append(dir_name)

            files = glob.glob(os.path.join(dir_path, '*'))
            cls_img_paths: list[str] = []
            for file_path in files:
                if not os.path.isfile(file_path):
                    continue
                ext = os.path.splitext(file_path)[1].lower()
                if ext not in img_exts:
                    continue
                cls_img_paths.append(file_path)

            if shuffle_instances:
                random.shuffle(cls_img_paths)
                if is_first_class:
                    print(f'Shuffled instance order within each class. This may affect if you specify insts_per_class to limit the number of instances per class, as it will take the first insts_per_class after shuffling.')
                    is_first_class = False

            if insts_per_class is not None:
                max_num = min(insts_per_class, len(cls_img_paths))
                cls_img_paths = cls_img_paths[:max_num]

            img_paths.extend(cls_img_paths)
            cls_ids.extend([len(unique_cls_labels) - 1] * len(cls_img_paths))

        self._img_paths = img_paths
        self._unique_cls_labels = unique_cls_labels
        self._cls_ids = torch.tensor(cls_ids, dtype=torch.long)

    @property
    def num_classes(self) -> int:
        """Number of unique classes in the dataset."""

        return len(self._unique_cls_labels)

    @property
    def classes(self) -> list[str]:
        """List of class names in the dataset."""

        return self._unique_cls_labels

    def __len__(self) -> int:
        """Total number of images in the dataset."""

        return len(self._img_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, str]:
        """Get the image, label, and image path at the specified index."""

        img_path = self._img_paths[index]
        cls_id = self._cls_ids[index]

        # Load image and apply transforms
        img = PIL.Image.open(img_path).convert('RGB')

        if self._transform is not None:
            img = self._transform(img)
        if self._target_transform is not None:
            cls_id = self._target_transform(cls_id)

        if self._return_img_path:
            return img, cls_id, img_path

        return img, cls_id

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Dataset class for Visual Atoms


class VisualAtomsDataset(torch.utils.data.Dataset):
    """A dataset class for Visual Atoms, proposed by Takashima et al. in "Visual Atoms: Pre-training Vision Transformers with Sinusoidal Waves" (CVPR 2023)."""

    # autopep8: off
    PARAM_KEY_LINE_NAMI1         : typing.Final[str]       = 'nami1'
    PARAM_KEY_LINE_NAMI2         : typing.Final[str]       = 'nami2'
    PARAM_KEY_NUM_LINES          : typing.Final[str]       = 'num_lines'
    PARAM_KEY_NUM_VERTICES       : typing.Final[str]       = 'num_vertices'
    PARAM_KEY_OVAL_RATE_X        : typing.Final[str]       = 'oval_rate_x'
    PARAM_KEY_OVAL_RATE_Y        : typing.Final[str]       = 'oval_rate_y'
    PARAM_KEY_PERLIN_NOISE_COEFF : typing.Final[str]       = 'perlin_noise_coefficient'
    # autopep8: on

    PARAM_KEYS: typing.Final[list[str]] = [
        PARAM_KEY_LINE_NAMI1,
        PARAM_KEY_LINE_NAMI2,
        PARAM_KEY_NUM_LINES,
        PARAM_KEY_NUM_VERTICES,
        PARAM_KEY_OVAL_RATE_X,
        PARAM_KEY_OVAL_RATE_Y,
        PARAM_KEY_PERLIN_NOISE_COEFF,
    ]

    def __init__(self, dset: str, transform: typing.Callable | None = None, target_transform: typing.Callable | None = None, normalize: bool = True):
        """Initialize the Visual Atoms dataset.

        Parameters
        ----------
        dset : str
            Path to the root directory of the Visual Atoms dataset.
            This directory should contain an 'images' subdirectory with class subdirectories, and a 'dataset_state.json' file.
        transform : typing.Callable | None, optional
            A function/transform that takes in an PIL image and returns a transformed version. E.g., `transforms.Compose([...])`. If None, no transformation is applied.
        target_transform : typing.Callable | None, optional
            A function/transform that takes in the target parameter and transforms it. If None, no transformation is applied.
        normalize : bool, optional
            Whether to normalize the parameter values to the range [-1, 1].
        """

        super().__init__()

        self._transform = transform
        self._target_transform = target_transform

        # Get image paths and class labels
        dirs = glob.glob(os.path.join(dset, 'images', '*'))
        dirs = [dir_path for dir_path in dirs if os.path.isdir(dir_path)]
        dirs = sorted(dirs)
        self._num_classes = len(dirs)

        self._img_paths = []
        self._cls_labels = []
        self._str_cls_labels = []
        self._unique_cls_labels = []

        for dir_path in dirs:
            assert os.path.isdir(dir_path), f'Expected directory, but got file: {dir_path}'
            dir_name = os.path.basename(dir_path)

            img_paths = glob.glob(os.path.join(dir_path, '*'))
            img_paths = [img_path for img_path in img_paths if os.path.isfile(img_path)]
            assert len(img_paths) > 0, f'No images found in directory: {dir_path}'

            self._img_paths.extend(img_paths)
            self._cls_labels.extend([len(self._unique_cls_labels)] * len(img_paths))
            self._str_cls_labels.extend([dir_name] * len(img_paths))
            self._unique_cls_labels.append(dir_name)

        assert len(self._img_paths) > 0, 'No images found in the dataset.'
        assert len(self._img_paths) == len(self._str_cls_labels), 'Mismatch between number of images and labels.'

        self._cls_labels = torch.tensor(self._cls_labels, dtype=torch.long)

        # Get dataset state
        state_file = os.path.abspath(os.path.join(dset, 'dataset_state.json'))
        assert os.path.isfile(state_file), f'Dataset state file (dataset_state.json) not found: {state_file}'

        with open(state_file, 'r') as file:
            dset_state = json.load(file)

        # Read params
        dset_params = dset_state['params']
        params_list: list[list[float]] = []
        for cls_label in self._str_cls_labels:
            param_values = []
            for param_key in self.PARAM_KEYS:
                param_values.append(dset_params[cls_label][param_key])
            params_list.append(param_values)
        params = torch.tensor(params_list, dtype=torch.float32)  # (N, 7)

        # Normalize params if required
        if normalize:
            # NOTE: For more details on the parameter ranges, refer to the Visual Atoms paper and implementation.
            # https://github.com/masora1030/CVPR2023-FDSL-on-VisualAtom/blob/de80c6972f99a57238633ba13b5a06a846272f30/visual_atomic_renderer/render_VisualAtom.py

            # Read param ranges
            dset_args = dset_state['args']

            # nami1
            nami1_min = dset_args['nami_1_min']
            nami1_max = dset_args['nami_1_max']
            params[:, 0] = (params[:, 0] - nami1_min) / (nami1_max - nami1_min) * 2.0 - 1.0

            # nami2
            nami2_min = dset_args['nami_2_min']
            nami2_max = dset_args['nami_2_max']
            params[:, 1] = (params[:, 1] - nami2_min) / (nami2_max - nami2_min) * 2.0 - 1.0

            # num_lines
            num_lines_min = dset_args['min_num_lines']
            num_lines_max = dset_args['max_num_lines']
            params[:, 2] = (params[:, 2] - num_lines_min) / (num_lines_max - num_lines_min) * 2.0 - 1.0

            # num_vertices
            num_vertices_min = dset_args['min_num_vertices']
            num_vertices_max = dset_args['max_num_vertices']
            params[:, 3] = (params[:, 3] - num_vertices_min) / (num_vertices_max - num_vertices_min) * 2.0 - 1.0

            # oval_rate_x, oval_rate_y
            oval_rate_min = 1.0
            oval_rate_max = dset_args['max_oval_rate']
            params[:, 4] = (params[:, 4] - oval_rate_min) / (oval_rate_max - oval_rate_min) * 2.0 - 1.0
            params[:, 5] = (params[:, 5] - oval_rate_min) / (oval_rate_max - oval_rate_min) * 2.0 - 1.0

            # perlin_noise_coefficient
            perlin_noise_coeff_min = dset_args['perlin_min']
            perlin_noise_coeff_max = perlin_noise_coeff_min + 4.0
            params[:, 6] = (params[:, 6] - perlin_noise_coeff_min) / (perlin_noise_coeff_max - perlin_noise_coeff_min) * 2.0 - 1.0

        assert params.ndim == 2 and params.shape[1] == len(self.PARAM_KEYS), f'Params shape mismatch: expected (?, {len(self.PARAM_KEYS)}), but got {params.shape}'
        self._params = params  # (N, 7)

    @property
    def num_classes(self) -> int:
        """Number of unique classes in the dataset."""

        return self._num_classes

    @property
    def classes(self) -> list[str]:
        """List of class names in the dataset."""

        return self._str_cls_labels

    def __len__(self) -> int:
        """Total number of images in the dataset."""

        return len(self._img_paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Get the image, label, and parameter at the specified index."""

        img_path = self._img_paths[index]
        cls_idx = self._cls_labels[index]

        param = self._params[index]  # (7,)

        # Load image and apply transforms
        img = PIL.Image.open(img_path).convert('RGB')
        if self._transform is not None:
            img = self._transform(img)

        # Apply target transform to param if specified
        if self._target_transform is not None:
            param = self._target_transform(param)

        return dict(imgs=img, cls_labels=cls_idx, params=param)

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PK BatchSampler for metric learning


class PKBatchSampler(torch.utils.data.Sampler):
    """BatchSampler that yields batches containing P classes with K instances each (PK sampling) for metric learning."""

    def __init__(
        self,
        labels: torch.Tensor,
        P: int,
        K: int,
        rank: int = 0,
        num_replicas: int = 1,
        shuffle: bool = True,
        seed: int = 0,
    ):
        """BatchSampler that yields index lists corresponding to P classes × K instances per batch.
        This implementation assumes that each class has the exact same number of instances.

        Parameters
        ----------
        labels : torch.Tensor
            1D tensor of length N containing class indices for each sample in the dataset.
        P : int
            Number of distinct classes per batch.
        K : int
            Number of instances per class in each batch.
        rank : int, optional
            Rank of the current process for distributed training.
        num_replicas : int, optional
            Total number of processes for distributed training.
        shuffle : bool, optional
            Whether to shuffle the classes and instance order.
        seed : int, optional
            Base random seed for this sampler.
        """

        assert labels.ndim == 1, "labels must be a 1D tensor"
        assert P > 0 and K > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas

        self.labels = labels.cpu().long().numpy()
        self.P = P
        self.K = K
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = seed

        # Build class -> indices mapping
        class_to_indices: dict[int, list[int]] = {}
        for idx, y in enumerate(self.labels):
            class_to_indices.setdefault(int(y), []).append(int(idx))

        self.class_to_indices = class_to_indices
        self.classes = sorted(list(class_to_indices.keys()))

        # Sanity check
        assert len(self.classes) > 0, "No classes found in labels."

    def __len__(self) -> int:
        # Sampler is effectively infinite; length is undefined.
        # DataLoader does not require __len__ for Iterable-style samplers.
        return 2 ** 31 - 1  # large placeholder to avoid some edge cases

    def __iter__(self):
        # Rank-dependent RNG to avoid identical sampling across ranks.
        global_rng = np.random.RandomState(self.seed)
        rng = np.random.RandomState(self.seed + self.rank * 100003)

        all_classes = np.array(self.classes, dtype=np.int64)

        # Infinite loop: will be stopped by training_loop
        while True:
            # Split classes for this rank; we will iterate over them in chunks of size P.
            classes = all_classes.copy()  # start from the full list to ensure global shuffling is consistent across ranks
            if self.shuffle:
                global_rng.shuffle(classes)
            if self.num_replicas > 1:
                classes_for_rank = classes[self.rank::self.num_replicas]
            else:
                classes_for_rank = classes

            # Sanity check: if a rank receives no classes, this is a configuration error.
            if classes_for_rank.size == 0:
                raise RuntimeError(
                    f'PKBatchSampler: no classes assigned to rank {self.rank}. num_replicas={self.num_replicas} may be too large.'
                )

            # Shuffle classes for this rank locally to get different class combinations each epoch
            if self.shuffle:
                rng.shuffle(classes_for_rank)

            # Per-class pools: shuffled index queues to avoid frequent replacement sampling
            pools: dict[int, list[int]] = {int(c): [] for c in classes_for_rank}

            # Helper: get K indices for class c
            def sample_k_from_class(c: int) -> list[int]:
                idxs = self.class_to_indices[c]
                if len(idxs) >= self.K:
                    # Use pool + reshuffling without replacement as long as possible
                    pool = pools[c]
                    if len(pool) < self.K:
                        # Refuel the pool by shuffling the indices of this class
                        perm = rng.permutation(idxs)
                        pool.extend(int(x) for x in perm)
                    chosen = pool[: self.K]
                    pools[c] = pool[self.K:]
                    return chosen
                else:
                    raise RuntimeError(f'Class {c} has only {len(idxs)} instances, which is less than K={self.K}. Consider setting K to a smaller value or using a different sampling strategy for this class.')

            num_classes = len(classes_for_rank)

            if num_classes >= self.P:
                # Iterate classes in chunks of size P
                pos = 0
                while pos + self.P <= num_classes:
                    # Scan next P classes
                    selected_classes = classes_for_rank[pos: pos + self.P]
                    pos += self.P

                    batch_indices: list[int] = []
                    for c in selected_classes:
                        batch_indices.extend(sample_k_from_class(int(c)))

                    # Yield 1 batch (P classes × K instances)
                    yield batch_indices
            else:
                raise RuntimeError(f'Number of classes assigned to rank {self.rank} is {num_classes}, which is less than P={self.P}. Consider setting P to a smaller value or using a different sampling strategy for this rank.')

            # After exhausting one pass over classes (Case 1), loop continues and reshuffles classes again.


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
