"""Utility script for ImageNet feature analysis.

License
-------

Copyright (c) 2026 Shinei Arakawa
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
"""

from __future__ import annotations

import json

import nltk
import numpy as np
from nltk.corpus import wordnet
from nltk.corpus.reader.wordnet import Synset

import torch_utils.util as util

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ImageNet utility


class ImageNetMetaData:
    def __init__(self):
        nltk.download('wordnet', quiet=True)

        with util.open_url('https://raw.githubusercontent.com/pytorch/vision/refs/heads/main/gallery/assets/imagenet_class_index.json') as file:
            self._class_idx = json.load(file)
        with util.open_url('https://raw.githubusercontent.com/s-kumano/imagenet-japanese-labels/refs/heads/main/imagenet_labels.jp.txt') as file:
            self._human_label_jp: list[str] = []
            for i in range(len(self._class_idx)):
                line = file.readline()
                line_utf8 = line.decode('utf-8')
                self._human_label_jp.append(line_utf8.strip())

    @property
    def num_classes(self) -> int:
        return len(self._class_idx)

    def get_synset_id_raw(self, class_id: str | int) -> str:
        return self._class_idx[str(class_id)][0]

    def get_synset_id(self, class_id: str | int) -> Synset:
        raw_id = self.get_synset_id_raw(class_id)

        pos = raw_id[0]
        offset = int(raw_id[1:])

        synset_id: Synset = wordnet.synset_from_pos_and_offset(pos, offset)

        return synset_id

    def get_human_label(self, class_id: str | int) -> str:
        return self._class_idx[str(class_id)][1]

    def get_human_label_jp(self, class_id: str | int) -> str:
        return self._human_label_jp[int(class_id)]

    def get_cls_id_from_synset_id_raw(self, synset_id: str) -> int | None:
        for class_id_str, entry in self._class_idx.items():
            if entry[0] == synset_id:
                return int(class_id_str)
        return None


def load_imagenet_meta_data() -> ImageNetMetaData:
    """Load ImageNet meta data.

    Returns
    -------
    ImageNetMetaData
            The ImageNet meta data object.
    """

    return ImageNetMetaData()

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# WordNet utility


def build_imagenet_synset_list(meta: ImageNetMetaData, num_classes: int = 1000) -> list[Synset]:
    synset_list = []
    for class_id in range(num_classes):
        synset = meta.get_synset_id(class_id)
        synset_list.append(synset)
    return synset_list


def compute_wup_similarity_matrix(synsets: list[Synset]) -> np.ndarray:
    num_cls = len(synsets)
    sim = np.zeros((num_cls, num_cls), dtype=np.float32)

    for i in range(num_cls):
        sim[i, i] = 1.0  # Self-similarity
        for j in range(i + 1, num_cls):
            s = synsets[i].wup_similarity(synsets[j])
            if s is None:
                s = 0.0
            sim[i, j] = s
            sim[j, i] = s

    return sim


def similarity_to_distance(sim: np.ndarray) -> np.ndarray:
    dist = 1.0 - sim
    return dist


def compute_near_far_indices(
    dist: np.ndarray,
    k_near: int = 3,
    k_far: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    n = dist.shape[0]
    near = np.zeros((n, k_near), dtype=np.int64)
    far = np.zeros((n, k_far), dtype=np.int64)

    for i in range(n):
        order = np.argsort(dist[i])
        order = order[order != i]

        near[i, :] = order[:k_near]
        far[i, :] = order[-k_far:]

    return near, far


def build_near_far_class_sets(
    meta: ImageNetMetaData,
    num_classes: int = 1000,
    k_near: int = 3,
    k_far: int = 3,
):
    """Build near and far class sets based on Wu-Palmer similarity.

    Parameters
    ----------
    meta : ImageNetMetaData
        The ImageNet meta data object.
    num_classes : int, optional
        Number of classes to consider (default is 1000).
    k_near : int, optional
        Number of near classes to select (default is 3).
    k_far : int, optional
        Number of far classes to select (default is 3).
    """

    synsets = build_imagenet_synset_list(meta, num_classes=num_classes)

    sim = compute_wup_similarity_matrix(synsets)
    dist = similarity_to_distance(sim)

    near, far = compute_near_far_indices(dist, k_near=k_near, k_far=k_far)

    tuples = []

    for i in range(num_classes):
        anchor = i
        near_i = near[i].tolist()
        far_i = far[i].tolist()
        tuples.append((anchor, near_i, far_i))

    return tuples, dist, sim


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
