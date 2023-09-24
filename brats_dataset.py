from torch.utils.data import Dataset
import os

from brats_reader import BratsReader
import logging
import numpy as np
import time
import random


class BratsDataset(Dataset):

    def __init__(self, root_dir, prefix="BraTS2021_", use_cache=False, cache_ratio=1.0):
        directories = list(sorted(os.listdir(root_dir)))

        self.sample_ids = []
        self.sample_id_to_index = {}
        for dir in directories:
            if dir.startswith(prefix):
                sample_id = int(dir[len(prefix):])
                self.sample_ids.append(sample_id)
                self.sample_id_to_index[sample_id] = len(self.sample_ids) - 1
        assert len(self.sample_ids) == len(self.sample_id_to_index)

        self.reader = BratsReader(root_dir, prefix)
        self.segmentation_mapping = np.array([0, 1, 2, -1, 3])  # label=3 is not used
        self.valid_labels = frozenset([0, 1, 2, 3])

        self.cache = [None] * len(self.sample_ids) if use_cache else None
        self.cache_ratio = cache_ratio

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        if self.cache is not None and self.cache[idx] is not None:
            return self.cache[idx]

        sample_id = self.sample_ids[idx]
        mri, segmentation = self.load_sample(sample_id)

        if (self.cache is not None) and (random.random() < self.cache_ratio):
            self.cache[idx] = (mri, segmentation)

        return mri, segmentation

    def get_sample_id(self, sample_idx):
        return self.sample_ids[sample_idx]

    def load_sample(self, sample_id):
        """
        Returns tuple:
            1) MRI: [C=4, H, W, D] (depth is clipped)
            2) Segmentation: [C=1, H, W, D] (depth is clipped)
        """
        mri = self.reader.load_mri_as_numpy(sample_id)
        mri = clip_4d(mri)
        logging.info(f"MRI image shape: {mri.shape}")

        segmentation = self.reader.load_segmentation_as_numpy(sample_id)
        segmentation = segmentation[np.newaxis, ...]  # add the "channel" dimension
        segmentation = self.convert_segmentation(segmentation)
        segmentation = clip_4d(segmentation)
        t = time.time()
        uniq = np.unique(segmentation)
        logging.debug(f"unique() took: {time.time() - t:.1f}s")
        logging.info(
            f"Target segmentation shape: {segmentation.shape}, values: {uniq}")
        return mri, segmentation

    def convert_segmentation(self, segmentation):
        t = time.time()
        result = self.segmentation_mapping[segmentation]
        logging.debug(f"Segmentation mapping took: {time.time() - t:.1f}s")
        return result

    def _check_segmentation(self, segmentation):
        for label in np.ravel(segmentation):
            if label not in self.valid_labels:
                raise ValueError(f"Invalid segmentation label: {label}")


def clip_4d(tensor):
    """
    Clips last dimension from 155 to 144 to make it compatible with the network.
    Accepts [C, H, W, D].
    """
    t = time.time()
    assert len(tensor.shape) == 4
    assert tensor.shape[3] == 155
    result = tensor[:, :, :, 5:-6]
    logging.debug(f"Clipping took: {time.time() - t:.1f}s")
    return result


def unclip_4d(tensor):
    """Accepts [C, H, W, D]."""
    assert len(tensor.shape) == 4
    assert tensor.shape[3] == 144
    unclipped = np.zeros([*tensor.shape[:-1], 155], dtype=tensor.dtype)
    unclipped[:, :, :, 5:-6] = tensor
    return unclipped
