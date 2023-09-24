import logging
import time

import nibabel as nib
import numpy as np


class BratsReader:
    modalities = ['flair', 't1', 't1ce', 't2']

    def __init__(self, brats_path, prefix="BraTS2021_"):
        self.brats_path = brats_path
        self.prefix = prefix

    def load_mri_as_numpy(self, id):
        """Returns [C, H, W, D]"""
        id_str = f"{id:05d}"
        all_modalities = []
        for modality in self.modalities:
            img_file = f"{self.brats_path}/{self.prefix}{id_str}/{self.prefix}{id_str}_{modality}.nii.gz"
            all_modalities.append(self._load_image_as_numpy(img_file))

        t = time.time()
        mri_4d = np.stack(all_modalities, dtype=np.float32)
        logging.debug(f"np.stack() took: {time.time() - t:.1f}s")

        return mri_4d  # [C, H, W, D]

    def load_segmentation_as_numpy(self, id):
        """Returns [H, W, D]"""
        id_str = f"{id:05d}"
        img_file = f"{self.brats_path}/{self.prefix}{id_str}/{self.prefix}{id_str}_seg.nii.gz"
        segmentation = self._load_image_as_numpy(img_file).astype(np.int8)
        return segmentation  # [H, W, D]

    def _load_image_as_numpy(self, filename):
        """Returns [H, W, D]"""
        start = time.time()
        img = nib.load(filename)
        result = img.get_fdata(dtype=np.float32)
        logging.info(f"Loaded image: {filename} (took {time.time() - start:.1f}s)")
        return result
