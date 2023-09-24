import torch
import logging
import brats_dataset as bd
import numpy as np


def segmentation_3d_from_prediction_4d(prediction_4d, unclip=False):
    """"
    Accepts [C, H, W, D] prediction.
    Returns [H, W, D] segmentation.
    """
    assert len(prediction_4d.shape) == 4
    logging.info(f"prediction_4d shape: {prediction_4d.shape}")
    prediction_4d = torch.as_tensor(prediction_4d, dtype=torch.float32)
    if unclip:
        prediction_4d = torch.as_tensor(bd.unclip_4d(prediction_4d.numpy()),
                                        dtype=prediction_4d.dtype, device=prediction_4d.device)
    logging.info(f"prediction_4d shape: {prediction_4d.shape}")
    result = torch.argmax(prediction_4d, dim=0).int()
    # result = result[0]
    logging.info(f"Prediction shape: {result.shape}")
    assert len(result.shape) == 3
    return result


def sample_to_dataset(sample, device):
    """Takes one sample [C, H, W, D], returns PyTorch dataset [B=1, C, H, W, D]."""
    assert len(sample.shape) == 4
    dataset = sample[np.newaxis, ...]
    dataset = torch.as_tensor(dataset, device=device, dtype=torch.float)
    assert len(dataset.shape) == 5
    return dataset


def segmentation_sample_to_dataset(sample, device):
    """Takes one sample [C, H, W, D], returns PyTorch dataset [B=1, C, H, W, D]."""
    dataset = sample_to_dataset(sample, device)
    return dataset.to(torch.int8).to(device)
