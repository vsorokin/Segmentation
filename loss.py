import logging

import numpy as np
import torch
import torch.nn as nn
import time


class DiceLoss(nn.Module):
    def __init__(self, device):
        super(DiceLoss, self).__init__()
        self.device = device

    def forward(self, input, target):
        return self.dice_loss(input, target)

    def dice_loss(self, input, target):
        """
        Args:
            input: a tensor of shape [B, C, H, W, D].
            target: a tensor of shape [B, 1, H, W, D].
        """
        t = time.time()
        num_classes = 4
        smooth = 1.

        assert len(input.shape) == 5
        assert len(target.shape) == 5
        assert input.shape[1] == num_classes
        assert target.shape[1] == 1

        # input = input[:, 1:, :, :, :]  # ignoring background class
        # print(f"Loss 1: target shape: {target.shape}")
        target = target.squeeze(1)  # [B, H, W, D]
        # print(f"Loss 2: target shape: {target.shape}")
        eye = torch.eye(num_classes, device=self.device)
        # print(f"Loss 2.5: eye shape: {eye.shape}")
        target_onehot = eye[target.to(torch.int)]  # [B, H, W, D, C]
        # print(f"Loss 3: target_onehot shape: {target_onehot.shape}")
        target_onehot = target_onehot.permute(0, 4, 1, 2, 3).float()  # [B, C, H, W, D]
        # print(f"Loss 4: target_onehot shape: {target_onehot.shape}")
        assert target_onehot.shape[1] == num_classes
        # target_onehot = target_onehot[:, 1:, :, :, :]  # ignoring background class
        dims = (0, 2, 3, 4)  # sum all but the channel
        numerator = torch.sum(input * target_onehot, dims)
        denominator = torch.sum(input + target_onehot, dims)
        dice_loss = (2. * numerator + smooth) / (denominator + smooth)
        dice_loss_avg = dice_loss.mean()
        result = 1. - dice_loss_avg
        logging.info(
            f"Dice loss: {result:.4f} (Dice score: {dice_loss_avg:.4f}, per channel: {self.format_tensor(dice_loss)}), took: {time.time() - t:.1f}s")
        return result

    def format_tensor(self, tensor):
        return "[" + " ".join([f"{v:.4f}" for v in tensor.tolist()]) + "]"

    def calculate_dice_loss(self, input_prediction_4d, target_segmentation_3d):
        assert len(input_prediction_4d.shape) == 4
        assert len(target_segmentation_3d.shape) == 3
        assert input_prediction_4d[0].shape == target_segmentation_3d.shape

        return self.dice_loss(
            input=torch.as_tensor(input_prediction_4d[np.newaxis, ...]),
            target=torch.as_tensor(target_segmentation_3d[np.newaxis, np.newaxis, ...])
        )


def main():
    import torch
    from loss import DiceLoss
    pred = torch.as_tensor([[[[[0.]]], [[[1.]]], [[[0]]], [[[0]]]]], dtype=torch.float)
    target = torch.as_tensor([[[[[1]]]]], dtype=torch.int)
    dice_loss = DiceLoss()
    r = dice_loss(pred, target)
    print("dice loss:", r)
