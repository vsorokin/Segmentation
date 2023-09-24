import torch
import logging
from model import UNet
import time


class Inference:
    def __init__(self, checkpoint_file, device, unet_filters=30):
        self.device = device
        logging.info(f"Loading model from [{checkpoint_file}]")
        checkpoint = torch.load(checkpoint_file, map_location=torch.device(device))

        self.model = UNet(n=unet_filters).to(device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def predict(self, dataset):
        """
        Dataset is [B, C, H, W, D].
        Returns [B, C, H, W, D] on CPU.
        """
        assert len(dataset.shape) == 5

        logging.info(f"Running inference...")
        t = time.time()
        with torch.inference_mode():
            pred = self.model(dataset.to(self.device))
        logging.info(
            f"Prediction shape: {pred.shape} (prediction took: {time.time() - t:.1f}s)")
        return pred.cpu()
