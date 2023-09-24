from model import UNet
import numpy as np
import torch
import logging
import time
from loss import DiceLoss
from torch.optim import SGD
from brats_dataset import BratsDataset
from torch.utils.data import DataLoader


class Training:

    def __init__(self,
                 metadata,
                 result_path,
                 device,
                 brats_dir,
                 brats_validation_dir,
                 batch_size,
                 use_cache,
                 cache_ratio,
                 use_cache_for_validation,
                 data_loading_num_workers,
                 checkpoint_to_load,
                 epochs,
                 learning_rate,
                 momentum,
                 epochs_per_checkpoint,
                 epochs_per_permanent_checkpoint,
                 unet_filters=30,
                 **kwargs):
        assert len(kwargs) == 0
        self.result_path = result_path
        self.metadata = metadata
        self.device = torch.device(device)
        self.epochs = epochs
        self.epochs_per_checkpoint = epochs_per_checkpoint
        self.epochs_per_permanent_checkpoint = epochs_per_permanent_checkpoint

        dataset = BratsDataset(brats_dir, use_cache=use_cache, cache_ratio=cache_ratio)
        self.data_loader = DataLoader(dataset,
                                      batch_size=batch_size,
                                      num_workers=data_loading_num_workers,
                                      shuffle=True)

        val_dataset = BratsDataset(brats_validation_dir, use_cache=use_cache_for_validation)
        self.val_data_loader = DataLoader(val_dataset,
                                          batch_size=batch_size,
                                          num_workers=data_loading_num_workers,
                                          shuffle=False) if brats_validation_dir else None

        self.model = UNet(n=unet_filters).to(self.device)  # .float()?

        # optimizer = Adam(model.parameters(), lr=learning_rate)
        self.optimizer = SGD(self.model.parameters(), lr=learning_rate, momentum=momentum,
                             nesterov=True)
        self.dice_loss = DiceLoss(device=self.device)

        if checkpoint_to_load is not None:
            logging.info(f"LOADING CHECKPOINT: [{checkpoint_to_load}]")
            checkpoint = torch.load(checkpoint_to_load)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.losses = checkpoint['losses']
            self.val_losses = checkpoint['val_losses']
            self.epoch = checkpoint['epoch'] + 1  # next epoch
            self.num_epochs = checkpoint['num_epochs']
            self.model.train()
            logging.info("\n===== CONTINUING FROM A CHECKPOINT =====\n")
            logging.info(
                f"Checkpoint loaded from [{checkpoint_to_load}]. Starting from epoch {self.epoch}.")
        else:
            self.losses = []
            self.val_losses = []
            self.epoch = 1
            self.num_epochs = epochs

    def run(self):
        while self.epoch <= self.num_epochs:
            logging.info(f"\n\n======== Epoch {self.epoch} of {self.num_epochs} ========")
            start_epoch = time.time()

            # Training
            logging.info(f"Running training")
            self.model.train()
            epoch_loss = self._run_iteration(self.data_loader)
            self.losses.append(epoch_loss)

            # Validation
            if self.val_data_loader:
                logging.info(f"Running validation")
                self.model.eval()
                with torch.no_grad():
                    epoch_val_loss = self._run_iteration(self.val_data_loader, validation=True)
                    self.val_losses.append(epoch_val_loss)

            logging.info(f"Epoch took: {time.time() - start_epoch:.1f}s")

            # Saving checkpoints
            # Current:
            if (self.epoch % self.epochs_per_checkpoint == 0) or (
                    self.epoch == self.num_epochs):
                self.save_checkpoint("checkpoint.pth")
            # Permanent:
            if self.epoch % self.epochs_per_permanent_checkpoint == 0:
                self.save_checkpoint(f"checkpoint_epoch{self.epoch}.pth")

            self.epoch += 1

    def _run_iteration(self, data_loader, validation=False):
        mini_batch_losses = []
        mini_batch_num = 1
        stage_name = "Validation" if validation else "Training"
        for input_batch, target_batch in data_loader:
            logging.info(
                f"\n--- {stage_name}: mini-batch {mini_batch_num} (epoch {self.epoch}) ---")
            t = time.time()
            input_batch = input_batch.float().to(self.device)
            target_batch = target_batch.to(torch.int8).to(self.device)
            logging.debug(f"Sending to device [{self.device}] took: {time.time() - t:.1f}s")

            start_inference = time.time()
            output = self.model(input_batch)
            logging.info(
                f"Inference done, output shape: {output.shape}, took: {time.time() - start_inference:.1f}s")

            loss = self.dice_loss(output, target_batch)
            mini_batch_loss = loss.item()
            logging.info(f"Mini-batch loss: {mini_batch_loss}")
            mini_batch_losses.append(mini_batch_loss)

            if not validation:
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            mini_batch_num += 1

        epoch_loss = np.mean(mini_batch_losses)
        logging.info(f"Epoch loss ({stage_name.lower()}): {epoch_loss}")
        return epoch_loss

    def save_checkpoint(self, checkpoint_file_name):
        checkpoint = {'epoch': self.epoch,
                      'num_epochs': self.num_epochs,
                      'model_state_dict': self.model.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict(),
                      'losses': self.losses,
                      'val_losses': self.val_losses,
                      'metadata': self.metadata}
        start_checkpoint = time.time()
        checkpoint_file = f"{self.result_path}/{checkpoint_file_name}"
        torch.save(checkpoint, checkpoint_file)
        logging.info(
            f"Checkpoint saved to [{checkpoint_file}] (took: {time.time() - start_checkpoint:.1f}s)")
