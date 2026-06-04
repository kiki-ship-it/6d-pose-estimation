"""Training utilities and trainer."""
import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any
from pathlib import Path
from tqdm import tqdm


class Trainer:
    """Training helper class."""

    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, criterion: nn.Module, device: str = 'cuda', work_dir: str = './'):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def train(self, train_loader: DataLoader, val_loader: DataLoader = None, num_epochs: int = 100, save_interval: int = 5):
        self.model.to(self.device)

        for epoch in range(1, num_epochs + 1):
            self.model.train()
            running_loss = 0.0

            for batch in tqdm(train_loader, desc=f"Training Epoch {epoch}/{num_epochs}"):
                images = batch['image'].to(self.device)
                keypoints_gt = batch['keypoints'].to(self.device)

                # Forward
                outputs = self.model(images)
                pred_keypoints = outputs['keypoints']

                # Loss (L2 on keypoint coordinates)
                loss = self.criterion(pred_keypoints, keypoints_gt)

                # Backprop
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * images.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)
            print(f"Epoch {epoch}/{num_epochs}, Loss: {epoch_loss:.6f}")

            # Validation step
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                print(f"Validation Loss: {val_loss:.6f}")

            # Save checkpoint
            if epoch % save_interval == 0 or epoch == num_epochs:
                ckpt_path = self.work_dir / f'checkpoint_epoch_{epoch}.pth'
                torch.save({'epoch': epoch, 'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}, ckpt_path)

    def validate(self, val_loader: DataLoader):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                images = batch['image'].to(self.device)
                keypoints_gt = batch['keypoints'].to(self.device)

                outputs = self.model(images)
                pred_keypoints = outputs['keypoints']
                loss = self.criterion(pred_keypoints, keypoints_gt)
                running_loss += loss.item() * images.size(0)

        val_loss = running_loss / len(val_loader.dataset)
        return val_loss
