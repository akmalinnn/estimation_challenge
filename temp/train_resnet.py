#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from efficientnet_pytorch import EfficientNet
from torchvision import models
from pathlib import Path
from collections import defaultdict
import numpy as np
import json
import multiprocessing
import matplotlib.pyplot as plt
import random
from sklearn.utils.class_weight import compute_sample_weight

# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model parameters
in_c = 2  # Number of input channels
num_c = 1 # Number of classes to predict

# Directory containing optical flow files
of_dir = "data/flow_diff"
# JSON file containing speed labels
label_json = "data/data_filtered.json"
# File to save used data
used_data_file = "used_data.json"

class OFDataset(Dataset):
    def __init__(self, of_dir, label_json, used_data_file, oversample=True):
        self.of_dir = Path(of_dir)
        self.labels = json.load(open(label_json))
        self.files = sorted(self.of_dir.glob("diff_flow_*.npy"))
        self.valid_indices = []
        self.used_data = {}
        self.label_values = []
        self.oversample = oversample

        for i in range(len(self.files) - 2):
            try:
                file1, file2, file3 = self.files[i:i+3]

                img1 = file1.stem.replace("diff_flow_", "").replace(".png", "") + ".jpg"
                img2 = file2.stem.replace("diff_flow_", "").replace(".png", "") + ".jpg"
                img3 = file3.stem.replace("diff_flow_", "").replace(".png", "") + ".jpg"

                # Extract video ID
                vid1, vid2, vid3 = img1.split("_frame_")[0], img2.split("_frame_")[0], img3.split("_frame_")[0]

                # Ensure frames are from the same video
                if vid1 != vid2 or vid2 != vid3:
                    continue

                # Ensure labels exist
                if img1 in self.labels and img2 in self.labels and img3 in self.labels:
                    # Compute acceleration label
                    speed1 = self.labels[img1]["speed"]
                    speed2 = self.labels[img2]["speed"]
                    speed3 = self.labels[img3]["speed"]
                    label = ((speed2 + speed3) / 2) - ((speed1 + speed2) / 2)
                    
                    # Discretize the label for oversampling
                    discretized_label = self.discretize_label(label)
                    
                    if discretized_label == 0 and random.random() > 0.08:  # Downsample zeros
                        continue

                    self.valid_indices.append(i)
                    self.used_data[str(file1)] = label
                    self.label_values.append(discretized_label)

            except Exception as e:
                print(f"Skipping due to error: {e}")

        # Save used data
        with open(used_data_file, "w") as f:
            json.dump(self.used_data, f, indent=4)
            
        # Compute sample weights for oversampling
        if self.oversample:
            self.sample_weights = self._compute_sample_weights()
        else:
            self.sample_weights = None

    def discretize_label(self, label):
        """Discretize label into bins for better class balancing"""
        if abs(label) < 0.5:
            return 0  # no change / very mild
        elif label >= 0.5 and label < 3:
            return 1  # small acceleration
        elif label >= 3:
            return 2  # large acceleration
        elif label <= -0.5 and label > -3:
            return -1  # small deceleration
        elif label <= -3:
            return -2  # large deceleration

    def _compute_sample_weights(self):
        """Compute sample weights for oversampling minority classes"""
        # Count samples per class
        class_counts = defaultdict(int)
        for label in self.label_values:
            class_counts[label] += 1
            
        # Compute weights inversely proportional to class frequencies
        weights = []
        for label in self.label_values:
            weights.append(1.0 / class_counts[label])
            
        return weights

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        true_idx = self.valid_indices[idx]
        file1, file2, file3 = self.files[true_idx:true_idx+3]

        img1 = file1.stem.replace("diff_flow_", "").replace(".png", "") + ".jpg"
        img2 = file2.stem.replace("diff_flow_", "").replace(".png", "") + ".jpg"
        img3 = file3.stem.replace("diff_flow_", "").replace(".png", "") + ".jpg"

        # Extract video ID to ensure they match
        vid1, vid2, vid3 = img1.split("_frame_")[0], img2.split("_frame_")[0], img3.split("_frame_")[0]
        assert vid1 == vid2 == vid3, "Error: Different videos in same sample!"

        # Get speed labels
        speed1 = self.labels[img1]["speed"]
        speed2 = self.labels[img2]["speed"]
        speed3 = self.labels[img3]["speed"]

        # Compute acceleration label
        label = (((speed2 + speed3) / 2) - ((speed1 + speed2) / 2))

        # Load optical flow data
        of_tensor = np.load(file1)
        if of_tensor.shape[-1] == 3:
            of_tensor = of_tensor[:, :, :2]  # Keep only 2 channels

        of_tensor = np.transpose(of_tensor, (2, 0, 1))  # Convert to (C, H, W)
        of_tensor = torch.tensor(of_tensor, dtype=torch.float32)

        return of_tensor, torch.tensor(label, dtype=torch.float32)

# Custom collate function to filter out None values
def custom_collate(batch):
    batch = [b for b in batch if b is not None]  
    if not batch:  
        return torch.zeros((1, in_c, 64, 64)), torch.tensor([0.0])
    return torch.utils.data.default_collate(batch)

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)

    # Load dataset with oversampling
    ds = OFDataset(of_dir, label_json, used_data_file, oversample=True)

    # labels = []
    # for i in range(len(ds)):
    #     sample = ds[i]
    #     if sample is not None:
    #         _, label = sample
    #         labels.append(label.item())

    # labels = np.array(labels)
    # num_zeros = np.sum(labels == 0)

    # # Plot histogram
    # plt.figure(figsize=(12, 6))
    # plt.hist(labels, bins=100, edgecolor='black', alpha=0.7)
    # plt.text(0, num_zeros + 5, f"0 Count: {num_zeros}", fontsize=12, color='red', ha='center')
    # plt.xlabel("Label Value (Acceleration)")
    # plt.ylabel("Frequency")
    # plt.title("Distribution of Labels Before Oversampling")
    # plt.grid(axis='y', linestyle='--', alpha=0.7)
    # plt.show()

    # print(f"Number of zero labels: {num_zeros}")

    # Train-test split
    train_size = int(0.8 * len(ds))
    val_size = len(ds) - train_size
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size])

    # Create weighted samplers for oversampling
    train_indices = train_ds.indices
    train_labels = [ds.label_values[i] for i in train_indices]
    train_weights = [ds.sample_weights[i] for i in train_indices]
    
    train_sampler = WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_indices),
        replacement=True
    )


    sampled_indices = list(train_sampler)
    sampled_labels = [ds.label_values[train_ds.indices[i]] for i in sampled_indices]

    plt.figure(figsize=(12, 6))
    plt.hist(sampled_labels, bins=100, edgecolor='black', alpha=0.7)
    plt.xlabel("Label Value (Acceleration)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Labels After Oversampling")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    num_zeros_after = np.sum(np.array(sampled_labels) == 0)
    print(f"Number of zero labels after oversampling: {num_zeros_after}")

    # Number of CPU workers
    cpu_cores = min(8, multiprocessing.cpu_count())
    
    # Use the weighted sampler for training data
    train_dl = DataLoader(
        train_ds,
        batch_size=32,
        sampler=train_sampler,
        num_workers=cpu_cores,
        pin_memory=True,
        collate_fn=custom_collate
    )
    
    # Regular sampler for validation
    val_dl = DataLoader(
        val_ds,
        batch_size=32,
        num_workers=cpu_cores,
        pin_memory=True,
        collate_fn=custom_collate
    )

    # Load model
    model = models.resnet34(pretrained=True)
    model.conv1 = nn.Conv2d(in_c, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Sequential(
        nn.Dropout(0.6),  
        nn.Linear(model.fc.in_features, num_c)
    )
    model.to(device)

    # Training settings with adjusted loss function
    opt = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
    
    # Use Huber loss which is less sensitive to outliers than MSE
    criterion = nn.MSELoss()

    train_losses, val_losses_epoch = [], []
    epochs = 1000

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_train_loss = 0

        for batch in train_dl:
            of_tensor, label = batch
            of_tensor, label = of_tensor.to(device), label.to(device)
            opt.zero_grad()
            pred = torch.squeeze(model(of_tensor))
            loss = criterion(pred, label)
            loss.backward()
            opt.step()
            epoch_train_loss += loss.item()

        train_loss = epoch_train_loss / len(train_dl)
        train_losses.append(train_loss)

        # Validation Loop
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for batch in val_dl:
                of_tensor, label = batch
                of_tensor, label = of_tensor.to(device), label.to(device)
                pred = torch.squeeze(model(of_tensor))
                loss = criterion(pred, label)
                epoch_val_loss += loss.item()

        val_loss = epoch_val_loss / len(val_dl)
        val_losses_epoch.append(val_loss)
        scheduler.step(val_loss)

        print(f"Epoch {epoch}: Train Loss = {train_loss:.8f}, Validation Loss = {val_loss:.8f}")

        if epoch % 100 == 0:
            # Save model
            torch.save(model.state_dict(), f"resnet_oversample_epoch{epoch}.pth")

            # Plot training curves
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
            plt.plot(range(1, len(val_losses_epoch) + 1), val_losses_epoch, label='Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.savefig(f"loss_plot_oversample_epoch{epoch}.png")
            plt.close()

    # Final model save
    torch.save(model.state_dict(), f"resnet_oversample_final.pth")