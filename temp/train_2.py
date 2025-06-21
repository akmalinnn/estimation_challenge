#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from efficientnet_pytorch import EfficientNet
from pathlib import Path
import numpy as np
import json
import multiprocessing
import matplotlib.pyplot as plt
import random


# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model parameters
v = 0     # Model version
in_c = 2  # Number of input channels
num_c = 1 # Number of classes to predict

# Directory containing optical flow files
of_dir = "data/flow_diff"
# JSON file containing speed labels
label_json = "data/data_filtered.json"

class OFDataset(Dataset):
    def __init__(self, of_dir, label_json):
        self.of_dir = Path(of_dir)
        self.labels = json.load(open(label_json))
        self.files = sorted(self.of_dir.glob("diff_flow_*.npy"))  # Ensure correct file order
        self.valid_indices = []

        zero_count = 0

        for i in range(len(self.files) - 2):
            try:
                file1, file2, file3 = self.files[i:i+3]

                img1 = file1.stem.replace("diff_flow_", "").replace(".png", "") + ".jpg"
                img2 = file2.stem.replace("diff_flow_", "").replace(".png", "") + ".jpg"
                img3 = file3.stem.replace("diff_flow_", "").replace(".png", "") + ".jpg"

                # Extract video ID (assuming format: "video_frame_xxxx.jpg")
                vid1, vid2, vid3 = img1.split("_frame_")[0], img2.split("_frame_")[0], img3.split("_frame_")[0]

                # Ensure frames are from the same video
                if vid1 != vid2 or vid2 != vid3:
                    continue  # Skip this index

                # Ensure labels exist
                if img1 in self.labels and img2 in self.labels and img3 in self.labels:
                    # Compute acceleration label
                    speed1 = self.labels[img1]["speed"]
                    speed2 = self.labels[img2]["speed"]
                    speed3 = self.labels[img3]["speed"]
                    label = ((speed2 + speed3) / 2) - ((speed1 + speed2) / 2)

                    # Print files with label >10 or <-10
                    # if label > 10 or label < -10:
                    #     print(f"{img1}, {img2}, {img3} -> Label: {label}")

                    # Count how many labels are zero
                    if label == 0:
                        zero_count += 1
                        if random.random() > 0.02 : # Remove most zeros
                            continue
                        zero_count += 1

                    self.valid_indices.append(i)

            except Exception as e:
                print(f"Skipping due to error: {e}")

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
        label = ((speed2 + speed3) / 2) - ((speed1 + speed2) / 2)

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

    # Load dataset inside __main__
    ds = OFDataset(of_dir, label_json)

    labels = []

    for i in range(len(ds)):
        sample = ds[i]
        if sample is not None:
            _, label = sample
            labels.append(label.item())

    labels = np.array(labels)  # Konversi ke array untuk efisiensi

    # Hitung jumlah label dengan nilai 0
    num_zeros = np.sum(labels == 0)

    # Plot histogram dengan jumlah lebih banyak bins untuk detail
    plt.figure(figsize=(12, 6))
    plt.hist(labels, bins=100, edgecolor='black', alpha=0.7)

    # Tambahkan teks jumlah 0 ke dalam plot
    plt.text(0, num_zeros + 5, f"0 Count: {num_zeros}", fontsize=12, color='red', ha='center')

    plt.xlabel("Label Value (Acceleration)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Labels")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    print(f"Jumlah label dengan nilai 0: {num_zeros}")


    # Train-test split
    ds_size = len(ds)
    split = int(np.floor(0.8 * ds_size))
    indices = list(range(ds_size))
    train_idx, val_idx = indices[:split], indices[split:]

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    # Number of CPU workers
    cpu_cores = min(4, multiprocessing.cpu_count())

    train_dl = DataLoader(ds, batch_size=8, sampler=train_sampler, num_workers=cpu_cores, 
                          pin_memory=True, collate_fn=custom_collate)
    val_dl = DataLoader(ds, batch_size=8, sampler=val_sampler, num_workers=cpu_cores, 
                        pin_memory=True, collate_fn=custom_collate)

    # Load EfficientNet model
    model = EfficientNet.from_pretrained(f'efficientnet-b{v}')
    model._conv_stem.in_channels = in_c  # Override in_channels
    model._conv_stem.weight = nn.Parameter(torch.randn(model._conv_stem.weight.shape[0], in_c, 
                                                        model._conv_stem.weight.shape[2], model._conv_stem.weight.shape[3]))
    model._fc = nn.Linear(model._fc.in_features, num_c)  # Adjust output layer
    model.to(device)

    # Training settings
    opt = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=10)
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
            torch.save(model.state_dict(), f"efficientnet_b{v}_epoch{epoch}.pth")

      
            plt.figure(figsize=(10, 5))

            # Plot training and validation loss with markers for better visibility
            plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', marker='o', linestyle='-', markersize=4)
            plt.plot(range(1, len(val_losses_epoch) + 1), val_losses_epoch, label='Validation Loss', marker='s', linestyle='-', markersize=4)

            # Labels and title
            plt.xlabel('Epochs', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.title('Training and Validation Loss', fontsize=14)

            # Adjust y-axis limits dynamically with a small margin
            min_loss = min(min(train_losses), min(val_losses_epoch))
            max_loss = max(max(train_losses), max(val_losses_epoch))
            plt.ylim(min_loss - 0.0001, max_loss + 0.0001)

            # Generate custom y-ticks with more decimal places
            yticks = np.linspace(min_loss, max_loss, num=6)  # Adjust the number of ticks as needed
            plt.yticks(yticks, [f'{tick:.6f}' for tick in yticks])  # Format with 6 decimal places

            # Improve grid for better visibility
            plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

            # Show legend
            plt.legend(fontsize=10)

            # Save the figure with high DPI for clarity
            plt.savefig(f"loss_plot_epoch{epoch}.png", dpi=300, bbox_inches='tight')

            plt.close()

    # Final model save
    torch.save(model.state_dict(), f"efficientnet_b{v}_final.pth")
