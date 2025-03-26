import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import timm
from pathlib import Path
import numpy as np
import json
import multiprocessing
import matplotlib.pyplot as plt

# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model Configuration
resnet_version = "resnet18"  # Choose ResNet model
in_c = 2  # Optical flow has 2 channels
num_c = 1 # Predicting a single value (speed)

# Load ResNet model using timm
model = timm.create_model(resnet_version, pretrained=True, num_classes=num_c)
model.conv1 = nn.Conv2d(in_c, model.conv1.out_channels, 
                         kernel_size=model.conv1.kernel_size, 
                         stride=model.conv1.stride, 
                         padding=model.conv1.padding, 
                         bias=False)

# Add dropout for regularization
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    model.fc
)
model.to(device)

# Dataset Class
class OFDataset(Dataset):
    def __init__(self, of_dir, label_f, normalize=True):
        self.of_dir = Path(of_dir)
        with open(label_f, "r") as f:
            self.labels = json.load(f)

        self.files = sorted(self.labels.keys())
        self.normalize = normalize

        speeds = [self.labels[f]["speed"] for f in self.files]
        self.min_speed = min(speeds)
        self.max_speed = max(speeds)
        
        print(f"Speed Normalization: min={self.min_speed}, max={self.max_speed}")
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        npy_path = self.of_dir / f"{file_name.replace('.jpg', '.npy')}"

        if not npy_path.exists():
            raise FileNotFoundError(f"Missing optical flow file: {npy_path}")

        of_array = np.load(npy_path).astype(np.float32)
        of_tensor = torch.tensor(of_array, dtype=torch.float32)
        if of_tensor.dim() == 3:
            of_tensor = of_tensor.permute(2, 0, 1)

        speed = self.labels[file_name]["speed"]
        
        if self.normalize:
            speed = (speed - self.min_speed) / (self.max_speed - self.min_speed)
        speed_label = torch.tensor([speed], dtype=torch.float32)
        return of_tensor, speed_label

# Dataset Directory and JSON Label File
of_dir = "data/flow"
labels_f = "data/data_filtered.json"
ds = OFDataset(of_dir, labels_f)

# Split Dataset
ds_size = len(ds)
indices = list(range(ds_size))
train_split = 0.8
split = int(np.floor(train_split * ds_size))
train_idx, val_idx = indices[:split], indices[split:]

train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)

cpu_cores = min(4, multiprocessing.cpu_count())
train_dl = DataLoader(ds, batch_size=16, sampler=train_sampler, num_workers=cpu_cores, pin_memory=(device == 'cuda'))
val_dl = DataLoader(ds, batch_size=16, sampler=val_sampler, num_workers=cpu_cores, pin_memory=(device == 'cuda'))

# Training Configuration
epochs = 1000
criterion = nn.MSELoss()
opt = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)

train_losses = []
val_losses_epoch = []

if __name__ == '__main__':
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_train_loss = 0
        for of_tensor, label in train_dl:
            of_tensor, label = of_tensor.to(device), label.to(device)
            opt.zero_grad()
            pred = torch.squeeze(model(of_tensor))
            loss = criterion(pred, label)
            loss.backward()
            opt.step()
            epoch_train_loss += loss.item()
        
        train_loss = epoch_train_loss / len(train_dl)
        train_losses.append(train_loss)

        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for of_tensor, label in val_dl:
                of_tensor, label = of_tensor.to(device), label.to(device)
                pred = torch.squeeze(model(of_tensor))
                loss = criterion(pred, label)
                epoch_val_loss += loss.item()
        
        val_loss = epoch_val_loss / len(val_dl)
        val_losses_epoch.append(val_loss)

        scheduler.step(val_loss)

        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Validation Loss = {val_loss:.4f}")

        if epoch % 100 == 0:
            model_path = f"{resnet_version}_epoch{epoch}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"Model disimpan: {model_path}")
            
            plt.figure(figsize=(8, 5))
            plt.plot(range(1, epoch+1), train_losses, label="Train Loss", marker="o")
            plt.plot(range(1, epoch+1), val_losses_epoch, label="Validation Loss", marker="s")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training and Validation Loss")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"loss_plot_epoch{epoch}.png", dpi=300)
            print(f"Loss plot disimpan sebagai loss_plot_epoch{epoch}.png")

    final_model_path = f"{resnet_version}_final.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"Model final disimpan: {final_model_path}")
