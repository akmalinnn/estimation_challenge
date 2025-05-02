from torch.utils.data import Dataset
import numpy as np
import json
from pathlib import Path
import torch
from torchvision import transforms 


class OFDataset(Dataset):
    def __init__(self, of_dir, label_json, transform=None):

        self.of_dir = Path(of_dir)
        self.labels = json.load(open(label_json))
        self.files = sorted(self.of_dir.glob("diff_flow_*.npy"))
        self.samples = []

        self.transform = transform

        # print(f"Total .npy files found: {len(self.files)}")
        # print(f"Example .npy file name: {[f.name for f in self.files[:3]]}")
        # print(f"Example label keys: {list(self.labels.keys())[:5]}")


        for i in range(len(self.files) - 2):
            file1, file2, file3 = self.files[i:i+3]
            img1 = file1.stem.replace("diff_flow_", "").replace(".png", "") + ".jpg"
            img2 = file2.stem.replace("diff_flow_", "").replace(".png", "") + ".jpg"
            img3 = file3.stem.replace("diff_flow_", "").replace(".png", "") + ".jpg"
            if img1 in self.labels and img2 in self.labels and img3 in self.labels:
                speed1 = self.labels[img1]["speed"]
                speed2 = self.labels[img2]["speed"]
                speed3 = self.labels[img3]["speed"]
                label = ((speed2 + speed3) / 2) - ((speed1 + speed2) / 2)
                self.samples.append((file1, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file, label = self.samples[idx]
        of_tensor = np.load(file)
        if of_tensor.shape[-1] == 3:
            of_tensor = of_tensor[:, :, :2]
        of_tensor = np.transpose(of_tensor, (2, 0, 1))
        of_tensor = torch.tensor(of_tensor, dtype=torch.float32)

        if self.transform:
            of_tensor = self.transform(of_tensor)
        return of_tensor, torch.tensor(label, dtype=torch.float32)
