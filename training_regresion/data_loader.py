import torch
import torch.utils.data as data
import numpy as np
import os
import json
import random
from collections import Counter

class OFDatasetRegression(data.Dataset):

    def __init__(self, json_file, root_dir, split):
        self.root_dir = root_dir

        if isinstance(split, str):
            self.split = [split]
        else:
            self.split = split

        with open(json_file, 'r') as f:
            self.data_dict = json.load(f)

        positive_samples = []
        negative_samples = []

        for fname in self.data_dict:
            info = self.data_dict[fname]

            if info["data_type"] in self.split and info["speed"] != 0.0:
                file_path = os.path.join(self.root_dir, fname)

                if os.path.exists(file_path):
                    sample = (file_path, info["speed"])
                    if info["speed"] > 0:
                        positive_samples.append(sample)
                    elif info["speed"] < 0:
                        negative_samples.append(sample)

        # Balance counts
        min_count = min(len(positive_samples), len(negative_samples))
        positive_samples = random.sample(positive_samples, min_count)
        negative_samples = random.sample(negative_samples, min_count)


        # positive_samples = positive_samples
        # negative_samples = negative_samples
        

        # Combine and shuffle
        self.samples = positive_samples + negative_samples
        random.shuffle(self.samples)

        print(f"Balanced regression samples (non-zero only): {len(self.samples)}")
        print(f"Positive labels: {len(positive_samples)}")
        print(f"Negative labels: {len(negative_samples)}")

        print("\nSample preview (file path and speed):")
        for i in range(min(10, len(self.samples))):  
            file_path, speed = self.samples[i]
            print(f"File: {file_path}, Speed: {speed}")

    def __getitem__(self, index):
        file_path, speed = self.samples[index]
        data = np.load(file_path)
        tensor = torch.tensor(data, dtype=torch.float32).permute(2, 0, 1)
        return tensor, torch.tensor(speed, dtype=torch.float32)

    def __len__(self):
        return len(self.samples)