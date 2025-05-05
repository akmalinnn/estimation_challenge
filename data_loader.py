import torch
import torch.utils.data as data
import numpy as np
import os
import json
import random
from collections import Counter

class OFDatasetClf(data.Dataset):
    def __init__(self, json_file, root_dir, split, balance=True):
        self.root_dir = root_dir
        self.balance = balance

        # Handle splitting
        if isinstance(split, str):
            self.split = [split]
        else:
            self.split = split

        # Load the dataset from the JSON file
        with open(json_file, 'r') as f:
            self.data_dict = json.load(f)

        self.samples = []
    
        label_counts = Counter()

        for fname in self.data_dict:
            info = self.data_dict[fname]

            # Filter by the desired split
            if info["data_type"] in self.split:
                file_path = os.path.join(self.root_dir, fname)

                if os.path.exists(file_path):
                    label = 0 if info["speed"] == 0.0 else 1

                    self.samples.append((file_path, label))
                    label_counts[label] += 1

        print(f"Loaded samples: {len(self.samples)}")
        print(f"Label counts: {dict(label_counts)}")

        # print("\n (file and label):")
        # for i in range(min(10, len(self.samples))): 
        #     file_path, label = self.samples[i]
        #     print(f"File: {file_path}, Label: {label}")

        if self.balance:
            # minimum number of class
            min_count = min(label_counts.values())

            balanced_samples = []
            for label in [0, 1]:
                label_samples = [sample for sample in self.samples if sample[1] == label]
                balanced_samples.extend(random.sample(label_samples, min_count))  # Random sampling

            self.samples = balanced_samples
            print(f"\nBalanced dataset with {len(self.samples)} samples.")
        

    def __getitem__(self, index):
        file_path, label = self.samples[index]
        data = np.load(file_path)

        tensor = torch.tensor(data, dtype=torch.float32).permute(2, 0, 1)
        # print(f"Fetching sample: {file_path}, Label: {label}")
        return tensor, label

    def __len__(self):
        return len(self.samples)


class OFDatasetEstimation(data.Dataset):

    def __init__(self, json_file, root_dir, split):
        self.root_dir = root_dir

        if isinstance(split, str):
            self.split = [split]
        else:
            self.split = split

        with open(json_file, 'r') as f:
            self.data_dict = json.load(f)

        self.samples = []

        for fname in self.data_dict:
            info = self.data_dict[fname]

           
            if info["data_type"] in self.split and info["speed"] != 0.0:
                file_path = os.path.join(self.root_dir, fname)

                if os.path.exists(file_path):
                    self.samples.append((file_path, info["speed"]))

        print(f"Loaded regression samples (non-zero only): {len(self.samples)}")
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