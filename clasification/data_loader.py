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
        # print(f"Loaded samples: {len(self.samples)}")

        for fname in self.data_dict:
            info = self.data_dict[fname]

            # Filter by the desired split
            if info["data_type"] in self.split:
                if -15.0 <= info["speed"] <= 15.0:
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


        #BALANCE DATASET

        if self.balance:
            min_count = min(label_counts.values())

            balanced_samples = []
            for label in set(label_counts.keys()):  # otomatis ambil semua label yang ada
                label_samples = [sample for sample in self.samples if sample[1] == label]

                if len(label_samples) >= min_count and min_count > 0:
                    balanced_samples.extend(random.sample(label_samples, min_count))
                else:
                    # Kalau jumlah sampel kurang dari min_count, pakai semua yang ada
                    balanced_samples.extend(label_samples)

            self.samples = balanced_samples
            print(f"\nBalanced dataset with {len(self.samples)} samples.")

        

    def __getitem__(self, index):
        file_path, label = self.samples[index]
        data = np.load(file_path)

        #print(f"Loaded numpy data shape: {data.shape}")

        tensor = torch.tensor(data, dtype=torch.float32).permute(2, 0, 1)
        #print(f"Tensor shape after permute: {tensor.shape}")
        #print(f"Fetching sample: {file_path}, Label: {label}")
        label_tensor = torch.tensor(label, dtype=torch.float32)
        #print(f"Label tensor: {label_tensor}, shape: {label_tensor.shape}")

        return tensor, label_tensor

    def __len__(self):
        return len(self.samples)
