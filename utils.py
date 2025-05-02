import torch
from torch.utils.data import DataLoader
import numpy as np

def calculate_normalisation_params(dataset):
    """
    Calculates mean and std per channel directly from the dataset.
    Assumes each sample returns (tensor[C,H,W], label).
    """
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)
    flow_batch, _ = next(iter(loader))  #  [N, C, H, W]

    mean = flow_batch.mean(dim=[0, 2, 3])  
    std = flow_batch.std(dim=[0, 2, 3])    

    return mean.tolist(), std.tolist()