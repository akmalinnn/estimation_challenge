import torch
from torch import optim
from torch.utils.data import DataLoader
from cnn.resnet import ResNet
from train_reg import train_reg
from data_loader import OFDatasetRegression
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
import os

if __name__ == '__main__':
    of_dir = '../data/flow_diff'
    label_json = '../data/flow_diff.json'
    batch_size = 8
    epochs = 100
    lr = 1e-5
    model_file = '../result/training_reg/model_plot/'
   

    train_dataset = OFDatasetRegression(json_file=label_json, root_dir=of_dir, split=["training","validation"])
    val_dataset = OFDatasetRegression(json_file=label_json, root_dir=of_dir, split="validation")
    test_dataset = OFDatasetRegression(json_file=label_json, root_dir=of_dir, split="testing")

    OF_MEAN = [0.5, 0.5]
    OF_STD = [0.5, 0.5]

    transform_train = transforms.Compose([
        transforms.RandomRotation(degrees=5),               
        transforms.RandomResizedCrop((360, 640), scale=(0.8, 1.0)),
        transforms.Normalize(mean=OF_MEAN, std=OF_STD)
    ])

    transform = transforms.Compose([
        transforms.Normalize(mean=OF_MEAN, std=OF_STD)
    ])

    
    train_dataset.transform = transform_train
    val_dataset.transform = transform
    test_dataset.transform = transform

    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # model networks
    model = ResNet(n=3, shortcuts=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.25)


    train_reg(
        model=model, 
        epochs=epochs, 
        train_loader=train_loader, 
        val_loader=val_loader,
        test_loader=test_loader, 
        optimizer=optimizer, 
        MODEL_PATH=model_file, 
        scheduler=scheduler
    )

