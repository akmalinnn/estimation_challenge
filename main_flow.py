import torch
from torch import optim
from torch.utils.data import random_split, DataLoader
from resnet import ResNet
from train import train
from data_loader import OFDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from utils import calculate_normalisation_params
import os

if __name__ == '__main__':
    of_dir = 'data/flow_diff'
    label_json = 'data/data_filtered.json'
    batch_size = 16
    epochs = 1000
    lr = 1e-5
    model_file = 'pretrained/resnetfinal.pt'
    results_file = 'results/of_results.csv'

    dataset = OFDataset(of_dir, label_json, transform=None)

    # OF_MEAN, OF_STD = calculate_normalisation_params(dataset)
    # print(f'Computed mean: {OF_MEAN}')
    # print(f'Computed std: {OF_STD}')
    OF_MEAN = [0.0019, 0.0005]
    OF_STD = [5.2, 1.8]


    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize(mean=OF_MEAN, std=OF_STD)
    ])

    transform_test = transforms.Compose([
        transforms.Normalize(mean=OF_MEAN, std=OF_STD)
    ])


    train_size = int(0.7 * len(dataset))  
    val_size = int(0.2 * len(dataset))    
    test_size = len(dataset) - train_size - val_size  
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

    train_ds.dataset.transform = transform_train
    val_ds.dataset.transform = transform_test
    test_ds.dataset.transform = transform_test


    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # means, stds = calculate_normalisation_params(train_loader, val_loader)
    # print(f'means: {means}')
    # print(f'stds: {stds}')


    # Initialize the model
    model = ResNet(n=3, shortcuts=True)
    
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.25)

    # Start training
    # train(model, epochs, train_loader, val_loader, criterion, optimizer,
    #       RESULTS_PATH=results_file, scheduler=scheduler, MODEL_PATH=model_file)

    train(model=model, epochs=epochs, train_loader=train_loader, val_loader=val_loader, 
    optimizer=optimizer, RESULTS_PATH=results_file, MODEL_PATH=model_file, scheduler=scheduler
    )


    # 70% train, 20% val, 10% test
    # process training:
    # train dan val

    # process inference:
    # test.