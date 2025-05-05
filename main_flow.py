import torch
from torch import optim
from torch.utils.data import DataLoader
from resnet import ResNet
from train import train, train_classifier
from data_loader import OFDatasetEstimation, OFDatasetClf
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from utils import calculate_normalisation_params
import os

if __name__ == '__main__':
    of_dir = 'data/flow_diff'
    label_json = 'data/flow_diff.json'
    batch_size = 16
    epochs = 50
    lr = 1e-5
    model_file = 'pretrained/resnetfinal.pt'
    results_file = 'results/of_results.csv'

    # for clasification
    train_dataset = OFDatasetClf(json_file=label_json, root_dir=of_dir, split="training")
    val_dataset = OFDatasetClf(json_file=label_json, root_dir=of_dir, split="validation")
    # test_dataset = OFDataset(json_file=label_json, root_dir=of_dir, split="test")

    # train_dataset = OFDatasetEstimation(json_file=label_json, root_dir=of_dir, split="training")
    # val_dataset = OFDatasetEstimation(json_file=label_json, root_dir=of_dir, split="validation")

    # Calculate or use predefined normalization parameters
    # OF_MEAN, OF_STD = calculate_normalisation_params(train_dataset)
    OF_MEAN = [0.5, 0.5]
    OF_STD = [0.5, 0.5]

    transform = transforms.Compose([
        transforms.Normalize(mean=OF_MEAN, std=OF_STD)
    ])

    
    train_dataset.transform = transform
    val_dataset.transform = transform
    # test_dataset.transform = transform

    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # model networks
    model = ResNet(n=3, shortcuts=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.25)

    # Start training
    train_classifier(
        model=model, 
        epochs=epochs, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        optimizer=optimizer, 
        RESULTS_PATH=results_file, 
        MODEL_PATH=model_file, 
        scheduler=scheduler
    )


        # train(model, epochs, train_loader, val_loader, optimizer,
        #   RESULTS_PATH, MODEL_PATH=None, scheduler=None):

    # train(
    #     model=model, 
    #     epochs=epochs, 
    #     train_loader=train_loader, 
    #     val_loader=val_loader, 
    #     optimizer=optimizer, 
    #     RESULTS_PATH=results_file, 
    #     MODEL_PATH=model_file, 
    #     scheduler=scheduler
    # )

    # 70% train, 20% val, 10% test
    # process training:
    # train dan val

    # process inference:
    # test.