import torch

import flow_viz

import matplotlib.pyplot as plt
import cv2
import numpy as np

from torch import optim
from torch.utils.data import DataLoader
from cnn.resnet import ResNet
from cnn.inceptionv4 import Inceptionv4
from cnn.mobilenetv2 import mobilenetv2
from cnn.mobilenet import MobileNetV1
from cnn.mobilenetv3 import mobilenetv3_large, mobilenetv3_small
from cnn.efficientnet import EfficientNet

from train import train_classifier
from data_loader import OFDatasetClf
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms

import os



if __name__ == '__main__':
    of_dir = '../data/flow_diff'
    label_json = '../data/flow_diff.json'
    batch_size = 8
    epochs = 50
    lr = 1e-5
    model_file = '../result/training_clf/model_plot/'


    

    # for clasification
    train_dataset = OFDatasetClf(json_file=label_json, root_dir=of_dir, split="training")
    val_dataset = OFDatasetClf(json_file=label_json, root_dir=of_dir, split="validation")
    test_dataset = OFDatasetClf(json_file=label_json, root_dir=of_dir, split="testing")
   


    OF_MEAN = [0.5, 0.5]
    OF_STD = [0.5, 0.5]
 


    transform_train = transforms.Compose([            
        transforms.RandomRotation(degrees=5), 
        transforms.RandomHorizontalFlip(p=0.5),              
        transforms.RandomResizedCrop((360, 640), scale=(0.7, 1.0), ratio=(16/9, 16/9),),  
        transforms.Normalize(mean=OF_MEAN, std=OF_STD), 
    ])

    
    transform = transforms.Compose([            
        transforms.Normalize(mean=OF_MEAN, std=OF_STD), 
    ])

        
    train_dataset.transform = transform_train
    val_dataset.transform = transform
    test_dataset.transform = transform

    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # model networks
    model = ResNet(n=3, shortcuts=True)
    # model = Inceptionv4(in_channels=2, classes=1)
    # model = mobilenetv2(num_classes=1)
    # model = mobilenetv3_large(num_classes=1)
    # model = MobileNetV1(num_classes=1)
    # model = EfficientNet.from_name('efficientnet-b0', in_channels=2, num_classes=1)

    # Optimizer 
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.25)
    # scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    

    # Start training
    train_classifier(
        model=model, 
        epochs=epochs, 
        train_loader=train_loader, 
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer, 
        MODEL_PATH=model_file, 
        scheduler=scheduler
    )


   