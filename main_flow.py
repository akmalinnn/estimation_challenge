import torch
from torch import optim
from torch.utils.data import random_split, DataLoader
from resnet import ResNet
from train import train
from data_loader import OFDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

if __name__ == '__main__':
    of_dir = 'data/flow_diff'
    label_json = 'data/data_filtered.json'
    batch_size = 16
    epochs = 1000
    lr = 1e-4
    model_file = 'pretrained/resnetfinal.pt'
    results_file = 'results/of_results.csv'

    dataset = OFDataset(of_dir, label_json)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    model = ResNet(n=5, shortcuts=True)
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Start training
    train(model, epochs, train_loader, test_loader, criterion, optimizer,
          RESULTS_PATH=results_file, scheduler=scheduler, MODEL_PATH=model_file)
