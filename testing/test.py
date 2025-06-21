import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from data_loader import OFDatasetClf, OFDatasetRegression
from resnet import ResNet
from efficientnet import EfficientNet
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import numpy as np

def run_inference():
    # Configuration
    of_dir = '../data/flow_diff'
    label_json = '../data/flow_diff.json'
    model_file = '../result/training_clf/batch size 8/resnet 3/model_CLF_epoch_0030.pt'
    batch_size = 16
    OF_MEAN = [0.5, 0.5]
    OF_STD = [0.5, 0.5]

    
    transform = transforms.Compose([
        transforms.Normalize(mean=OF_MEAN, std=OF_STD)
    ])

    
    test_dataset = OFDatasetClf(json_file=label_json, root_dir=of_dir, split="testing")
    test_dataset.transform = transform
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    # Load model
    model = ResNet(n=3, shortcuts=True)
    # model = EfficientNet.from_name('efficientnet-b0', in_channels=2, num_classes=1)
    model.load_state_dict(torch.load(model_file, map_location='cpu'))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Running Inference"):
            inputs = inputs.to(device)
            labels = labels.to(device).float()

            outputs = model(inputs).squeeze()
            preds = ((outputs) > 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Evaluation
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)

    print(f"\nTest Results:")
    print(f"Accuracy:  {acc:.4f}")

    # Show all predictions vs labels
    print("\nPredictions vs Ground Truth:")
    for i in range(len(all_preds)):
        pred = int(all_preds[i])
        true = int(all_labels[i])
        print(f"Sample {i+1}: Predicted = {pred}, Ground Truth = {true}")

from data_loader import OFDatasetRegression 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def run_inference_regression():
    
    of_dir = 'data/flow_diff'
    label_json = 'data/flow_diff.json'
    model_file = 'pretrained\model_REG_epoch_0050.pt'
    batch_size = 16
    OF_MEAN = [0.5, 0.5]
    OF_STD = [0.5, 0.5]

    
    transform = transforms.Compose([
        transforms.Normalize(mean=OF_MEAN, std=OF_STD)
    ])

    # Load regression dataset (non-zero only)
    test_dataset = OFDatasetRegression(json_file=label_json, root_dir=of_dir, split="testing")
    test_dataset.transform = transform
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # Load model
    model = ResNet(n=3, shortcuts=True)  
    model.load_state_dict(torch.load(model_file, map_location='cpu'))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Running Regression Inference"):
            inputs = inputs.to(device)
            labels = labels.to(device).float()

            outputs = model(inputs).squeeze()
            all_preds.append(outputs.cpu())
            all_labels.append(labels.cpu())

    # Concatenate tensors
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Calculate metrics using PyTorch
    mae = torch.mean(torch.abs(all_preds - all_labels)).item()
    rmse = torch.sqrt(torch.mean((all_preds - all_labels) ** 2)).item()
    mse = torch.mean((all_preds - all_labels) ** 2).item()

    # Show results
    print(f"\nRegression Test Results:")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MSE: {mse:.4f}")

    print("\nPredictions vs Ground Truth:")
    for i in range(len(all_preds)):
        print(f"Sample {i+1}: Predicted = {all_preds[i].item():.2f}, Ground Truth = {all_labels[i].item():.2f}")

if __name__ == "__main__":
    run_inference()
    # run_inference_regression()
