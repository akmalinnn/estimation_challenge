import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from data_loader import OFDataset, OFDatasetEstimation
from resnet import ResNet
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import numpy as np

def run_inference():
    # Configuration
    of_dir = 'data/flow_diff'
    label_json = 'data/flow_diff.json'
    model_file = 'pretrained/model_epoch_0050.pt'
    batch_size = 16
    OF_MEAN = [0.5, 0.5]
    OF_STD = [0.5, 0.5]

    
    transform = transforms.Compose([
        transforms.Normalize(mean=OF_MEAN, std=OF_STD)
    ])

    
    test_dataset = OFDataset(json_file=label_json, root_dir=of_dir, split="testing")
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
        for inputs, labels in tqdm(test_loader, desc="Running Inference"):
            inputs = inputs.to(device)
            labels = labels.to(device).float()

            outputs = model(inputs).squeeze()
            preds = (torch.sigmoid(outputs) > 0.5).float()

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
    model_file = 'pretrained/model_epoch_0050.pt'
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
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert to NumPy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Metrics
    mae = mean_absolute_error(all_labels, all_preds)
    rmse = mean_squared_error(all_labels, all_preds, squared=False)
    r2 = r2_score(all_labels, all_preds)

    print(f"\nRegression Test Results:")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")

    # Show all predictions vs ground truth
    print("\nPredictions vs Ground Truth:")
    for i in range(len(all_preds)):
        print(f"Sample {i+1}: Predicted = {all_preds[i]:.2f}, Ground Truth = {all_labels[i]:.2f}")



if __name__ == "__main__":
    run_inference()
    run_inference_regression()
