import os
import json
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from PIL import Image

# ================================
# 1. Dataset Class (Same as train.py)
# ================================
class OpticalFlowDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, json_path, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        with open(json_path, "r") as f:
            self.labels = json.load(f)

        self.image_files = list(self.labels.keys())

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        label = self.labels[img_name]["speed"]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)


# ================================
# 2. Load Model
# ================================
def load_model(model_path, device):
    model = models.efficientnet_b0(pretrained=False)  
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 128),
        nn.ReLU(),
        nn.Linear(128, 1)  # Output one value (speed)
    )
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model


# ================================
# 3. Test Function
# ================================
def test(model, test_loader, device):
    criterion = nn.MSELoss()
    total_loss = 0

    with torch.no_grad():
        for images, speeds in test_loader:
            images, speeds = images.to(device), speeds.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, speeds)
            total_loss += loss.item()

    print(f"Test Loss (MSE): {total_loss / len(test_loader):.4f}")


# ================================
# 4. Run Testing
# ================================
if __name__ == "__main__":
    IMAGE_DIR = "data/images"
    JSON_PATH = "data/speed_labels.json"
    MODEL_PATH = "models/efficientnet_speed.pth"
    BATCH_SIZE = 16
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    test_dataset = OpticalFlowDataset(IMAGE_DIR, JSON_PATH, transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = load_model(MODEL_PATH, DEVICE)
    test(model, test_loader, DEVICE)
