import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

# ================================
# 1. Dataset Class
# ================================
class OpticalFlowDataset(Dataset):
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
# 2. Data Preparation
# ================================
def get_data_loaders(image_dir, json_path, batch_size=16, train_split=0.8):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    dataset = OpticalFlowDataset(image_dir, json_path, transform)
    
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# ================================
# 3. Load EfficientNet
# ================================
def get_model():
    model = models.efficientnet_b0(pretrained=True)  # Load pretrained EfficientNet-B0
    num_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Linear(num_features, 128),
        nn.ReLU(),
        nn.Linear(128, 1)  # Output one value (speed)
    )

    return model


# ================================
# 4. Train Function
# ================================
def train(model, train_loader, test_loader, device, epochs=10, lr=0.001):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, speeds in train_loader:
            images, speeds = images.to(device), speeds.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, speeds)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}")

    print("Training complete!")


# ================================
# 5. Save Model
# ================================
def save_model(model, filename="b0.pth"):
    torch.save(model.state_dict(), filename)
    print(f"Model saved as {filename}")


# ================================
# 6. Main Execution
# ================================
if __name__ == "__main__":
    IMAGE_DIR = "data/images"  # Change this
    JSON_PATH = "data/data_filtered.json"  # Change this
    BATCH_SIZE = 16
    EPOCHS = 10
    LEARNING_RATE = 0.001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_data_loaders(IMAGE_DIR, JSON_PATH, BATCH_SIZE)
    model = get_model()
    train(model, train_loader, test_loader, DEVICE, EPOCHS, LEARNING_RATE)
    save_model(model)
