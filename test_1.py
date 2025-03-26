import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models  # Import models
import numpy as np
import os

# Load trained model
MODEL_PATH = "models/efficientnet_b0_final.pth"  # Change to your trained model path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load EfficientNet-B0 model
model = models.efficientnet_b0(weights=None)  # Ensure correct model initialization
model.classifier = nn.Sequential(
    nn.Linear(model.classifier[1].in_features, 128),
    nn.ReLU(),
    nn.Linear(128, 1)  # Output speed
)

# Load model state_dict with strict=False to ignore mismatches
model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
model.to(device)
model.eval()

# Define transformation for optical flow input
def preprocess_npy(npy_path):
    flow = np.load(npy_path)  # Load .npy file (HxWx2 or CxHxW)
    
    if flow.shape[0] == 2:  # Ensure correct shape (CxHxW)
        flow = np.transpose(flow, (1, 2, 0))  # Convert to HxWxC if needed
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),  # Resize for EfficientNet
        transforms.Normalize(mean=[0.0, 0.0], std=[1.0, 1.0])  # Normalize flow values
    ])
    
    flow = transform(flow).unsqueeze(0).to(device)  # Convert to tensor and add batch dim
    return flow


def predict_speed(npy_path):
    """Predicts speed from an optical flow .npy file."""
    flow_tensor = preprocess_npy(npy_path)
    
    with torch.no_grad():
        speed = model(flow_tensor).item()  # Get prediction
    return speed

# Test a single optical flow file
npy_path = "00_0006_frame_00065.npy"  # Change this to your test .npy file
if os.path.exists(npy_path):
    predicted_speed = predict_speed(npy_path)
    print(f"Predicted Speed: {predicted_speed:.2f} km/h")
else:
    print(f"❌ Flow file not found: {npy_path}")