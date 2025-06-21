import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import os

# Load trained model
MODEL_PATH = "efficientnet_b0_epoch1000.pth"  # Update your model path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load EfficientNet-B0 model
model = models.efficientnet_b0(weights=None)  # Load model without pretrained weights

# Modify first convolutional layer to accept 2 channels instead of 3
in_features = model.features[0][0].in_channels  # Get original input channels (3)
out_features = model.features[0][0].out_channels  # Get output channels (32)
kernel_size = model.features[0][0].kernel_size
stride = model.features[0][0].stride
padding = model.features[0][0].padding

# Replace first conv layer with a new one that takes 2 input channels
model.features[0][0] = nn.Conv2d(2, out_features, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

# Modify classifier for single-output prediction
model.classifier = nn.Sequential(
    nn.Linear(model.classifier[1].in_features, 128),
    nn.ReLU(),
    nn.Linear(128, 1)  # Output: speed prediction
)

# Load trained model weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
model.to(device)
model.eval()

# Transformation function for optical flow input
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

# Prediction function
def predict_speed(npy_path):
    """Predicts speed from an optical flow .npy file."""
    flow_tensor = preprocess_npy(npy_path)
    
    with torch.no_grad():
        speed = model(flow_tensor).item()  # Get prediction
    return speed

# Test a single optical flow file
npy_path = "diff_flow_05_0012_frame_00155.png.npy"  # Change this to your test .npy file
if os.path.exists(npy_path):
    predicted_speed = predict_speed(npy_path)
    print(f"Predicted Speed: {predicted_speed:.9f} km/h")
else:
    print(f"❌ Flow file not found: {npy_path}")
