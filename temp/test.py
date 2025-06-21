import torch
import timm
import numpy as np
from pathlib import Path

# **Konfigurasi Model**
device = 'cuda' if torch.cuda.is_available() else 'cpu'
v = 0  # EfficientNet versi
in_c = 2  # Optical flow memiliki 2 channel
num_c = 1  # Model hanya memprediksi 1 nilai (kecepatan)

# **Muat Model dengan Arsitektur yang Sama**
model = timm.create_model(f'efficientnet_b{v}', pretrained=False, num_classes=num_c)
model.conv_stem = torch.nn.Conv2d(in_c, model.conv_stem.out_channels, 
                                  kernel_size=model.conv_stem.kernel_size, 
                                  stride=model.conv_stem.stride, 
                                  padding=model.conv_stem.padding, 
                                  bias=False)

# **Pastikan layer classifier sama seperti saat training**
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(0.3),  
    torch.nn.Linear(model.classifier.in_features, num_c)  # Samakan dengan model training
)

# **Load Model**
model.load_state_dict(torch.load("models/efficientnet_b0_final.pth", map_location=device), strict=False)
model.to(device)
model.eval()

# **Fungsi Prediksi Tanpa Normalisasi**
def predict_speed(npy_file):
    if not Path(npy_file).exists():
        raise FileNotFoundError(f"File {npy_file} tidak ditemukan.")

    # **Load optical flow file**
    of_array = np.load(npy_file).astype(np.float32)
    of_tensor = torch.tensor(of_array, dtype=torch.float32)

    # **Pastikan tensor berbentuk (C, H, W)**
    if of_tensor.dim() == 3:
        of_tensor = of_tensor.permute(2, 0, 1)  # (H, W, C) → (C, H, W)

    # **Tambahkan batch dimensi**
    of_tensor = of_tensor.unsqueeze(0).to(device)  # (1, C, H, W)

    # **Lakukan prediksi**
    with torch.no_grad():
        pred_speed = torch.squeeze(model(of_tensor)).item()

    return pred_speed

# **Contoh Penggunaan**
npy_test_file = "data/flow/00_0003_frame_00001.npy"
predicted_speed = predict_speed(npy_test_file)
print(f"Predicted Speed: {predicted_speed:.2f} km/h")
