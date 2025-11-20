#!/usr/bin/env python3
import os
import numpy as np
import torch
import torch.nn as nn  # (MỚI) Cần cho định nghĩa class
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# ======== 1. Tải lại kiến trúc model (MỚI - KIẾN TRÚC DCGAN) ========
# PyTorch cần định nghĩa lớp để tải trọng số
z_dim = 100
img_size = 49
img_channels = 1

# (MỚI) Đây là kiến trúc Generator của Mal-DCGAN
class Generator(nn.Module):
    """
    Kiến trúc Generator (DCGAN) - Sao chép từ file huấn luyện
    """
    def __init__(self, z_dim, img_channels, img_size): # img_size = 49
        super(Generator, self).__init__()
        ngf = 32 # Kích thước feature map của G
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, ngf, 5, 2, 1, bias=False), # (B, 64, 24, 24)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False), # (B, 128, 12, 12)
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False), # (B, 256, 6, 6)
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        bottleneck_dim = (ngf * 4) * 6 * 6 # 256 * 36 = 9216
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(bottleneck_dim + z_dim, ngf * 4, 6, 1, 0, bias=False), # (B, 256, 6, 6)
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False), # (B, 128, 12, 12)
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False), # (B, 64, 24, 24)
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, img_channels, 5, 2, 1, bias=False), # (B, 1, 49, 49)
            nn.Tanh() # Output trong [-1, 1]
        )
    def forward(self, real_x, z):
        x_encoded = self.encoder(real_x) # (B, 256, 6, 6)
        x_flat = x_encoded.view(x_encoded.size(0), -1) # (B, 9216)
        combined = torch.cat([x_flat, z], dim=1) # (B, 9216 + 100)
        combined_rs = combined.view(combined.size(0), -1, 1, 1) # (B, 9316, 1, 1)
        perturb_mask = self.decoder(combined_rs) # (B, 1, 49, 49)
        perturb_mask_scaled = perturb_mask * 0.5 
        adv = torch.clamp(real_x + perturb_mask_scaled, 0.0, 1.0)
        return adv, perturb_mask_scaled

# ======== 2. Cấu hình và đường dẫn (SỬA) ========
DATA_DIR = "/home/thangkb2024/processed"
# (MỚI) Thay đổi đường dẫn model
GEN_PATH = "generator_mal-dcgan_IMPROVED.pt" 
BLACKBOX_PATH = os.path.join(DATA_DIR, "blackbox.pkl")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# (MỚI) Thêm các kích thước cho DCGAN
input_dim_1d = 2381 # Kích thước 1D gốc
# img_size = 49 (Đã định nghĩa ở trên)
# img_channels = 1 (Đã định nghĩa ở trên)
padded_dim = img_size * img_size # 2401

print(f"Using device: {device}")

# ======== 3. (MỚI) Thêm các hàm Helper từ file huấn luyện ========
def pad_and_reshape(x_1d_batch):
    """
    Nhận batch 1D (B, 2381) và chuyển thành (B, 1, 49, 49)
    """
    B, _ = x_1d_batch.shape
    padding = np.zeros((B, padded_dim - input_dim_1d))
    x_1d_padded = np.concatenate([x_1d_batch, padding], axis=1)
    x_2d = x_1d_padded.reshape((B, img_channels, img_size, img_size))
    return x_2d

def query_blackbox_from_tensor(tensor_batch_4d):
    """
    Nhận tensor 4D (B, 1, 49, 49),
    Flatten -> Trim (loại bỏ padding) -> Query Blackbox
    """
    B = tensor_batch_4d.size(0)
    tensor_flat = tensor_batch_4d.view(B, -1) # (B, 2401)
    tensor_trimmed = tensor_flat[:, :input_dim_1d] # (B, 2381)
    arr = tensor_trimmed.detach().cpu().numpy()
    preds = blackbox.predict(arr.astype(np.float64)) # (MỚI) Thêm astype cho an toàn
    return preds # numpy array

# ======== 4. Tải Dữ liệu TEST (Giữ nguyên logic) ========
print("Loading FINAL TEST data (X_test.npy)...")
X_test_all = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_test_all = np.load(os.path.join(DATA_DIR, "y_test.npy"))

# Tách riêng các mẫu malware/benign 1D (dùng cho baseline)
X_test_mal_1d = X_test_all[y_test_all == 1]
X_test_ben_1d = X_test_all[y_test_all == 0]

print(f"Test set (1D) loaded: Malware {X_test_mal_1d.shape}, Benign {X_test_ben_1d.shape}")

# ======== 5. Tải Black-Box và Generator (SỬA) ========
print(f"Loading Black-Box model: {BLACKBOX_PATH}")
blackbox = joblib.load(BLACKBOX_PATH)

print(f"Loading BEST DCGAN Generator: {GEN_PATH}")
# (MỚI) Khởi tạo đúng class Generator (DCGAN)
G = Generator(z_dim, img_channels, img_size).to(device)
G.load_state_dict(torch.load(GEN_PATH, map_location=device))
G.eval() # Chuyển sang chế độ đánh giá

# ======== 6. Đánh giá Baseline (Giữ nguyên) ========
print("\n" + "="*40)
print(" 1. BASELINE PERFORMANCE (Original 1D Files)")
print("="*40)

# Dùng dữ liệu 1D gốc để kiểm tra Blackbox
preds_mal_original = blackbox.predict(X_test_mal_1d.astype(np.float64))
preds_ben_original = blackbox.predict(X_test_ben_1d.astype(np.float64))

detection_rate = np.mean(preds_mal_original == 1)
fp_rate = np.mean(preds_ben_original == 1)

print(f"🔥 Detection Rate (Malware): {detection_rate * 100:.2f}%")
print(f"🔥 False Positive Rate (Benign): {fp_rate * 100:.2f}%")

# ======== 7. Sinh Mẫu Đối Kháng và Đánh giá Evasion Rate (SỬA) ========
print("\n" + "="*40)
print(" 2. ADVERSARIAL PERFORMANCE (Generated Files)")
print("="*40)

# (MỚI) BƯỚC 1: Chuyển đổi dữ liệu malware 1D sang 2D (49x49)
print(f"Padding and Reshaping {X_test_mal_1d.shape[0]} malware test samples to 2D...")
X_test_mal_2d = pad_and_reshape(X_test_mal_1d)
X_test_mal_t = torch.tensor(X_test_mal_2d, dtype=torch.float32).to(device)
print(f"Reshaped malware tensor to: {X_test_mal_t.shape}")

# (MỚI) BƯỚC 2: Sinh mẫu đối kháng (Input là 4D, Output là 4D)
with torch.no_grad(): # Không cần tính gradient
    z_eval = torch.randn(X_test_mal_t.size(0), z_dim, device=device)
    # G (DCGAN) nhận tensor 4D (B, 1, 49, 49)
    adv_samples_4d, perturb_4d = G(X_test_mal_t, z_eval) 

# (MỚI) BƯỚC 3: Đưa mẫu đối kháng 4D vào hàm query (tự động flatten/trim)
print("Querying Black-Box with 4D adversarial samples...")
preds_adversarial = query_blackbox_from_tensor(adv_samples_4d)

# Evasion Rate là % mẫu mã độc (malware) bị black-box dự đoán nhầm là 0 (benign)
evasion_rate = np.mean(preds_adversarial == 0) 

print(f"🚀 EVASION RATE (Malware): {evasion_rate * 100:.2f}%")
print(f"   (Blackbox bị lừa, tin rằng {evasion_rate * 100:.2f}% malware là file sạch)")

# ======== 8. Đo lường sự thay đổi (Perturbation) (SỬA) ========
# (MỚI) perturb_4d giờ là tensor 4D (B, 1, 49, 49)
perturb_np = perturb_4d.cpu().numpy()
# Tính toán L1/L2 trên từng pixel (vẫn hợp lệ)
avg_perturb_l1 = np.mean(np.abs(perturb_np))
avg_perturb_l2 = np.mean(np.square(perturb_np)) # L2 là bình phương
print("\n" + "="*40)
print(" 3. PERTURBATION (Mức độ thay đổi file - trên pixel)")
print("="*40)
print(f"   Average L1 Perturbation (per pixel): {avg_perturb_l1:.6f}")
print(f"   Average L2 Perturbation (per pixel): {avg_perturb_l2:.6f}")

# ======== 9. Vẽ biểu đồ kết quả (Giữ nguyên) ========
try:
    labels = ['Detected (1)', 'Evasion (0)']
    
    # Dữ liệu cho biểu đồ
    original_counts = [np.sum(preds_mal_original == 1), np.sum(preds_mal_original == 0)]
    adversarial_counts = [np.sum(preds_adversarial == 1), np.sum(preds_adversarial == 0)]

    df = pd.DataFrame({
        'Sample Type': ['Original Malware', 'Original Malware', 'Adversarial Malware', 'Adversarial Malware'],
        'Prediction': labels * 2,
        'Count': original_counts + adversarial_counts
    })

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Sample Type', y='Count', hue='Prediction', data=df)
    plt.title('Black-Box Performance: Original vs Adversarial (Test Set)')
    plt.ylabel('Number of Samples')
    plt.savefig("evaluation_dcgan_sieucaitien.png") # (MỚI) Đổi tên file output
    print(f"\n✅ Đã lưu biểu đồ kết quả vào: evaluation_dcgan_sieucaitien.png")

except ImportError:
    print("\n(Vui lòng cài 'pip install pandas matplotlib seaborn' để vẽ biểu đồ)")

print("\n🎉 Đánh giá Mal-DCGAN hoàn tất!")
