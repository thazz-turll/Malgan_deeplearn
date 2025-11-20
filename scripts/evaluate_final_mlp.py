#!/usr/bin/env python3
import os
import numpy as np
import torch
import torch.nn as nn  # Cần cho định nghĩa class
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# ======== 1. Tải lại kiến trúc model (MỚI - KIẾN TRÚC MLP) ========
# PyTorch cần định nghĩa lớp để tải trọng số
z_dim = 100
# input_dim sẽ được load từ dữ liệu

# (MỚI) Đây là kiến trúc Generator của MalGAN (MLP)
class Generator(nn.Module):
    """
    Kiến trúc Generator (MLP) - Sao chép từ file huấn luyện
    """
    def __init__(self, input_dim, z_dim, p_max=0.5):
        super(Generator, self).__init__()
        self.p_max = p_max
        self.net = nn.Sequential(
            nn.Linear(input_dim + z_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, input_dim),
            nn.Tanh()
        )

    def forward(self, real_x, z):
        x = torch.cat([real_x, z], dim=1)
        perturb = self.net(x) * self.p_max
        adv = torch.clamp(real_x + perturb, 0.0, 1.0)
        return adv, perturb

# ======== 2. Cấu hình và đường dẫn (SỬA) ========
DATA_DIR = "/home/thangkb2024/processed"
# (SỬA) Đổi đường dẫn model sang file MLP (thay _BASE hoặc _IMPROVED nếu cần)
GEN_PATH = "generator_malgan_BASE_best.pt" 
BLACKBOX_PATH = os.path.join(DATA_DIR, "blackbox.pkl")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

# ======== 3. (MỚI) Bỏ các hàm Helper của DCGAN ========
# (Không cần pad_and_reshape)
# (Không cần query_blackbox_from_tensor)

# ======== 4. Tải Dữ liệu TEST (Giữ nguyên logic) ========
print("Loading FINAL TEST data (X_test.npy)...")
X_test_all = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_test_all = np.load(os.path.join(DATA_DIR, "y_test.npy"))

# Tách riêng các mẫu malware/benign 1D
X_test_mal_1d = X_test_all[y_test_all == 1]
X_test_ben_1d = X_test_all[y_test_all == 0]

# (MỚI) Lấy input_dim từ dữ liệu
input_dim = X_test_mal_1d.shape[1] 
print(f"Input dimension set to: {input_dim}")
print(f"Test set (1D) loaded: Malware {X_test_mal_1d.shape}, Benign {X_test_ben_1d.shape}")

# ======== 5. Tải Black-Box và Generator (SỬA) ========
print(f"Loading Black-Box model: {BLACKBOX_PATH}")
blackbox = joblib.load(BLACKBOX_PATH)

print(f"Loading BEST MLP Generator: {GEN_PATH}")
# (SỬA) Khởi tạo đúng class Generator (MLP)
G = Generator(input_dim, z_dim, p_max=0.5).to(device)
G.load_state_dict(torch.load(GEN_PATH, map_location=device))
G.eval() # Chuyển sang chế độ đánh giá

# ======== 6. Đánh giá Baseline (Giữ nguyên) ========
print("\n" + "="*40)
print(" 1. BASELINE PERFORMANCE (Original 1D Files)")
print("="*40)

# Dùng dữ liệu 1D gốc để kiểm tra Blackbox
preds_mal_original = blackbox.predict(X_test_mal_1d.astype(np.float64))
preds_ben_original = blackbox.predict(X_test_ben_1d.astype(np.float64))

# (SỬA) Đổi tên detection_rate thành tpr
tpr = np.mean(preds_mal_original == 1)
fp_rate = np.mean(preds_ben_original == 1)

print(f"🔥 True Positive Rate (TPR): {tpr * 100:.2f}%")
print(f"🔥 False Positive Rate (FPR): {fp_rate * 100:.2f}%")

# ======== 7. Sinh Mẫu Đối Kháng và Đánh giá Evasion Rate (SỬA) ========
print("\n" + "="*40)
print(" 2. ADVERSARIAL PERFORMANCE (Generated 1D Files)")
print("="*40)

# (SỬA) BƯỚC 1: Chuyển đổi numpy 1D sang tensor 1D
X_test_mal_t = torch.tensor(X_test_mal_1d, dtype=torch.float32).to(device)
print(f"Loaded malware tensor to: {X_test_mal_t.shape}")

# (SỬA) BƯỚC 2: Sinh mẫu đối kháng (Input 1D, Output 1D)
with torch.no_grad(): # Không cần tính gradient
    z_eval = torch.randn(X_test_mal_t.size(0), z_dim, device=device)
    # G (MLP) nhận tensor 2D (B, input_dim)
    adv_samples_1d_t, perturb_1d_t = G(X_test_mal_t, z_eval)

# (SỬA) BƯỚC 3: Đưa mẫu đối kháng 1D vào blackbox (query trực tiếp)
print("Querying Black-Box with 1D adversarial samples...")
adv_samples_1d_np = adv_samples_1d_t.cpu().numpy()
preds_adversarial = blackbox.predict(adv_samples_1d_np.astype(np.float64))

# ASR (Attack Success Rate) = Evasion Rate
asr = np.mean(preds_adversarial == 0) 

print(f"🚀 Attack Success Rate (ASR) / Evasion Rate: {asr * 100:.2f}%")
print(f"   (Blackbox bị lừa, tin rằng {asr * 100:.2f}% malware là file sạch)")

# ======== 8. Đo lường sự thay đổi (Perturbation) (SỬA) ========
# (SỬA) perturb_1d_t giờ là tensor 2D (B, input_dim)
perturb_np = perturb_1d_t.cpu().numpy()
# Tính toán L1/L2 trên từng đặc trưng
avg_perturb_l1 = np.mean(np.abs(perturb_np))
avg_perturb_l2 = np.mean(np.square(perturb_np)) # L2 là bình phương
print("\n" + "="*40)
print(" 3. PERTURBATION (Mức độ thay đổi file - trên vector)")
print("="*40)
print(f"   Average L1 Perturbation (per feature): {avg_perturb_l1:.6f}")
print(f"   Average L2 Perturbation (per feature): {avg_perturb_l2:.6f}")

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
    plt.savefig("evaluation_mlp_results.png") # (SỬA) Đổi tên file output
    print(f"\n✅ Đã lưu biểu đồ kết quả vào: evaluation_mlp_results.png")

except ImportError:
    print("\n(Vui lòng cài 'pip install pandas matplotlib seaborn' để vẽ biểu đồ)")

print("\n🎉 Đánh giá MalGAN (MLP) hoàn tất!")
