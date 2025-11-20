#!/usr/bin/env python3

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
import joblib

# ================== User data paths (Giữ nguyên) ==================
# !!! THAY ĐỔI ĐƯỜNG DẪN NÀY CHO PHÙ HỢP VỚI MÁY CỦA BẠN !!!
DATA_DIR = "/home/thangkb2024/processed" 
# ================================================================

X_TRAIN_PATH = os.path.join(DATA_DIR, "X_train.npy")
Y_TRAIN_PATH = os.path.join(DATA_DIR, "y_train.npy")
X_VAL_PATH = os.path.join(DATA_DIR, "X_val.npy")
Y_VAL_PATH = os.path.join(DATA_DIR, "y_val.npy")
BLACKBOX_PATH = os.path.join(DATA_DIR, "blackbox.pkl")

# Model save paths [CẢI TIẾN]
GEN_SAVE_PATH = "generator_mal-dcgan_IMPROVED.pt"
SUB_SAVE_PATH = "substitute_detector_dcgan_IMPROVED.pt"

# ================== Hyperparams (DCGAN) ==================
z_dim = 100
batch_size = 64 # Giảm batch size cho CNN
lr_G = 0.0002 # Learning rate chuẩn cho DCGAN
lr_S = 0.0002 # Learning rate chuẩn cho DCGAN
betas = (0.5, 0.999) # Betas chuẩn cho DCGAN
epochs = 20 # DCGAN cần nhiều epoch hơn
lambda_perturb = 0.01 # [CẢI TIẾN] Hệ số cân bằng cho loss nhiễu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Kích thước cho DCGAN
input_dim_1d = 2381 # Kích thước 1D gốc
img_size = 49
img_channels = 1
padded_dim = img_size * img_size # 2401

# ================== Hàm xử lý dữ liệu (MỚI) ==================
def pad_and_reshape(x_1d_batch):
    """
    Nhận batch 1D (B, 2381) và chuyển thành (B, 1, 49, 49)
    """
    B, _ = x_1d_batch.shape
    # 1. Tạo ma trận padding 0
    padding = np.zeros((B, padded_dim - input_dim_1d))
    # 2. Nối (Pad)
    x_1d_padded = np.concatenate([x_1d_batch, padding], axis=1)
    # 3. Reshape
    x_2d = x_1d_padded.reshape((B, img_channels, img_size, img_size))
    return x_2d

# ================== Load data (Cập nhật) ==================
print("Loading split training and validation data (1D)...")
X_train_all_1d = np.load(X_TRAIN_PATH)
y_train_all = np.load(Y_TRAIN_PATH)
X_val_all_1d = np.load(X_VAL_PATH)
y_val_all = np.load(Y_VAL_PATH)

# Tách riêng malware/benign từ tập TRAIN (1D)
X_mal_train_1d = X_train_all_1d[y_train_all == 1]
X_ben_train_1d = X_train_all_1d[y_train_all == 0]

# Tách riêng malware từ tập VAL (1D)
X_mal_val_1d = X_val_all_1d[y_val_all == 1]

# --- BƯỚC CHUYỂN ĐỔI SANG 2D ---
print("Padding and Reshaping data to 2D...")
X_mal_train_2d = pad_and_reshape(X_mal_train_1d)
X_ben_train_2d = pad_and_reshape(X_ben_train_1d)
X_mal_val_2d = pad_and_reshape(X_mal_val_1d)

print(f"Loaded Train 2D: Mal {X_mal_train_2d.shape}, Ben {X_ben_train_2d.shape}")
print(f"Loaded Val 2D (for eval): Mal {X_mal_val_2d.shape}")

# Convert to torch tensors (float)
X_mal_t = torch.tensor(X_mal_train_2d, dtype=torch.float32)
X_ben_t = torch.tensor(X_ben_train_2d, dtype=torch.float32)

mal_ds = TensorDataset(X_mal_t)
mal_loader = DataLoader(mal_ds, batch_size=batch_size, shuffle=True, drop_last=True)

ben_ds = TensorDataset(X_ben_t)
ben_loader = DataLoader(ben_ds, batch_size=batch_size, shuffle=True, drop_last=True)

# Dùng dữ liệu _val 2D để đánh giá
X_mal_val_t = torch.tensor(X_mal_val_2d, dtype=torch.float32).to(device)


# ================== Black-box detector (Cập nhật) ==================
print(f"Loading black-box detector from {BLACKBOX_PATH}...")
if not os.path.exists(BLACKBOX_PATH):
    print(f"Lỗi: Không tìm thấy model blackbox tại {BLACKBOX_PATH}")
    print("Vui lòng chạy script huấn luyện blackbox trước.")
    exit()
    
blackbox = joblib.load(BLACKBOX_PATH)

def query_blackbox_from_tensor(tensor_batch_4d):
    """
    Hàm MỚI: Nhận tensor 4D (B, 1, 49, 49),
    Flatten -> Trim (loại bỏ padding) -> Query Blackbox
    """
    B = tensor_batch_4d.size(0)
    # 1. Flatten 4D -> 2D
    tensor_flat = tensor_batch_4d.view(B, -1) # (B, 2401)
    
    # 2. Trim (loại bỏ padding)
    # Cắt từ 2401 về 2381
    tensor_trimmed = tensor_flat[:, :input_dim_1d] # (B, 2381)
    
    # 3. Query
    arr = tensor_trimmed.detach().cpu().numpy()
    preds = blackbox.predict(arr)
    return preds # numpy array

# ================== Models (KIẾN TRÚC DCGAN ĐÃ SỬA) ==================

class Generator(nn.Module):
    """
    Kiến trúc Generator (DCGAN) - ĐÃ SỬA LỖI UPSCALING
    Tạo ra MẶT NẠ NHIỄU (perturbation mask)
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
        
        # Trả về adv và MASK (để mô hình cải tiến có thể dùng)
        return adv, perturb_mask_scaled

class SubstituteDetector(nn.Module):
    """
    Kiến trúc Substitute Detector (DCGAN) - ĐÃ SỬA LỖI DOWNSCALING
    Hoạt động như một Discriminator / Bộ phân loại ảnh
    """
    def __init__(self, img_channels):
        super(SubstituteDetector, self).__init__()
        ndf = 32 # Kích thước feature map của D (S)
        self.net = nn.Sequential(
            # Input (B, 1, 49, 49)
            # LỚP ĐÃ SỬA: (từ 49x49 -> 24x24)
            nn.Conv2d(img_channels, ndf, 5, 2, 1, bias=False), # (B, 64, 24, 24)
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), # (B, 128, 12, 12)
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), # (B, 256, 6, 6)
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 4, 1, 6, 1, 0, bias=False), # (B, 1, 1, 1)
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).view(-1, 1) # Flatten về (B, 1)

# Instantiate
G = Generator(z_dim, img_channels, img_size).to(device)
S = SubstituteDetector(img_channels).to(device)

# Áp dụng weights_init (chuẩn của DCGAN)
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
print("Initializing weights...")
G.apply(weights_init)
S.apply(weights_init)

optimizer_G = optim.Adam(G.parameters(), lr=lr_G, betas=betas)
optimizer_S = optim.Adam(S.parameters(), lr=lr_S, betas=betas)
bce = nn.BCELoss()

# ================== Training loop (ĐÃ CẢI TIẾN) ==================
print("Start training Mal-DCGAN [IMPROVED] procedure...")
best_evasion_rate = 0.0 

for epoch in range(1, epochs + 1):
    G.train()
    S.train()

    # [CẢI TIẾN] Tách riêng loss G
    epoch_loss_G_adv = 0.0
    epoch_loss_G_perturb = 0.0
    epoch_loss_S = 0.0
    n_batches = 0

    ben_iter = iter(ben_loader)

    for mal_batch in mal_loader: # mal_batch giờ là (B, 1, 49, 49)
        real_mal = mal_batch[0].to(device)
        B = real_mal.size(0)
        n_batches += 1

        # ---- Generate adversarial samples ----
        z = torch.randn(B, z_dim, device=device)
        adv, perturb = G(real_mal, z) # adv giờ là (B, 1, 49, 49)

        # ---- Query blackbox (đã sửa) ----
        # Gửi tensor 4D vào hàm query mới
        adv_labels_bb = query_blackbox_from_tensor(adv) 

        # ---- Prepare substitute training batch ----
        try:
            benign_batch = next(ben_iter)[0].to(device) # (B, 1, 49, 49)
        except StopIteration:
            ben_iter = iter(ben_loader)
            benign_batch = next(ben_iter)[0].to(device)

        adv_tensor_for_S = adv.detach()
        adv_labels_tensor = torch.tensor(adv_labels_bb, dtype=torch.float32, device=device).view(-1,1)
        benign_labels_tensor = torch.zeros((benign_batch.size(0),1), dtype=torch.float32, device=device)

        m = min(adv_tensor_for_S.size(0), benign_batch.size(0))
        # train_X_S giờ là tensor 4D
        train_X_S = torch.cat([adv_tensor_for_S[:m], benign_batch[:m]], dim=0)
        train_y_S = torch.cat([adv_labels_tensor[:m], benign_labels_tensor[:m]], dim=0)

        # ---- Train Substitute detector S ----
        optimizer_S.zero_grad()
        preds_S = S(train_X_S) # S nhận ảnh 4D
        loss_S = bce(preds_S, train_y_S)
        loss_S.backward()
        optimizer_S.step()

        # ---- Train Generator G (ĐÃ CẢI TIẾN) ----
        optimizer_G.zero_grad()
        z2 = torch.randn(B, z_dim, device=device)
        
        # [CẢI TIẾN] Lấy cả adv2 và mặt nạ nhiễu perturb2
        adv2, perturb2 = G(real_mal, z2) # adv2 là ảnh 4D
        
        preds_sub_on_adv = S(adv2) # S nhận ảnh 4D
        target_ben = torch.zeros_like(preds_sub_on_adv, device=device)
        
        # [CẢI TIẾN] Tính 2 thành phần loss cho G
        # 1. Loss đối kháng (để đánh lừa S)
        loss_adv_detection = bce(preds_sub_on_adv, target_ben)
        # 2. Loss nhiễu (để giữ thay đổi ở mức tối thiểu)
        loss_perturb = torch.mean(torch.abs(perturb2))
        
        # Loss G tổng là tổng có trọng số
        loss_G = loss_adv_detection + lambda_perturb * loss_perturb
        
        loss_G.backward()
        optimizer_G.step()

        # [CẢI TIẾN] Ghi lại cả 3 loss
        epoch_loss_G_adv += loss_adv_detection.item()
        epoch_loss_G_perturb += loss_perturb.item()
        epoch_loss_S += loss_S.item()

    # End of epoch
    # [CẢI TIẾN] Tính trung bình và in ra
    avg_loss_G_adv = epoch_loss_G_adv / n_batches
    avg_loss_G_perturb = epoch_loss_G_perturb / n_batches
    avg_loss_S = epoch_loss_S / n_batches
    print(f"Epoch {epoch:03d}/{epochs}  Avg Loss_S: {avg_loss_S:.6f}  Avg Loss_G_Adv: {avg_loss_G_adv:.6f}  Avg Loss_G_Perturb: {avg_loss_G_perturb:.6f}")


    # ================== EVALUATION BLOCK (Logic giữ nguyên) ==================
    if epoch % 5 == 0:
        G.eval()
        with torch.no_grad():
            # Đánh giá trên tập VALIDATION 2D (X_mal_val_t)
            z_eval = torch.randn(X_mal_val_t.size(0), z_dim, device=device)
            adv_eval, _ = G(X_mal_val_t, z_eval) # adv_eval là (B_val, 1, 49, 49)
            
            # Query blackbox (dùng hàm query 4D mới)
            preds_bb = query_blackbox_from_tensor(adv_eval)
            
            evasion_rate = np.mean(preds_bb == 0)
            print(f"  -> Eval (epoch {epoch}): evasion_rate on VAL SET = {evasion_rate:.4f}")

            # ---- Lưu model tốt nhất ----
            if evasion_rate > best_evasion_rate:
                best_evasion_rate = evasion_rate
                torch.save(G.state_dict(), GEN_SAVE_PATH)
                print(f"  -> 🎉 New best model saved! Evasion: {evasion_rate*100:.2f}%")

# ================== Save final models ==================
torch.save(S.state_dict(), SUB_SAVE_PATH)
print(f"Final Substitute Detector saved to {SUB_SAVE_PATH}")
print(f"Best Generator saved to {GEN_SAVE_PATH} (Best Evasion Rate: {best_evasion_rate*100:.2f}%)")
print("Done.")
