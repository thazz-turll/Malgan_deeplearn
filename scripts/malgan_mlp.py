#!/usr/bin/env python3
"""
MalGAN training script (PyTorch) - BẢN CƠ SỞ (BASE)

- Huấn luyện G chỉ với loss đối kháng (loss_adv_detection).
- Đã sửa lỗi data leakage, query blackbox.
- Lưu model G tốt nhất dựa trên validation evasion_rate.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
import joblib

# ================== User data paths (CORRECTED) ==================
DATA_DIR = "/home/thangkb2024/processed"
X_TRAIN_PATH = os.path.join(DATA_DIR, "X_train.npy")
Y_TRAIN_PATH = os.path.join(DATA_DIR, "y_train.npy")
X_VAL_PATH = os.path.join(DATA_DIR, "X_val.npy")
Y_VAL_PATH = os.path.join(DATA_DIR, "y_val.npy")
BLACKBOX_PATH = os.path.join(DATA_DIR, "blackbox.pkl")

# Model save paths [SỬA]
GEN_SAVE_PATH = "generator_malgan_BASE_best.pt"
SUB_SAVE_PATH = "substitute_detector_BASE.pt"

# ================== Hyperparams ==================
z_dim = 100
batch_size = 128
lr_G = 1e-4
lr_S = 1e-4
epochs = 20 # Bạn có thể tăng lên 200 hoặc 300
# lambda_perturb = 1.0 # [ĐÃ BỎ] Không dùng trong mô hình cơ sở
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ================== Load data (CORRECTED) ==================
print("Loading split training and validation data...")
X_train_all = np.load(X_TRAIN_PATH)
y_train_all = np.load(Y_TRAIN_PATH)
X_val_all = np.load(X_VAL_PATH)
y_val_all = np.load(Y_VAL_PATH)

# Tách riêng malware/benign từ tập TRAIN (dùng để huấn luyện G và S)
X_mal_train = X_train_all[y_train_all == 1]
X_ben_train = X_train_all[y_train_all == 0]

# Tách riêng malware/benign từ tập VAL (dùng để đánh giá)
X_mal_val = X_val_all[y_val_all == 1]
# X_ben_val = X_val_all[y_val_all == 0] # Không cần thiết cho eval này

input_dim = X_mal_train.shape[1]
print(f"Loaded Train: Mal {X_mal_train.shape}, Ben {X_ben_train.shape}")
print(f"Loaded Val (for eval): Mal {X_mal_val.shape}")

# Convert to torch tensors (float)
# Dùng dữ liệu _train cho DataLoaders
X_mal_t = torch.tensor(X_mal_train, dtype=torch.float32)
X_ben_t = torch.tensor(X_ben_train, dtype=torch.float32)

mal_ds = TensorDataset(X_mal_t)
mal_loader = DataLoader(mal_ds, batch_size=batch_size, shuffle=True, drop_last=True)

ben_ds = TensorDataset(X_ben_t)
ben_loader = DataLoader(ben_ds, batch_size=batch_size, shuffle=True, drop_last=True)

# Dùng dữ liệu _val để đánh giá
X_mal_val_t = torch.tensor(X_mal_val, dtype=torch.float32).to(device)


# ================== Black-box detector (fixed) ==================
if not os.path.exists(BLACKBOX_PATH):
    print(f"Lỗi: Không tìm thấy model blackbox tại {BLACKBOX_PATH}")
    print("Vui lòng chạy script huấn luyện blackbox trước.")
    exit()

print("Loading black-box detector from", BLACKBOX_PATH)
blackbox = joblib.load(BLACKBOX_PATH)

# Black-box (đã sửa, không binarize)
def query_blackbox_from_tensor(tensor_batch):
    """
    tensor_batch: torch.Tensor (B, input_dim), continuous in [0,1]
    Gửi thẳng float tensor [0,1] vào blackbox (RF)
    """
    arr = tensor_batch.detach().cpu().numpy()
    preds = blackbox.predict(arr)
    return preds  # numpy array

# ================== Models (Copy y hệt) ==================
class Generator(nn.Module):
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

class SubstituteDetector(nn.Module):
    def __init__(self, input_dim):
        super(SubstituteDetector, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# Instantiate
G = Generator(input_dim, z_dim, p_max=0.5).to(device)
S = SubstituteDetector(input_dim).to(device)

optimizer_G = optim.Adam(G.parameters(), lr=lr_G, betas=(0.5, 0.999))
optimizer_S = optim.Adam(S.parameters(), lr=lr_S, betas=(0.5, 0.999))
bce = nn.BCELoss()

# ================== Training loop ==================
print("Start training MalGAN [BASE] procedure...")
best_evasion_rate = 0.0 # Theo dõi evasion rate tốt nhất

for epoch in range(1, epochs + 1):
    G.train()
    S.train()

    epoch_loss_G = 0.0
    epoch_loss_S = 0.0
    n_batches = 0

    ben_iter = iter(ben_loader)

    for mal_batch in mal_loader: # Chỉ lặp qua tập train
        real_mal = mal_batch[0].to(device)
        B = real_mal.size(0)
        n_batches += 1

        # ---- Generate adversarial samples ----
        z = torch.randn(B, z_dim, device=device)
        adv, perturb = G(real_mal, z)

        # Query blackbox (đã sửa)
        adv_labels_bb = query_blackbox_from_tensor(adv)

        # ---- Prepare substitute training batch ----
        try:
            benign_batch = next(ben_iter)[0].to(device)
        except StopIteration:
            ben_iter = iter(ben_loader)
            benign_batch = next(ben_iter)[0].to(device)

        adv_tensor_for_S = adv.detach()
        adv_labels_tensor = torch.tensor(adv_labels_bb, dtype=torch.float32, device=device).view(-1,1)
        benign_labels_tensor = torch.zeros((benign_batch.size(0),1), dtype=torch.float32, device=device)

        m = min(adv_tensor_for_S.size(0), benign_batch.size(0))
        train_X_S = torch.cat([adv_tensor_for_S[:m], benign_batch[:m]], dim=0)
        train_y_S = torch.cat([adv_labels_tensor[:m], benign_labels_tensor[:m]], dim=0)

        # ---- Train Substitute detector S ----
        optimizer_S.zero_grad()
        preds_S = S(train_X_S)
        loss_S = bce(preds_S, train_y_S)
        loss_S.backward()
        optimizer_S.step()

        # ---- Train Generator G (Bản thường) ----
        optimizer_G.zero_grad()
        z2 = torch.randn(B, z_dim, device=device)
        
        # Dù G trả về (adv2, perturb2), chúng ta chỉ cần adv2
        adv2, _ = G(real_mal, z2)
        
        preds_sub_on_adv = S(adv2)
        target_ben = torch.zeros_like(preds_sub_on_adv, device=device)
        
        # [SỬA ĐỔI CHÍNH] Loss cơ sở chỉ có loss đối kháng
        loss_G = bce(preds_sub_on_adv, target_ben)
        
        loss_G.backward()
        optimizer_G.step()

        epoch_loss_G += loss_G.item()
        epoch_loss_S += loss_S.item()

    # End of epoch
    avg_loss_G = epoch_loss_G / n_batches
    avg_loss_S = epoch_loss_S / n_batches
    print(f"Epoch {epoch:03d}/{epochs}  Avg Loss_G: {avg_loss_G:.6f}  Avg Loss_S: {avg_loss_S:.6f}")

    # ================== EVALUATION BLOCK (CORRECTED) ==================
    if epoch % 5 == 0:
        G.eval()
        with torch.no_grad():
            # Đánh giá trên tập VALIDATION (X_mal_val_t)
            z_eval = torch.randn(X_mal_val_t.size(0), z_dim, device=device)
            adv_eval, _ = G(X_mal_val_t, z_eval)
            
            # Query blackbox (đã sửa)
            preds_bb = blackbox.predict(adv_eval.cpu().numpy())
            
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
