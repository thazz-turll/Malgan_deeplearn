#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# ======== Load dữ liệu ========
X_mal = np.load("/home/thangkb2024/processed/X_mal.npy")
X_ben = np.load("/home/thangkb2024/processed/X_ben.npy")

# Chỉ dùng malware để sinh biến thể
X_mal = torch.tensor(X_mal, dtype=torch.float32)
dataset = TensorDataset(X_mal)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

input_dim = X_mal.shape[1]   # số đặc trưng (2381)
z_dim = 100                  # chiều của vector noise

# ======== Generator ========
class Generator(nn.Module):
    def __init__(self, input_dim, z_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, input_dim),
            nn.Sigmoid()  # đầu ra [0,1]
        )
    def forward(self, z):
        return self.model(z)

# ======== Detector (Discriminator) ========
class Detector(nn.Module):
    def __init__(self, input_dim):
        super(Detector, self).__init__()
        self.model = nn.Sequential(
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
        return self.model(x)

# ======== Khởi tạo ========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = Generator(input_dim, z_dim).to(device)
D = Detector(input_dim).to(device)

criterion = nn.BCELoss()
lr = 1e-4
optimizer_G = optim.Adam(G.parameters(), lr=lr)
optimizer_D = optim.Adam(D.parameters(), lr=lr)

# ======== Huấn luyện ========
epochs = 100
for epoch in range(epochs):
    for batch in dataloader:
        real_data = batch[0].to(device)
        batch_size = real_data.size(0)

        # Label thật & giả
        real_label = torch.ones(batch_size, 1, device=device)
        fake_label = torch.zeros(batch_size, 1, device=device)

        # --- Train Detector ---
        z = torch.randn(batch_size, z_dim, device=device)
        fake_data = G(z).detach()
        output_real = D(real_data)
        output_fake = D(fake_data)

        loss_D = criterion(output_real, real_label) + criterion(output_fake, fake_label)
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # --- Train Generator ---
        z = torch.randn(batch_size, z_dim, device=device)
        fake_data = G(z)
        output = D(fake_data)
        loss_G = criterion(output, real_label)  # G muốn đánh lừa D
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

    print(f"Epoch [{epoch+1}/{epochs}]  Loss_D: {loss_D.item():.4f}  Loss_G: {loss_G.item():.4f}")

torch.save(G.state_dict(), "generator.pt")
torch.save(D.state_dict(), "detector.pt")
print("✅ Huấn luyện hoàn tất, model đã lưu.")
