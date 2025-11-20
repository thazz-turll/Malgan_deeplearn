# Chạy trong Python shell hoặc file .py
import os
import ember
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

# đường dẫn
DATA_DIR = "/home/thangkb2024/ember2018/"
OUT_DIR = "/home/thangkb2024/processed"
os.makedirs(OUT_DIR, exist_ok=True)

# 1) load ember vectorized (bạn đã làm)
X_train, y_train, X_test, y_test = ember.read_vectorized_features(DATA_DIR)

# 2) tách malware / benign từ X_train (giữ như bạn)
X_mal = X_train[y_train == 1]
X_ben = X_train[y_train == 0]

print("Original sizes -> Malware:", X_mal.shape, "Benign:", X_ben.shape)

# 3) (TÙY CHỌN) giảm kích thước nếu cần để test nhanh
# Nếu muốn test nhanh: uncomment hai dòng dưới
X_mal = X_mal[:5000]
X_ben = X_ben[:5000]
print("After subsample -> Malware:", X_mal.shape, "Benign:", X_ben.shape)

# 4) Fit scaler 1 lần trên tổng hợp dữ liệu (malware + benign),
#    sau đó transform từng phần riêng biệt
scaler = MinMaxScaler()

# Fit trên toàn bộ tập train (mal + ben) để scale thống nhất
X_all = np.vstack([X_mal, X_ben])
scaler.fit(X_all)           # CHỈ fit 1 lần ở đây

# Transform từng phần bằng scaler đã fit
X_mal_scaled = scaler.transform(X_mal)
X_ben_scaled = scaler.transform(X_ben)

# 5) Lưu kết quả ra file .npy và lưu scaler để dùng lại later
np.save(os.path.join(OUT_DIR, "X_mal.npy"), X_mal_scaled)
np.save(os.path.join(OUT_DIR, "X_ben.npy"), X_ben_scaled)
joblib.dump(scaler, os.path.join(OUT_DIR, "minmax_scaler.joblib"))

print("Saved:", os.path.join(OUT_DIR, "X_mal.npy"), os.path.join(OUT_DIR, "X_ben.npy"))
print("Saved scaler to:", os.path.join(OUT_DIR, "minmax_scaler.joblib"))
