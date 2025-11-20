import numpy as np
import os
import ember
from sklearn.model_selection import train_test_split

# ======== Đường dẫn ========
DATA_DIR = "/home/thangkb2024/processed"

X_mal = np.load(os.path.join(DATA_DIR, "X_mal.npy"))
X_ben = np.load(os.path.join(DATA_DIR, "X_ben.npy"))

y_mal = np.ones(len(X_mal))
y_ben = np.zeros(len(X_ben))

# ======== Gộp dữ liệu lại ========
X = np.vstack([X_mal, X_ben])
y = np.hstack([y_mal, y_ben])

print("Tổng dữ liệu:", X.shape, y.shape)

# ======== Chia train / val / test ========
# 70% train, 15% val, 15% test
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42
)

# ======== Lưu ra file ========
np.save(os.path.join(DATA_DIR, "X_train.npy"), X_train)
np.save(os.path.join(DATA_DIR, "y_train.npy"), y_train)
np.save(os.path.join(DATA_DIR, "X_val.npy"), X_val)
np.save(os.path.join(DATA_DIR, "y_val.npy"), y_val)
np.save(os.path.join(DATA_DIR, "X_test.npy"), X_test)
np.save(os.path.join(DATA_DIR, "y_test.npy"), y_test)

print("✅ Đã chia dữ liệu và lưu tại:", DATA_DIR)
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
