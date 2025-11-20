from sklearn.ensemble import RandomForestClassifier
import numpy as np, joblib, os

DATA_DIR = "/home/thangkb2024/processed"
X_train = np.load(os.path.join(DATA_DIR,"X_train.npy"))
y_train = np.load(os.path.join(DATA_DIR,"y_train.npy"))
X_val = np.load(os.path.join(DATA_DIR,"X_val.npy"))
y_val = np.load(os.path.join(DATA_DIR,"y_val.npy"))

rf = RandomForestClassifier(n_estimators=200, max_depth=30, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
print("Validation accuracy:", rf.score(X_val, y_val))

joblib.dump(rf, os.path.join(DATA_DIR,"blackbox.pkl"))
