import numpy as np
import pickle
import json
import os
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# Paths
res_dir = r"C:\Projects\Graduate Project\Fusion\Others\Results"
mulde_path = os.path.join(res_dir, "test_multiscale_log_density.npz")
stgnf_path = os.path.join(res_dir, "stgnf_scores.pkl")

print("Loading data...")
# Load MULDE 16D
mulde_data = np.load(mulde_path)
mulde_16d = mulde_data['log_density']
labels = mulde_data['labels']
video_ids = mulde_data['video_ids']
frame_indices = mulde_data['frame_indices']

# Load STG-NF 1D
with open(stgnf_path, 'rb') as f:
    stgnf_data = pickle.load(f)

# Extract scores sequentially
stgnf_1d = []
for vid, f_idx in zip(video_ids, frame_indices):
    v = vid
    # We can just extract it since we verified order is exact
    pass

stgnf_1d = []
for k, v in stgnf_data['scores_by_video'].items():
    stgnf_1d.extend(v['anomaly_scores'])
stgnf_1d = np.array(stgnf_1d).reshape(-1, 1)

print(f"MULDE 16D shape: {mulde_16d.shape}")
print(f"STG-NF 1D shape: {stgnf_1d.shape}")

# Concatenate into 17D
combined_17d = np.concatenate([mulde_16d, stgnf_1d], axis=1)

# Normalization: Important so variance is standard
scaler = StandardScaler()
combined_17d_scaled = scaler.fit_transform(combined_17d)

# FIT GMM
print("Fitting GMM (N_Components=5) on test data (Unsupervised)...")
gmm = GaussianMixture(n_components=5, covariance_type="full", random_state=42, reg_covar=1e-3)
gmm.fit(combined_17d_scaled)

# Score samples
final_scores = -gmm.score_samples(combined_17d_scaled)
micro_auc = roc_auc_score(labels, final_scores)

print(f"\n[Result] 17D Intermediate Fusion Micro AUC: {micro_auc:.4f}")

# Single Model Baselines for comparison
# 1. STG-NF baseline
stgnf_auc = roc_auc_score(labels, stgnf_1d)
if stgnf_auc < 0.5:
    stgnf_auc = 1.0 - stgnf_auc
print(f"[Baseline] STG-NF 1D Micro AUC (Corrected): {stgnf_auc:.4f}")

# FIT CHEATING GMM (Normal Test Frames Only) to show upper bound
normal_idx = (labels == 0)
gmm_cheat = GaussianMixture(n_components=5, covariance_type="full", random_state=42, reg_covar=1e-3)
gmm_cheat.fit(combined_17d_scaled[normal_idx])
cheat_scores = -gmm_cheat.score_samples(combined_17d_scaled)
cheat_auc = roc_auc_score(labels, cheat_scores)
print(f"\n[Bound Test] 17D Fusion if trained ONLY on normal frames: {cheat_auc:.4f}")

# 2. MULDE baseline
mulde_scaler = StandardScaler()
mulde_scaled = mulde_scaler.fit_transform(mulde_16d)
gmm_mulde = GaussianMixture(n_components=5, covariance_type="full", random_state=42, reg_covar=1e-3)
gmm_mulde.fit(mulde_scaled)
mulde_scores = -gmm_mulde.score_samples(mulde_scaled)
mulde_auc = roc_auc_score(labels, mulde_scores)
print(f"[Baseline] MULDE 16D (unsupervised test) Micro AUC: {mulde_auc:.4f}")
