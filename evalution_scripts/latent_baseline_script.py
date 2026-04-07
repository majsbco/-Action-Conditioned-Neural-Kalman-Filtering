import torch
import torch.nn as nn
import numpy as np
import os
import glob
import re
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
import open3d as o3d

# =======================================================================
# 1. 模型组件 (与训练时保持一致)
# =======================================================================

def generate_template_points(M=2048):
    np.random.seed(42) 
    coords = np.random.randn(M, 3)
    radii = np.linalg.norm(coords, axis=1, keepdims=True)
    template_points = coords / radii
    return torch.from_numpy(template_points).float()

class Encoder(nn.Module):
    def __init__(self, latent_dim=32):
        super(Encoder, self).__init__()
        self.mlp_local = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.ReLU(),
            nn.Conv1d(128, 256, 1)
        )
        self.fc_global = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
    def forward(self, x):
        x = x.transpose(2, 1) 
        local_features = self.mlp_local(x)
        global_feature, _ = torch.max(local_features, dim=2, keepdim=False) 
        z = self.fc_global(global_feature)
        return z

class MeshReconstructionNet(nn.Module):
    def __init__(self, M=2048, latent_dim=32):
        super(MeshReconstructionNet, self).__init__()
        self.encoder = Encoder(latent_dim)
    def forward(self, O):
        z = self.encoder(O) 
        return z 

# =======================================================================
# 2. 卡尔曼滤波器 (针对 32 维 Latent Space)
# =======================================================================

class LatentKalmanFilter:
    def __init__(self, dim=32, q=1e-5, r=0.01): #q:物理惯性 r:对encoder怀疑度
        self.dim = dim
        self.x = np.zeros((dim, 2)) # [pos, vel]
        self.P = np.eye(2) * 1.0
        self.F = np.array([[1, 1], [0, 1]]) # 状态转移
        self.H = np.array([[1, 0]])         # 观测矩阵
        self.Q = np.eye(2) * q              # 过程噪声
        self.R = np.array([[r]])            # 测量噪声

    def step(self, z_obs):
        z_filtered = np.zeros(self.dim)
        for i in range(self.dim):
            x_pred = self.F @ self.x[i]
            P_pred = self.F @ self.P @ self.F.T + self.Q
            y = z_obs[i] - (self.H @ x_pred)
            S = self.H @ P_pred @ self.H.T + self.R
            K = P_pred @ self.H.T @ np.linalg.inv(S)
            self.x[i] = x_pred + K @ y
            self.P = (np.eye(2) - K @ self.H) @ P_pred
            z_filtered[i] = self.x[i, 0]
        return z_filtered

# =======================================================================
# 3. 数据加载
# =======================================================================

class LatentExtractionDataset(Dataset):
    def __init__(self, file_paths, N_input): 
        self.file_paths = file_paths 
        self.N = N_input 
    def __len__(self):
        return len(self.file_paths)
    def __getitem__(self, idx):
        occ_file_path, _ = self.file_paths[idx]
        occ_pcd_o3d = o3d.io.read_point_cloud(occ_file_path)
        occ_pcd_raw = np.asarray(occ_pcd_o3d.points)
        num_points_raw = occ_pcd_raw.shape[0]
        if num_points_raw == 0:
            occ_pcd = torch.zeros((self.N, 3)).float()
        else:
            choice = np.random.choice(num_points_raw, self.N, replace=(num_points_raw < self.N))
            occ_pcd = torch.from_numpy(occ_pcd_raw[choice, :]).float()
        return occ_pcd

# =======================================================================
# 4. 主循环与分析
# =======================================================================

def analyze_latent_with_kalman():
    PROJECT_ROOT = "D:/Cu_BEM_2/tshirt-data/tshirt_out_3"
    DATASET_ROOT = os.path.join(PROJECT_ROOT, "dataset_arm")
    WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "trained_armloss_weights.pth")
    
    LATENT_DIM = 32
    N_INPUT_POINTS = 3000
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MeshReconstructionNet(latent_dim=LATENT_DIM).to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device), strict=False)
    model.eval()

    occluded_files = glob.glob(os.path.join(DATASET_ROOT, "occ_frame_*.ply"))
    FILE_PATHS = [(p, int(re.search(r"occ_frame_(\d+)", os.path.basename(p)).group(1))) for p in occluded_files]
    FILE_PATHS.sort(key=lambda x: x[1]) 

    dataset = LatentExtractionDataset(FILE_PATHS, N_INPUT_POINTS)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 初始化滤波器
    kf = LatentKalmanFilter(dim=LATENT_DIM, q=1e-5, r=0.05)

    all_raw = []
    all_filt = []

    print("--- 正在提取潜向量并应用卡尔曼滤波 ---")
    with torch.no_grad():
        for i, occ_pcd in enumerate(loader):
            z_raw = model(occ_pcd.to(device)).cpu().numpy()[0]
            z_filt = kf.step(z_raw)
            all_raw.append(z_raw)
            all_filt.append(z_filt)
            if i % 100 == 0: print(f"进度: {i}/{len(dataset)}")

    Z_raw = np.array(all_raw)
    Z_filt = np.array(all_filt)

    # --- PCA 单维度分析 ---
    max_k = 10
    pca = PCA(n_components=max_k)
    
    # 我们基于原始数据拟合 PCA，然后观察滤波对这些主成分轴的影响
    pca.fit(Z_raw)
    Z_raw_pca = pca.transform(Z_raw)
    Z_filt_pca = pca.transform(Z_filt)

    jitter_raw = []
    jitter_filt = []
    
    print("\n[高维 PCA 量化指标对比 - 卡尔曼滤波效果]")
    print("-" * 80)
    for i in range(max_k):
        j_r = np.mean(np.abs(np.diff(Z_raw_pca[:, i], n=2)))
        j_f = np.mean(np.abs(np.diff(Z_filt_pca[:, i], n=2)))
        jitter_raw.append(j_r)
        jitter_filt.append(j_f)
        improvement = (j_r - j_f) / (j_r + 1e-9) * 100
        print(f"PC {i+1:2d} | 方差: {pca.explained_variance_ratio_[i]*100:5.2f}% | 原始抖动: {j_r:.6f} | 滤波后: {j_f:.6f} | 优化率: {improvement:5.1f}%")

    # --- 绘图 ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 子图 1: 每一维 PC 的抖动值对比 (柱状图)
    x = np.arange(1, max_k + 1)
    width = 0.35
    ax1.bar(x - width/2, jitter_raw, width, label='Raw (Encoder)', color='salmon', alpha=0.7)
    ax1.bar(x + width/2, jitter_filt, width, label='With Kalman Filter', color='mediumseagreen', alpha=0.8)
    ax1.set_title("Jitter Comparison per PC Axis (Lower is Smoother)", fontsize=14)
    ax1.set_xlabel("Principal Component Index")
    ax1.set_ylabel("Jitter Score (2nd Order Diff)")
    ax1.set_xticks(x)
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    # 子图 2: 信息效率 (方差 / 抖动)
    # 这个指标反映了单位噪声下包含的信息量。卡尔曼滤波应能显著提高高维 PC 的效率。
    quality_raw = pca.explained_variance_ratio_ / (np.array(jitter_raw) + 1e-9)
    quality_filt = pca.explained_variance_ratio_ / (np.array(jitter_filt) + 1e-9)
    
    ax2.plot(x, quality_raw, 'o--', color='salmon', label='Raw Efficiency')
    ax2.plot(x, quality_filt, 's-', color='mediumseagreen', linewidth=2, label='Filtered Efficiency')
    ax2.set_title("Information Efficiency (Variance / Jitter)", fontsize=14)
    ax2.set_xlabel("Principal Component Index")
    ax2.set_ylabel("Quality Score")
    ax2.set_xticks(x)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(PROJECT_ROOT, 'pca_kalman_dimension_report.png')
    plt.savefig(save_path, dpi=150)
    print(f"\n量化报告图表已保存至: {save_path}")
    plt.show()

if __name__ == '__main__':
    analyze_latent_with_kalman()