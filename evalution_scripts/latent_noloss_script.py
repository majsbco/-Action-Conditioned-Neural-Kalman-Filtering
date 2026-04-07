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
# 1. 模型组件 (需与训练时完全一致)
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

class Decoder(nn.Module):
    def __init__(self, template_points, latent_dim=32):
        super(Decoder, self).__init__()
        self.template_points = template_points 
        self.M = template_points.shape[0] 
        self.mlp_reconstruction = nn.Sequential(
            nn.Linear(3 + latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 3) 
        )
    def forward(self, z):
        B = z.size(0)
        z_broadcast = z.unsqueeze(1).repeat(1, self.M, 1)
        q_broadcast = self.template_points.to(z.device).unsqueeze(0).repeat(B, 1, 1)
        input_features = torch.cat([q_broadcast, z_broadcast], dim=2)
        p_hat = self.mlp_reconstruction(input_features)
        return p_hat

class MeshReconstructionNet(nn.Module):
    def __init__(self, M=2048, latent_dim=32):
        super(MeshReconstructionNet, self).__init__()
        template_points = generate_template_points(M)
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(template_points, latent_dim)
    def forward(self, O):
        z = self.encoder(O) 
        p_hat = self.decoder(z) 
        return p_hat, z 

# =======================================================================
# 2. 数据加载类
# =======================================================================

class LatentExtractionDataset(Dataset):
    def __init__(self, file_paths, N_input): 
        self.file_paths = file_paths 
        self.N = N_input 
    def __len__(self):
        return len(self.file_paths)
    def __getitem__(self, idx):
        occ_file_path, time_index = self.file_paths[idx]
        occ_pcd_o3d = o3d.io.read_point_cloud(occ_file_path)
        occ_pcd_raw = np.asarray(occ_pcd_o3d.points)
        num_points_raw = occ_pcd_raw.shape[0]
        if num_points_raw == 0:
            occ_pcd = torch.zeros((self.N, 3)).float()
        else:
            choice = np.random.choice(num_points_raw, self.N, replace=(num_points_raw < self.N))
            occ_pcd = torch.from_numpy(occ_pcd_raw[choice, :]).float()
        return occ_pcd, time_index 

# =======================================================================
# 3. 单维度量化分析主循环
# =======================================================================

def analyze_latent_no_smooth_pro():
    # 配置路径（确保指向没有平滑损失的权重文件）
    PROJECT_ROOT = "D:/Cu_BEM_2/tshirt-data/tshirt_out_4"
    DATASET_ROOT = os.path.join(PROJECT_ROOT, "dataset_arm")
    WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "trained_armloss_weights.pth")
    
    N_INPUT_POINTS = 3000
    LATENT_DIM = 32 
    BATCH_SIZE = 32 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- 启动 PCA 单维度平滑度深度评估 (无平滑损失对照组) | 设备: {device.type} ---")
    
    # 1. 模型加载
    model = MeshReconstructionNet(M=2048, latent_dim=LATENT_DIM).to(device)
    if not os.path.exists(WEIGHTS_PATH):
        print(f"错误: 权重文件不存在 {WEIGHTS_PATH}")
        return
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()

    # 2. 提取潜向量
    occluded_files = glob.glob(os.path.join(DATASET_ROOT, "occ_frame_*.ply"))
    FILE_PATHS = [(p, int(re.search(r"occ_frame_(\d+)", os.path.basename(p)).group(1))) for p in occluded_files]
    FILE_PATHS.sort(key=lambda x: x[1]) 

    dataset = LatentExtractionDataset(FILE_PATHS, N_INPUT_POINTS)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_latents = []
    with torch.no_grad():
        for occ_pcd, _ in data_loader:
            _, z = model(occ_pcd.to(device)) 
            all_latents.append(z.cpu().numpy())

    Z = np.concatenate(all_latents, axis=0) 

    # --- 核心量化逻辑：单维度分析 ---
    max_k = 10
    pca = PCA(n_components=max_k)
    Z_pca = pca.fit_transform(Z)
    
    indiv_jitter = [] 
    indiv_velocity = [] 

    print("\n[单维度量化指标报告 - Baseline]")
    print("-" * 60)
    for i in range(max_k):
        component_data = Z_pca[:, i]
        v = np.mean(np.abs(np.diff(component_data)))
        indiv_velocity.append(v)
        j = np.mean(np.abs(np.diff(component_data, n=2)))
        indiv_jitter.append(j)
        
        print(f"PC {i+1:2d} | 方差占比: {pca.explained_variance_ratio_[i]*100:5.2f}% | 独立步长: {v:.6f} | 独立抖动: {j:.6f}")

    # --- 绘图展示 ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 子图 1: 每个主成分的独立抖动
    # 预测：在无平滑损失情况下，高维 PC 的抖动值会显著升高
    ax1.bar(range(1, max_k + 1), indiv_jitter, color='skyblue', edgecolor='black', alpha=0.8)
    ax1.set_title("Baseline: Jitter per PC Axis (Individual Smoothness)", fontsize=14)
    ax1.set_xlabel("Principal Component Index", fontsize=12)
    ax1.set_ylabel("Jitter Score (Lower is Smoother)", fontsize=12)
    ax1.set_xticks(range(1, max_k + 1))
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # 子图 2: 信息效率 (方差 / 抖动)
    # 预测：由于高维噪声大，曲线会下降得非常快
    quality_score = pca.explained_variance_ratio_ / (np.array(indiv_jitter) + 1e-9)
    ax2.plot(range(1, max_k + 1), quality_score, 'o-', color='orange', linewidth=2)
    ax2.fill_between(range(1, max_k + 1), quality_score, color='orange', alpha=0.1)
    ax2.set_title("Baseline: Information Efficiency (Variance / Jitter)", fontsize=14)
    ax2.set_xlabel("Principal Component Index", fontsize=12)
    ax2.set_ylabel("Quality Score (Higher is Better)", fontsize=12)
    ax2.set_xticks(range(1, max_k + 1))
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_p = os.path.join(PROJECT_ROOT, 'pca_no_smooth_dimension_report.png')
    plt.savefig(save_p, dpi=150)
    print(f"\n对照组量化分析完成。报告已保存至: {save_p}")
    plt.show()

if __name__ == '__main__':
    analyze_latent_no_smooth_pro()