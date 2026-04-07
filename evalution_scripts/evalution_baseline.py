import torch
import torch.nn as nn
import numpy as np
import os
import glob
import re
import random
import open3d as o3d
from torch.utils.data import Dataset, DataLoader

# =======================================================================
# 1. Kalman Filter (保持平滑逻辑)
# =======================================================================
class LatentKalmanFilterVectorized:
    def __init__(self, dim=32, q=1e-5, r=0.01):
        self.dim = dim
        self.is_initialized = False
        self.X = np.zeros((dim, 2, 1))
        self.P = np.tile(np.eye(2) * 1.0, (dim, 1, 1))
        self.F = np.tile(np.array([[1.0, 1.0], [0.0, 1.0]]), (dim, 1, 1))
        self.H = np.tile(np.array([[1.0, 0.0]]), (dim, 1, 1))
        self.Q = np.tile(np.eye(2) * q, (dim, 1, 1))
        self.R = np.tile(np.array([[r]]), (dim, 1, 1))

    def step(self, z_obs):
        z_obs = z_obs.reshape(self.dim, 1, 1)
        if not self.is_initialized:
            self.X[:, 0, 0] = z_obs[:, 0, 0]
            self.is_initialized = True
            return self.X[:, 0, 0].flatten()
        self.X = self.F @ self.X
        self.P = self.F @ self.P @ self.F.transpose(0, 2, 1) + self.Q
        y = z_obs - (self.H @ self.X)
        HT = self.H.transpose(0, 2, 1)
        S = self.H @ self.P @ HT + self.R
        K = self.P @ HT @ np.linalg.inv(S)
        self.X = self.X + K @ y
        I = np.tile(np.eye(2), (self.dim, 1, 1))
        self.P = (I - K @ self.H) @ self.P
        return self.X[:, 0, 0].flatten()

# =======================================================================
# 2. Model Structure
# =======================================================================
def generate_template_points(M=2048):
    np.random.seed(42)
    coords = np.random.randn(M, 3)
    return torch.from_numpy(coords / np.linalg.norm(coords, axis=1, keepdims=True)).float()

class Encoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(nn.Conv1d(3, 64, 1), nn.ReLU(), nn.Conv1d(64, 128, 1), nn.ReLU(), nn.Conv1d(128, 256, 1))
        self.fc = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, latent_dim))
    def forward(self, x):
        x = self.mlp(x.transpose(2, 1))
        x_max, _ = torch.max(x, dim=2)
        return self.fc(x_max)

class Decoder(nn.Module):
    def __init__(self, template_points, latent_dim=32):
        super().__init__()
        self.register_buffer('template_points', template_points)
        self.mlp = nn.Sequential(nn.Linear(3 + latent_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 3))
    def forward(self, z):
        B, M = z.size(0), self.template_points.size(0)
        z_ext = z.unsqueeze(1).repeat(1, M, 1)
        temp_ext = self.template_points.unsqueeze(0).repeat(B, 1, 1)
        return self.mlp(torch.cat([temp_ext, z_ext], dim=2))

class MeshReconstructionNet(nn.Module):
    def __init__(self, M=2048, latent_dim=32):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(generate_template_points(M), latent_dim)
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

# =======================================================================
# 3. Execution & Visualization
# =======================================================================
def run_target_visualization():
    PROJECT_ROOT = "D:/Cu_BEM_2/tshirt-data/tshirt_out_3"
    DATASET_ROOT = os.path.join(PROJECT_ROOT, "dataset_arm")
    WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "trained_arm_weights.pth")
    
    # --- 这里设置你想查看的帧 ---
    TARGET_FRAMES = [10, 50, 100, 150, 200] 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MeshReconstructionNet(M=2048, latent_dim=32).to(device)
    if os.path.exists(WEIGHTS_PATH):
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device), strict=False)
        print(f"Loaded: {WEIGHTS_PATH}")
    model.eval()

    kf = LatentKalmanFilterVectorized(dim=32, q=1e-6, r=0.05)
    
    # 获取并排序文件
    files = sorted(glob.glob(os.path.join(DATASET_ROOT, "occ_frame_*.ply")),
                   key=lambda x: int(re.search(r"occ_frame_(\d+)", x).group(1)))

    print(f"Starting inference. Will popup window for frames: {TARGET_FRAMES}")

    with torch.no_grad():
        for f_path in files:
            frame_id = int(re.search(r"occ_frame_(\d+)", f_path).group(1))
            
            # 读取并预处理
            pcd_in = o3d.io.read_point_cloud(f_path)
            pts = np.asarray(pcd_in.points)
            # 简单采样到3000点
            if len(pts) > 3000: pts = pts[np.random.choice(len(pts), 3000, replace=False)]
            in_tensor = torch.from_numpy(pts).float().unsqueeze(0).to(device)

            # 推理
            recon_raw, z_raw = model(in_tensor)
            
            # 卡尔曼平滑 (必须每一帧都跑，否则滤波状态会断)
            z_filt = kf.step(z_raw[0].cpu().numpy())

            # 只有目标帧才弹窗
            if frame_id in TARGET_FRAMES:
                z_filt_t = torch.from_numpy(z_filt).float().unsqueeze(0).to(device)
                recon_filt = model.decoder(z_filt_t)

                # 构造 Open3D 对象
                def make_o3d(points, color, offset):
                    p = o3d.geometry.PointCloud()
                    p.points = o3d.utility.Vector3dVector(points)
                    p.paint_uniform_color(color)
                    p.translate([offset, 0, 0])
                    return p

                # [左: 原始模型输出] | [中: 输入点云] | [右: KF平滑后输出]
                vis_raw = make_o3d(recon_raw[0].cpu().numpy(), [0.8, 0.2, 0.2], -0.6)
                vis_in  = make_o3d(pts, [0.2, 0.4, 0.8], 0)
                vis_kf  = make_o3d(recon_filt[0].cpu().numpy(), [0.2, 0.8, 0.2], 0.6)
                
                coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                
                print(f"Showing Frame {frame_id}. Close window to continue...")
                o3d.visualization.draw_geometries([vis_raw, vis_in, vis_kf, coord], 
                                                  window_name=f"Frame {frame_id} Comparison")

if __name__ == '__main__':
    run_target_visualization()