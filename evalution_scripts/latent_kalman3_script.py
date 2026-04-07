import torch
import torch.nn as nn
import numpy as np
import os
import glob
import re
import trimesh
import open3d as o3d
from scipy.spatial import cKDTree

# =======================================================================
# 1. 模型架构 (与 training_script_kalman_3 保持严格一致)
# =======================================================================

class PointNetEncoder(nn.Module):
    def __init__(self, latent_dim=64):
        super(PointNetEncoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, 512, 1), nn.BatchNorm1d(512), nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, latent_dim)
        )

    def forward(self, x):
        x = x.transpose(2, 1)
        x = self.mlp(x)
        x_max, _ = torch.max(x, dim=2)
        return self.fc(x_max)

class RecursiveTrackingNet(nn.Module):
    def __init__(self, M=2048, obs_latent_dim=64, grasp_latent_dim=16, 
                 damping=0.95, template_path=None):
        super(RecursiveTrackingNet, self).__init__()
        self.M = M
        self.damping = damping
        self.enc_obs = PointNetEncoder(obs_latent_dim)
        self.enc_grasp = nn.Sequential(
            nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, grasp_latent_dim)
        )
        self.f_gru = nn.GRUCell(obs_latent_dim + grasp_latent_dim, obs_latent_dim)
        self.ln_h = nn.LayerNorm(obs_latent_dim)
        
        self.gate = nn.Sequential(
            nn.Linear(obs_latent_dim + (obs_latent_dim + grasp_latent_dim), 128), 
            nn.ReLU(),
            nn.Linear(128, 1), 
            nn.Sigmoid()
        )
        
        mesh = trimesh.load(template_path, process=False)
        if isinstance(mesh, trimesh.Scene): mesh = list(mesh.geometry.values())[0]
        v = np.array(mesh.vertices[:M], dtype=np.float32)
        v_mean = np.mean(v, axis=0)
        v_centered = v - v_mean 
        self.register_buffer('template_tensor', torch.from_numpy(v_centered))
        
        self.decoder = nn.Sequential(
            nn.Linear(3 + obs_latent_dim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 3) 
        )

    def forward(self, o_t, g_t, h_t, h_prev=None):
        B = o_t.size(0)
        z_obs = self.enc_obs(o_t)
        z_grasp = self.enc_grasp(g_t)
        z_fused = torch.cat([z_obs, z_grasp], dim=-1)
        
        h_inertial = h_t
        if h_prev is not None:
            h_inertial = h_t + self.damping * (h_t - h_prev)
            
        alpha = self.gate(torch.cat([h_inertial, z_fused], dim=-1))
        h_gru = self.f_gru(z_fused, h_t)
        h_next = self.ln_h((1 - alpha) * h_inertial + alpha * h_gru)
        
        z_ext = h_next.unsqueeze(1).repeat(1, self.M, 1)
        t_ext = self.template_tensor.unsqueeze(0).repeat(B, 1, 1)
        
        delta_v = self.decoder(torch.cat([t_ext, z_ext], dim=-1))
        p_hat = t_ext + delta_v
        return p_hat, h_next

# =======================================================================
# 2. 核心评估函数 (镜像训练逻辑)
# =======================================================================

def chamfer_loss_aligned_eval(p1, p2):
    """
    镜像 training_script_kalman_3.py 中的 chamfer_loss_cm:
    1. 计算中心对齐 (Centroid Alignment)
    2. 计算平方距离均值 (MSE Loss)
    3. 同时输出物理意义上的线性距离 (Linear Dist)
    """
    # 转为对齐后的 numpy
    p1_centered = p1 - np.mean(p1, axis=0)
    p2_centered = p2 - np.mean(p2, axis=0)
    
    tree1 = cKDTree(p1_centered)
    tree2 = cKDTree(p2_centered)
    
    dist1, _ = tree1.query(p2_centered)
    dist2, _ = tree2.query(p1_centered)
    
    # 训练时的 Loss 是平方均值
    mse_loss = (np.mean(np.square(dist1)) + np.mean(np.square(dist2))) / 2.0
    # 物理误差是直接均值
    linear_dist = (np.mean(dist1) + np.mean(dist2)) / 2.0
    
    return mse_loss, linear_dist

def load_pc(path, n=2048):
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points, dtype=np.float32)
    if len(pts) == 0: return np.zeros((n, 3))
    if len(pts) >= n:
        idx = np.random.choice(len(pts), n, replace=False)
        pts = pts[idx]
    else:
        pts = np.tile(pts, (n // len(pts) + 1, 1))[:n]
    return pts

def parse_grasps(path):
    pts = []
    if not os.path.exists(path): return np.zeros((1000, 3), dtype=np.float32)
    with open(path, 'r') as f:
        content = f.read()
        matches = re.findall(r'X=\s*([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)', content)
        for m in matches: pts.append([float(m[0]), float(m[1]), float(m[2])])
    return np.array(pts, dtype=np.float32)

# =======================================================================
# 3. 推理主程序
# =======================================================================

def run_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 路径配置 ---
    BASE = r"D:\Cu_BEM_2\tshirt-data\tshirt_kalman_3"
    CKPT_PATH = "checkpoints/best_model_simplified2.pth"
    TEMPLATE_PATH = os.path.join(BASE, "tshirt_mech00001.obj")
    DATA_PATH = os.path.join(BASE, "dataset_arm")
    GRASP_FILE = os.path.join(BASE, "vertex_237_trajectory.txt")

    # 预设 Centroid (需根据训练脚本 dataset 类中的计算结果填入)
    GLOBAL_CENTROID = np.array([-5.8771, 115.0959, 6.6215], dtype=np.float32)

    # --- 加载模型 ---
    model = RecursiveTrackingNet(template_path=TEMPLATE_PATH).to(device)
    if os.path.exists(CKPT_PATH):
        ckpt = torch.load(CKPT_PATH, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"成功加载模型: {CKPT_PATH}")
    else:
        print(f"警告: 未找到模型文件 {CKPT_PATH}，将使用随机初始化权重进行结构测试。")
        
    model.eval()

    obs_files = sorted(glob.glob(os.path.join(DATA_PATH, "occ_frame_*.ply")), 
                       key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group()))
    grasps = parse_grasps(GRASP_FILE)
    
    h_t = torch.zeros(1, 64).to(device)
    h_prev = None
    
    mse_list = []
    linear_list = []

    print(f"开始推理，使用对齐评估模式...")
    
    max_frames = min(len(obs_files), len(grasps))
    for i in range(max_frames):
        raw_obs = load_pc(obs_files[i])
        raw_grasp = grasps[i]
        
        # 预处理：标准化 (减去全局均值)
        obs_norm = raw_obs - GLOBAL_CENTROID
        grasp_norm = raw_grasp - GLOBAL_CENTROID
        
        o_t = torch.from_numpy(obs_norm).float().unsqueeze(0).to(device)
        g_t = torch.from_numpy(grasp_norm).float().unsqueeze(0).to(device)

        with torch.no_grad():
            p_hat_norm, h_next = model(o_t, g_t, h_t, h_prev)
            
            # 评估逻辑：对齐中心后的 MSE 和 Linear 误差
            pred_np = p_hat_norm[0].cpu().numpy()
            gt_np = o_t[0].cpu().numpy()
            
            mse_val, linear_val = chamfer_loss_aligned_eval(pred_np, gt_np)
            
            mse_list.append(mse_val)
            linear_list.append(linear_val)

            h_prev = h_t.clone()
            h_t = h_next

        if (i+1) % 50 == 0:
            print(f"Frame {i+1:03d} | Loss (MSE): {mse_val:.4f} | Dist (Linear): {linear_val:.4f} cm")

    # --- 最终统计 ---
    avg_mse = np.mean(mse_list)
    avg_linear = np.mean(linear_list)
    
    print("\n" + "="*40)
    print(f"评估总结 (对齐后):")
    print(f"平均 Chamfer Loss (MSE, 对应训练值): {avg_mse:.6f}")
    print(f"平均 物理 CD 误差 (Linear): {avg_linear:.6f} cm")
    print(f"平均 物理 CD 误差 (Linear): {avg_linear * 10:.2f} mm")
    print("-" * 40)
    print(f"验证: sqrt(Loss) = {np.sqrt(avg_mse):.4f} cm")
    print("注: 如果 Loss 接近 2.17，说明模型形状预测极其精准。")
    print("="*40)

if __name__ == "__main__":
    run_inference()