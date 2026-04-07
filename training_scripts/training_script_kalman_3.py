import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import glob
import re
from torch.utils.data import Dataset, DataLoader
import open3d as o3d
import trimesh

# =======================================================================
# 1. 模型架构 (严格遵循 Kalman + Residual 逻辑)
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
        # x: [B, N, 3] -> [B, 3, N]
        x = x.transpose(2, 1)
        x = self.mlp(x)
        x = torch.max(x, 2)[0]
        return self.fc(x)

class KalmanTracker(nn.Module):
    def __init__(self, template_path, latent_dim=64):
        super(KalmanTracker, self).__init__()
        
        # 加载模板
        mesh = trimesh.load(template_path, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = list(mesh.geometry.values())[0]
        
        pts = np.array(mesh.vertices, dtype=np.float32)
        # 模板也需要初始化在原点 (减去自身中心)
        self.v_mean = np.mean(pts, axis=0)
        pts_centered = pts - self.v_mean
        
        self.register_buffer('template', torch.from_numpy(pts_centered).float())
        self.num_pts = pts_centered.shape[0]

        self.obs_encoder = PointNetEncoder(latent_dim)
        self.gate_gen = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Sigmoid()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 3, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 3) 
        )

    def forward(self, obs, goal, h_t=None, h_prev=None):
        batch_size = obs.shape[0]
        f_obs = self.obs_encoder(obs)
        f_goal = self.obs_encoder(goal)
        
        combined_f = torch.cat([f_obs, f_goal], dim=-1)
        g_t = self.gate_gen(combined_f)
        
        if h_t is None: h_t = f_obs
        if h_prev is None: h_prev = f_obs
            
        h_next = g_t * h_t + (1 - g_t) * h_prev
        
        h_expand = h_next.unsqueeze(1).expand(-1, self.num_pts, -1)
        template_expand = self.template.unsqueeze(0).expand(batch_size, -1, -1)
        
        decoder_input = torch.cat([h_expand, template_expand], dim=-1)
        offsets = self.decoder(decoder_input.view(-1, decoder_input.shape[-1]))
        offsets = offsets.view(batch_size, self.num_pts, 3)
        
        p_hat = template_expand + offsets
        return p_hat, h_next, offsets

# =======================================================================
# 2. 损失函数
# =======================================================================

def chamfer_loss_precision(p1, p2, offsets):
    dist_sq = torch.cdist(p1, p2)**2
    loss_1 = torch.mean(torch.min(dist_sq, dim=2)[0])
    loss_2 = torch.mean(torch.min(dist_sq, dim=1)[0])
    reg_loss = torch.mean(offsets**2)
    return (loss_1 + loss_2), reg_loss

# =======================================================================
# 3. 数据集 (每一帧动态对齐几何中心)
# =======================================================================

class ArmDataset(Dataset):
    def __init__(self, root_dir):
        # 匹配 occ_frame_*.ply
        self.files = sorted(glob.glob(os.path.join(root_dir, "occ_frame_*.ply")),
                           key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group()))
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 加载当前帧点云
        pcd = o3d.io.read_point_cloud(self.files[idx])
        pts = np.asarray(pcd.points, dtype=np.float32)
        
        # 采样至固定点数
        if len(pts) >= 2048:
            idx_sample = np.random.choice(len(pts), 2048, replace=False)
            pts = pts[idx_sample]
        else:
            pts = np.tile(pts, (2048 // len(pts) + 1, 1))[:2048]

        # --- 核心修改：动态对齐几何中心 ---
        current_centroid = np.mean(pts, axis=0)
        pts_norm = pts - current_centroid # 强制当前帧中心为 (0,0,0)
        
        # 加载目标帧 (下一帧)
        goal_idx = min(idx + 1, len(self.files) - 1)
        pcd_g = o3d.io.read_point_cloud(self.files[goal_idx])
        pts_g = np.asarray(pcd_g.points, dtype=np.float32)
        if len(pts_g) >= 2048:
            pts_g = pts_g[np.random.choice(len(pts_g), 2048, replace=False)]
        else:
            pts_g = np.tile(pts_g, (2048 // len(pts_g) + 1, 1))[:2048]
            
        # 目标帧也减去当前帧的中心，保持它们在同一局部参考系下
        pts_g_norm = pts_g - current_centroid
        
        return torch.from_numpy(pts_norm), torch.from_numpy(pts_g_norm)

# =======================================================================
# 4. 训练主循环
# =======================================================================

def train():
    BASE = r"Cu_BEM_2\tshirt-data\tshirt_kalman_3"
    TEMPLATE_PATH = os.path.join(BASE, "tshirt_mech00001.obj")
    DATA_PATH = os.path.join(BASE, "dataset_arm")
    SAVE_PATH = "checkpoints/kalman_best_aligned.pth"
    
    os.makedirs("checkpoints", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = KalmanTracker(TEMPLATE_PATH).to(device)
    dataset = ArmDataset(DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    MAX_EPOCHS = 500
    SEQ_LEN = 5
    best_loss = float('inf')

    for epoch in range(MAX_EPOCHS):
        model.train()
        epoch_recon = 0
        num_iters = 0
        
        # 简单的滑动窗口训练逻辑
        for i in range(0, len(dataset) - SEQ_LEN, SEQ_LEN):
            h_t = None
            h_prev = None
            seq_loss_val = 0
            
            for t in range(SEQ_LEN):
                obs, goal = dataset[i+t]
                obs, goal = obs.unsqueeze(0).to(device), goal.unsqueeze(0).to(device)
                
                p_hat, h_next, offsets = model(obs, goal, h_t, h_prev)
                recon, reg = chamfer_loss_precision(p_hat, obs, offsets)
                
                total_loss = recon + 0.1 * reg
                seq_loss_val += total_loss
                
                h_prev = h_t.detach() if h_t is not None else None
                h_t = h_next
            
            optimizer.zero_grad()
            (seq_loss_val / SEQ_LEN).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            h_t = h_t.detach()
            epoch_recon += recon.item()
            num_iters += 1

        avg_loss = epoch_recon / num_iters if num_iters > 0 else 0
        scheduler.step(avg_loss)
        
        # 计算物理意义上的 RMS 误差 (单位与点云一致)
        physical_error = np.sqrt(avg_loss)
        print(f"Epoch [{epoch+1:03d}] CD Loss: {avg_loss:.6f} | RMS Error: {physical_error:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'loss': avg_loss
            }, SAVE_PATH)
            print(f"--> [SAVE] 模型已更新: {SAVE_PATH}")

if __name__ == "__main__":
    train()