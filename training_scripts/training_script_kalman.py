import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import glob
import re
from torch.utils.data import Dataset, DataLoader
import open3d as o3d
from scipy.spatial import cKDTree
import trimesh  # 用于加载OBJ文件

# =======================================================================
# 1. 简化模型架构 (使用单一模板)
# =======================================================================

class PointNetEncoder(nn.Module):
    """观测点云编码器"""
    def __init__(self, latent_dim=64):
        super(PointNetEncoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, 512, 1), nn.BatchNorm1d(512), nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(512, 256), 
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

    def forward(self, x):
        x = x.transpose(2, 1)
        x = self.mlp(x)
        x_max, _ = torch.max(x, dim=2)
        return self.fc(x_max)

class RecursiveTrackingNet(nn.Module):
    """
    简化版本：使用单一模板
    """
    def __init__(self, M=2048, latent_dim=64, damping=0.95, template_path=None):
        """
        Args:
            template_path (str): 单一模板文件路径
        """
        super(RecursiveTrackingNet, self).__init__()
        self.latent_dim = latent_dim
        self.M = M
        self.damping = damping 
        
        self.enc_obs = PointNetEncoder(latent_dim)
        self.f_gru = nn.GRUCell(latent_dim, latent_dim)
        
        # 门控网络
        self.gate = nn.Sequential(
            nn.Linear(latent_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # --- 核心修改：加载单一模板 ---
        print(f"[模型初始化] 加载单一模板...")
        if template_path and os.path.exists(template_path):
            self.template = self._load_template_from_obj(template_path, M)
            print(f"  -> 已加载模板从: {template_path}")
        else:
            # 如果没有提供模板路径，使用默认的球面模板
            self.template = self._generate_template_points(M)
            print(f"  -> 创建默认球面模板")
        
        # 注册为缓冲区
        self.register_buffer('template_tensor', self.template)
        # --- 核心修改结束 ---
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(3 + latent_dim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 3)
        )

    def _generate_template_points(self, M):
        """备用的默认球面采样方法"""
        np.random.seed(42)
        coords = np.random.randn(M, 3)
        radii = np.linalg.norm(coords, axis=1, keepdims=True)
        return torch.from_numpy(coords / radii).float()

    def _load_template_from_obj(self, obj_path, target_num_points):
        """
        从OBJ文件加载网格顶点，并采样得到固定数量的模板点
        """
        # 1. 使用trimesh加载OBJ网格文件
        mesh = trimesh.load(obj_path)
        vertices = mesh.vertices.astype(np.float32)
        print(f"  -> 已加载顶点数: {len(vertices)}")

        # 2. 采样到目标点数
        if len(vertices) > target_num_points:
            # 使用最远点采样(FPS)获取分布均匀的特征点子集
            print(f"  -> 使用最远点采样(FPS)从 {len(vertices)} 个顶点中采样 {target_num_points} 个点...")
            template_points = self._farthest_point_sampling(vertices, target_num_points)
        else:
            # 如果网格顶点数不足，则直接使用所有顶点，并通过重复来补足数量
            template_points = vertices
            if len(template_points) < target_num_points:
                repeat_times = (target_num_points // len(template_points)) + 1
                template_points = np.tile(template_points, (repeat_times, 1))[:target_num_points]
            print(f"  -> 顶点数不足，通过重复使用顶点补足至 {len(template_points)} 个点。")
        
        print(f"  -> 最终模板点云形状: {template_points.shape}")
        return torch.from_numpy(template_points).float()

    def _farthest_point_sampling(self, points, n_samples):
        """最远点采样 (Farthest Point Sampling, FPS) 实现"""
        n_points = points.shape[0]
        sampled_indices = np.zeros(n_samples, dtype=np.int64)
        distances = np.full(n_points, np.inf)
        
        first_idx = np.random.randint(n_points)
        sampled_indices[0] = first_idx
        
        for i in range(1, n_samples):
            last_selected = points[sampled_indices[i-1]]
            dist_to_last = np.linalg.norm(points - last_selected, axis=1)
            distances = np.minimum(distances, dist_to_last)
            sampled_indices[i] = np.argmax(distances)
        
        return points[sampled_indices]

    def forward(self, o_t, h_prev, h_prev_prev=None):
        """
        前向传播（简化：移除帧索引参数，使用单一模板）
        """
        B = o_t.size(0)
        z_t = self.enc_obs(o_t)
        
        # 惯性预测
        if h_prev_prev is not None:
            velocity = h_prev - h_prev_prev
            h_inertial = h_prev + self.damping * velocity
        else:
            h_inertial = h_prev
            
        # 序列更新 (GRU滤波)
        h_gru = self.f_gru(z_t, h_prev)
        
        # 门控
        gate_input = torch.cat([h_inertial, z_t], dim=-1)
        alpha = self.gate(gate_input)
        h_t = (1 - alpha) * h_inertial + alpha * h_gru
        
        # --- 核心修改：直接使用单一模板 ---
        z_ext = h_t.unsqueeze(1).repeat(1, self.M, 1)
        template_ext = self.template_tensor.unsqueeze(0).repeat(B, 1, 1)
        
        p_hat = self.decoder(torch.cat([template_ext, z_ext], dim=-1))
        
        # 返回预测点云和更新后的隐状态
        return p_hat, h_t

# =======================================================================
# 2. 损失函数与数据集
# =======================================================================

def chamfer_loss(pred, gt):
    """倒角距离损失函数"""
    dist = torch.cdist(pred, gt)
    d2gt = torch.min(dist, dim=2)[0]
    d2pred = torch.min(dist, dim=1)[0]
    return torch.mean(d2gt**2) + torch.mean(d2pred**2)

def smoothness_loss(h_current, h_previous):
    """
    平滑度损失函数 - 惩罚相邻帧潜状态之间的剧烈变化
    """
    if h_previous is None:
        return torch.tensor(0.0, device=h_current.device)
    return torch.mean((h_current - h_previous)**2)

class PointcloudSeqDataset(Dataset):
    def __init__(self, folder, n_points=2048):
        raw_files = glob.glob(os.path.join(folder, "occ_frame_*.ply"))
        self.files = sorted(raw_files, 
                           key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group()) 
                           if re.search(r'\d+', os.path.basename(x)) else 0)
        self.n_points = n_points

    def __len__(self): 
        return len(self.files)

    def __getitem__(self, idx):
        pcd = o3d.io.read_point_cloud(self.files[idx])
        pts = np.asarray(pcd.points, dtype=np.float32)
        if len(pts) == 0: 
            return torch.zeros((self.n_points, 3))
        choice = np.random.choice(len(pts), self.n_points, replace=len(pts) < self.n_points)
        return torch.from_numpy(pts[choice])

# =======================================================================
# 3. 训练控制逻辑
# =======================================================================

def main():
    DATA_PATH = r"Cu_BEM_2\tshirt-data\tshirt_kalman_3\dataset_arm"
    SAVE_DIR = "checkpoints"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # --- 核心修改：定义单一模板路径 ---
    TEMPLATE_PATH = r"D:\Cu_BEM_2\tshirt-data\tshirt_kalman_3\tshirt_mech00001.obj"
    
    # 训练参数
    SMOOTHNESS_WEIGHT = 0.5  
    print(f"平滑损失权重设置为: λ_smooth = {SMOOTHNESS_WEIGHT}")
    
    LATENT_DIM = 64
    SEQ_LEN = 10
    MAX_EPOCHS = 500
    LR = 1e-4
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 修改：初始化模型时传入单一模板路径 ---
    model = RecursiveTrackingNet(latent_dim=LATENT_DIM, template_path=TEMPLATE_PATH).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=15)
    
    dataset = PointcloudSeqDataset(DATA_PATH)
    best_loss = float('inf')
    start_epoch = 0

    checkpoint_path = os.path.join(SAVE_DIR, "best_model_single_template.pth")
    if os.path.exists(checkpoint_path):
        print(f"加载检查点 (使用单一模板)...")
        ckpt = torch.load(checkpoint_path)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        best_loss = ckpt.get('loss', float('inf'))
        start_epoch = ckpt.get('epoch', 0)
        if start_epoch >= MAX_EPOCHS: 
            MAX_EPOCHS = start_epoch + 500
        print(f"  继续从第 {start_epoch} 轮训练。")
    else:
        print("未找到现有检查点，将开始新的训练。")
    
    print(f"开始训练: 潜空间维度={LATENT_DIM}, 设备={device}")
    print(f"使用单一模板: {TEMPLATE_PATH}")
    print(f"平滑损失权重: {SMOOTHNESS_WEIGHT}")

    for epoch in range(start_epoch, MAX_EPOCHS):
        model.train()
        epoch_total_loss = 0
        epoch_recon_loss = 0
        epoch_smooth_loss = 0
        num_iters = 0
        
        # 每一轮初始化隐状态
        h_t = torch.zeros(1, LATENT_DIM).to(device)
        h_prev = None
        
        for i in range(0, len(dataset) - SEQ_LEN, SEQ_LEN):
            optimizer.zero_grad()
            seq_loss = 0
            seq_recon_loss = 0
            seq_smooth_loss = 0
            
            # 时间步循环
            for t in range(SEQ_LEN):
                current_global_idx = i + t
                o_t = dataset[current_global_idx].unsqueeze(0).to(device)
                
                # --- 修改：前向传播不再传入帧索引参数 ---
                p_hat, h_next = model(o_t, h_t, h_prev)
                
                # 计算组合损失
                recon_loss = chamfer_loss(p_hat, o_t)
                smooth_loss = smoothness_loss(h_next, h_prev)
                total_loss = recon_loss + SMOOTHNESS_WEIGHT * smooth_loss
                
                # 累加各项损失用于统计
                seq_loss += total_loss
                seq_recon_loss += recon_loss.item()
                seq_smooth_loss += smooth_loss.item() if isinstance(smooth_loss, torch.Tensor) else 0.0
                
                # 每隔一定帧数，打印损失详情
                if current_global_idx % 200 == 0:
                    print(f"  帧 {current_global_idx}: "
                          f"重建损失: {recon_loss.item():.6f}, "
                          f"平滑损失: {smooth_loss.item() if isinstance(smooth_loss, torch.Tensor) else 0.0:.6f}, "
                          f"总损失: {total_loss.item():.6f}")
                
                # 记录状态
                h_prev = h_t.detach()
                h_t = h_next
            
            # 反向传播
            (seq_loss / SEQ_LEN).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # 截断反向传播，保持隐状态
            h_t = h_t.detach()
            h_prev = h_prev.detach() if h_prev is not None else None
            
            # 累加epoch统计
            epoch_total_loss += seq_loss.item()
            epoch_recon_loss += seq_recon_loss
            epoch_smooth_loss += seq_smooth_loss
            num_iters += 1

        # 计算平均损失
        avg_epoch_loss = epoch_total_loss / (num_iters * SEQ_LEN) if num_iters > 0 else 0
        avg_recon_loss = epoch_recon_loss / (num_iters * SEQ_LEN) if num_iters > 0 else 0
        avg_smooth_loss = epoch_smooth_loss / (num_iters * SEQ_LEN) if num_iters > 0 else 0
        
        scheduler.step(avg_epoch_loss)
        
        # 打印详细的epoch统计
        print(f"Epoch [{epoch+1:04d}/{MAX_EPOCHS}] "
              f"总损失: {avg_epoch_loss:.6f}, "
              f"重建损失: {avg_recon_loss:.6f}, "
              f"平滑损失: {avg_smooth_loss:.6f}")
        
        is_best = avg_epoch_loss < best_loss
        if is_best:
            best_loss = avg_epoch_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'template_path': TEMPLATE_PATH,  # 保存单一模板路径
                'smoothness_weight': SMOOTHNESS_WEIGHT,
            }, checkpoint_path)
            print(f"  -> 保存最佳模型至 {checkpoint_path}")
            
        print(f"Epoch [{epoch+1:04d}/{MAX_EPOCHS}] 平均损失: {avg_epoch_loss:.6f} {'[BEST]' if is_best else ''}")

if __name__ == "__main__":
    main()