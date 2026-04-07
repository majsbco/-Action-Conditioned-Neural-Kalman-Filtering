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
import trimesh

# =======================================================================
# 1. 简化模型架构 (移除动态模板，使用单一模板)
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

class GraspPointEncoder(nn.Module):
    """抓取点编码器"""
    def __init__(self, grasp_latent_dim=16):
        super(GraspPointEncoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, grasp_latent_dim)
        )
        
    def forward(self, grasp_points):
        if grasp_points.dim() == 1:
            grasp_points = grasp_points.unsqueeze(0)
        return self.mlp(grasp_points)

class FusionModule(nn.Module):
    """多模态融合模块"""
    def __init__(self, obs_latent_dim=64, grasp_latent_dim=16, fused_dim=64):
        super(FusionModule, self).__init__()
        self.fusion_proj = nn.Sequential(
            nn.Linear(obs_latent_dim + grasp_latent_dim, fused_dim),
            nn.ReLU(),
            nn.Linear(fused_dim, fused_dim)
        )
        
    def forward(self, z_obs, z_grasp):
        if z_obs.dim() == 1:
            z_obs = z_obs.unsqueeze(0)
        if z_grasp.dim() == 1:
            z_grasp = z_grasp.unsqueeze(0)
            
        z_fused = torch.cat([z_obs, z_grasp], dim=-1)
        return self.fusion_proj(z_fused)

class RecursiveTrackingNet(nn.Module):
    """
    简化版本：
    1. 移除动态模板切换，使用单一模板
    2. 状态转移方程：h_t = Φ(h_{t-1}, Fusion(z_obs, z_grasp))
    """
    def __init__(self, M=2048, obs_latent_dim=64, grasp_latent_dim=16, 
                 damping=0.95, template_path=None):
        super(RecursiveTrackingNet, self).__init__()
        self.obs_latent_dim = obs_latent_dim
        self.grasp_latent_dim = grasp_latent_dim
        self.M = M
        self.damping = damping 
        
        # 编码器
        self.enc_obs = PointNetEncoder(obs_latent_dim)
        self.enc_grasp = GraspPointEncoder(grasp_latent_dim)
        self.fusion = FusionModule(obs_latent_dim, grasp_latent_dim, obs_latent_dim)
        
        # 状态转移
        self.f_gru = nn.GRUCell(obs_latent_dim, obs_latent_dim)
        
        # 门控网络
        self.gate = nn.Sequential(
            nn.Linear(obs_latent_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # 单一模板
        print(f"[模型初始化] 加载单一模板...")
        if template_path and os.path.exists(template_path):
            self.template = self._load_template_from_obj(template_path, M)
            print(f"  -> 已加载模板从: {template_path}")
        else:
            self.template = self._generate_template_points(M)
            print(f"  -> 创建默认球面模板")
        
        # 注册为缓冲区
        self.register_buffer('template_tensor', self.template)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(3 + obs_latent_dim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 3)
        )

    def _generate_template_points(self, M):
        np.random.seed(42)
        coords = np.random.randn(M, 3)
        radii = np.linalg.norm(coords, axis=1, keepdims=True)
        return torch.from_numpy(coords / radii).float()

    def _load_template_from_obj(self, obj_path, target_num_points):
        mesh = trimesh.load(obj_path)
        vertices = mesh.vertices.astype(np.float32)
        
        if len(vertices) > target_num_points:
            template_points = self._farthest_point_sampling(vertices, target_num_points)
        else:
            template_points = vertices
            if len(template_points) < target_num_points:
                repeat_times = (target_num_points // len(template_points)) + 1
                template_points = np.tile(template_points, (repeat_times, 1))[:target_num_points]
        
        return torch.from_numpy(template_points).float()

    def _farthest_point_sampling(self, points, n_samples):
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

    def forward(self, o_t, g_t, h_prev, h_prev_prev=None):
        """
        前向传播
        
        Args:
            o_t: 观测点云 [B, M, 3]
            g_t: 抓取点坐标 [B, 3] 或 [3]
            h_prev: 前一时刻隐状态 [B, latent_dim]
            h_prev_prev: 前前时刻隐状态 [B, latent_dim]
            
        Returns:
            p_hat: 预测点云 [B, M, 3]
            h_next: 更新后的隐状态 [B, latent_dim]
        """
        B = o_t.size(0)
        
        # 编码
        z_obs = self.enc_obs(o_t)
        z_grasp = self.enc_grasp(g_t)
        z_fused = self.fusion(z_obs, z_grasp)
        
        # 惯性预测
        if h_prev_prev is not None:
            velocity = h_prev - h_prev_prev
            h_inertial = h_prev + self.damping * velocity
        else:
            h_inertial = h_prev
            
        # GRU状态转移
        h_gru = self.f_gru(z_fused, h_prev)
        
        # 门控融合
        gate_input = torch.cat([h_inertial, z_fused], dim=-1)
        alpha = self.gate(gate_input)
        h_next = (1 - alpha) * h_inertial + alpha * h_gru
        
        # 使用单一模板解码
        z_ext = h_next.unsqueeze(1).repeat(1, self.M, 1)
        template_ext = self.template_tensor.unsqueeze(0).repeat(B, 1, 1)
        
        p_hat = self.decoder(torch.cat([template_ext, z_ext], dim=-1))
        
        return p_hat, h_next

# =======================================================================
# 2. 损失函数与数据集
# =======================================================================

def chamfer_loss(pred, gt):
    dist = torch.cdist(pred, gt)
    d2gt = torch.min(dist, dim=2)[0]
    d2pred = torch.min(dist, dim=1)[0]
    return torch.mean(d2gt**2) + torch.mean(d2pred**2)

def smoothness_loss(h_current, h_previous):
    if h_previous is None:
        return torch.tensor(0.0, device=h_current.device)
    return torch.mean((h_current - h_previous)**2)

class GraspPointSequence:
    """抓取点序列管理器"""
    def __init__(self, grasp_file_path):
        self.times = []
        self.positions = []
        self._load_grasp_file(grasp_file_path)
        
    def _load_grasp_file(self, file_path):
        """更健壮的解析方法"""
        print(f"正在读取抓取点文件: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 使用正则表达式匹配所有 T= 和 X= 对
        # 改进的正则表达式模式，更精确地匹配格式
        pattern = r'T=\s*([\d\.]+)[\s\S]*?X=\s*([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)'
        matches = re.findall(pattern, content)
        
        print(f"正则表达式匹配到 {len(matches)} 个数据对")
        
        for match in matches:
            try:
                time = float(match[0])
                x = float(match[1])
                y = float(match[2])
                z = float(match[3])
                
                self.times.append(time)
                self.positions.append([x, y, z])
                
            except ValueError as e:
                print(f"警告: 无法解析匹配 {match}: {e}")
        
        print(f"成功加载抓取点序列: {len(self.positions)} 帧")
        
        if len(self.positions) == 0:
            # 尝试回退解析方法
            self._fallback_parse(content)
            
    def _fallback_parse(self, content):
        """回退解析方法，当正则表达式无法匹配时使用"""
        print("尝试回退解析方法...")
        
        lines = content.strip().split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # 跳过空行
            if not line:
                i += 1
                continue
                
            # 查找时间行
            if line.startswith('T='):
                try:
                    # 提取时间
                    time_str = line[2:].strip()  # 去掉'T='
                    time = float(time_str)
                    
                    # 查找下一个非空行，应该是坐标行
                    for j in range(i+1, len(lines)):
                        next_line = lines[j].strip()
                        if not next_line:
                            continue
                            
                        if next_line.startswith('X='):
                            # 提取坐标
                            coord_str = next_line[2:].strip()  # 去掉'X='
                            coords = list(map(float, coord_str.split()))
                            
                            if len(coords) == 3:
                                self.times.append(time)
                                self.positions.append(coords)
                                i = j  # 跳过坐标行
                                break
                            break
                        
                except ValueError as e:
                    print(f"警告: 解析行时出错: {line}")
                except IndexError as e:
                    print(f"警告: 索引错误: {e}")
            
            i += 1
        
        print(f"回退解析方法加载了 {len(self.positions)} 帧数据")
        
    def get_grasp_point_at_frame(self, frame_idx):
        if frame_idx < 0 or frame_idx >= len(self.positions):
            raise IndexError(f"帧索引 {frame_idx} 超出范围 [0, {len(self.positions)-1}]")
        return self.positions[frame_idx]

class PointcloudSeqDataset(Dataset):
    def __init__(self, folder, grasp_sequence, n_points=2048):
        raw_files = glob.glob(os.path.join(folder, "occ_frame_*.ply"))
        self.files = sorted(raw_files, 
                           key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group()) 
                           if re.search(r'\d+', os.path.basename(x)) else 0)
        self.n_points = n_points
        self.grasp_sequence = grasp_sequence
        
        # 对齐数据
        min_len = min(len(self.files), len(grasp_sequence.positions))
        self.files = self.files[:min_len]
        print(f"数据集大小: {min_len} 帧")
        
    def __len__(self): 
        return len(self.files)

    def __getitem__(self, idx):
        # 加载点云
        pcd = o3d.io.read_point_cloud(self.files[idx])
        pts = np.asarray(pcd.points, dtype=np.float32)
        if len(pts) == 0: 
            pts = np.zeros((self.n_points, 3), dtype=np.float32)
        else:
            choice = np.random.choice(len(pts), self.n_points, 
                                     replace=len(pts) < self.n_points)
            pts = pts[choice]
        
        # 获取对应的抓取点坐标
        grasp_pt = self.grasp_sequence.get_grasp_point_at_frame(idx)
        
        # 修复：正确处理列表类型的抓取点
        if isinstance(grasp_pt, list):
            # 如果是列表，先转换为numpy数组
            grasp_pt_np = np.array(grasp_pt, dtype=np.float32)
        elif hasattr(grasp_pt, 'astype'):
            # 如果是numpy数组，直接转换类型
            grasp_pt_np = grasp_pt.astype(np.float32)
        else:
            # 其他情况，尝试转换
            grasp_pt_np = np.array(grasp_pt, dtype=np.float32)
        
        return {
            'points': torch.from_numpy(pts),
            'grasp_point': torch.from_numpy(grasp_pt_np),  # 使用转换后的numpy数组
            'frame_idx': idx
        }

# =======================================================================
# 3. 训练控制逻辑
# =======================================================================

def main():
    # 路径配置
    DATA_PATH = r"Cu_BEM_2\tshirt-data\tshirt_kalman_3\dataset_arm"
    GRASP_FILE = r"Cu_BEM_2\tshirt-data\tshirt_kalman_3\vertex_237_trajectory.txt"
    TEMPLATE_PATH = r"Cu_BEM_2\tshirt-data\tshirt_kalman_3\tshirt_mech00001.obj"  # 单一模板
    SAVE_DIR = "checkpoints"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 训练参数
    SMOOTHNESS_WEIGHT = 0.5
    OBS_LATENT_DIM = 64
    GRASP_LATENT_DIM = 16
    SEQ_LEN = 10
    MAX_EPOCHS = 300
    LR = 1e-4
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载数据
    print("加载抓取点序列...")
    grasp_sequence = GraspPointSequence(GRASP_FILE)
    dataset = PointcloudSeqDataset(DATA_PATH, grasp_sequence)
    
    # 初始化模型
    model = RecursiveTrackingNet(
        obs_latent_dim=OBS_LATENT_DIM,
        grasp_latent_dim=GRASP_LATENT_DIM,
        template_path=TEMPLATE_PATH
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=15)
    
    best_loss = float('inf')
    start_epoch = 0

    checkpoint_path = os.path.join(SAVE_DIR, "best_model_simplified.pth")
    if os.path.exists(checkpoint_path):
        print("加载检查点...")
        ckpt = torch.load(checkpoint_path)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        best_loss = ckpt.get('loss', float('inf'))
        start_epoch = ckpt.get('epoch', 0)
        if start_epoch >= MAX_EPOCHS: 
            MAX_EPOCHS = start_epoch + 500
        print(f"继续从第 {start_epoch} 轮训练")
    else:
        print("开始新的训练")
    
    print(f"训练配置:")
    print(f"  观测潜空间维度: {OBS_LATENT_DIM}")
    print(f"  抓取点潜空间维度: {GRASP_LATENT_DIM}")
    print(f"  设备: {device}")
    print(f"  平滑损失权重: {SMOOTHNESS_WEIGHT}")
    print(f"  数据集大小: {len(dataset)} 帧")
    print(f"  模板: {TEMPLATE_PATH}")

    for epoch in range(start_epoch, MAX_EPOCHS):
        model.train()
        epoch_total_loss = 0
        epoch_recon_loss = 0
        epoch_smooth_loss = 0
        num_iters = 0
        
        # 初始化隐状态
        h_t = torch.zeros(1, OBS_LATENT_DIM).to(device)
        h_prev = None
        
        for i in range(0, len(dataset) - SEQ_LEN, SEQ_LEN):
            optimizer.zero_grad()
            seq_loss = 0
            seq_recon_loss = 0
            seq_smooth_loss = 0
            
            for t in range(SEQ_LEN):
                current_global_idx = i + t
                
                # 获取当前帧数据
                data = dataset[current_global_idx]
                o_t = data['points'].unsqueeze(0).to(device)
                g_t = data['grasp_point'].to(device)
                
                # 前向传播
                p_hat, h_next = model(o_t, g_t, h_t, h_prev)
                
                # 计算损失
                recon_loss = chamfer_loss(p_hat, o_t)
                smooth_loss = smoothness_loss(h_next, h_prev)
                total_loss = recon_loss + SMOOTHNESS_WEIGHT * smooth_loss
                
                # 累加损失
                seq_loss += total_loss
                seq_recon_loss += recon_loss.item()
                seq_smooth_loss += smooth_loss.item() if isinstance(smooth_loss, torch.Tensor) else 0.0
                
                # 记录状态
                h_prev = h_t.detach()
                h_t = h_next
            
            # 反向传播
            (seq_loss / SEQ_LEN).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # 截断反向传播
            h_t = h_t.detach()
            h_prev = h_prev.detach() if h_prev is not None else None
            
            # 统计
            epoch_total_loss += seq_loss.item()
            epoch_recon_loss += seq_recon_loss
            epoch_smooth_loss += seq_smooth_loss
            num_iters += 1

        # 计算平均损失
        avg_epoch_loss = epoch_total_loss / (num_iters * SEQ_LEN) if num_iters > 0 else 0
        avg_recon_loss = epoch_recon_loss / (num_iters * SEQ_LEN) if num_iters > 0 else 0
        avg_smooth_loss = epoch_smooth_loss / (num_iters * SEQ_LEN) if num_iters > 0 else 0
        
        scheduler.step(avg_epoch_loss)
        
        # 打印统计
        print(f"Epoch [{epoch+1:04d}/{MAX_EPOCHS}] "
              f"总损失: {avg_epoch_loss:.6f}, "
              f"重建损失: {avg_recon_loss:.6f}, "
              f"平滑损失: {avg_smooth_loss:.6f}")
        
        # 保存最佳模型
        is_best = avg_epoch_loss < best_loss
        if is_best:
            best_loss = avg_epoch_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'template_path': TEMPLATE_PATH,
            }, checkpoint_path)
            print(f"  -> 保存最佳模型至 {checkpoint_path}")

if __name__ == "__main__":
    main()