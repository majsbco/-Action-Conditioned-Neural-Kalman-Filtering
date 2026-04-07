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
import trimesh
import sys

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# =======================================================================
# 1. 模型定义 (包含两种版本的模型)
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
    """抓取点编码器（仅带抓取点模型使用）"""
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
    """多模态融合模块（仅带抓取点模型使用）"""
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

# =======================================================================
# 2. 不带抓取点的模型（版本1）
# =======================================================================

class RecursiveTrackingNetWithoutGrasp(nn.Module):
    """
    不带抓取点输入的模型
    状态转移方程：h_t = Φ(h_{t-1}, z_obs)
    """
    def __init__(self, M=2048, latent_dim=64, damping=0.95, template_path=None):
        super(RecursiveTrackingNetWithoutGrasp, self).__init__()
        self.latent_dim = latent_dim
        self.M = M
        self.damping = damping
        
        # 编码器
        self.enc_obs = PointNetEncoder(latent_dim)
        
        # 状态转移
        self.f_gru = nn.GRUCell(latent_dim, latent_dim)
        
        # 门控网络
        self.gate = nn.Sequential(
            nn.Linear(latent_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # 单一模板
        if template_path and os.path.exists(template_path):
            self.template = self._load_template_from_obj(template_path, M)
        else:
            self.template = self._generate_template_points(M)
        
        # 注册为缓冲区
        self.register_buffer('template_tensor', self.template)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(3 + latent_dim, 512), nn.ReLU(),
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

    def forward(self, o_t, h_prev, h_prev_prev=None):
        """
        前向传播（不带抓取点输入）
        
        Args:
            o_t: 观测点云 [B, M, 3]
            h_prev: 前一时刻隐状态 [B, latent_dim]
            h_prev_prev: 前前时刻隐状态 [B, latent_dim]
            
        Returns:
            p_hat: 预测点云 [B, M, 3]
            h_next: 更新后的隐状态 [B, latent_dim]
            z_obs: 观测编码 [B, latent_dim]
        """
        B = o_t.size(0)
        
        # 编码
        z_obs = self.enc_obs(o_t)  # 观测编码
        
        # 惯性预测
        if h_prev_prev is not None:
            velocity = h_prev - h_prev_prev
            h_inertial = h_prev + self.damping * velocity
        else:
            h_inertial = h_prev
            
        # GRU状态转移
        h_gru = self.f_gru(z_obs, h_prev)
        
        # 门控融合
        gate_input = torch.cat([h_inertial, z_obs], dim=-1)
        alpha = self.gate(gate_input)
        h_next = (1 - alpha) * h_inertial + alpha * h_gru
        
        # 使用单一模板解码
        z_ext = h_next.unsqueeze(1).repeat(1, self.M, 1)
        template_ext = self.template_tensor.unsqueeze(0).repeat(B, 1, 1)
        
        p_hat = self.decoder(torch.cat([template_ext, z_ext], dim=-1))
        
        return p_hat, h_next, z_obs

# =======================================================================
# 3. 带抓取点的模型（版本2）
# =======================================================================

class RecursiveTrackingNetWithGrasp(nn.Module):
    """
    带抓取点输入的模型
    状态转移方程：h_t = Φ(h_{t-1}, Fusion(z_obs, z_grasp))
    """
    def __init__(self, M=2048, obs_latent_dim=64, grasp_latent_dim=16, 
                 damping=0.95, template_path=None):
        super(RecursiveTrackingNetWithGrasp, self).__init__()
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
        if template_path and os.path.exists(template_path):
            self.template = self._load_template_from_obj(template_path, M)
        else:
            self.template = self._generate_template_points(M)
        
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
        前向传播（带抓取点输入）
        
        Args:
            o_t: 观测点云 [B, M, 3]
            g_t: 抓取点坐标 [B, 3] 或 [3]
            h_prev: 前一时刻隐状态 [B, latent_dim]
            h_prev_prev: 前前时刻隐状态 [B, latent_dim]
            
        Returns:
            p_hat: 预测点云 [B, M, 3]
            h_next: 更新后的隐状态 [B, latent_dim]
            z_obs: 观测编码 [B, latent_dim]
        """
        B = o_t.size(0)
        
        # 编码
        z_obs = self.enc_obs(o_t)  # 观测编码
        z_grasp = self.enc_grasp(g_t)  # 抓取点编码
        z_fused = self.fusion(z_obs, z_grasp)  # 融合编码
        
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
        
        return p_hat, h_next, z_obs

# =======================================================================
# 4. 抓取点序列管理器
# =======================================================================

class GraspPointSequence:
    """抓取点序列管理器"""
    def __init__(self, grasp_file_path):
        self.times = []
        self.positions = []
        self._load_grasp_file(grasp_file_path)
        
    def _load_grasp_file(self, file_path):
        """读取抓取点轨迹文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 使用正则表达式匹配所有 T= 和 X= 对
        pattern = r'T=\s*([\d\.]+)[\s\S]*?X=\s*([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)'
        matches = re.findall(pattern, content)
        
        for match in matches:
            try:
                time = float(match[0])
                x = float(match[1])
                y = float(match[2])
                z = float(match[3])
                
                self.times.append(time)
                self.positions.append([x, y, z])
                
            except ValueError as e:
                pass
        
        if len(self.positions) == 0:
            # 尝试回退解析方法
            self._fallback_parse(content)
            
    def _fallback_parse(self, content):
        """回退解析方法，当正则表达式无法匹配时使用"""
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
                    pass
                except IndexError as e:
                    pass
            
            i += 1
        
    def get_grasp_point_at_frame(self, frame_idx):
        if frame_idx < 0 or frame_idx >= len(self.positions):
            raise IndexError(f"帧索引 {frame_idx} 超出范围 [0, {len(self.positions)-1}]")
        return self.positions[frame_idx]

# =======================================================================
# 5. 数据加载
# =======================================================================

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
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        pcd = o3d.io.read_point_cloud(self.files[idx])
        pts = np.asarray(pcd.points, dtype=np.float32)
        if len(pts) == 0:
            pts = np.zeros((self.n_points, 3), dtype=np.float32)
        else:
            choice = np.random.choice(len(pts), self.n_points, replace=len(pts) < self.n_points)
            pts = pts[choice]
        
        # 获取对应的抓取点坐标
        grasp_pt = self.grasp_sequence.get_grasp_point_at_frame(idx)
        
        return {
            'points': torch.from_numpy(pts),
            'grasp_point': torch.tensor(grasp_pt, dtype=torch.float32),
            'filename': self.files[idx]
        }

# =======================================================================
# 6. 对比评估函数
# =======================================================================

def compare_jitter_between_models():
    """
    对比不带抓取点和带抓取点模型的抖动值效果
    """
    # 配置参数
    DATA_PATH = r"D:\Cu_BEM_2\tshirt-data\tshirt_kalman_3\dataset_arm"
    GRASP_FILE = r"D:\Cu_BEM_2\tshirt-data\tshirt_kalman_3\vertex_237_trajectory.txt"
    
    # 模型checkpoint路径 - 根据您的训练脚本调整
    MODEL_WITHOUT_GRASP = "checkpoints/best_model_single_template.pth"  # 不带抓取点的模型
    MODEL_WITH_GRASP = "checkpoints/best_model_simplified.pth"  # 带抓取点的模型
    
    # 模板路径
    TEMPLATE_PATH = r"D:\Cu_BEM_2\tshirt-data\tshirt_kalman_3\tshirt_mech00001.obj"
    
    OBS_LATENT_DIM = 64
    GRASP_LATENT_DIM = 16
    MAX_PC_COMPONENTS = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print("="*60)
    print("对比：不带抓取点 vs 带抓取点模型的抖动值效果")
    print("="*60)
    
    # 加载抓取点序列
    print("1. 加载抓取点序列...")
    grasp_sequence = GraspPointSequence(GRASP_FILE)
    
    # 加载点云数据集
    print("\n2. 加载点云数据集...")
    dataset = PointcloudSeqDataset(DATA_PATH, grasp_sequence)
    all_frames = []
    all_grasp_points = []
    
    for i in range(len(dataset)):
        data = dataset[i]
        all_frames.append(data['points'].unsqueeze(0).to(device))
        all_grasp_points.append(data['grasp_point'].to(device))
    
    T = len(all_frames)
    print(f"   已加载 {T} 帧点云数据和抓取点序列")
    
    # =======================================================================
    # 加载不带抓取点的模型
    # =======================================================================
    print("\n3. 加载不带抓取点的模型...")
    if os.path.exists(MODEL_WITHOUT_GRASP):
        checkpoint_without = torch.load(MODEL_WITHOUT_GRASP, map_location=device)
        state_dict_without = checkpoint_without['model_state_dict']
        
        # 初始化模型
        model_without = RecursiveTrackingNetWithoutGrasp(
            latent_dim=OBS_LATENT_DIM,
            template_path=TEMPLATE_PATH
        ).to(device)
        
        # 修复状态字典
        keys_to_remove = []
        for key in state_dict_without.keys():
            if 'current_template' in key or ('template' in key and 'templates' not in key):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            if key in state_dict_without:
                del state_dict_without[key]
        
        # 加载状态字典
        model_without.load_state_dict(state_dict_without, strict=False)
        model_without.eval()
        print(f"   已加载不带抓取点模型: {MODEL_WITHOUT_GRASP}")
    else:
        print(f"   错误: 不带抓取点的模型文件不存在: {MODEL_WITHOUT_GRASP}")
        return
    
    # =======================================================================
    # 加载带抓取点的模型
    # =======================================================================
    print("\n4. 加载带抓取点的模型...")
    if os.path.exists(MODEL_WITH_GRASP):
        checkpoint_with = torch.load(MODEL_WITH_GRASP, map_location=device)
        state_dict_with = checkpoint_with['model_state_dict']
        
        # 初始化模型
        model_with = RecursiveTrackingNetWithGrasp(
            obs_latent_dim=OBS_LATENT_DIM,
            grasp_latent_dim=GRASP_LATENT_DIM,
            template_path=TEMPLATE_PATH
        ).to(device)
        
        # 修复状态字典
        keys_to_remove = []
        for key in state_dict_with.keys():
            if 'current_template' in key or ('template' in key and 'templates' not in key):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            if key in state_dict_with:
                del state_dict_with[key]
        
        # 加载状态字典
        model_with.load_state_dict(state_dict_with, strict=False)
        model_with.eval()
        print(f"   已加载带抓取点模型: {MODEL_WITH_GRASP}")
    else:
        print(f"   错误: 带抓取点的模型文件不存在: {MODEL_WITH_GRASP}")
        return
    
    # 存储结果
    results = {
        'without_grasp': {'Z_filtered': [], 'Z_obs': []},  # 不带抓取点
        'with_grasp': {'Z_filtered': [], 'Z_obs': []}      # 带抓取点
    }
    
    # 初始化模型状态
    h_t_without = torch.zeros(1, OBS_LATENT_DIM).to(device)
    h_prev_without = None
    
    h_t_with = torch.zeros(1, OBS_LATENT_DIM).to(device)
    h_prev_with = None
    
    print("\n5. 正在提取潜向量并进行滤波...")
    with torch.no_grad():
        for t in range(min(T, 500)):  # 只处理前500帧以加快速度
            o_t = all_frames[t]
            g_t = all_grasp_points[t]
            
            # 不带抓取点的模型推理
            _, h_next_without, z_obs_without = model_without(o_t, h_t_without, h_prev_without)
            h_next_without_np = h_next_without.squeeze(0).cpu().numpy()
            z_obs_without_np = z_obs_without.squeeze(0).cpu().numpy()
            
            results['without_grasp']['Z_filtered'].append(h_next_without_np)
            results['without_grasp']['Z_obs'].append(z_obs_without_np)
            
            # 带抓取点的模型推理
            _, h_next_with, z_obs_with = model_with(o_t, g_t, h_t_with, h_prev_with)
            h_next_with_np = h_next_with.squeeze(0).cpu().numpy()
            z_obs_with_np = z_obs_with.squeeze(0).cpu().numpy()
            
            results['with_grasp']['Z_filtered'].append(h_next_with_np)
            results['with_grasp']['Z_obs'].append(z_obs_with_np)
            
            # 更新状态
            h_prev_without = h_t_without.detach()
            h_t_without = h_next_without
            
            h_prev_with = h_t_with.detach()
            h_t_with = h_next_with
            
            if (t+1) % 100 == 0 or (t+1) == min(T, 500):
                print(f"   进度: {t+1}/{min(T, 500)}")
    
    # 转换为numpy数组
    for method in ['without_grasp', 'with_grasp']:
        results[method]['Z_filtered'] = np.array(results[method]['Z_filtered'])
        results[method]['Z_obs'] = np.array(results[method]['Z_obs'])
    
    # 基于观测数据拟合PCA
    print("\n6. 进行PCA分析...")
    all_obs = np.vstack([results['without_grasp']['Z_obs'], results['with_grasp']['Z_obs']])
    pca = PCA(n_components=MAX_PC_COMPONENTS)
    pca.fit(all_obs)
    
    # 对滤波后的数据进行PCA变换
    for method in ['without_grasp', 'with_grasp']:
        results[method]['Z_filtered_pca'] = pca.transform(results[method]['Z_filtered'])
        results[method]['Z_obs_pca'] = pca.transform(results[method]['Z_obs'])
    
    # 计算抖动值
    print("\n7. 计算抖动值...")
    jitter_results = {}
    
    for method in ['without_grasp', 'with_grasp']:
        jitter_results[method] = {
            'obs_jitter': [],      # 观测序列的抖动
            'filtered_jitter': [], # 滤波后序列的抖动
            'obs_velocity_std': 0, # 观测序列速度标准差
            'filtered_velocity_std': 0, # 滤波后序列速度标准差
            'obs_max_velocity': 0, # 观测序列最大速度
            'filtered_max_velocity': 0, # 滤波后序列最大速度
        }
        
        # 计算每个主成分的抖动
        for i in range(MAX_PC_COMPONENTS):
            # 观测序列抖动
            obs_jitter = np.mean(np.abs(np.diff(results[method]['Z_obs_pca'][:, i], n=2)))
            # 滤波后序列抖动
            filtered_jitter = np.mean(np.abs(np.diff(results[method]['Z_filtered_pca'][:, i], n=2)))
            
            jitter_results[method]['obs_jitter'].append(obs_jitter)
            jitter_results[method]['filtered_jitter'].append(filtered_jitter)
        
        # 计算速度统计
        # 观测序列速度
        obs_velocity = np.diff(results[method]['Z_obs_pca'], axis=0)
        jitter_results[method]['obs_velocity_std'] = np.std(obs_velocity)
        jitter_results[method]['obs_max_velocity'] = np.max(np.abs(obs_velocity))
        
        # 滤波后序列速度
        filtered_velocity = np.diff(results[method]['Z_filtered_pca'], axis=0)
        jitter_results[method]['filtered_velocity_std'] = np.std(filtered_velocity)
        jitter_results[method]['filtered_max_velocity'] = np.max(np.abs(filtered_velocity))
    
    # 打印汇总统计
    print("\n" + "="*60)
    print("抖动值对比统计")
    print("="*60)
    
    print(f"\n全局平均抖动值:")
    print(f"  不带抓取点模型 - 观测: {np.mean(jitter_results['without_grasp']['obs_jitter']):.6f}")
    print(f"  不带抓取点模型 - 滤波后: {np.mean(jitter_results['without_grasp']['filtered_jitter']):.6f}")
    print(f"  带抓取点模型 - 观测: {np.mean(jitter_results['with_grasp']['obs_jitter']):.6f}")
    print(f"  带抓取点模型 - 滤波后: {np.mean(jitter_results['with_grasp']['filtered_jitter']):.6f}")
    
    print(f"\n速度标准差:")
    print(f"  不带抓取点模型 - 观测: {jitter_results['without_grasp']['obs_velocity_std']:.6f}")
    print(f"  不带抓取点模型 - 滤波后: {jitter_results['without_grasp']['filtered_velocity_std']:.6f}")
    print(f"  带抓取点模型 - 观测: {jitter_results['with_grasp']['obs_velocity_std']:.6f}")
    print(f"  带抓取点模型 - 滤波后: {jitter_results['with_grasp']['filtered_velocity_std']:.6f}")
    
    print(f"\n最大绝对速度:")
    print(f"  不带抓取点模型 - 观测: {jitter_results['without_grasp']['obs_max_velocity']:.6f}")
    print(f"  不带抓取点模型 - 滤波后: {jitter_results['without_grasp']['filtered_max_velocity']:.6f}")
    print(f"  带抓取点模型 - 观测: {jitter_results['with_grasp']['obs_max_velocity']:.6f}")
    print(f"  带抓取点模型 - 滤波后: {jitter_results['with_grasp']['filtered_max_velocity']:.6f}")
    
    # 计算改善率
    without_filtered_jitter = np.mean(jitter_results['without_grasp']['filtered_jitter'])
    with_filtered_jitter = np.mean(jitter_results['with_grasp']['filtered_jitter'])
    
    jitter_improvement = (without_filtered_jitter - with_filtered_jitter) / without_filtered_jitter * 100
    
    print(f"\n抖动改善率:")
    print(f"  带抓取点模型相比不带抓取点模型的抖动改善: {jitter_improvement:.1f}%")
    
    # 可视化对比
    print("\n8. 生成可视化图表...")
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 子图1: 抖动值对比柱状图
    x = np.arange(1, MAX_PC_COMPONENTS + 1)
    width = 0.35
    
    axes[0, 0].bar(x - width/2, jitter_results['without_grasp']['filtered_jitter'], width, 
                   label='不带抓取点', color='orange', alpha=0.8)
    axes[0, 0].bar(x + width/2, jitter_results['with_grasp']['filtered_jitter'], width, 
                   label='带抓取点', color='dodgerblue', alpha=0.8)
    
    axes[0, 0].set_title('各主成分抖动值对比 (滤波后)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('主成分索引', fontsize=10)
    axes[0, 0].set_ylabel('抖动值', fontsize=10)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([f'PC{i}' for i in x])
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # 在柱状图上添加改善率标签
    for i in range(MAX_PC_COMPONENTS):
        improvement = (jitter_results['without_grasp']['filtered_jitter'][i] - 
                      jitter_results['with_grasp']['filtered_jitter'][i]) / \
                      jitter_results['without_grasp']['filtered_jitter'][i] * 100
        if improvement > 0:
            axes[0, 0].text(x[i] + width/2, 
                          jitter_results['with_grasp']['filtered_jitter'][i] + 0.001, 
                          f'+{improvement:.0f}%', ha='center', va='bottom', 
                          fontsize=8, color='green')
    
    # 子图2: 速度统计对比
    categories = ['速度标准差', '最大绝对速度']
    without_values = [
        jitter_results['without_grasp']['filtered_velocity_std'],
        jitter_results['without_grasp']['filtered_max_velocity']
    ]
    with_values = [
        jitter_results['with_grasp']['filtered_velocity_std'],
        jitter_results['with_grasp']['filtered_max_velocity']
    ]
    
    x_pos = np.arange(len(categories))
    axes[0, 1].bar(x_pos - 0.2, without_values, 0.4, label='不带抓取点', color='orange', alpha=0.8)
    axes[0, 1].bar(x_pos + 0.2, with_values, 0.4, label='带抓取点', color='dodgerblue', alpha=0.8)
    
    axes[0, 1].set_title('速度统计对比 (滤波后)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('统计指标', fontsize=10)
    axes[0, 1].set_ylabel('值', fontsize=10)
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(categories)
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # 子图3: 前几个主成分的时间序列
    axes[1, 0].plot(results['without_grasp']['Z_filtered_pca'][:, 0], 
                   label='不带抓取点 (PC1)', color='orange', alpha=0.8)
    axes[1, 0].plot(results['with_grasp']['Z_filtered_pca'][:, 0], 
                   label='带抓取点 (PC1)', color='dodgerblue', alpha=0.8)
    axes[1, 0].set_title('第一主成分时间序列对比 (滤波后)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('时间帧', fontsize=10)
    axes[1, 0].set_ylabel('PC1值', fontsize=10)
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # 子图4: 第二主成分的时间序列
    axes[1, 1].plot(results['without_grasp']['Z_filtered_pca'][:, 1], 
                   label='不带抓取点 (PC2)', color='orange', alpha=0.8)
    axes[1, 1].plot(results['with_grasp']['Z_filtered_pca'][:, 1], 
                   label='带抓取点 (PC2)', color='dodgerblue', alpha=0.8)
    axes[1, 1].set_title('第二主成分时间序列对比 (滤波后)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('时间帧', fontsize=10)
    axes[1, 1].set_ylabel('PC2值', fontsize=10)
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    # 设置总标题
    plt.suptitle(f'不带抓取点 vs 带抓取点模型抖动值对比\n抖动改善率: {jitter_improvement:.1f}%', 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    
    # 保存图表
    save_path = f'jitter_comparison_both_models.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n对比图表已保存至: {save_path}")
    plt.show()
    
    # 打印详细评估结论
    print("\n" + "="*60)
    print("评估结论:")
    
    if jitter_improvement > 0:
        print(f"✓ 带抓取点模型的平滑度优于不带抓取点模型！")
        print(f"  - 抖动改善率: {jitter_improvement:.1f}% (带抓取点更平滑)")
        
        # 检查其他指标
        velocity_std_improvement = (jitter_results['without_grasp']['filtered_velocity_std'] - 
                                   jitter_results['with_grasp']['filtered_velocity_std']) / \
                                   jitter_results['without_grasp']['filtered_velocity_std'] * 100
        max_velocity_improvement = (jitter_results['without_grasp']['filtered_max_velocity'] - 
                                   jitter_results['with_grasp']['filtered_max_velocity']) / \
                                   jitter_results['without_grasp']['filtered_max_velocity'] * 100
        
        if velocity_std_improvement > 0:
            print(f"  - 速度标准差改善: {velocity_std_improvement:.1f}%")
        if max_velocity_improvement > 0:
            print(f"  - 最大速度改善: {max_velocity_improvement:.1f}%")
    else:
        print(f"✗ 带抓取点模型的平滑度未表现出优势")
        print(f"  - 抖动改善率: {jitter_improvement:.1f}%")
    
    print("="*60)
    
    return jitter_results

# =======================================================================
# 7. 主程序入口
# =======================================================================
if __name__ == "__main__":
    print("开始对比不带抓取点和带抓取点模型的抖动值效果")
    print("="*60)
    
    # 运行对比评估
    jitter_results = compare_jitter_between_models()
    
    if jitter_results:
        print("\n对比完成！")
    else:
        print("\n对比过程中出现错误。")