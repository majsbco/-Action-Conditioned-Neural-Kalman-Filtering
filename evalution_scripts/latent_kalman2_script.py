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

# 添加当前目录到路径，以便导入training_script_kalman_2中的模型
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# =======================================================================
# 1. 从训练脚本中导入必要的模型组件
# =======================================================================

class PointNetEncoder(nn.Module):
    """观测点云编码器（从训练脚本导入）"""
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
    """抓取点编码器（从训练脚本导入）"""
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
    """多模态融合模块（从训练脚本导入）"""
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
    新版本模型架构（包含抓取点输入）
    状态转移方程：h_t = Φ(h_{t-1}, Fusion(z_obs, z_grasp))
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
        
        # 加载单一模板
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
# 2. 匀速运动卡尔曼滤波器基线定义
# =======================================================================

class ConstantVelocityKFBaseline:
    """
    匀速运动卡尔曼滤波器基线
    在64维潜空间的每一维上独立运行
    """
    def __init__(self, dim=64, dt=1.0, process_noise=1e-5, measurement_noise=0.05):
        self.dim = dim
        self.dt = dt
        # 为每个维度维护独立的状态和协方差矩阵
        self.states = None  # 形状: (dim, 2) 每个维度有[位置, 速度]
        self.Ps = None      # 形状: (dim, 2, 2) 每个维度有2x2协方差矩阵
        
        # 状态转移矩阵
        self.F = np.array([[1, self.dt], [0, 1]])
        # 观测矩阵
        self.H = np.array([[1, 0]])
        # 过程噪声协方差矩阵
        self.Q = np.eye(2) * process_noise
        # 测量噪声协方差
        self.R = measurement_noise
        
        self.initialized = False

    def update(self, measurement):
        if not self.initialized:
            # 初始化状态和协方差
            self.states = np.zeros((self.dim, 2))
            self.states[:, 0] = measurement  # 初始位置设为测量值
            self.Ps = np.tile(np.eye(2), (self.dim, 1, 1))  # 每个维度一个单位矩阵
            self.initialized = True
            return measurement
            
        filtered = np.zeros(self.dim)
        
        for i in range(self.dim):
            # 预测步骤
            x_pred = self.F @ self.states[i]
            P_pred = self.F @ self.Ps[i] @ self.F.T + self.Q
            
            # 更新步骤
            y = measurement[i] - (self.H @ x_pred)  # 测量残差
            S = self.H @ P_pred @ self.H.T + self.R  # 残差协方差
            K = (P_pred @ self.H.T) / S  # 卡尔曼增益，现在形状是(2, 1)或(2,)
            
            # 更新状态估计
            self.states[i] = x_pred + K.flatten() * y
            # 更新协方差估计
            self.Ps[i] = (np.eye(2) - np.outer(K, self.H)) @ P_pred
            
            filtered[i] = self.states[i, 0]
            
        return filtered

# =======================================================================
# 3. 抓取点序列管理器
# =======================================================================

class GraspPointSequence:
    """抓取点序列管理器"""
    def __init__(self, grasp_file_path):
        self.times = []
        self.positions = []
        self._load_grasp_file(grasp_file_path)
        
    def _load_grasp_file(self, file_path):
        """读取抓取点轨迹文件"""
        print(f"正在读取抓取点文件: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 使用正则表达式匹配所有 T= 和 X= 对
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

# =======================================================================
# 4. 数据加载
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
        print(f"数据集大小: {min_len} 帧")
        
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
# 5. GRU滤波器与KF基线对比评估函数
# =======================================================================

def evaluate_gru_vs_kf_baseline():
    """
    评估GRU滤波器与匀速运动卡尔曼滤波器基线的对比
    使用新模型架构（包含抓取点输入）
    """
    DATA_PATH = r"D:\Cu_BEM_2\tshirt-data\tshirt_kalman_3\dataset_arm"
    GRASP_FILE = r"D:\Cu_BEM_2\tshirt-data\tshirt_kalman_3\vertex_237_trajectory.txt"
    
    # 训练后的模型checkpoint路径
    MODEL_AFTER = "checkpoints/best_model_simplified.pth"  # 新模型checkpoint
    
    OBS_LATENT_DIM = 64
    GRASP_LATENT_DIM = 16
    MAX_PC_COMPONENTS = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print("="*60)
    print("GRU滤波器 vs 匀速运动卡尔曼滤波器基线对比")
    print("="*60)
    
    # 加载抓取点序列
    print("1. 加载抓取点序列...")
    grasp_sequence = GraspPointSequence(GRASP_FILE)
    
    # 加载训练后的GRU模型
    print("2. 加载训练后的GRU模型（包含抓取点输入）...")
    
    # 模板路径
    TEMPLATE_PATH = r"D:\Cu_BEM_2\tshirt-data\tshirt_kalman_3\tshirt_mech00001.obj"
    
    # 初始化新模型
    model_after = RecursiveTrackingNet(
        obs_latent_dim=OBS_LATENT_DIM,
        grasp_latent_dim=GRASP_LATENT_DIM,
        template_path=TEMPLATE_PATH
    ).to(device)
    
    # 加载checkpoint
    if os.path.exists(MODEL_AFTER):
        checkpoint_after = torch.load(MODEL_AFTER, map_location=device)
        state_dict_after = checkpoint_after['model_state_dict']
        
        # 修复状态字典
        keys_to_remove = []
        for key in state_dict_after.keys():
            if 'current_template' in key or ('template' in key and 'templates' not in key):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            if key in state_dict_after:
                del state_dict_after[key]
                print(f"   已删除意外键: {key}")
        
        # 加载状态字典
        missing_keys, unexpected_keys = model_after.load_state_dict(state_dict_after, strict=False)
        if missing_keys:
            print(f"   警告: 以下键缺失: {missing_keys}")
        if unexpected_keys:
            print(f"   警告: 以下键意外: {unexpected_keys}")
        
        # 从checkpoint中获取平滑度损失权重
        smoothness_weight = checkpoint_after.get('loss', 0.5)  # 如果没有明确存储，使用默认值
        print(f"   平滑度损失权重: {smoothness_weight}")
    else:
        print(f"   警告: 检查点文件不存在: {MODEL_AFTER}")
        print(f"   将使用随机初始化的模型")
        smoothness_weight = 0.5
    
    model_after.eval()
    print(f"   GRU模型已加载: {MODEL_AFTER}")
    print(f"   模型架构: 包含抓取点输入")
    print(f"   观测潜空间维度: {OBS_LATENT_DIM}")
    print(f"   抓取点潜空间维度: {GRASP_LATENT_DIM}")
    
    # 初始化匀速运动卡尔曼滤波器基线
    print("\n3. 初始化匀速运动卡尔曼滤波器基线...")
    kf_baseline = ConstantVelocityKFBaseline(
        dim=OBS_LATENT_DIM, 
        process_noise=1e-5, 
        measurement_noise=0.05
    )
    print(f"   KF基线已初始化: dt=1.0, Q={1e-5}, R={0.05}")
    
    # 加载点云数据集
    print("\n4. 加载点云数据集...")
    dataset = PointcloudSeqDataset(DATA_PATH, grasp_sequence)
    all_frames = []
    all_grasp_points = []
    
    for i in range(len(dataset)):
        data = dataset[i]
        all_frames.append(data['points'].unsqueeze(0).to(device))
        all_grasp_points.append(data['grasp_point'].to(device))
    
    T = len(all_frames)
    print(f"   已加载 {T} 帧点云数据和抓取点序列")
    
    # 存储结果
    results = {
        'gru': {'Z_raw': [], 'Z_filtered': []},  # GRU: raw=z_obs观测, filtered=h_t滤波后
        'kf': {'Z_raw': [], 'Z_filtered': []}    # KF: raw=z_obs观测, filtered=KF滤波后
    }
    
    # 初始化GRU模型状态
    h_t_gru = torch.zeros(1, OBS_LATENT_DIM).to(device)
    h_prev_gru = None
    
    print("\n5. 正在提取潜向量并进行滤波...")
    with torch.no_grad():
        for t in range(T):
            o_t = all_frames[t]
            g_t = all_grasp_points[t]
            
            # GRU滤波器前向传播（包含抓取点输入）
            _, h_next_gru, z_obs = model_after(o_t, g_t, h_t_gru, h_prev_gru)
            
            z_obs_np = z_obs.squeeze(0).cpu().numpy()  # GRU观测编码 z_obs
            h_next_gru_np = h_next_gru.squeeze(0).cpu().numpy()  # GRU滤波后 h_t
            
            # 对相同的观测z_obs进行匀速KF基线滤波
            z_obs_kf_filtered = kf_baseline.update(z_obs_np)  # KF滤波后
            
            # 存储结果
            results['gru']['Z_raw'].append(z_obs_np)
            results['gru']['Z_filtered'].append(h_next_gru_np)
            
            results['kf']['Z_raw'].append(z_obs_np)  # 与GRU使用相同的观测
            results['kf']['Z_filtered'].append(z_obs_kf_filtered)
            
            # 更新GRU状态
            h_prev_gru = h_t_gru.detach()
            h_t_gru = h_next_gru
            
            if (t+1) % 100 == 0 or (t+1) == T:
                print(f"   进度: {t+1}/{T}")
    
    # 转换为numpy数组
    for method in ['gru', 'kf']:
        results[method]['Z_raw'] = np.array(results[method]['Z_raw'])
        results[method]['Z_filtered'] = np.array(results[method]['Z_filtered'])
    
    # 基于GRU观测数据拟合PCA（保持一致）
    pca = PCA(n_components=MAX_PC_COMPONENTS)
    pca.fit(results['gru']['Z_raw'])
    
    # 对两个方法的数据进行PCA变换
    for method in ['gru', 'kf']:
        results[method]['Z_raw_pca'] = pca.transform(results[method]['Z_raw'])
        results[method]['Z_filtered_pca'] = pca.transform(results[method]['Z_filtered'])
    
    # 计算每个主成分轴的抖动值
    jitter_gru_raw = []
    jitter_gru_filtered = []
    jitter_kf_raw = []
    jitter_kf_filtered = []
    
    print("\n6. 计算抖动分数...")
    for i in range(MAX_PC_COMPONENTS):
        # GRU模型的抖动
        j_gru_raw = np.mean(np.abs(np.diff(results['gru']['Z_raw_pca'][:, i], n=2)))
        j_gru_filtered = np.mean(np.abs(np.diff(results['gru']['Z_filtered_pca'][:, i], n=2)))
        
        # KF基线的抖动
        j_kf_raw = np.mean(np.abs(np.diff(results['kf']['Z_raw_pca'][:, i], n=2)))
        j_kf_filtered = np.mean(np.abs(np.diff(results['kf']['Z_filtered_pca'][:, i], n=2)))
        
        jitter_gru_raw.append(j_gru_raw)
        jitter_gru_filtered.append(j_gru_filtered)
        jitter_kf_raw.append(j_kf_raw)
        jitter_kf_filtered.append(j_kf_filtered)
    
    # 计算信息效率 (方差解释率 / 抖动值)
    variance_ratios = pca.explained_variance_ratio_
    
    quality_gru_raw = variance_ratios / (np.array(jitter_gru_raw) + 1e-9)
    quality_gru_filtered = variance_ratios / (np.array(jitter_gru_filtered) + 1e-9)
    quality_kf_raw = variance_ratios / (np.array(jitter_kf_raw) + 1e-9)
    quality_kf_filtered = variance_ratios / (np.array(jitter_kf_filtered) + 1e-9)
    
    # 打印汇总统计
    print("\n" + "="*60)
    print("GRU滤波器 vs 匀速KF基线 对比评估报告")
    print("="*60)
    print(f"数据集大小: {T} 帧")
    print(f"平滑度损失权重: {smoothness_weight}")
    print(f"模型架构: 包含抓取点输入")
    
    avg_jitter_gru_raw = np.mean(jitter_gru_raw)
    avg_jitter_gru_filtered = np.mean(jitter_gru_filtered)
    avg_jitter_kf_raw = np.mean(jitter_kf_raw)
    avg_jitter_kf_filtered = np.mean(jitter_kf_filtered)
    
    avg_quality_gru_filtered = np.mean(quality_gru_filtered)
    avg_quality_kf_filtered = np.mean(quality_kf_filtered)
    
    print(f"\n平均抖动 (所有PC维度):")
    print(f"  GRU - 原始观测: {avg_jitter_gru_raw:.6f}")
    print(f"  GRU - 滤波后: {avg_jitter_gru_filtered:.6f}")
    print(f"  KF基线 - 原始观测: {avg_jitter_kf_raw:.6f}")
    print(f"  KF基线 - 滤波后: {avg_jitter_kf_filtered:.6f}")
    
    # 计算GRU滤波后相对于KF滤波后的改善率
    jitter_improvement = (avg_jitter_kf_filtered - avg_jitter_gru_filtered) / avg_jitter_kf_filtered * 100
    quality_improvement = (avg_quality_gru_filtered - avg_quality_kf_filtered) / avg_quality_kf_filtered * 100
    
    print(f"\nGRU滤波后相比KF滤波后的改善率:")
    print(f"  抖动改善: {jitter_improvement:.1f}% (正值表示GRU更平滑)")
    print(f"  信息效率改善: {quality_improvement:.1f}% (正值表示GRU更优)")
    
    # 可视化对比 - 只生成两个核心子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 子图1: 滤波后每维度抖动值对比柱状图
    x = np.arange(1, MAX_PC_COMPONENTS + 1)  # PC索引从1开始
    width = 0.35
    
    # 绘制GRU滤波后和KF滤波后的抖动值
    ax1.bar(x - width/2, jitter_gru_filtered, width, label='GRU滤波器 (滤波后)', 
            color='dodgerblue', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.bar(x + width/2, jitter_kf_filtered, width, label='匀速KF基线 (滤波后)', 
            color='darkviolet', alpha=0.7, edgecolor='black', linewidth=0.5)
    
    ax1.set_title("Jitter Comparison per Principal Component\nFiltered Output (Lower is Better)", 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel("Principal Component Index", fontsize=12)
    ax1.set_ylabel("Jitter Score (Mean Absolute 2nd Order Diff)", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'PC{i}' for i in x])
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    ax1.set_axisbelow(True)
    
    # 在柱状图上添加GRU相比KF的改善率标签
    for i in range(MAX_PC_COMPONENTS):
        improvement = (jitter_kf_filtered[i] - jitter_gru_filtered[i]) / jitter_kf_filtered[i] * 100
        if improvement > 0:
            # 在GRU柱子上方添加改善百分比
            ax1.text(x[i] - width/2, jitter_gru_filtered[i] + 0.01, 
                    f'+{improvement:.0f}%', ha='center', va='bottom', 
                    fontsize=8, fontweight='bold', color='green')
        elif improvement < 0:
            # 在GRU柱子上方添加恶化百分比
            ax1.text(x[i] - width/2, jitter_gru_filtered[i] + 0.01, 
                    f'{improvement:.0f}%', ha='center', va='bottom', 
                    fontsize=8, fontweight='bold', color='red')
    
    # 子图2: 滤波后信息效率对比曲线
    ax2.plot(x, quality_gru_filtered, 'D-', color='dodgerblue', linewidth=2.5, 
             markersize=7, markerfacecolor='white', markeredgecolor='dodgerblue',
             markeredgewidth=1.5, label='GRU滤波器 (滤波后)')
    ax2.plot(x, quality_kf_filtered, 'v:', color='darkviolet', linewidth=2, 
             markersize=6, markerfacecolor='white', markeredgecolor='darkviolet',
             markeredgewidth=1.5, label='匀速KF基线 (滤波后)')
    
    ax2.set_title("Information Efficiency (Variance Explained / Jitter)\nFiltered Output (Higher is Better)", 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel("Principal Component Index", fontsize=12)
    ax2.set_ylabel("Quality Score", fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'PC{i}' for i in x])
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_axisbelow(True)
    
    # 添加平均信息效率的参考线
    ax2.axhline(y=avg_quality_gru_filtered, color='dodgerblue', linestyle='--', alpha=0.5, 
                linewidth=1, label=f'GRU平均: {avg_quality_gru_filtered:.2f}')
    ax2.axhline(y=avg_quality_kf_filtered, color='darkviolet', linestyle='--', alpha=0.5, 
                linewidth=1, label=f'KF平均: {avg_quality_kf_filtered:.2f}')
    
    # 添加改进标签
    ax2.text(0.02, 0.95, f'GRU相比KF改善: {quality_improvement:.1f}%', 
             transform=ax2.transAxes, fontsize=10, fontweight='bold',
             verticalalignment='top', bbox=dict(boxstyle='round', 
             facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # 设置总标题
    plt.suptitle(f'GRU滤波器(带抓取点) vs 匀速KF基线对比\n'
                 f'抖动改善: {jitter_improvement:.1f}%, 信息效率改善: {quality_improvement:.1f}%', 
                 fontsize=16, y=1.02)
    plt.tight_layout()
    
    # 保存图表
    save_path = f'gru_vs_kf_comparison_with_grasp.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n评估图表已保存至: {save_path}")
    plt.show()
    
    # 打印详细评估结论
    print("\n" + "="*60)
    print("评估结论:")
    if jitter_improvement > 0 and quality_improvement > 0:
        print(f"✓ GRU滤波器(带抓取点输入)全面优于匀速KF基线！")
        print(f"  - 抖动改善: {jitter_improvement:.1f}% (GRU更平滑)")
        print(f"  - 信息效率改善: {quality_improvement:.1f}% (GRU综合性能更优)")
    elif jitter_improvement > 0:
        print(f"⚠ GRU滤波器(带抓取点输入)部分优于KF基线：抖动改善但信息效率下降")
        print(f"  - 抖动改善: {jitter_improvement:.1f}% (GRU更平滑)")
        print(f"  - 但信息效率下降: {abs(quality_improvement):.1f}%")
    elif quality_improvement > 0:
        print(f"⚠ GRU滤波器(带抓取点输入)部分优于KF基线：信息效率改善但抖动增加")
        print(f"  - 信息效率改善: {quality_improvement:.1f}% (GRU综合性能更优)")
        print(f"  - 但抖动增加: {abs(jitter_improvement):.1f}%")
    else:
        print(f"✗ GRU滤波器(带抓取点输入)未表现出优势")
        print(f"  - 抖动变化: {jitter_improvement:.1f}%")
        print(f"  - 信息效率变化: {quality_improvement:.1f}%")
    
    print("="*60)
    
    # 返回详细结果
    return {
        'jitter_gru_filtered': jitter_gru_filtered,
        'jitter_kf_filtered': jitter_kf_filtered,
        'quality_gru_filtered': quality_gru_filtered,
        'quality_kf_filtered': quality_kf_filtered,
        'jitter_improvement': jitter_improvement,
        'quality_improvement': quality_improvement,
        'smoothness_weight': smoothness_weight,
        'results': results
    }

# =======================================================================
# 6. 主程序入口
# =======================================================================
if __name__ == "__main__":
    print("开始评估GRU滤波器(带抓取点输入)与匀速KF基线的对比...")
    print("="*60)
    
    # 检查模型文件是否存在
    MODEL_AFTER = "checkpoints/best_model_simplified.pth"
    
    if not os.path.exists(MODEL_AFTER):
        print(f"警告: 训练后的模型文件不存在: {MODEL_AFTER}")
        print("请先使用包含抓取点输入的训练脚本训练模型。")
        print("\n您可以运行以下步骤:")
        print("1. 运行 training_script_kalman_2.py 训练模型")
        print("2. 确保检查点保存在: checkpoints/best_model_simplified.pth")
        print("3. 再次运行本评估脚本")
        
        # 询问是否继续使用随机初始化的模型进行评估
        response = input("\n是否继续使用随机初始化的模型进行评估? (y/n): ")
        if response.lower() == 'y':
            # 运行评估
            evaluation_results = evaluate_gru_vs_kf_baseline()
            print("\n评估完成！")
            
            # 打印主要结果
            if evaluation_results:
                print(f"\n主要评估指标:")
                print(f"  GRU相比KF的抖动改善率: {evaluation_results.get('jitter_improvement', 0):.1f}%")
                print(f"  GRU相比KF的信息效率改善率: {evaluation_results.get('quality_improvement', 0):.1f}%")
                print(f"  平滑度损失权重: {evaluation_results.get('smoothness_weight', 0)}")
    else:
        # 运行完整的GRU vs KF对比评估
        evaluation_results = evaluate_gru_vs_kf_baseline()
        print("\n评估完成！")
        
        # 打印主要结果
        print(f"\n主要评估指标:")
        print(f"  GRU相比KF的抖动改善率: {evaluation_results['jitter_improvement']:.1f}%")
        print(f"  GRU相比KF的信息效率改善率: {evaluation_results['quality_improvement']:.1f}%")
        print(f"  平滑度损失权重: {evaluation_results['smoothness_weight']}")