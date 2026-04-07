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
import trimesh  # 用于加载OBJ文件

# =======================================================================
# 1. 模型定义 (动态模板版本)
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
    动态模板版本，与训练脚本完全一致
    """
    def __init__(self, M=2048, latent_dim=64, damping=0.95, template_dict=None):
        super(RecursiveTrackingNet, self).__init__()
        self.latent_dim = latent_dim
        self.M = M
        self.damping = damping
        
        self.enc_obs = PointNetEncoder(latent_dim)
        self.f_gru = nn.GRUCell(latent_dim, latent_dim)
        
        self.gate = nn.Sequential(
            nn.Linear(latent_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # --- 动态模板初始化 ---
        self.template_dict = template_dict or {}
        self.templates = {}  # 存储加载后的模板张量
        self.template_names = ['flat', 'folded']  # 预设的模板名称
        
        if not self.template_dict:
            print(f"[模型初始化] 警告: 未提供模板字典，将创建默认的球面模板。")
            default_template = self._generate_template_points(M)
            self.templates['default'] = default_template
            self.template_names = ['default']
        else:
            # 确保必须的模板存在
            for name in self.template_names:
                if name in self.template_dict:
                    obj_path = self.template_dict[name]
                    if os.path.exists(obj_path):
                        template = self._load_template_from_obj(obj_path, M)
                        self.templates[name] = template
                        print(f"  -> 已加载模板 '{name}' 从: {obj_path}")
                    else:
                        raise FileNotFoundError(f"模板文件不存在: {obj_path}")
                else:
                    raise KeyError(f"模板字典中必须包含 '{name}' 键。")
        
        # 当前活跃模板，初始化为平整模板
        self.current_template_name = 'flat'
        self.current_template = self.templates['flat']
        print(f"[模型初始化] 模板库加载完成: {list(self.templates.keys())}")
        # --- 动态模板初始化结束 ---
        
        self.decoder = nn.Sequential(
            nn.Linear(3 + latent_dim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 3)
        )

    def _generate_template_points(self, M):
        """备用的默认球面采样方法。"""
        np.random.seed(42)
        coords = np.random.randn(M, 3)
        radii = np.linalg.norm(coords, axis=1, keepdims=True)
        return torch.from_numpy(coords / radii).float()

    def _load_template_from_obj(self, obj_path, target_num_points):
        """
        从OBJ文件加载网格顶点，并采样得到固定数量的模板点。
        关键：严格遵循控制变量法，不进行任何归一化操作，保持顶点原始坐标。
        """
        # 1. 使用trimesh加载OBJ网格文件
        mesh = trimesh.load(obj_path)
        vertices = mesh.vertices.astype(np.float32)  # 获取顶点坐标 [V, 3]
        print(f"  -> 已加载顶点数: {len(vertices)}")

        # 2. 采样到目标点数 (target_num_points = M = 2048)
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
        # 注意：此处没有进行任何归一化操作，直接返回原始坐标
        return torch.from_numpy(template_points).float()

    def _farthest_point_sampling(self, points, n_samples):
        """
        最远点采样 (Farthest Point Sampling, FPS) 实现。
        从点云 `points` 中选取 `n_samples` 个分布最远的点。
        """
        n_points = points.shape[0]
        sampled_indices = np.zeros(n_samples, dtype=np.int64)
        distances = np.full(n_points, np.inf)  # 所有点到已采样点集的最短距离
        
        # 随机选择第一个点
        first_idx = np.random.randint(n_points)
        sampled_indices[0] = first_idx
        
        for i in range(1, n_samples):
            # 获取上一个被选中的点
            last_selected = points[sampled_indices[i-1]]
            # 计算所有点到上一个选中点的欧氏距离
            dist_to_last = np.linalg.norm(points - last_selected, axis=1)
            # 更新每个点到已采样点集的最短距离
            distances = np.minimum(distances, dist_to_last)
            # 选择距离最大的点（即离已采样点集最远的点）作为下一个采样点
            sampled_indices[i] = np.argmax(distances)
        
        return points[sampled_indices]

    def select_template_by_frame(self, frame_idx):
        """
        根据帧号选择模板，并确保返回的模板张量与模型在同一设备上。
        规则：在 [250, 480] 帧区间内使用'folded'模板，否则使用'flat'模板。
        """
        if 250 <= frame_idx <= 480:
            selected_name = 'folded'
        else:
            selected_name = 'flat'

        if selected_name != self.current_template_name:
            self.current_template_name = selected_name
            # 关键修复：获取模板后，将其移动到与模型参数相同的设备上
            selected_template_tensor = self.templates[selected_name]
            # 使用 next(self.parameters()).device 获取模型所在的设备
            model_device = next(self.parameters()).device
            self.current_template = selected_template_tensor.to(model_device)
        else:
            # 即使模板名称不变，也需要确保当前模板在正确的设备上
            model_device = next(self.parameters()).device
            if self.current_template.device != model_device:
                self.current_template = self.current_template.to(model_device)
        
        return self.current_template, self.current_template_name

    def forward(self, o_t, h_prev, h_prev_prev=None, current_frame_idx=0):
        """
        前向传播，增加 current_frame_idx 参数以支持动态模板选择。
        返回: p_hat, h_t, z_t, used_tmpl_name
        其中h_t是经过动力学预测和GRU更新后的最终潜状态
        """
        B = o_t.size(0)
        z_t = self.enc_obs(o_t)  # 观测编码
        
        # 1. 动力学预测 (惯性预测)
        if h_prev_prev is not None:
            velocity = h_prev - h_prev_prev
            h_inertial = h_prev + self.damping * velocity
        else:
            h_inertial = h_prev
            
        # 2. GRU更新
        h_gru = self.f_gru(z_t, h_prev)
        
        # 3. 门控融合
        gate_input = torch.cat([h_inertial, z_t], dim=-1)
        alpha = self.gate(gate_input)
        alpha_scaled = 0.7 + 0.3 * alpha  # 强制alpha更大，更信任当前观测
        h_t = (1 - alpha_scaled) * h_inertial + alpha_scaled * h_gru
        
        # 4. 动态选择模板
        selected_template, used_tmpl_name = self.select_template_by_frame(current_frame_idx)
        
        # 5. 坐标重构
        z_ext = h_t.unsqueeze(1).repeat(1, self.M, 1)
        template_ext = selected_template.unsqueeze(0).repeat(B, 1, 1)
        p_hat = self.decoder(torch.cat([template_ext, z_ext], dim=-1))
        
        return p_hat, h_t, z_t, used_tmpl_name

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
# 3. 数据加载
# =======================================================================

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
        return torch.from_numpy(pts[choice]), self.files[idx]

# =======================================================================
# 4. GRU滤波器与KF基线对比评估函数
# =======================================================================

def evaluate_gru_vs_kf_baseline():
    """
    评估GRU滤波器与匀速运动卡尔曼滤波器基线的对比
    训练后的GRU滤波器 vs 传统匀速KF基线
    """
    DATA_PATH = r"D:\Cu_BEM_2\tshirt-data\tshirt_kalman_2\dataset_arm"
    
    # 训练后的模型checkpoint路径
    MODEL_AFTER = "checkpoints/best_model_v2_dim64_DynamicTemplate_smooth.pth"  # 训练后（平滑度损失）
    
    LATENT_DIM = 64
    MAX_PC_COMPONENTS = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print("="*60)
    print("GRU滤波器 vs 匀速运动卡尔曼滤波器基线对比")
    print("="*60)
    
    # 加载训练后的GRU模型
    print("1. 加载训练后的GRU模型（有平滑度损失）...")
    checkpoint_after = torch.load(MODEL_AFTER, map_location=device)
    
    if 'template_dict' in checkpoint_after and checkpoint_after['template_dict'] is not None:
        TEMPLATE_DICT = checkpoint_after['template_dict']
        print(f"   模板字典: {list(TEMPLATE_DICT.keys())}")
    else:
        # 如果checkpoint中没有模板字典，使用默认的
        TEMPLATE_DICT = {
            'flat': r"D:\Cu_BEM_2\tshirt-data\tshirt_kalman_2\tshirt_mech00001.obj",
            'folded': r"D:\Cu_BEM_2\tshirt-data\tshirt_kalman_2\tshirt_mech00250.obj",
        }
        print(f"   使用默认模板字典: {list(TEMPLATE_DICT.keys())}")
    
    model_after = RecursiveTrackingNet(
        latent_dim=LATENT_DIM, 
        template_dict=TEMPLATE_DICT
    ).to(device)
    
    # 修复状态字典
    state_dict_after = checkpoint_after['model_state_dict']
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
    
    model_after.eval()
    print(f"   GRU模型已加载: {MODEL_AFTER}")
    
    # 初始化匀速运动卡尔曼滤波器基线
    print("\n2. 初始化匀速运动卡尔曼滤波器基线...")
    kf_baseline = ConstantVelocityKFBaseline(
        dim=LATENT_DIM, 
        process_noise=1e-5, 
        measurement_noise=0.05
    )
    print(f"   KF基线已初始化: dt=1.0, Q={1e-5}, R={0.05}")
    
    # 从checkpoint中获取平滑度损失权重
    smoothness_weight = checkpoint_after.get('smoothness_weight', 0.5)
    print(f"   平滑度损失权重: {smoothness_weight}")
    
    # 加载点云数据集
    dataset = PointcloudSeqDataset(DATA_PATH)
    all_frames = []
    for i in range(len(dataset)):
        pc, fname = dataset[i]
        all_frames.append(pc.unsqueeze(0).to(device))
    T = len(all_frames)
    print(f"\n3. 已加载 {T} 帧点云数据")
    
    # 存储结果
    results = {
        'gru': {'Z_raw': [], 'Z_filtered': [], 'template_usage': []},  # GRU: raw=z_t观测, filtered=h_t滤波后
        'kf': {'Z_raw': [], 'Z_filtered': []}  # KF: raw=z_t观测, filtered=KF滤波后
    }
    
    # 初始化GRU模型状态
    h_t_gru = torch.zeros(1, LATENT_DIM).to(device)
    h_prev_gru = None
    
    print("\n4. 正在提取潜向量并进行滤波...")
    with torch.no_grad():
        for t in range(T):
            o_t = all_frames[t]
            
            # GRU滤波器前向传播
            _, h_next_gru, z_t_gru, used_tmpl = model_after(o_t, h_t_gru, h_prev_gru, current_frame_idx=t)
            
            z_t_gru_np = z_t_gru.squeeze(0).cpu().numpy()  # GRU观测编码 z_t
            h_next_gru_np = h_next_gru.squeeze(0).cpu().numpy()  # GRU滤波后 h_t
            
            # 对相同的观测z_t进行匀速KF基线滤波
            z_t_kf_filtered = kf_baseline.update(z_t_gru_np)  # KF滤波后
            
            # 存储结果
            results['gru']['Z_raw'].append(z_t_gru_np)
            results['gru']['Z_filtered'].append(h_next_gru_np)
            results['gru']['template_usage'].append(used_tmpl)
            
            results['kf']['Z_raw'].append(z_t_gru_np)  # 与GRU使用相同的观测
            results['kf']['Z_filtered'].append(z_t_kf_filtered)
            
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
    
    print("\n5. 计算抖动分数...")
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
    
    # 模板使用统计
    print("\n模板使用统计 (GRU模型):")
    flat_count = sum(1 for tmpl in results['gru']['template_usage'] if tmpl == 'flat')
    folded_count = sum(1 for tmpl in results['gru']['template_usage'] if tmpl == 'folded')
    print(f"  'flat'模板: {flat_count}帧 ({flat_count/T*100:.1f}%)")
    print(f"  'folded'模板: {folded_count}帧 ({folded_count/T*100:.1f}%)")
    
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
    plt.suptitle(f'GRU滤波器 vs 匀速KF基线对比 (λ={smoothness_weight})\n'
                 f'抖动改善: {jitter_improvement:.1f}%, 信息效率改善: {quality_improvement:.1f}%', 
                 fontsize=16, y=1.02)
    plt.tight_layout()
    
    # 保存图表
    save_path = f'gru_vs_kf_comparison_lambda_{smoothness_weight}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n评估图表已保存至: {save_path}")
    plt.show()
    
    # 打印详细评估结论
    print("\n" + "="*60)
    print("评估结论:")
    if jitter_improvement > 0 and quality_improvement > 0:
        print(f"✓ GRU滤波器全面优于匀速KF基线！")
        print(f"  - 抖动改善: {jitter_improvement:.1f}% (GRU更平滑)")
        print(f"  - 信息效率改善: {quality_improvement:.1f}% (GRU综合性能更优)")
    elif jitter_improvement > 0:
        print(f"⚠ GRU滤波器部分优于KF基线：抖动改善但信息效率下降")
        print(f"  - 抖动改善: {jitter_improvement:.1f}% (GRU更平滑)")
        print(f"  - 但信息效率下降: {abs(quality_improvement):.1f}%")
    elif quality_improvement > 0:
        print(f"⚠ GRU滤波器部分优于KF基线：信息效率改善但抖动增加")
        print(f"  - 信息效率改善: {quality_improvement:.1f}% (GRU综合性能更优)")
        print(f"  - 但抖动增加: {abs(jitter_improvement):.1f}%")
    else:
        print(f"✗ GRU滤波器未表现出优势")
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
# 5. 主程序入口
# =======================================================================
if __name__ == "__main__":
    print("开始评估GRU滤波器与匀速KF基线的对比...")
    print("="*60)
    
    # 检查模型文件是否存在
    MODEL_AFTER = "checkpoints/best_model_v2_dim64_DynamicTemplate_smooth.pth"
    
    if not os.path.exists(MODEL_AFTER):
        print(f"错误: 训练后的模型文件不存在: {MODEL_AFTER}")
        print("请先使用添加了平滑度损失函数的训练脚本训练模型。")
        print("\n您可以运行以下步骤:")
        print("1. 修改 training_script_kalman.py 添加平滑度损失")
        print("2. 运行训练脚本进行微调训练")
        print("3. 再次运行本评估脚本")
    else:
        # 运行完整的GRU vs KF对比评估
        evaluation_results = evaluate_gru_vs_kf_baseline()
        print("\n评估完成！")
        
        # 打印主要结果
        print(f"\n主要评估指标:")
        print(f"  GRU相比KF的抖动改善率: {evaluation_results['jitter_improvement']:.1f}%")
        print(f"  GRU相比KF的信息效率改善率: {evaluation_results['quality_improvement']:.1f}%")
        print(f"  平滑度损失权重: {evaluation_results['smoothness_weight']}")