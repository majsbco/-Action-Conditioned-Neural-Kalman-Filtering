import torch
import torch.nn as nn
import numpy as np
import os
import glob
import re
from sklearn.decomposition import PCA
import open3d as o3d
import trimesh
from scipy.fft import fft, fftfreq
import warnings
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
warnings.filterwarnings('ignore')

# =======================================================================
# 1. 模型定义
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

# =======================================================================
# 2. 不带抓取点的GRU滤波器 (M5)
# =======================================================================

class RecursiveTrackingNetWithoutGrasp(nn.Module):
    """不带抓取点输入的模型 (M5)"""
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
        """
        B = o_t.size(0)
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
        
        return h_next, z_obs

# =======================================================================
# 3. 带抓取点的GRU滤波器 (M4)
# =======================================================================

class RecursiveTrackingNetWithGrasp(nn.Module):
    """带抓取点输入的模型 (M4)"""
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
        
        return h_next, z_obs

# =======================================================================
# 4. 匀速运动卡尔曼滤波器 (M3)
# =======================================================================

class ConstantVelocityKalmanFilter:
    """匀速运动卡尔曼滤波器 (M3)"""
    def __init__(self, dim=64, dt=1.0, process_noise=1e-5, measurement_noise=0.05):
        self.dim = dim
        self.dt = dt
        self.states = None
        self.Ps = None
        
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
            self.states[:, 0] = measurement
            self.Ps = np.tile(np.eye(2), (self.dim, 1, 1))
            self.initialized = True
            return measurement
            
        filtered = np.zeros(self.dim)
        
        for i in range(self.dim):
            # 预测步骤
            x_pred = self.F @ self.states[i]
            P_pred = self.F @ self.Ps[i] @ self.F.T + self.Q
            
            # 更新步骤
            y = measurement[i] - (self.H @ x_pred)
            S = self.H @ P_pred @ self.H.T + self.R
            K = (P_pred @ self.H.T) / S
            
            # 更新状态估计
            self.states[i] = x_pred + K.flatten() * y
            self.Ps[i] = (np.eye(2) - np.outer(K, self.H)) @ P_pred
            
            filtered[i] = self.states[i, 0]
            
        return filtered

# =======================================================================
# 5. 平滑损失监督模型 (M2)
# =======================================================================

class SmoothLossSupervisedModel:
    """平滑损失监督模型 (M2)"""
    def __init__(self, latent_dim=64, window_size=3):
        self.latent_dim = latent_dim
        self.window_size = window_size
        
    def smooth_sequence(self, sequence):
        """对序列进行移动平均平滑"""
        if len(sequence) < self.window_size:
            return sequence.copy()
        
        smoothed = np.zeros_like(sequence)
        half_window = self.window_size // 2
        
        for i in range(len(sequence)):
            start = max(0, i - half_window)
            end = min(len(sequence), i + half_window + 1)
            smoothed[i] = np.mean(sequence[start:end], axis=0)
        
        return smoothed

# =======================================================================
# 6. 评估指标计算函数
# =======================================================================

def compute_velocity_statistics(sequence):
    """计算速度统计：速度标准差和最大绝对速度"""
    if len(sequence) <= 1:
        return 0.0, 0.0
    
    velocity = np.diff(sequence, axis=0)
    velocity_std = np.std(velocity)
    max_velocity = np.max(np.abs(velocity))
    
    return velocity_std, max_velocity

def compute_jitter_score(sequence):
    """计算抖动值：二阶差分的绝对值的均值"""
    if len(sequence) <= 2:
        return 0.0
    
    second_order_diff = np.diff(sequence, n=2, axis=0)
    jitter = np.mean(np.abs(second_order_diff))
    
    return jitter

def compute_high_freq_energy_ratio(sequence, sample_rate=1.0):
    """计算高频能量占比（>Fs/4）"""
    if len(sequence) <= 1:
        return 0.0
    
    # 对每个维度计算频谱
    n = len(sequence)
    freqs = fftfreq(n, d=1.0/sample_rate)
    
    total_energy = 0
    high_freq_energy = 0
    
    for dim in range(sequence.shape[1]):
        signal = sequence[:, dim]
        fft_values = fft(signal)
        power_spectrum = np.abs(fft_values) ** 2
        
        # 计算总能量
        total_energy += np.sum(power_spectrum)
        
        # 计算高频能量（频率 > 采样频率/4）
        high_freq_mask = np.abs(freqs) > sample_rate / 4
        high_freq_energy += np.sum(power_spectrum[high_freq_mask])
    
    if total_energy == 0:
        return 0.0
    
    return high_freq_energy / total_energy

def compute_information_efficiency(sequence, variance_ratios):
    """计算信息效率：方差解释率 / (抖动值 + ε)"""
    if len(sequence) <= 2:
        return 0.0
    
    # 对每个主成分计算抖动
    n_components = min(len(variance_ratios), sequence.shape[1])
    jitters = []
    
    for i in range(n_components):
        pc_sequence = sequence[:, i]
        if len(pc_sequence) > 2:
            jitter = compute_jitter_score(pc_sequence.reshape(-1, 1))
            jitters.append(jitter)
        else:
            jitters.append(0.0)
    
    jitters = np.array(jitters)
    # 计算信息效率
    efficiency = variance_ratios[:n_components] / (jitters + 1e-9)
    
    return np.mean(efficiency)

def plot_pc_temporal_evolution(results, variance_ratios, n_components_to_plot=2, max_frames=700):
    """
    绘制五个模型在PC1, PC2上随时间变化的曲线图
    
    参数:
        results: 字典，包含每个模型的潜状态序列
        variance_ratios: 方差解释率列表
        n_components_to_plot: 要绘制的主成分数量
        max_frames: 最大绘制帧数
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 模型颜色映射
    model_colors = {
        'M1': '#1f77b4',  # 蓝色
        'M2': '#ff7f0e',  # 橙色
        'M3': '#2ca02c',  # 绿色
        'M4': '#d62728',  # 红色
        'M5': '#9467bd'   # 紫色
    }
    
    # 模型线型映射
    model_line_styles = {
        'M1': '-',  # 实线
        'M2': '--', # 虚线
        'M3': ':',  # 点线
        'M4': '-',  # 实线
        'M5': '--'  # 虚线
    }
    
    # 模型标签
    model_labels = {
        'M1': 'M1: Original Observation',
        'M2': 'M2: Smooth Loss Supervised',
        'M3': 'M3: Constant Velocity KF',
        'M4': 'M4: GRU with Grasp Point',
        'M5': 'M5: GRU without Grasp Point'
    }
    
    # 只取前n_components_to_plot个主成分
    n_components_to_plot = min(n_components_to_plot, len(variance_ratios))
    
    # 对每个模型的序列进行PCA变换
    pca = PCA(n_components=n_components_to_plot)
    
    # 使用M1的观测序列拟合PCA
    pca.fit(results['M1'])
    
    # 获取每个模型的PCA变换序列
    pca_sequences = {}
    for model_name in ['M1', 'M2', 'M3', 'M4', 'M5']:
        sequence = results[model_name]
        sequence_pca = pca.transform(sequence)
        pca_sequences[model_name] = sequence_pca
    
    # 创建图表
    fig, axes = plt.subplots(n_components_to_plot, 1, figsize=(16, 6*n_components_to_plot), dpi=100)
    
    # 确保axes是数组
    if n_components_to_plot == 1:
        axes = [axes]
    
    # 时间轴
    time_axis = np.arange(min(max_frames, len(results['M1'])))

    lines_for_legend = []
    labels_for_legend = []
    
    # 绘制每个主成分的时间序列
    for i in range(n_components_to_plot):
        ax = axes[i]
        
        for model_name in ['M1', 'M2', 'M3', 'M4', 'M5']:
            line, = ax.plot(time_axis, pca_sequences[model_name][:max_frames, i], 
                           linewidth=2,
                           linestyle=model_line_styles[model_name],
                           color=model_colors[model_name],
                           label=model_labels[model_name],
                           alpha=0.8)
            if i == 0:
                lines_for_legend.append(line)
                labels_for_legend.append(model_labels[model_name])
        
        # 设置子图属性
        ax.set_title(f'PC{i+1} Temporal Evolution (Variance Ratio: {variance_ratios[i]:.4f})', 
                     fontsize=16, fontweight='bold')
        ax.set_xlabel('Frame Index', fontsize=18)
        ax.set_ylabel(f'PC{i+1} Value', fontsize=18)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        ax.tick_params(axis='both', which='major', labelsize=16)
        
        # 添加图例
        fig.legend(lines_for_legend, labels_for_legend,     
               loc='upper center', 
               bbox_to_anchor=(0.5, 0.98), 
               ncol=3, 
               fontsize=18,
               frameon=True)
        plt.tight_layout()
        fig.subplots_adjust(top=0.87)
   
    # 保存图表
    save_path = 'pc_temporal_evolution.png'
    plt.savefig(save_path, dpi=500, bbox_inches='tight', pad_inches=0.1, 
            facecolor='white', edgecolor='none')
    print(f"\n主成分时间演化图已保存至: {save_path}")
    
    # 显示图表
    plt.show()
    
    return pca_sequences

def plot_comparison_subplots(results, variance_ratios, n_components=10):
    """
    绘制两个子图：
    1. 各模型在不同PC上的抖动值
    2. 各模型在不同PC上的信息效率
    """
    # 设置中文字体，避免特殊字符显示问题
    plt.rcParams['font.sans-serif'] = ['Arial']  
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    
    # 模型颜色映射
    model_colors = {
        'M1': '#1f77b4',  # 蓝色
        'M2': '#ff7f0e',  # 橙色
        'M3': '#2ca02c',  # 绿色
        'M4': '#d62728',  # 红色
        'M5': '#9467bd'   # 紫色
    }
    
    # 模型标记映射
    model_markers = {
        'M1': 'o',  # 圆形
        'M2': 's',  # 方形
        'M3': '^',  # 三角形
        'M4': 'D',  # 菱形
        'M5': 'v'   # 倒三角形
    }
    
    model_labels = {
        'M1': 'M1: Original Observation',
        'M2': 'M2: Smooth Loss Supervised',
        'M3': 'M3: Constant Velocity KF',
        'M4': 'M4: GRU with Grasp Point',
        'M5': 'M5: GRU without Grasp Point'
    }
    
    # 只取前n_components个主成分
    n_components = min(n_components, len(variance_ratios))
    
    # 对每个模型的序列进行PCA变换
    pca = PCA(n_components=n_components)
    
    # 使用M1的观测序列拟合PCA
    pca.fit(results['M1'])
    
    # 存储每个模型在每个PC上的抖动值和信息效率
    pc_jitters = {model: [] for model in ['M1', 'M2', 'M3', 'M4', 'M5']}
    pc_efficiencies = {model: [] for model in ['M1', 'M2', 'M3', 'M4', 'M5']}
    
    # 对每个主成分计算每个模型的抖动值和信息效率
    for model_name in ['M1', 'M2', 'M3', 'M4', 'M5']:
        # 对序列进行PCA变换
        sequence = results[model_name]
        sequence_pca = pca.transform(sequence)
        
        for i in range(n_components):
            # 提取第i个主成分的序列
            pc_sequence = sequence_pca[:, i]
            
            # 计算该PC的抖动值
            if len(pc_sequence) > 2:
                second_order_diff = np.diff(pc_sequence, n=2)
                jitter = np.mean(np.abs(second_order_diff))
            else:
                jitter = 0.0
            
            # 计算该PC的信息效率
            variance_ratio = variance_ratios[i] if i < len(variance_ratios) else 0
            efficiency = variance_ratio / (jitter + 1e-9)
            
            pc_jitters[model_name].append(jitter)
            pc_efficiencies[model_name].append(efficiency)
    
    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=100)
    
    # 子图1：各模型在不同PC上的抖动值
    ax1 = axes[0]
    x = np.arange(1, n_components + 1)
    
    for model_name in ['M1', 'M2', 'M3', 'M4', 'M5']:
        jitters = pc_jitters[model_name]
        ax1.plot(x, jitters, 
                marker=model_markers[model_name],
                markersize=8,
                linewidth=2,
                label=model_labels[model_name], 
                color=model_colors[model_name],
                alpha=0.8)
    
    ax1.set_xlabel('Principal Component Index', fontsize=16)
    ax1.set_ylabel('Jitter Value (mean(|second-order diff|))', fontsize=16)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'PC{i}' for i in x], fontsize=12)
    ax1.tick_params(axis='y', labelsize=12)
    ax1.legend(fontsize=15, loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)
    
    # 添加网格线
    ax1.yaxis.grid(True, alpha=0.2)
    
    # 在子图1上标注最小值
    for i in range(n_components):
        if i < 3:  # 只在前3个PC上添加标注，避免重叠
            pc_jitter_values = [pc_jitters[model][i] for model in ['M1', 'M2', 'M3', 'M4', 'M5']]
            min_idx = np.argmin(pc_jitter_values)
            min_model = ['M1', 'M2', 'M3', 'M4', 'M5'][min_idx]
            
            # 在最小值点上添加标记
            ax1.plot(x[i], pc_jitter_values[min_idx], 
                    marker='*', 
                    markersize=12, 
                    markerfacecolor='yellow', 
                    markeredgecolor='black', 
                    markeredgewidth=1)
    
    # 子图2：各模型在不同PC上的信息效率
    ax2 = axes[1]
    
    for model_name in ['M1', 'M2', 'M3', 'M4', 'M5']:
        efficiencies = pc_efficiencies[model_name]
        ax2.plot(x, efficiencies, 
                marker=model_markers[model_name],
                markersize=8,
                linewidth=2,
                label=model_labels[model_name], 
                color=model_colors[model_name],
                alpha=0.8)
    
    ax2.set_xlabel('Principal Component Index', fontsize=16)
    ax2.set_ylabel('Information Efficiency (Variance Ratio / (Jitter+ε))', fontsize=16)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'PC{i+1}' for i in x], fontsize=12)
    ax2.tick_params(axis='y', labelsize=12)
    ax2.legend(fontsize=15, loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)
    
    # 添加网格线
    ax2.yaxis.grid(True, alpha=0.2)
    
    # 在子图2上标注最大值
    for i in range(n_components):
        if i < 3:  # 只在前3个PC上添加标注，避免重叠
            pc_efficiency_values = [pc_efficiencies[model][i] for model in ['M1', 'M2', 'M3', 'M4', 'M5']]
            max_idx = np.argmax(pc_efficiency_values)
            max_model = ['M1', 'M2', 'M3', 'M4', 'M5'][max_idx]
            
            # 在最大值点上添加标记
            ax2.plot(x[i], pc_efficiency_values[max_idx], 
                    marker='*', 
                    markersize=12, 
                    markerfacecolor='yellow', 
                    markeredgecolor='black', 
                    markeredgewidth=1)
    
    # 调整子图间距
    plt.tight_layout()
    
    # 保存图表
    save_path = 'model_performance_comparison_subplots.png'
    plt.savefig(save_path, dpi=500, bbox_inches='tight', facecolor='white')
    print(f"\n对比子图已保存至: {save_path}")
    
    # 显示图表
    plt.show()
    
    # 打印详细的PC分析结果
    print("\n" + "="*60)
    print("Detailed PC Analysis Results")
    print("="*60)
    
    # 打印前3个PC的详细对比
    print(f"\nTop {min(3, n_components)} Principal Components Analysis:")
    for i in range(min(3, n_components)):
        print(f"\nPC{i+1} (Variance Ratio: {variance_ratios[i]:.4f}):")
        
        # 抖动值排名
        print("  Jitter Ranking (from low to high):")
        pc_jitter_dict = {}
        for model in ['M1', 'M2', 'M3', 'M4', 'M5']:
            pc_jitter_dict[model] = pc_jitters[model][i]
        
        sorted_jitters = sorted(pc_jitter_dict.items(), key=lambda x: x[1])
        for rank, (model, jitter) in enumerate(sorted_jitters, 1):
            model_short = model.replace('GRU with Grasp Point', 'M4').replace('GRU without Grasp Point', 'M5')
            print(f"    {rank}. {model_short}: {jitter:.6f}")
        
        # 信息效率排名
        print("  Information Efficiency Ranking (from high to low):")
        pc_efficiency_dict = {}
        for model in ['M1', 'M2', 'M3', 'M4', 'M5']:
            pc_efficiency_dict[model] = pc_efficiencies[model][i]
        
        sorted_efficiencies = sorted(pc_efficiency_dict.items(), key=lambda x: x[1], reverse=True)
        for rank, (model, efficiency) in enumerate(sorted_efficiencies, 1):
            model_short = model.replace('GRU with Grasp Point', 'M4').replace('GRU without Grasp Point', 'M5')
            print(f"    {rank}. {model_short}: {efficiency:.6f}")
        
        # 计算M4相对于M5的改善率
        m4_jitter = pc_jitters['M4'][i]
        m5_jitter = pc_jitters['M5'][i]
        m4_efficiency = pc_efficiencies['M4'][i]
        m5_efficiency = pc_efficiencies['M5'][i]
        
        if m5_jitter > 0:
            jitter_improvement = (m5_jitter - m4_jitter) / m5_jitter * 100
        else:
            jitter_improvement = 0
        
        if m5_efficiency > 0:
            efficiency_improvement = (m4_efficiency - m5_efficiency) / m5_efficiency * 100
        else:
            efficiency_improvement = 0
        
        print(f"  M4 vs M5 Improvement on PC{i+1}:")
        print(f"    Jitter Improvement: {jitter_improvement:+.1f}% (positive means M4 is smoother)")
        print(f"    Efficiency Improvement: {efficiency_improvement:+.1f}% (positive means M4 is better)")
    
    return pc_jitters, pc_efficiencies

# =======================================================================
# 7. 主分析函数
# =======================================================================

def analyze_all_models():
    # 配置参数
    DATA_PATH = r"D:\Cu_BEM_2\tshirt-data\tshirt_kalman_3\dataset_arm"
    TEMPLATE_PATH = r"D:\Cu_BEM_2\tshirt-data\tshirt_kalman_3\tshirt_mech00001.obj"
    
    # 模型checkpoint路径
    MODEL_WITHOUT_GRASP_PATH = "checkpoints/best_model_single_template.pth"  # M5
    MODEL_WITH_GRASP_PATH = "checkpoints/best_model_simplified.pth"  # M4
    
    # 模型参数
    OBS_LATENT_DIM = 64
    GRASP_LATENT_DIM = 16
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print("="*60)
    print("五种模型平滑度指标对比分析")
    print("="*60)
    
    # 加载点云数据
    print("\n1. 加载点云数据集...")
    raw_files = glob.glob(os.path.join(DATA_PATH, "occ_frame_*.ply"))
    files = sorted(raw_files,
                  key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group())
                  if re.search(r'\d+', os.path.basename(x)) else 0)
    
    # 限制处理帧数（加快分析速度）
    max_frames = 700
    files = files[:max_frames]
    
    all_frames = []
    for i, f_path in enumerate(files):
        pcd = o3d.io.read_point_cloud(f_path)
        pts = np.asarray(pcd.points, dtype=np.float32)
        if len(pts) == 0:
            pts = np.zeros((2048, 3), dtype=np.float32)
        else:
            choice = np.random.choice(len(pts), 2048, replace=len(pts) < 2048)
            pts = pts[choice]
        all_frames.append(torch.from_numpy(pts).float().unsqueeze(0).to(device))
    
    T = len(all_frames)
    print(f"   已加载 {T} 帧点云数据")
    
    # 加载抓取点序列（仅M4需要）
    print("\n2. 加载抓取点序列...")
    GRASP_FILE = r"D:\Cu_BEM_2\tshirt-data\tshirt_kalman_3\vertex_237_trajectory.txt"
    
    def load_grasp_points(file_path):
        """加载抓取点序列"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        pattern = r'T=\s*([\d\.]+)[\s\S]*?X=\s*([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)'
        matches = re.findall(pattern, content)
        
        positions = []
        for match in matches:
            try:
                x = float(match[1])
                y = float(match[2])
                z = float(match[3])
                positions.append([x, y, z])
            except ValueError:
                pass
        
        return positions[:T]  # 只取与点云相同数量的帧
    
    grasp_positions = load_grasp_points(GRASP_FILE)
    all_grasp_points = []
    for pos in grasp_positions:
        all_grasp_points.append(torch.tensor(pos, dtype=torch.float32).to(device))
    
    print(f"   已加载 {len(all_grasp_points)} 帧抓取点数据")
    
    # =======================================================================
    # 初始化所有模型
    # =======================================================================
    
    # M5: 不带抓取点的GRU滤波器
    print("\n3. 加载不带抓取点的GRU滤波器 (M5)...")
    if os.path.exists(MODEL_WITHOUT_GRASP_PATH):
        checkpoint_without = torch.load(MODEL_WITHOUT_GRASP_PATH, map_location=device)
        state_dict_without = checkpoint_without['model_state_dict']
        
        # 初始化模型
        model_without_grasp = RecursiveTrackingNetWithoutGrasp(
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
        model_without_grasp.load_state_dict(state_dict_without, strict=False)
        model_without_grasp.eval()
        print(f"   已加载不带抓取点模型: {MODEL_WITHOUT_GRASP_PATH}")
        
        # M1: 原始观测 - 使用M5的编码器
        encoder = model_without_grasp.enc_obs
        encoder.eval()
    else:
        print(f"   错误: 不带抓取点的模型文件不存在: {MODEL_WITHOUT_GRASP_PATH}")
        return None, None, None
    
    # M4: 带抓取点的GRU滤波器
    print("\n4. 加载带抓取点的GRU滤波器 (M4)...")
    if os.path.exists(MODEL_WITH_GRASP_PATH):
        checkpoint_with = torch.load(MODEL_WITH_GRASP_PATH, map_location=device)
        state_dict_with = checkpoint_with['model_state_dict']
        
        # 初始化模型
        model_with_grasp = RecursiveTrackingNetWithGrasp(
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
        model_with_grasp.load_state_dict(state_dict_with, strict=False)
        model_with_grasp.eval()
        print(f"   已加载带抓取点模型: {MODEL_WITH_GRASP_PATH}")
    else:
        print(f"   错误: 带抓取点的模型文件不存在: {MODEL_WITH_GRASP_PATH}")
        return None, None, None
    
    # M2: 平滑损失监督模型
    smooth_model = SmoothLossSupervisedModel(latent_dim=OBS_LATENT_DIM, window_size=3)
    
    # M3: 匀速卡尔曼滤波器
    kf_model = ConstantVelocityKalmanFilter(dim=OBS_LATENT_DIM)
    
    # =======================================================================
    # 运行所有模型
    # =======================================================================
    
    print("\n5. 运行所有模型并提取潜状态序列...")
    
    # 存储结果
    results = {
        'M1': [],  # 原始观测
        'M2': [],  # 平滑损失监督
        'M3': [],  # 匀速卡尔曼滤波器
        'M4': [],  # 带抓取点的GRU滤波器
        'M5': []   # 不带抓取点的GRU滤波器
    }
    
    # 初始化GRU模型状态
    h_t_with_grasp = torch.zeros(1, OBS_LATENT_DIM).to(device)
    h_prev_with_grasp = None
    
    h_t_without_grasp = torch.zeros(1, OBS_LATENT_DIM).to(device)
    h_prev_without_grasp = None
    
    with torch.no_grad():
        for t in range(T):
            o_t = all_frames[t]  # 形状: [1, 2048, 3]
            
            # M1: 原始观测编码 - 使用M5的编码器
            z_obs = encoder(o_t)  # 注意：encoder内部会进行transpose
            z_obs_np = z_obs.squeeze(0).cpu().numpy()
            results['M1'].append(z_obs_np)
            
            # M3: 卡尔曼滤波器更新
            kf_filtered = kf_model.update(z_obs_np)
            results['M3'].append(kf_filtered)
            
            # M4: 带抓取点的GRU滤波器
            if t < len(all_grasp_points):
                g_t = all_grasp_points[t]
            else:
                g_t = torch.zeros(3, dtype=torch.float32).to(device)
            
            h_next_with_grasp, _ = model_with_grasp(o_t, g_t, h_t_with_grasp, h_prev_with_grasp)
            h_next_with_grasp_np = h_next_with_grasp.squeeze(0).cpu().numpy()
            results['M4'].append(h_next_with_grasp_np)
            
            # 更新M4状态
            h_prev_with_grasp = h_t_with_grasp.detach()
            h_t_with_grasp = h_next_with_grasp
            
            # M5: 不带抓取点的GRU滤波器
            h_next_without_grasp, _ = model_without_grasp(o_t, h_t_without_grasp, h_prev_without_grasp)
            h_next_without_grasp_np = h_next_without_grasp.squeeze(0).cpu().numpy()
            results['M5'].append(h_next_without_grasp_np)
            
            # 更新M5状态
            h_prev_without_grasp = h_t_without_grasp.detach()
            h_t_without_grasp = h_next_without_grasp
            
            if (t+1) % 50 == 0 or (t+1) == T:
                print(f"   进度: {t+1}/{T}")
    
    # M2: 对M1的结果进行平滑
    print("\n6. 计算M2（平滑损失监督模型）结果...")
    M1_sequence = np.array(results['M1'])
    M2_smoothed = smooth_model.smooth_sequence(M1_sequence)
    results['M2'] = M2_smoothed.tolist()
    
    # 转换为numpy数组
    for model_name in results:
        results[model_name] = np.array(results[model_name])
    
    # =======================================================================
    # 计算评估指标
    # =======================================================================
    
    print("\n7. 计算评估指标...")
    
    # 基于M1（原始观测）拟合PCA
    pca = PCA(n_components=OBS_LATENT_DIM)
    pca.fit(results['M1'])
    variance_ratios = pca.explained_variance_ratio_
    
    # 存储指标结果
    metrics = {}
    
    for model_name in ['M1', 'M2', 'M3', 'M4', 'M5']:
        sequence = results[model_name]
        
        # 对序列进行PCA变换
        sequence_pca = pca.transform(sequence)
        
        # 计算指标
        velocity_std, max_velocity = compute_velocity_statistics(sequence_pca)
        jitter = compute_jitter_score(sequence_pca)
        high_freq_ratio = compute_high_freq_energy_ratio(sequence_pca)
        info_efficiency = compute_information_efficiency(sequence_pca, variance_ratios)
        
        metrics[model_name] = {
            'velocity_std': velocity_std,
            'max_velocity': max_velocity,
            'jitter': jitter,
            'high_freq_ratio': high_freq_ratio,
            'info_efficiency': info_efficiency
        }
    
    # =======================================================================
    # 打印结果表格
    # =======================================================================
    
    print("\n" + "="*60)
    print("模型潜空间平滑度与效率对比表")
    print("="*60)
    print("\n评估指标 | M1: 原始观测 | M2: 平滑损失监督 | M3: 匀速卡尔曼滤波器 | M4: GRU递归滤波器（带抓取点） | M5: GRU递归滤波器（不带抓取点）")
    print("-" * 120)
    
    # 一阶差分（速度）统计
    print("一阶差分（速度）统计")
    print("速度标准差 (全局)")
    row = f"{' ':12} | "
    for model_name in ['M1', 'M2', 'M3', 'M4', 'M5']:
        row += f"{metrics[model_name]['velocity_std']:.6f} | "
    print(row)
    
    print("最大绝对速度 (全局)")
    row = f"{' ':12} | "
    for model_name in ['M1', 'M2', 'M3', 'M4', 'M5']:
        row += f"{metrics[model_name]['max_velocity']:.6f} | "
    print(row)
    
    print("\n二阶差分（抖动）统计")
    print("平均抖动值 (全局)")
    row = f"{' ':12} | "
    for model_name in ['M1', 'M2', 'M3', 'M4', 'M5']:
        row += f"{metrics[model_name]['jitter']:.6f} | "
    print(row)
    
    print("\n频谱特征")
    print("高频能量占比 (>Fs/4)")
    row = f"{' ':12} | "
    for model_name in ['M1', 'M2', 'M3', 'M4', 'M5']:
        row += f"{metrics[model_name]['high_freq_ratio']:.6f} | "
    print(row)
    
    print("\n信息保真度")
    print("信息效率 (全局平均)")
    row = f"{' ':12} | "
    for model_name in ['M1', 'M2', 'M3', 'M4', 'M5']:
        row += f"{metrics[model_name]['info_efficiency']:.6f} | "
    print(row)
    
    print("\n" + "="*60)
    
    # =======================================================================
    # 绘制子图
    # =======================================================================
    
    print("\n8. 绘制对比子图...")
    try:
        # 绘制PC1和PC2的时间演化图
        print("   绘制PC1和PC2时间演化图...")
        pca_sequences = plot_pc_temporal_evolution(results, variance_ratios, n_components_to_plot=2, max_frames=T)
        
        # 绘制抖动值和信息效率对比图
        print("   绘制抖动值和信息效率对比图...")
        pc_jitters, pc_efficiencies = plot_comparison_subplots(results, variance_ratios, n_components=10)
        print("   子图绘制完成")
    except Exception as e:
        print(f"   绘制子图时出错: {e}")
        import traceback
        traceback.print_exc()
    
    # =======================================================================
    # 保存结果到文件
    # =======================================================================
    
    print("\n9. 保存详细结果到文件...")
    output_file = "model_comparison_results.txt"
    with open(output_file, 'w') as f:
        f.write("模型潜空间平滑度与效率对比表\n")
        f.write("="*60 + "\n\n")
        
        f.write("模型说明:\n")
        f.write("M1: 原始观测 - 未经任何处理的观测编码序列\n")
        f.write("M2: 平滑损失监督模型 - 对原始观测序列进行移动平均平滑\n")
        f.write("M3: 匀速卡尔曼滤波器 - 基于匀速运动假设的经典卡尔曼滤波\n")
        f.write("M4: GRU递归滤波器（带抓取点） - 包含抓取点输入的状态转移模型\n")
        f.write("M5: GRU递归滤波器（不带抓取点） - 不包含抓取点输入的状态转移模型\n\n")
        
        f.write("评估指标 | M1: 原始观测 | M2: 平滑损失监督 | M3: 匀速卡尔曼滤波器 | M4: GRU递归滤波器（带抓取点） | M5: GRU递归滤波器（不带抓取点）\n")
        f.write("-" * 120 + "\n")
        
        f.write("一阶差分（速度）统计\n")
        f.write("速度标准差 (全局)\n")
        row = f"{' ':12} | "
        for model_name in ['M1', 'M2', 'M3', 'M4', 'M5']:
            row += f"{metrics[model_name]['velocity_std']:.6f} | "
        f.write(row + "\n")
        
        f.write("最大绝对速度 (全局)\n")
        row = f"{' ':12} | "
        for model_name in ['M1', 'M2', 'M3', 'M4', 'M5']:
            row += f"{metrics[model_name]['max_velocity']:.6f} | "
        f.write(row + "\n")
        
        f.write("\n二阶差分（抖动）统计\n")
        f.write("平均抖动值 (全局)\n")
        row = f"{' ':12} | "
        for model_name in ['M1', 'M2', 'M3', 'M4', 'M5']:
            row += f"{metrics[model_name]['jitter']:.6f} | "
        f.write(row + "\n")
        
        f.write("\n频谱特征\n")
        f.write("高频能量占比 (>Fs/4)\n")
        row = f"{' ':12} | "
        for model_name in ['M1', 'M2', 'M3', 'M4', 'M5']:
            row += f"{metrics[model_name]['high_freq_ratio']:.6f} | "
        f.write(row + "\n")
        
        f.write("\n信息保真度\n")
        f.write("信息效率 (全局平均)\n")
        row = f"{' ':12} | "
        for model_name in ['M1', 'M2', 'M3', 'M4', 'M5']:
            row += f"{metrics[model_name]['info_efficiency']:.6f} | "
        f.write(row + "\n")
        
        f.write("\n" + "="*60 + "\n")
        
        # 添加详细说明
        f.write("\n指标说明:\n")
        f.write("1. 速度标准差: 序列一阶差分（速度）的标准差，值越小表示速度变化越稳定\n")
        f.write("2. 最大绝对速度: 序列一阶差分（速度）的最大绝对值，值越小表示最大突变越小\n")
        f.write("3. 平均抖动值: 序列二阶差分（加速度）的绝对值的均值，核心平滑度指标，值越小越平滑\n")
        f.write("4. 高频能量占比: 序列FFT后频率高于采样频率1/4的成分能量占比，值越低表明高频噪声越少\n")
        f.write("5. 信息效率: 方差解释率/(抖动值+ε)，平衡平滑度与信息量，值越大越好\n")
    
    print(f"   详细结果已保存到: {output_file}")
    print("\n分析完成！")
    
    return metrics, results, variance_ratios

# =======================================================================
# 8. 主程序入口
# =======================================================================

if __name__ == "__main__":
    print("开始五种模型平滑度指标对比分析")
    print("="*60)
    
    # 检查模型文件是否存在
    MODEL_WITHOUT_GRASP_PATH = "checkpoints/best_model_single_template.pth"
    MODEL_WITH_GRASP_PATH = "checkpoints/best_model_simplified.pth"
    
    missing_models = []
    if not os.path.exists(MODEL_WITHOUT_GRASP_PATH):
        missing_models.append(f"M5 (不带抓取点): {MODEL_WITHOUT_GRASP_PATH}")
    if not os.path.exists(MODEL_WITH_GRASP_PATH):
        missing_models.append(f"M4 (带抓取点): {MODEL_WITH_GRASP_PATH}")
    
    if missing_models:
        print("警告: 以下模型文件不存在:")
        for model in missing_models:
            print(f"  - {model}")
        print("\n请确保模型文件存在，或修改模型路径。")
        print("分析将继续使用可用的模型...")
    
    # 运行分析
    try:
        metrics, results, variance_ratios = analyze_all_models()
        
        if metrics is not None and results is not None:
            print("\n" + "="*60)
            print("分析总结:")
            print("="*60)
            
            # 找出最优模型
            best_jitter_model = min(['M1', 'M2', 'M3', 'M4', 'M5'], 
                                    key=lambda x: metrics[x]['jitter'])
            best_efficiency_model = max(['M1', 'M2', 'M3', 'M4', 'M5'], 
                                        key=lambda x: metrics[x]['info_efficiency'])
            
            print(f"最优平滑度模型: {best_jitter_model} (抖动值: {metrics[best_jitter_model]['jitter']:.6f})")
            print(f"最优信息效率模型: {best_efficiency_model} (信息效率: {metrics[best_efficiency_model]['info_efficiency']:.6f})")
            
            # 对比M4和M5
            m4_jitter = metrics['M4']['jitter']
            m5_jitter = metrics['M5']['jitter']
            jitter_improvement = (m5_jitter - m4_jitter) / m5_jitter * 100 if m5_jitter > 0 else 0
            
            m4_efficiency = metrics['M4']['info_efficiency']
            m5_efficiency = metrics['M5']['info_efficiency']
            efficiency_improvement = (m4_efficiency - m5_efficiency) / m5_efficiency * 100 if m5_efficiency > 0 else 0
            
            print(f"\nM4 vs M5 对比:")
            print(f"  抖动改善率: {jitter_improvement:.1f}% (正值表示M4更平滑)")
            print(f"  信息效率改善率: {efficiency_improvement:.1f}% (正值表示M4更优)")
        else:
            print("\n分析失败，无法获取结果。")
            
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()