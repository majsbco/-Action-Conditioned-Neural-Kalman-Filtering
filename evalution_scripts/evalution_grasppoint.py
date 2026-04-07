import torch
import torch.nn as nn
import numpy as np
import os
import glob
import re
import open3d as o3d
from scipy.spatial import cKDTree
import trimesh
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

# =======================================================================
# 1. 模型定义 (严格遵循两个训练脚本)
# =======================================================================

class PointNetEncoderForM4(nn.Module):
    """M4模型的观测点云编码器 - 与training_script_kalman_2.py一致"""
    def __init__(self, latent_dim=64):
        super(PointNetEncoderForM4, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, 512, 1), nn.BatchNorm1d(512), nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(512, 256),  # fc.0
            nn.ReLU(),  # fc.1
            nn.Linear(256, latent_dim)  # fc.2
        )

    def forward(self, x):
        x = x.transpose(2, 1)
        x = self.mlp(x)
        x_max, _ = torch.max(x, dim=2)
        return self.fc(x_max)

class PointNetEncoderForM5(nn.Module):
    """M5模型的观测点云编码器 - 与training_script_kalman_3.py完全一致"""
    def __init__(self, latent_dim=64):
        super(PointNetEncoderForM5, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, 512, 1), nn.BatchNorm1d(512), nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(512, 256),  # fc.0
            nn.ReLU(),  # fc.1
            nn.LayerNorm(256),  # fc.2 - LayerNorm层
            nn.Linear(256, latent_dim)  # fc.3
        )

    def forward(self, x):
        x = x.transpose(2, 1)
        x = self.mlp(x)
        x = torch.max(x, 2)[0]  # 与训练脚本一致
        return self.fc(x)

# =======================================================================
# 1.1 M5模型 (KalmanTracker) - 遵循 training_script_kalman_3.py
# =======================================================================

class KalmanTrackerM5(nn.Module):
    """M5模型 - 不带抓取点输入，遵循Kalman逻辑"""
    def __init__(self, template_path, latent_dim=64):
        super(KalmanTrackerM5, self).__init__()
        
        # 加载模板
        mesh = trimesh.load(template_path, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = list(mesh.geometry.values())[0]
        
        pts = np.array(mesh.vertices, dtype=np.float32)
        # 减去中心
        v_mean = np.mean(pts, axis=0)
        pts_centered = pts - v_mean
        
        self.num_pts = pts_centered.shape[0]
        self.register_buffer('template', torch.from_numpy(pts_centered).float())

        # 编码器 - 使用M5专用编码器
        self.obs_encoder = PointNetEncoderForM5(latent_dim)
        
        # 门控生成器
        self.gate_gen = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),  # 输入128，输出64
            nn.Sigmoid()
        )
        
        # 解码器
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
        
        if h_t is None: 
            h_t = f_obs
        if h_prev is None: 
            h_prev = f_obs
            
        h_next = g_t * h_t + (1 - g_t) * h_prev
        
        h_expand = h_next.unsqueeze(1).expand(-1, self.num_pts, -1)
        template_expand = self.template.unsqueeze(0).expand(batch_size, -1, -1)
        
        decoder_input = torch.cat([h_expand, template_expand], dim=-1)
        offsets = self.decoder(decoder_input.view(-1, decoder_input.shape[-1]))
        offsets = offsets.view(batch_size, self.num_pts, 3)
        
        p_hat = template_expand + offsets
        return p_hat, h_next, offsets

# =======================================================================
# 1.2 M4模型 (RecursiveTrackingNetWithGrasp) - 遵循 training_script_kalman_2.py
# =======================================================================

class GraspPointEncoder(nn.Module):
    """抓取点编码器"""
    def __init__(self, grasp_latent_dim=16):
        super(GraspPointEncoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 16),  # 0
            nn.ReLU(),  # 1
            nn.Linear(16, 32),  # 2
            nn.ReLU(),  # 3
            nn.Linear(32, grasp_latent_dim)  # 4
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
            nn.Linear(obs_latent_dim + grasp_latent_dim, fused_dim),  # 0
            nn.ReLU(),  # 1
            nn.Linear(fused_dim, fused_dim)  # 2
        )
        
    def forward(self, z_obs, z_grasp):
        if z_obs.dim() == 1:
            z_obs = z_obs.unsqueeze(0)
        if z_grasp.dim() == 1:
            z_grasp = z_grasp.unsqueeze(0)
            
        z_fused = torch.cat([z_obs, z_grasp], dim=-1)
        return self.fusion_proj(z_fused)

class RecursiveTrackingNetWithGraspM4(nn.Module):
    """M4模型 - 带抓取点输入"""
    def __init__(self, M=2048, obs_latent_dim=64, grasp_latent_dim=16, 
                 damping=0.95, template_path=None):
        super(RecursiveTrackingNetWithGraspM4, self).__init__()
        self.obs_latent_dim = obs_latent_dim
        self.grasp_latent_dim = grasp_latent_dim
        self.M = M
        self.damping = damping 
        
        # 编码器 - 使用M4专用编码器
        self.enc_obs = PointNetEncoderForM4(obs_latent_dim)
        self.enc_grasp = GraspPointEncoder(grasp_latent_dim)
        self.fusion = FusionModule(obs_latent_dim, grasp_latent_dim, obs_latent_dim)
        
        # 状态转移
        self.f_gru = nn.GRUCell(obs_latent_dim, obs_latent_dim)
        
        # 门控网络
        self.gate = nn.Sequential(
            nn.Linear(obs_latent_dim * 2, 128),  # 0
            nn.ReLU(),  # 1
            nn.Linear(128, 1),  # 2
            nn.Sigmoid()  # 3
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
            nn.Linear(3 + obs_latent_dim, 512),  # 0
            nn.ReLU(),  # 1
            nn.Linear(512, 512),  # 2
            nn.ReLU(),  # 3
            nn.Linear(512, 3)  # 4
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
        前向传播 - 与训练脚本完全一致
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
        
        # 解码
        z_ext = h_next.unsqueeze(1).repeat(1, self.M, 1)
        template_ext = self.template_tensor.unsqueeze(0).repeat(B, 1, 1)
        
        decoder_input = torch.cat([template_ext, z_ext], dim=-1)
        p_hat = self.decoder(decoder_input.view(-1, decoder_input.shape[-1]))
        p_hat = p_hat.view(B, self.M, 3)
        
        return p_hat, h_next

# =======================================================================
# 2. 辅助函数
# =======================================================================

def load_point_cloud(file_path, n_points=2048):
    """加载点云文件并采样到固定点数"""
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points, dtype=np.float32)
    
    if len(points) == 0:
        return np.zeros((n_points, 3), dtype=np.float32)
    
    if len(points) > n_points:
        indices = np.random.choice(len(points), n_points, replace=False)
        points = points[indices]
    elif len(points) < n_points:
        points = np.tile(points, (n_points // len(points) + 1, 1))[:n_points]
    
    return points

def parse_grasp_trajectory(file_path):
    """解析抓取点轨迹文件"""
    positions = []
    if not os.path.exists(file_path):
        print(f"警告: 抓取点文件不存在: {file_path}")
        return np.zeros((1000, 3), dtype=np.float32)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    matches = re.findall(r'X=\s*([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)', content)
    for match in matches:
        positions.append([float(match[0]), float(match[1]), float(match[2])])
    
    return np.array(positions, dtype=np.float32)

def compute_template_diagonal(template_path):
    """计算模板包围盒体对角线长度 (单位: cm)"""
    if not os.path.exists(template_path):
        print(f"错误: 模板文件不存在: {template_path}")
        return None
    
    mesh = trimesh.load(template_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = list(mesh.geometry.values())[0]
    
    vertices = mesh.vertices.astype(np.float32)
    if len(vertices) == 0:
        print(f"错误: 模板文件为空: {template_path}")
        return None
    
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    diagonal_length = np.linalg.norm(max_coords - min_coords)
    
    return diagonal_length

# ======================== 关键修改点 ======================== #
# 原来的 compute_grasp_point_distance 函数替换为新的区域计算函数
def compute_grasp_region_average_distance(pred_points, gt_points, grasp_point, template_diagonal, region_radius_cm=5.0, min_points=5):
    """
    计算抓取点球形区域内的平均倒角距离百分比
    
    参数:
        pred_points: 预测点云 [N, 3] (单位: cm)
        gt_points: 真实点云 [M, 3] (单位: cm)
        grasp_point: 抓取点坐标 [3] (单位: cm)
        template_diagonal: 模板对角线长度 (单位: cm)
        region_radius_cm: 球形区域半径 (单位: cm，默认5.0cm)
        min_points: 区域内的最小点数要求
    
    返回:
        区域平均误差百分比
    """
    # 1. 中心对齐 (与之前一致)
    pred_centered = pred_points - np.mean(pred_points, axis=0)
    gt_centered = gt_points - np.mean(gt_points, axis=0)
    grasp_centered = grasp_point - np.mean(gt_points, axis=0)
    
    # 2. 创建KDTree用于区域查询
    tree_pred = cKDTree(pred_centered)
    tree_gt = cKDTree(gt_centered)
    
    # 3. 查找抓取点半径内的所有点索引
    indices_pred = tree_pred.query_ball_point(grasp_centered, region_radius_cm)
    indices_gt = tree_gt.query_ball_point(grasp_centered, region_radius_cm)
    
    # 检查结果是否为嵌套列表（单个查询点的情况）
    if indices_pred and isinstance(indices_pred[0], list):
        indices_pred = indices_pred[0]
    if indices_gt and isinstance(indices_gt[0], list):
        indices_gt = indices_gt[0]
    
    # 4. 检查区域内是否有足够的点
    if len(indices_pred) < min_points or len(indices_gt) < min_points:
        # 如果点数不足，退回使用最近点计算
        dist_pred, idx_pred = tree_pred.query(grasp_centered.reshape(1, 3))
        dist_gt, idx_gt = tree_gt.query(grasp_centered.reshape(1, 3))
        grasp_distance = np.linalg.norm(pred_centered[idx_pred[0]] - gt_centered[idx_gt[0]])
        
        # 调试信息
        if len(indices_pred) < min_points or len(indices_gt) < min_points:
            print(f"  警告: 半径{region_radius_cm}cm内点数不足 (pred:{len(indices_pred)}, gt:{len(indices_gt)}), 退回最近点计算")
    else:
        # 5. 提取区域内的点
        region_points_pred = pred_centered[indices_pred]
        region_points_gt = gt_centered[indices_gt]
        
        # 6. 计算两个区域点集之间的倒角距离
        # 方法1: 计算区域中心之间的距离
        center_pred = np.mean(region_points_pred, axis=0)
        center_gt = np.mean(region_points_gt, axis=0)
        grasp_distance = np.linalg.norm(center_pred - center_gt)
    
    # 7. 转换为百分比
    region_cd_percentage = (grasp_distance / template_diagonal) * 100
    
    return region_cd_percentage

def compute_step_errors(grasp_errors, num_steps=50):
    """将帧级误差转换为步级误差"""
    total_frames = len(grasp_errors)
    frames_per_step = total_frames // num_steps
    
    step_errors = []
    for i in range(num_steps):
        start_idx = i * frames_per_step
        end_idx = (i + 1) * frames_per_step if i < num_steps - 1 else total_frames
        step_avg_error = np.mean(grasp_errors[start_idx:end_idx])
        step_errors.append(step_avg_error)
    
    return np.array(step_errors)

def plot_m4_heatmap_alone(step_errors_m4, max_percentage=10.0, colormap='viridis', 
                         bar_width=1.0, fig_width=20, fig_height=3):
    """绘制M4模型热力图 (单独的图片)"""
    num_steps = len(step_errors_m4)
    time_axis = np.arange(num_steps)
    
    # 获取颜色映射
    cmap = cm.get_cmap(colormap)
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    
    # 绘制横向热力图
    for i, (t, error) in enumerate(zip(time_axis, step_errors_m4)):
        # 计算归一化值
        norm_error = np.clip(error / max_percentage, 0, 1)
        # 获取颜色
        color = cmap(norm_error)
        # 绘制矩形
        ax.add_patch(plt.Rectangle((t - bar_width/2, 0), bar_width, 1, 
                                  color=color, edgecolor='none'))
    
    ax.set_xlim(time_axis[0] - bar_width/2, time_axis[-1] + bar_width/2)
    ax.set_ylim(0, 1)
    ax.set_ylabel('M4\n(with grasp)', fontsize=18, fontweight='bold')
    ax.set_xlabel('Step Index', fontsize=16)
    ax.set_title('M4: Grasp Point CD% by Steps (0-10%)', fontsize=20, fontweight='bold', pad=20)
    
    # 设置y轴
    ax.set_yticks([])
    
    # 设置x轴刻度
    n_ticks = min(20, num_steps)
    tick_indices = np.linspace(0, num_steps-1, n_ticks, dtype=int)
    tick_labels = [str(i+1) for i in tick_indices]
    ax.set_xticks(time_axis[tick_indices])
    ax.set_xticklabels(tick_labels, fontsize=14)
    
    # 添加网格线
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    save_path = 'm4_grasp_point_heatmap_by_steps.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nM4抓取点热力图已保存至: {save_path}")
    
    # 显示图表
    plt.show()
    
    return fig

def plot_m5_heatmap_alone(step_errors_m5, max_percentage=10.0, colormap='viridis',
                         bar_width=1.0, fig_width=20, fig_height=3):
    """绘制M5模型热力图 (单独的图片)"""
    num_steps = len(step_errors_m5)
    time_axis = np.arange(num_steps)
    
    # 获取颜色映射
    cmap = cm.get_cmap(colormap)
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    
    # 绘制横向热力图
    for i, (t, error) in enumerate(zip(time_axis, step_errors_m5)):
        # 计算归一化值
        norm_error = np.clip(error / max_percentage, 0, 1)
        # 获取颜色
        color = cmap(norm_error)
        # 绘制矩形
        ax.add_patch(plt.Rectangle((t - bar_width/2, 0), bar_width, 1, 
                                  color=color, edgecolor='none'))
    
    ax.set_xlim(time_axis[0] - bar_width/2, time_axis[-1] + bar_width/2)
    ax.set_ylim(0, 1)
    ax.set_ylabel('M5\n(without grasp)', fontsize=18, fontweight='bold')
    ax.set_xlabel('Step Index', fontsize=16)
    ax.set_title('M5: Grasp Point CD% by Steps (0-10%)', fontsize=20, fontweight='bold', pad=20)
    
    # 设置y轴
    ax.set_yticks([])
    
    # 设置x轴刻度
    n_ticks = min(20, num_steps)
    tick_indices = np.linspace(0, num_steps-1, n_ticks, dtype=int)
    tick_labels = [str(i+1) for i in tick_indices]
    ax.set_xticks(time_axis[tick_indices])
    ax.set_xticklabels(tick_labels, fontsize=14)
    
    # 添加网格线
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    save_path = 'm5_grasp_point_heatmap_by_steps.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nM5抓取点热力图已保存至: {save_path}")
    
    # 显示图表
    plt.show()
    
    return fig

def plot_colorbar_alone(max_percentage=10.0, colormap='viridis', fig_width=12, fig_height=1.5):
    """绘制独立的基准颜色条图片"""
    # 获取颜色映射
    cmap = cm.get_cmap(colormap)
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    
    # 创建归一化器
    norm = Normalize(vmin=0, vmax=max_percentage)
    
    # 创建颜色条
    cb = ColorbarBase(ax, cmap=cmap, norm=norm, orientation='horizontal')
    cb.set_label(f'CD as % of Template Diagonal (0-{max_percentage}%)', fontsize=12, fontweight='bold')
    
    # 设置刻度
    ticks = np.linspace(0, max_percentage, 6)
    cb.set_ticks(ticks)
    cb.set_ticklabels([f'{tick:.1f}%' for tick in ticks], fontsize=14)
    
    # 设置标题
    ax.set_title('Color Scale Reference', fontsize=20, fontweight='bold', pad=20)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    save_path = 'colorbar_reference_by_steps.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n基准颜色条已保存至: {save_path}")
    
    # 显示图表
    plt.show()
    
    return fig

# =======================================================================
# 3. 主分析函数
# =======================================================================

def analyze_grasp_point_error_by_steps():
    """
    分析抓取点误差的步级演化
    将700帧数据划分为50个步骤，比较M4和M5模型
    """
    # 配置参数
    DATA_PATH = r"D:\Cu_BEM_2\tshirt-data\tshirt_kalman_3\dataset_arm"
    TEMPLATE_PATH = r"D:\Cu_BEM_2\tshirt-data\tshirt_kalman_3\tshirt_mech00001.obj"
    
    # 模型checkpoint路径
    MODEL_M5_PATH = "checkpoints/kalman_best_aligned.pth"  # M5
    MODEL_M4_PATH = "checkpoints/best_model_simplified.pth"  # M4
    
    GRASP_FILE = r"D:\Cu_BEM_2\tshirt-data\tshirt_kalman_3\vertex_237_trajectory.txt"
    
    # 训练脚本中的全局中心点
    GLOBAL_CENTROID = np.array([-5.8771, 115.0959, 6.6215], dtype=np.float32)
    
    # 步级参数
    NUM_STEPS = 50
    TOTAL_FRAMES = 700
    
    # 区域计算参数 - 新增参数
    REGION_RADIUS_CM = 10  # 球形区域半径 5cm
    MIN_POINTS_IN_REGION = 10  # 区域内的最小点数要求
    
    # 可视化参数
    COLORMAP = 'viridis'
    MAX_PERCENTAGE = 5.0  # 调整为0-5%范围
    BAR_WIDTH = 1.0
    FIG_WIDTH = 20
    FIG_HEIGHT = 3
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print("="*60)
    print("抓取点误差步级演化分析 (区域平均方法)")
    print("="*60)
    print(f"总帧数: {TOTAL_FRAMES}")
    print(f"步数: {NUM_STEPS}")
    print(f"每步帧数: {TOTAL_FRAMES // NUM_STEPS}")
    print(f"区域半径: {REGION_RADIUS_CM} cm")
    print(f"颜色条范围: 0.0% - {MAX_PERCENTAGE:.1f}%")
    print("="*60)
    
    # 1. 计算模板对角线
    print("\n1. 计算模板包围盒体对角线...")
    template_diagonal = compute_template_diagonal(TEMPLATE_PATH)
    if template_diagonal is None:
        return
    
    print(f"   模板对角线长度: {template_diagonal:.6f} cm")
    print(f"   5cm半径占模板对角线的: {5.0/template_diagonal*100:.2f}%")
    
    # 2. 加载点云数据
    print("\n2. 加载点云数据集...")
    raw_files = glob.glob(os.path.join(DATA_PATH, "occ_frame_*.ply"))
    files = sorted(raw_files,
                  key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group())
                  if re.search(r'\d+', os.path.basename(x)) else 0)
    
    files = files[:TOTAL_FRAMES]  # 只取前700帧
    
    all_frames_raw = []
    for i, f_path in enumerate(files):
        pts = load_point_cloud(f_path, 2048)
        all_frames_raw.append(pts)
    
    T = len(all_frames_raw)
    print(f"   已加载 {T} 帧点云数据")
    
    # 3. 加载抓取点序列
    print("\n3. 加载抓取点序列...")
    grasp_positions_raw = parse_grasp_trajectory(GRASP_FILE)[:T]
    print(f"   已加载 {len(grasp_positions_raw)} 帧抓取点数据")
    
    # 4. 数据预处理: 中心对齐
    print("\n4. 数据预处理: 中心对齐...")
    all_frames = []
    all_grasp_points = []
    for i in range(T):
        pts_norm = all_frames_raw[i] - GLOBAL_CENTROID
        all_frames.append(torch.from_numpy(pts_norm).float().unsqueeze(0).to(device))
        
        grasp_norm = grasp_positions_raw[i] - GLOBAL_CENTROID
        all_grasp_points.append(torch.from_numpy(grasp_norm).float().to(device))
    
    # 5. 加载模型
    print("\n5. 加载模型...")
    
    # M5模型
    if os.path.exists(MODEL_M5_PATH):
        checkpoint = torch.load(MODEL_M5_PATH, map_location=device)
        model_m5 = KalmanTrackerM5(template_path=TEMPLATE_PATH, latent_dim=64).to(device)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # 打印模型结构以验证
        print("M5模型结构验证:")
        print(f"  obs_encoder.fc.2类型: {type(model_m5.obs_encoder.fc[2])}")
        print(f"  obs_encoder.fc.2.weight形状: {model_m5.obs_encoder.fc[2].weight.shape}")
        print(f"  obs_encoder.fc.3.weight形状: {model_m5.obs_encoder.fc[3].weight.shape}")
        
        # 加载状态字典
        model_m5.load_state_dict(state_dict, strict=True)
        model_m5.eval()
        print(f"   已成功加载M5模型: {MODEL_M5_PATH}")
    else:
        print(f"   错误: M5模型文件不存在: {MODEL_M5_PATH}")
        return
    
    # M4模型
    if os.path.exists(MODEL_M4_PATH):
        checkpoint = torch.load(MODEL_M4_PATH, map_location=device)
        model_m4 = RecursiveTrackingNetWithGraspM4(
            obs_latent_dim=64, grasp_latent_dim=16, template_path=TEMPLATE_PATH
        ).to(device)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # 打印模型结构以验证
        print("\nM4模型结构验证:")
        print(f"  enc_obs.fc.2.weight形状: {model_m4.enc_obs.fc[2].weight.shape}")
        
        # 加载状态字典
        model_m4.load_state_dict(state_dict, strict=True)
        model_m4.eval()
        print(f"   已成功加载M4模型: {MODEL_M4_PATH}")
    else:
        print(f"   错误: M4模型文件不存在: {MODEL_M4_PATH}")
        return
    
    # 6. 运行模型并计算误差
    print("\n6. 运行模型并计算区域平均误差...")
    print(f"   区域半径: {REGION_RADIUS_CM} cm")
    print(f"   最小点数要求: {MIN_POINTS_IN_REGION}")
    
    # 存储结果
    grasp_errors_m4 = []
    grasp_errors_m5 = []
    
    # 存储区域点数信息用于调试
    region_stats = {
        'm4_points_pred': [],
        'm4_points_gt': [],
        'm5_points_pred': [],
        'm5_points_gt': []
    }
    
    # 初始化状态
    h_t_m4 = torch.zeros(1, 64).to(device)
    h_prev_m4 = None
    
    h_t_m5 = torch.zeros(1, 64).to(device)
    h_prev_m5 = None
    
    with torch.no_grad():
        for t in range(T):
            o_t = all_frames[t]
            g_t = all_grasp_points[t]
            
            # 对于M5模型，需要obs和goal两个输入
            if t < T - 1:
                goal_t = all_frames[t + 1]
            else:
                goal_t = o_t
            
            # 运行M5模型
            p_hat_m5_norm, h_next_m5, _ = model_m5(o_t, goal_t, h_t_m5, h_prev_m5)
            p_hat_m5 = p_hat_m5_norm[0].cpu().numpy()
            
            # 运行M4模型
            p_hat_m4_norm, h_next_m4 = model_m4(o_t, g_t.unsqueeze(0), h_t_m4, h_prev_m4)
            p_hat_m4 = p_hat_m4_norm[0].cpu().numpy()
            
            o_t_np = o_t[0].cpu().numpy()
            g_t_np = g_t.cpu().numpy()
            
            # ======================== 关键修改点 ======================== #
            # 计算抓取点区域平均误差百分比 (使用新的区域计算方法)
            grasp_error_m4 = compute_grasp_region_average_distance(
                p_hat_m4, o_t_np, g_t_np, template_diagonal, 
                region_radius_cm=REGION_RADIUS_CM, min_points=MIN_POINTS_IN_REGION
            )
            grasp_error_m5 = compute_grasp_region_average_distance(
                p_hat_m5, o_t_np, g_t_np, template_diagonal,
                region_radius_cm=REGION_RADIUS_CM, min_points=MIN_POINTS_IN_REGION
            )
            
            grasp_errors_m4.append(grasp_error_m4)
            grasp_errors_m5.append(grasp_error_m5)
            
            # 更新状态
            h_prev_m4, h_t_m4 = h_t_m4, h_next_m4
            h_prev_m5, h_t_m5 = h_t_m5, h_next_m5
            
            if (t+1) % 50 == 0 or (t+1) == T:
                print(f"   进度: {t+1}/{T}")
    
    # 转换为numpy数组
    grasp_errors_m4 = np.array(grasp_errors_m4)
    grasp_errors_m5 = np.array(grasp_errors_m5)
    
    # 7. 将帧级误差转换为步级误差
    print(f"\n7. 将帧级误差转换为步级误差 ({NUM_STEPS}个步骤)...")
    
    step_errors_m4 = compute_step_errors(grasp_errors_m4, NUM_STEPS)
    step_errors_m5 = compute_step_errors(grasp_errors_m5, NUM_STEPS)
    
    print(f"   每个步骤包含 {T // NUM_STEPS} 帧")
    print(f"   M4步级误差范围: [{np.min(step_errors_m4):.2f}%, {np.max(step_errors_m4):.2f}%]")
    print(f"   M5步级误差范围: [{np.min(step_errors_m5):.2f}%, {np.max(step_errors_m5):.2f}%]")
    
    # 8. 绘制三张独立的图片
    print("\n8. 绘制三张独立的图片...")
    
    # 绘制M4热力图
    fig_m4 = plot_m4_heatmap_alone(
        step_errors_m4, 
        max_percentage=MAX_PERCENTAGE, colormap=COLORMAP,
        bar_width=BAR_WIDTH, fig_width=FIG_WIDTH, fig_height=FIG_HEIGHT
    )
    
    # 绘制M5热力图
    fig_m5 = plot_m5_heatmap_alone(
        step_errors_m5, 
        max_percentage=MAX_PERCENTAGE, colormap=COLORMAP,
        bar_width=BAR_WIDTH, fig_width=FIG_WIDTH, fig_height=FIG_HEIGHT
    )
    
    # 绘制独立颜色条
    fig_colorbar = plot_colorbar_alone(
        max_percentage=MAX_PERCENTAGE, colormap=COLORMAP,
        fig_width=12, fig_height=1.5
    )
    
    # 9. 输出统计结果
    print("\n" + "="*60)
    print("抓取点区域平均误差统计结果")
    print("="*60)
    
    print(f"分析总帧数: {T}")
    print(f"步数: {NUM_STEPS}")
    print(f"每步帧数: {T // NUM_STEPS}")
    print(f"模板对角线长度: {template_diagonal:.6f} cm")
    print(f"区域半径: {REGION_RADIUS_CM} cm")
    print(f"颜色条范围: 0-{MAX_PERCENTAGE:.1f}%")
    
    print(f"\nM4 (带抓取点) 模型抓取点区域误差统计:")
    print(f"  平均帧级误差: {np.mean(grasp_errors_m4):.2f}%")
    print(f"  平均步级误差: {np.mean(step_errors_m4):.2f}%")
    print(f"  最大步级误差: {np.max(step_errors_m4):.2f}% (步 {np.argmax(step_errors_m4) + 1})")
    print(f"  最小步级误差: {np.min(step_errors_m4):.2f}% (步 {np.argmin(step_errors_m4) + 1})")
    print(f"  标准差: {np.std(step_errors_m4):.2f}%")
    
    print(f"\nM5 (不带抓取点) 模型抓取点区域误差统计:")
    print(f"  平均帧级误差: {np.mean(grasp_errors_m5):.2f}%")
    print(f"  平均步级误差: {np.mean(step_errors_m5):.2f}%")
    print(f"  最大步级误差: {np.max(step_errors_m5):.2f}% (步 {np.argmax(step_errors_m5) + 1})")
    print(f"  最小步级误差: {np.min(step_errors_m5):.2f}% (步 {np.argmin(step_errors_m5) + 1})")
    print(f"  标准差: {np.std(step_errors_m5):.2f}%")
    
    # 比较M4和M5
    mean_step_improvement = ((np.mean(step_errors_m5) - np.mean(step_errors_m4)) / np.mean(step_errors_m5)) * 100
    max_step_improvement = ((np.max(step_errors_m5) - np.max(step_errors_m4)) / np.max(step_errors_m5)) * 100
    
    print(f"\nM4 vs M5 步级对比:")
    print(f"  平均步级误差改善: {mean_step_improvement:+.1f}% (正值表示M4更优)")
    print(f"  最大步级误差改善: {max_step_improvement:+.1f}% (正值表示M4更优)")
    
    # 10. 保存详细结果到文件
    print("\n9. 保存详细结果到文件...")
    output_file = "grasp_region_error_results_by_steps.txt"
    with open(output_file, 'w') as f:
        f.write("抓取点区域平均误差步级演化分析结果\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"分析总帧数: {T}\n")
        f.write(f"步数: {NUM_STEPS}\n")
        f.write(f"每步帧数: {T // NUM_STEPS}\n")
        f.write(f"模板对角线长度: {template_diagonal:.6f} cm\n")
        f.write(f"区域半径: {REGION_RADIUS_CM} cm\n")
        f.write(f"颜色条范围: 0-{MAX_PERCENTAGE:.1f}%\n\n")
        
        f.write("M4 (带抓取点) 模型抓取点区域误差统计:\n")
        f.write(f"  平均帧级误差: {np.mean(grasp_errors_m4):.2f}%\n")
        f.write(f"  平均步级误差: {np.mean(step_errors_m4):.2f}%\n")
        f.write(f"  最大步级误差: {np.max(step_errors_m4):.2f}% (步 {np.argmax(step_errors_m4) + 1})\n")
        f.write(f"  最小步级误差: {np.min(step_errors_m4):.2f}% (步 {np.argmin(step_errors_m4) + 1})\n")
        f.write(f"  标准差: {np.std(step_errors_m4):.2f}%\n\n")
        
        f.write("M5 (不带抓取点) 模型抓取点区域误差统计:\n")
        f.write(f"  平均帧级误差: {np.mean(grasp_errors_m5):.2f}%\n")
        f.write(f"  平均步级误差: {np.mean(step_errors_m5):.2f}%\n")
        f.write(f"  最大步级误差: {np.max(step_errors_m5):.2f}% (步 {np.argmax(step_errors_m5) + 1})\n")
        f.write(f"  最小步级误差: {np.min(step_errors_m5):.2f}% (步 {np.argmin(step_errors_m5) + 1})\n")
        f.write(f"  标准差: {np.std(step_errors_m5):.2f}%\n\n")
        
        f.write("M4 vs M5 步级对比:\n")
        f.write(f"  平均步级误差改善: {mean_step_improvement:+.1f}%\n")
        f.write(f"  最大步级误差改善: {max_step_improvement:+.1f}%\n\n")
        
        f.write("逐步抓取点区域误差数据:\n")
        f.write("步索引, M4步级误差(%), M5步级误差(%)\n")
        for i in range(NUM_STEPS):
            f.write(f"{i+1}, {step_errors_m4[i]:.2f}, {step_errors_m5[i]:.2f}\n")
    
    print(f"   详细结果已保存到: {output_file}")
    print("\n分析完成！")
    
    return step_errors_m4, step_errors_m5, grasp_errors_m4, grasp_errors_m5

# =======================================================================
# 4. 主程序入口
# =======================================================================

if __name__ == "__main__":
    print("开始抓取点区域平均误差步级演化分析")
    print("="*60)
    
    # 检查模型文件是否存在
    MODEL_M5_PATH = "checkpoints/kalman_best_aligned.pth"
    MODEL_M4_PATH = "checkpoints/best_model_simplified.pth"
    
    missing_models = []
    if not os.path.exists(MODEL_M5_PATH):
        missing_models.append(f"M5: {MODEL_M5_PATH}")
    if not os.path.exists(MODEL_M4_PATH):
        missing_models.append(f"M4: {MODEL_M4_PATH}")
    
    if missing_models:
        print("警告: 以下模型文件不存在:")
        for model in missing_models:
            print(f"  - {model}")
        print("\n请确保模型文件存在，或修改模型路径。")
    
    # 运行分析
    try:
        step_errors_m4, step_errors_m5, grasp_errors_m4, grasp_errors_m5 = analyze_grasp_point_error_by_steps()
        
        if step_errors_m4 is not None and step_errors_m5 is not None:
            print("\n" + "="*60)
            print("分析总结:")
            print("="*60)
            
            # 找出最优模型
            if np.mean(step_errors_m4) < np.mean(step_errors_m5):
                print(f"最优模型: M4 (带抓取点)")
                print(f"  平均步级误差: {np.mean(step_errors_m4):.2f}% vs M5: {np.mean(step_errors_m5):.2f}%")
                improvement = ((np.mean(step_errors_m5) - np.mean(step_errors_m4)) / np.mean(step_errors_m5)) * 100
                print(f"  相对改进: {improvement:.1f}%")
            else:
                print(f"最优模型: M5 (不带抓取点)")
                print(f"  平均步级误差: {np.mean(step_errors_m5):.2f}% vs M4: {np.mean(step_errors_m4):.2f}%")
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()