import torch
import torch.nn as nn
import numpy as np
import os
import glob
import re
import open3d as o3d
from scipy.spatial import cKDTree
import trimesh
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize, LogNorm
import matplotlib
import warnings

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

# 修复MatplotlibDeprecationWarning
matplotlib.colormaps.get_cmap = lambda x: cm.get_cmap(x)

# =======================================================================
# 1. 模型定义 (与您的训练脚本完全一致)
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
    更新版本：包含抓取点输入
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
        前向传播（更新：增加抓取点输入）
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
# 2. 辅助函数
# =======================================================================

def compute_gradient_sensitivity_simple(model, o_t, g_t, h_prev, h_prev_prev=None):
    """
    简化但正确的梯度计算函数
    """
    # 将抓取点设置为需要梯度
    g_t_requires_grad = g_t.clone().detach().requires_grad_(True)
    
    # 临时将模型设置为训练模式
    original_mode = model.training
    model.train()
    
    # 前向传播
    pred_points, _ = model(o_t, g_t_requires_grad, h_prev, h_prev_prev)
    
    B, N, D = pred_points.shape
    
    # 创建一个与预测点云相同形状的单位权重
    # 这将计算所有点对所有坐标的梯度
    weights = torch.ones_like(pred_points)
    
    # 计算加权和
    weighted_sum = torch.sum(pred_points * weights)
    
    # 计算梯度
    gradient = torch.autograd.grad(
        outputs=weighted_sum,
        inputs=g_t_requires_grad,
        create_graph=False,
        retain_graph=True
    )[0]
    
    if gradient is None:
        print("警告: 整体梯度计算返回None")
        # 尝试另一种方法：计算每个点的梯度
        gradient_vectors = torch.zeros((B, N, 3), device=pred_points.device)
        
        for i in range(N):
            # 计算第i个点对g_t的梯度
            point_grad = torch.autograd.grad(
                outputs=pred_points[0, i].sum(),  # 点的三个坐标之和
                inputs=g_t_requires_grad,
                retain_graph=True,
                allow_unused=True
            )[0]
            
            if point_grad is not None:
                gradient_vectors[0, i, :] = point_grad
            else:
                print(f"点{i}的梯度为None")
        
        gradient_norms = torch.norm(gradient_vectors[0], dim=1)
    else:
        # 梯度形状应为[3]，但我们需要每个点的梯度
        # 这里我们假设所有点对g_t的梯度相同（简化假设）
        gradient_vectors = gradient.unsqueeze(0).unsqueeze(0).expand(B, N, 3)
        gradient_norms = torch.norm(gradient_vectors[0], dim=1)
    
    # 恢复模型原来的模式
    model.train(original_mode)
    
    return pred_points.detach(), gradient_norms.detach(), gradient_vectors.detach()

def compute_gradient_per_point(model, o_t, g_t, h_prev, h_prev_prev=None):
    """
    计算每个点对抓取点的梯度
    """
    # 将抓取点设置为需要梯度
    g_t_requires_grad = g_t.clone().detach().requires_grad_(True)
    
    # 临时将模型设置为训练模式
    original_mode = model.training
    model.train()
    
    # 前向传播
    pred_points, _ = model(o_t, g_t_requires_grad, h_prev, h_prev_prev)
    
    B, N, D = pred_points.shape
    
    # 为每个点计算梯度
    gradient_vectors = torch.zeros((B, N, 3), device=pred_points.device)
    
    for i in range(N):
        # 为第i个点创建梯度计算图
        point_i = pred_points[0, i]  # 形状[3]
        
        # 计算这个点的梯度
        grad_i = torch.autograd.grad(
            outputs=point_i,  # 输出是三维向量
            inputs=g_t_requires_grad,
            grad_outputs=torch.ones_like(point_i),  # 每个输出分量的权重
            retain_graph=(i < N-1),  # 最后一个点后不保留图
            allow_unused=True
        )[0]
        
        if grad_i is not None:
            gradient_vectors[0, i, :] = grad_i
        else:
            print(f"警告: 点{i}的梯度为None")
    
    # 计算梯度范数
    gradient_norms = torch.norm(gradient_vectors[0], dim=1)
    
    # 恢复模型原来的模式
    model.train(original_mode)
    
    return pred_points.detach(), gradient_norms.detach(), gradient_vectors.detach()

def compute_gradient_heatmap(gradient_norms, colormap='viridis', use_log_scale=True):
    """
    将梯度范数转换为热力图颜色
    """
    # 转换为numpy
    gradient_norms_np = gradient_norms.cpu().numpy() if torch.is_tensor(gradient_norms) else gradient_norms
    
    # 避免零或负值
    eps = 1e-8
    gradient_norms_np = gradient_norms_np + eps
    
    if use_log_scale:
        # 使用对数尺度
        gradient_norms_np = np.log10(gradient_norms_np)
        min_val = np.min(gradient_norms_np)
        max_val = np.max(gradient_norms_np)
        
        # 归一化到[0, 1]
        if max_val > min_val:
            normalized_gradients = (gradient_norms_np - min_val) / (max_val - min_val)
        else:
            normalized_gradients = np.zeros_like(gradient_norms_np)
    else:
        # 使用线性尺度
        min_val = np.min(gradient_norms_np)
        max_val = np.max(gradient_norms_np)
        
        # 归一化到[0, 1]
        if max_val > min_val:
            normalized_gradients = (gradient_norms_np - min_val) / (max_val - min_val)
        else:
            normalized_gradients = np.zeros_like(gradient_norms_np)
    
    # 使用指定的颜色映射
    cmap = plt.cm.get_cmap(colormap)
    heatmap_colors = cmap(normalized_gradients)[:, :3]  # 只取RGB，忽略alpha通道
    
    return heatmap_colors, normalized_gradients, (min_val, max_val)

def create_gradient_colorbar(min_val, max_val, colormap='viridis', use_log_scale=True, save_path='gradient_colorbar.png'):
    """
    创建梯度颜色条图例（横版）
    """
    fig, ax = plt.subplots(figsize=(8, 1.2))
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.4, top=0.7)
    
    cmap = plt.cm.get_cmap(colormap)
    
    if use_log_scale:
        norm = LogNorm(vmin=10**min_val, vmax=10**max_val)
        ticks = np.logspace(np.log10(10**min_val), np.log10(10**max_val), 7)
        tick_labels = [f'{tick:.1e}' for tick in ticks]
    else:
        norm = Normalize(vmin=10**min_val, vmax=10**max_val) if min_val < 0 else Normalize(vmin=min_val, vmax=max_val)
        ticks = np.linspace(min_val, max_val, 7)
        tick_labels = [f'{tick:.1e}' for tick in ticks]
    
    cb = ColorbarBase(ax, cmap=cmap, norm=norm, orientation='horizontal')
    cb.set_label('Gradient Magnitude (Sensitivity to grasp point)', fontsize=12, fontweight='bold')
    
    cb.set_ticks(ticks)
    cb.set_ticklabels(tick_labels, fontsize=10)
    
    scale_type = "Log Scale" if use_log_scale else "Linear Scale"
    title = f'Gradient Sensitivity ({scale_type})'
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"梯度颜色条已保存至: {save_path}")
    plt.close()

def load_point_cloud(file_path, n_points=2048):
    """
    加载点云文件并采样到固定点数
    """
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points, dtype=np.float32)
    
    if len(points) == 0:
        return np.zeros((n_points, 3), dtype=np.float32)
    
    if len(points) > n_points:
        indices = np.random.choice(len(points), n_points, replace=False)
        points = points[indices]
    elif len(points) < n_points:
        indices = np.random.choice(len(points), n_points, replace=True)
        points = points[indices]
    
    return points

def load_grasp_trajectory(file_path):
    """
    加载抓取点轨迹文件
    """
    if not os.path.exists(file_path):
        print(f"警告: 抓取点文件不存在: {file_path}")
        return None
    
    positions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    pattern = r'X=\s*([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)'
    matches = re.findall(pattern, content)
    
    for match in matches:
        try:
            x = float(match[0])
            y = float(match[1])
            z = float(match[2])
            positions.append([x, y, z])
        except ValueError:
            continue
    
    return np.array(positions, dtype=np.float32)

# =======================================================================
# 3. 梯度敏感度可视化主函数
# =======================================================================

def run_gradient_sensitivity_visualization():
    """
    计算并可视化预测点云对抓取点输入的梯度敏感度
    """
    DATA_PATH = r"D:\Cu_BEM_2\tshirt-data\tshirt_kalman_3\dataset_arm"
    MODEL_PATH = "checkpoints/best_model_simplified.pth"
    TEMPLATE_PATH = r"D:\Cu_BEM_2\tshirt-data\tshirt_kalman_3\tshirt_mech00001.obj"
    GRASP_FILE = r"D:\Cu_BEM_2\tshirt-data\tshirt_kalman_3\vertex_237_trajectory.txt"
    
    TARGET_FRAMES = [50,250,450,500, 600,700]
    
    COLORMAP = 'viridis'
    USE_LOG_SCALE = True
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print("="*60)
    print("梯度敏感度可视化 (修正版)")
    print("="*60)
    print(f"使用对数尺度: {USE_LOG_SCALE}")
    print("="*60)
    
    # 1. 加载模型
    print("\n1. 加载模型...")
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 模型文件不存在: {MODEL_PATH}")
        return
    
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    state_dict = checkpoint['model_state_dict']
    
    model = RecursiveTrackingNet(
        obs_latent_dim=64,
        grasp_latent_dim=16,
        template_path=TEMPLATE_PATH
    ).to(device)
    
    keys_to_remove = []
    for key in state_dict.keys():
        if 'current_template' in key or ('template' in key and 'templates' not in key):
            keys_to_remove.append(key)
    
    for key in keys_to_remove:
        if key in state_dict:
            del state_dict[key]
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"   模型加载成功: {MODEL_PATH}")
    
    # 2. 加载抓取点轨迹
    print("\n2. 加载抓取点轨迹...")
    grasp_trajectory = load_grasp_trajectory(GRASP_FILE)
    if grasp_trajectory is not None:
        print(f"   已加载 {len(grasp_trajectory)} 个抓取点")
    else:
        print(f"   警告: 无法加载抓取点轨迹")
        grasp_trajectory = np.zeros((1000, 3), dtype=np.float32)
    
    # 3. 获取点云文件列表
    print("\n3. 扫描点云文件...")
    raw_files = glob.glob(os.path.join(DATA_PATH, "occ_frame_*.ply"))
    
    files = []
    for f in raw_files:
        basename = os.path.basename(f)
        if re.match(r'^occ_frame_\d+\.ply$', basename):
            files.append(f)
    
    if not files:
        print(f"错误: 在 {DATA_PATH} 中没有找到有效的点云文件")
        return
    
    files = sorted(files, 
                   key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group())
                   if re.search(r'\d+', os.path.basename(x)) else 0)
    
    total_frames = len(files)
    print(f"   找到 {total_frames} 个点云文件 (帧 0-{total_frames-1})")
    
    valid_frames = []
    for frame_idx in TARGET_FRAMES:
        if 0 <= frame_idx < total_frames:
            valid_frames.append(frame_idx)
        else:
            print(f"警告: 帧 {frame_idx} 超出有效范围 [0, {total_frames-1}]")
    
    if not valid_frames:
        print("错误: 没有有效的帧索引")
        return
    
    print(f"\n将计算以下帧的梯度敏感度: {valid_frames}")
    
    # 4. 递归运行到第一个目标帧
    print("\n4. 递归运行到第一个目标帧...")
    h_prev = None
    h_prev_prev = None
    
    first_target = min(valid_frames)
    for frame_idx in range(first_target):
        if frame_idx % 50 == 0:
            print(f"  处理进度: {frame_idx}/{first_target}")
        
        file_path = files[frame_idx]
        input_points = load_point_cloud(file_path, n_points=2048)
        
        if grasp_trajectory is not None and frame_idx < len(grasp_trajectory):
            grasp_point = grasp_trajectory[frame_idx]
        else:
            grasp_point = np.zeros(3, dtype=np.float32)
        
        input_tensor = torch.from_numpy(input_points).float().unsqueeze(0).to(device)
        grasp_tensor = torch.from_numpy(grasp_point).float().to(device)
        
        if h_prev is None:
            h_prev = torch.zeros(1, 64).to(device)
        
        with torch.no_grad():
            _, h_next = model(input_tensor, grasp_tensor, h_prev, h_prev_prev)
        
        h_prev_prev = h_prev
        h_prev = h_next
    
    print(f"   已运行到帧 {first_target-1}")
    
    # 5. 计算目标帧的梯度敏感度
    all_gradient_stats = []
    
    for frame_idx in valid_frames:
        print(f"\n=== 计算帧 {frame_idx} 的梯度敏感度 ===")
        print(f"    文件: {os.path.basename(files[frame_idx])}")
        
        if frame_idx > first_target:
            for i in range(first_target, frame_idx):
                file_path = files[i]
                input_points = load_point_cloud(file_path, n_points=2048)
                
                if grasp_trajectory is not None and i < len(grasp_trajectory):
                    grasp_point = grasp_trajectory[i]
                else:
                    grasp_point = np.zeros(3, dtype=np.float32)
                
                input_tensor = torch.from_numpy(input_points).float().unsqueeze(0).to(device)
                grasp_tensor = torch.from_numpy(grasp_point).float().to(device)
                
                with torch.no_grad():
                    _, h_next = model(input_tensor, grasp_tensor, h_prev, h_prev_prev)
                
                h_prev_prev = h_prev
                h_prev = h_next
        
        file_path = files[frame_idx]
        input_points = load_point_cloud(file_path, n_points=2048)
        
        if grasp_trajectory is not None and frame_idx < len(grasp_trajectory):
            grasp_point = grasp_trajectory[frame_idx]
        else:
            grasp_point = np.zeros(3, dtype=np.float32)
        
        input_tensor = torch.from_numpy(input_points).float().unsqueeze(0).to(device)
        grasp_tensor = torch.from_numpy(grasp_point).float().to(device)
        
        print("    计算梯度敏感度...")
        
        # 使用修正后的梯度计算函数
        pred_points, gradient_norms, gradient_vectors = compute_gradient_per_point(
            model, input_tensor, grasp_tensor, h_prev, h_prev_prev
        )
        
        # 检查梯度是否有效
        if torch.all(gradient_norms == 0):
            print("    警告: 梯度全为零，尝试备用方法...")
            pred_points, gradient_norms, gradient_vectors = compute_gradient_sensitivity_simple(
                model, input_tensor, grasp_tensor, h_prev, h_prev_prev
            )
        
        # 将梯度范数转换为热力图颜色
        heatmap_colors, normalized_gradients, (min_val, max_val) = compute_gradient_heatmap(
            gradient_norms, colormap=COLORMAP, use_log_scale=USE_LOG_SCALE
        )
        
        gradient_stats = {
            'frame_idx': frame_idx,
            'min_gradient': float(torch.min(gradient_norms)),
            'max_gradient': float(torch.max(gradient_norms)),
            'mean_gradient': float(torch.mean(gradient_norms)),
            'std_gradient': float(torch.std(gradient_norms)),
            'min_val': min_val,
            'max_val': max_val
        }
        all_gradient_stats.append(gradient_stats)
        
        print(f"    梯度统计:")
        print(f"      最小值: {gradient_stats['min_gradient']:.2e}")
        print(f"      最大值: {gradient_stats['max_gradient']:.2e}")
        print(f"      平均值: {gradient_stats['mean_gradient']:.2e}")
        print(f"      标准差: {gradient_stats['std_gradient']:.2e}")
        
        # 可视化
        pred_np = pred_points[0].cpu().numpy()
        
        geometries = []
        
        pred_pcd = o3d.geometry.PointCloud()
        pred_pcd.points = o3d.utility.Vector3dVector(pred_np)
        pred_pcd.colors = o3d.utility.Vector3dVector(heatmap_colors)
        geometries.append(pred_pcd)
        
        grasp_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        grasp_sphere.compute_vertex_normals()
        grasp_sphere.paint_uniform_color([1.0, 0.0, 0.0])
        grasp_sphere.translate(grasp_point)
        geometries.append(grasp_sphere)
        
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        geometries.append(coord_frame)
        
        print("    正在显示梯度敏感度可视化窗口...")
        o3d.visualization.draw_geometries(
            geometries,
            window_name=f"Frame {frame_idx} | Gradient Sensitivity",
            width=1200,
            height=800
        )
        
        with torch.no_grad():
            _, h_next = model(input_tensor, grasp_tensor, h_prev, h_prev_prev)
        
        h_prev_prev = h_prev
        h_prev = h_next
        first_target = frame_idx + 1
    
    # 6. 生成梯度颜色条
    print("\n5. 生成梯度颜色条...")
    if all_gradient_stats:
        all_min_vals = [stats['min_val'] for stats in all_gradient_stats]
        all_max_vals = [stats['max_val'] for stats in all_gradient_stats]
        
        global_min_val = min(all_min_vals)
        global_max_val = max(all_max_vals)
        
        create_gradient_colorbar(
            global_min_val, global_max_val, 
            colormap=COLORMAP, 
            use_log_scale=USE_LOG_SCALE,
            save_path='gradient_sensitivity_colorbar.png'
        )
    
    # 7. 输出梯度统计结果
    print("\n" + "="*60)
    print("梯度敏感度统计结果")
    print("="*60)
    
    if all_gradient_stats:
        print(f"分析帧: {[stats['frame_idx'] for stats in all_gradient_stats]}")
        
        for stats in all_gradient_stats:
            print(f"\n帧 {stats['frame_idx']}:")
            print(f"  梯度最小值: {stats['min_gradient']:.2e}")
            print(f"  梯度最大值: {stats['max_gradient']:.2e}")
            print(f"  梯度平均值: {stats['mean_gradient']:.2e}")
            print(f"  梯度标准差: {stats['std_gradient']:.2e}")
        
        all_min_gradients = [stats['min_gradient'] for stats in all_gradient_stats]
        all_max_gradients = [stats['max_gradient'] for stats in all_gradient_stats]
        all_mean_gradients = [stats['mean_gradient'] for stats in all_gradient_stats]
        
        print(f"\n总体统计:")
        print(f"  最小梯度: {min(all_min_gradients):.2e}")
        print(f"  最大梯度: {max(all_max_gradients):.2e}")
        print(f"  平均梯度: {np.mean(all_mean_gradients):.2e}")
        
        if min(all_min_gradients) > 0:
            gradient_range = max(all_max_gradients) / min(all_min_gradients)
            print(f"  梯度变化范围: {gradient_range:.2e} 倍")
    
    print(f"\n完成 {len(valid_frames)} 帧的梯度敏感度可视化")
    print("梯度颜色条已保存为: gradient_sensitivity_colorbar.png")

# =======================================================================
# 4. 主程序入口
# =======================================================================

if __name__ == "__main__":
    print("开始梯度敏感度可视化 (修正版)")
    print("="*60)
    run_gradient_sensitivity_visualization()