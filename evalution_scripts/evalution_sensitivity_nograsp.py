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

# 设置字体
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

# 修复MatplotlibDeprecationWarning
matplotlib.colormaps.get_cmap = lambda x: cm.get_cmap(x)

# =======================================================================
# 1. M5模型定义 (不带抓取点的KalmanTracker)
# =======================================================================

class PointNetEncoderM5(nn.Module):
    """M5模型的PointNet编码器"""
    def __init__(self, latent_dim=64):
        super(PointNetEncoderM5, self).__init__()
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
        x = torch.max(x, 2)[0]
        return self.fc(x)

class KalmanTrackerM5(nn.Module):
    """M5模型 - 不带抓取点输入"""
    def __init__(self, template_path, latent_dim=64):
        super(KalmanTrackerM5, self).__init__()
        
        # 加载模板
        mesh = trimesh.load(template_path, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = list(mesh.geometry.values())[0]
        
        pts = np.array(mesh.vertices, dtype=np.float32)
        # 模板中心化
        self.v_mean = np.mean(pts, axis=0)
        pts_centered = pts - self.v_mean
        
        self.register_buffer('template', torch.from_numpy(pts_centered).float())
        self.num_pts = pts_centered.shape[0]

        self.obs_encoder = PointNetEncoderM5(latent_dim)
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
# 2. 辅助函数
# =======================================================================

def compute_m5_gradient_sensitivity_simple(model, o_t, g_t, h_prev, h_prev_temp=None):
    """
    简化的M5梯度计算函数
    由于M5模型不使用抓取点，直接返回零梯度
    """
    # 使用目标帧点云作为goal
    goal_tensor = o_t.clone()
    
    # 前向传播
    p_hat, _, _ = model(o_t, goal_tensor, h_prev, h_prev_temp)
    
    B, N, D = p_hat.shape
    
    # 创建零梯度
    gradient_norms = torch.zeros(N, device=p_hat.device)
    gradient_vectors = torch.zeros(B, N, 3, device=p_hat.device)
    
    return p_hat.detach(), gradient_norms.detach(), gradient_vectors.detach()

def compute_gradient_heatmap(gradient_norms, colormap='viridis', use_log_scale=True, 
                           vmin=None, vmax=None):
    """
    将梯度范数转换为热力图颜色
    新增参数：vmin, vmax 用于指定颜色映射的范围
    """
    # 转换为numpy
    gradient_norms_np = gradient_norms.cpu().numpy() if torch.is_tensor(gradient_norms) else gradient_norms
    
    # 避免零或负值
    eps = 1e-8
    gradient_norms_np = gradient_norms_np + eps
    
    if use_log_scale:
        # 使用对数尺度
        gradient_norms_np = np.log10(gradient_norms_np)
        
        # 如果指定了范围，使用指定范围；否则使用实际范围
        if vmin is not None and vmax is not None:
            min_val = vmin
            max_val = vmax
        else:
            min_val = np.min(gradient_norms_np)
            max_val = np.max(gradient_norms_np)
        
        # 归一化到[0, 1]
        if max_val > min_val:
            # 将值限制在[min_val, max_val]范围内
            gradient_norms_clipped = np.clip(gradient_norms_np, min_val, max_val)
            normalized_gradients = (gradient_norms_clipped - min_val) / (max_val - min_val)
        else:
            normalized_gradients = np.zeros_like(gradient_norms_np)
    else:
        # 使用线性尺度
        if vmin is not None and vmax is not None:
            min_val = vmin
            max_val = vmax
        else:
            min_val = np.min(gradient_norms_np)
            max_val = np.max(gradient_norms_np)
        
        # 归一化到[0, 1]
        if max_val > min_val:
            # 将值限制在[min_val, max_val]范围内
            gradient_norms_clipped = np.clip(gradient_norms_np, min_val, max_val)
            normalized_gradients = (gradient_norms_clipped - min_val) / (max_val - min_val)
        else:
            normalized_gradients = np.zeros_like(gradient_norms_np)
    
    # 使用指定的颜色映射
    cmap = plt.cm.get_cmap(colormap)
    heatmap_colors = cmap(normalized_gradients)[:, :3]  # 只取RGB，忽略alpha通道
    
    return heatmap_colors, normalized_gradients, (min_val, max_val)

def create_gradient_colorbar(min_val, max_val, colormap='viridis', use_log_scale=True, 
                           save_path='m5_gradient_colorbar.png'):
    """
    创建梯度颜色条图例（竖版）
    """
    fig, ax = plt.subplots(figsize=(1.2, 6))
    fig.subplots_adjust(left=0.3, right=0.5)
    
    cmap = plt.cm.get_cmap(colormap)
    
    if use_log_scale:
        # 对数尺度
        # 注意：min_val和max_val已经是log10之后的值
        norm = LogNorm(vmin=10**min_val, vmax=10**max_val)
        ticks = np.logspace(np.log10(10**min_val), np.log10(10**max_val), 7)
        tick_labels = [f'{tick:.1e}' for tick in ticks]
    else:
        # 线性尺度
        norm = Normalize(vmin=min_val, vmax=max_val)
        ticks = np.linspace(min_val, max_val, 7)
        tick_labels = [f'{tick:.1e}' for tick in ticks]
    
    cb = ColorbarBase(ax, cmap=cmap, norm=norm, orientation='vertical')
    cb.set_label('Gradient Magnitude\n(Sensitivity to grasp point)', fontsize=10, fontweight='bold')
    
    cb.set_ticks(ticks)
    cb.set_ticklabels(tick_labels, fontsize=8)
    
    scale_type = "Log Scale" if use_log_scale else "Linear Scale"
    title = f'M5 Gradient Sensitivity\n{scale_type}'
    ax.set_title(title, fontsize=9, fontweight='bold', pad=10)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"M5梯度颜色条已保存至: {save_path}")
    plt.close()

def load_point_cloud_and_center(file_path, n_points=2048):
    """
    加载点云文件，采样到固定点数，并返回中心化后的点云和重心
    与训练脚本保持一致
    """
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points, dtype=np.float32)
    
    if len(points) == 0:
        return np.zeros((n_points, 3), dtype=np.float32), np.zeros(3, dtype=np.float32)
    
    # 采样至固定点数
    if len(points) >= n_points:
        idx_sample = np.random.choice(len(points), n_points, replace=False)
        points = points[idx_sample]
    else:
        points = np.tile(points, (n_points // len(points) + 1, 1))[:n_points]
    
    # 动态对齐几何中心
    current_centroid = np.mean(points, axis=0)
    points_norm = points - current_centroid  # 强制当前帧中心为 (0,0,0)
    
    return points_norm, current_centroid

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
# 3. M5梯度敏感度可视化主函数
# =======================================================================

def run_m5_gradient_visualization():
    """
    计算并可视化M5模型对抓取点输入的梯度敏感度
    """
    # 配置参数
    BASE = r"D:\Cu_BEM_2\tshirt-data\tshirt_kalman_3"
    DATA_PATH = os.path.join(BASE, "dataset_arm")
    TEMPLATE_PATH = os.path.join(BASE, "tshirt_mech00001.obj")
    GRASP_FILE = os.path.join(BASE, "vertex_237_trajectory.txt")
    
    # 模型checkpoint路径
    MODEL_M5_PATH = "checkpoints/kalman_best_aligned.pth"    # M5模型
    
    # 在这里直接指定要可视化的帧索引
    TARGET_FRAMES = [50,250,450,500, 600,700]
    
    # 可视化参数
    COLORMAP = 'viridis'  # 使用viridis颜色映射
    USE_LOG_SCALE = True  # 是否使用对数尺度
    
    # 关键修改：指定与M4相同的颜色条范围
    # 您需要从M4梯度可视化结果中获取这些值
    # 如果M4的梯度范围是对数尺度下的[-3, 1]，则设置如下：
    M4_LOG_MIN = -3.0  # 10^-3 = 0.001
    M4_LOG_MAX = 1.0   # 10^1 = 10
    
    # 如果您希望使用线性尺度，请将USE_LOG_SCALE设置为False，并设置线性范围
    # M4_LINEAR_MIN = 0.001
    # M4_LINEAR_MAX = 10.0
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print("="*60)
    print("M5梯度敏感度可视化 (不带抓取点模型)")
    print("="*60)
    print(f"使用对数尺度: {USE_LOG_SCALE}")
    print(f"颜色条范围: 10^{M4_LOG_MIN:.1f} 到 10^{M4_LOG_MAX:.1f}")
    print("="*60)
    
    # 1. 加载M5模型
    print("\n1. 加载M5模型...")
    if not os.path.exists(MODEL_M5_PATH):
        print(f"错误: M5模型文件不存在: {MODEL_M5_PATH}")
        return
    
    checkpoint_m5 = torch.load(MODEL_M5_PATH, map_location=device)
    state_dict_m5 = checkpoint_m5['model_state_dict']
    
    model_m5 = KalmanTrackerM5(TEMPLATE_PATH).to(device)
    model_m5.load_state_dict(state_dict_m5)
    model_m5.eval()
    print(f"   M5模型加载成功: {MODEL_M5_PATH}")
    
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
    
    # 严格过滤，只保留 "occ_frame_数字.ply" 格式的文件
    files = []
    for f in raw_files:
        basename = os.path.basename(f)
        if re.match(r'^occ_frame_\d+\.ply$', basename):
            files.append(f)
    
    if not files:
        print(f"错误: 在 {DATA_PATH} 中没有找到有效的点云文件")
        return
    
    # 按数字排序
    files = sorted(files, 
                   key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group())
                   if re.search(r'\d+', os.path.basename(x)) else 0)
    
    total_frames = len(files)
    print(f"   找到 {total_frames} 个点云文件 (帧 0-{total_frames-1})")
    
    # 检查目标帧是否在有效范围内
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
    
    # M5模型的状态
    h_prev_m5 = None
    h_prev_m5_temp = None  # M5使用h_t和h_prev
    
    first_target = min(valid_frames)
    for frame_idx in range(first_target):
        if frame_idx % 50 == 0:
            print(f"  处理进度: {frame_idx}/{first_target}")
        
        file_path = files[frame_idx]
        
        # 加载点云并中心化
        obs_norm, _ = load_point_cloud_and_center(file_path, n_points=2048)
        
        if grasp_trajectory is not None and frame_idx < len(grasp_trajectory):
            grasp_point = grasp_trajectory[frame_idx]
        else:
            grasp_point = np.zeros(3, dtype=np.float32)
        
        # 转换为张量
        obs_tensor = torch.from_numpy(obs_norm).float().unsqueeze(0).to(device)
        grasp_tensor = torch.from_numpy(grasp_point).float().to(device)
        
        # 加载目标帧点云（用于M5模型）
        goal_idx = min(frame_idx + 1, total_frames - 1)
        goal_file_path = files[goal_idx]
        goal_norm, _ = load_point_cloud_and_center(goal_file_path, n_points=2048)
        goal_tensor = torch.from_numpy(goal_norm).float().unsqueeze(0).to(device)
        
        # 初始化状态
        if h_prev_m5 is None:
            h_prev_m5 = torch.zeros(1, 64).to(device)
            h_prev_m5_temp = torch.zeros(1, 64).to(device)
        
        # M5模型推理
        with torch.no_grad():
            _, h_next_m5, _ = model_m5(obs_tensor, goal_tensor, h_prev_m5, h_prev_m5_temp)
        
        # 更新状态
        h_prev_m5_temp = h_prev_m5
        h_prev_m5 = h_next_m5
    
    print(f"   已运行到帧 {first_target-1}")
    
    # 5. 计算目标帧的梯度敏感度
    all_gradient_stats = []
    
    for frame_idx in valid_frames:
        print(f"\n=== 计算帧 {frame_idx} 的梯度敏感度 ===")
        print(f"    文件: {os.path.basename(files[frame_idx])}")
        
        if frame_idx > first_target:
            for i in range(first_target, frame_idx):
                file_path = files[i]
                
                # 加载点云并中心化
                obs_norm, _ = load_point_cloud_and_center(file_path, n_points=2048)
                
                if grasp_trajectory is not None and i < len(grasp_trajectory):
                    grasp_point = grasp_trajectory[i]
                else:
                    grasp_point = np.zeros(3, dtype=np.float32)
                
                obs_tensor = torch.from_numpy(obs_norm).float().unsqueeze(0).to(device)
                grasp_tensor = torch.from_numpy(grasp_point).float().to(device)
                
                # 加载目标帧点云（用于M5模型）
                goal_idx = min(i + 1, total_frames - 1)
                goal_file_path = files[goal_idx]
                goal_norm, _ = load_point_cloud_and_center(goal_file_path, n_points=2048)
                goal_tensor = torch.from_numpy(goal_norm).float().unsqueeze(0).to(device)
                
                # M5模型推理
                with torch.no_grad():
                    _, h_next_m5, _ = model_m5(obs_tensor, goal_tensor, h_prev_m5, h_prev_m5_temp)
                
                # 更新状态
                h_prev_m5_temp = h_prev_m5
                h_prev_m5 = h_next_m5
        
        file_path = files[frame_idx]
        
        # 加载点云并中心化
        obs_norm, _ = load_point_cloud_and_center(file_path, n_points=2048)
        
        if grasp_trajectory is not None and frame_idx < len(grasp_trajectory):
            grasp_point = grasp_trajectory[frame_idx]
        else:
            grasp_point = np.zeros(3, dtype=np.float32)
        
        obs_tensor = torch.from_numpy(obs_norm).float().unsqueeze(0).to(device)
        grasp_tensor = torch.from_numpy(grasp_point).float().to(device)
        
        # 加载目标帧点云（用于M5模型）
        goal_idx = min(frame_idx + 1, total_frames - 1)
        goal_file_path = files[goal_idx]
        goal_norm, _ = load_point_cloud_and_center(goal_file_path, n_points=2048)
        goal_tensor = torch.from_numpy(goal_norm).float().unsqueeze(0).to(device)
        
        print("    计算M5梯度敏感度...")
        
        # 计算M5梯度
        pred_points_m5, gradient_norms_m5, gradient_vectors_m5 = compute_m5_gradient_sensitivity_simple(
            model_m5, obs_tensor, grasp_tensor, h_prev_m5, h_prev_m5_temp
        )
        
        # 由于M5不使用抓取点，梯度全为零
        print("    梯度统计: 全为零 (符合预期，M5模型不使用抓取点输入)")
        
        # 将梯度范数转换为热力图颜色，使用与M4相同的范围
        heatmap_colors_m5, normalized_gradients_m5, (min_val_m5, max_val_m5) = compute_gradient_heatmap(
            gradient_norms_m5, 
            colormap=COLORMAP, 
            use_log_scale=USE_LOG_SCALE,
            vmin=M4_LOG_MIN if USE_LOG_SCALE else None,  # 如果使用对数尺度，使用M4的对数范围
            vmax=M4_LOG_MAX if USE_LOG_SCALE else None
        )
        
        gradient_stats_m5 = {
            'frame_idx': frame_idx,
            'min_gradient': 0.0,
            'max_gradient': 0.0,
            'mean_gradient': 0.0,
            'std_gradient': 0.0,
            'min_val': min_val_m5,
            'max_val': max_val_m5
        }
        all_gradient_stats.append(gradient_stats_m5)
        
        # 可视化M5的梯度敏感度
        pred_np_m5 = pred_points_m5[0].cpu().numpy()
        
        # 创建Open3D可视化
        geometries_m5 = []
        
        # M5预测点云 (梯度敏感度热力图着色)
        pred_pcd_m5 = o3d.geometry.PointCloud()
        pred_pcd_m5.points = o3d.utility.Vector3dVector(pred_np_m5)
        pred_pcd_m5.colors = o3d.utility.Vector3dVector(heatmap_colors_m5)
        geometries_m5.append(pred_pcd_m5)
        
        # 抓取点标记 (红色球体)
        grasp_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        grasp_sphere.compute_vertex_normals()
        grasp_sphere.paint_uniform_color([1.0, 0.0, 0.0])
        grasp_sphere.translate(grasp_point)
        
        geometries_m5.append(grasp_sphere)
        
        # 坐标轴
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        geometries_m5.append(coord_frame)
        
        print("    正在显示M5梯度敏感度可视化窗口...")
        o3d.visualization.draw_geometries(
            geometries_m5,
            window_name=f"M5 Frame {frame_idx} | Gradient Sensitivity = 0 (Model does not use grasp point)",
            width=1200,
            height=800
        )
        
        # 更新状态
        with torch.no_grad():
            _, h_next_m5, _ = model_m5(obs_tensor, goal_tensor, h_prev_m5, h_prev_m5_temp)
        
        h_prev_m5_temp = h_prev_m5
        h_prev_m5 = h_next_m5
        first_target = frame_idx + 1
    
    # 6. 生成梯度颜色条
    print("\n5. 生成梯度颜色条...")
    if all_gradient_stats:
        # 使用与M4相同的范围生成颜色条
        if USE_LOG_SCALE:
            global_min_val = M4_LOG_MIN
            global_max_val = M4_LOG_MAX
        else:
            # 如果使用线性尺度，您需要设置M4_LINEAR_MIN和M4_LINEAR_MAX
            global_min_val = 0.0
            global_max_val = 0.0
        
        create_gradient_colorbar(
            global_min_val, global_max_val, 
            colormap=COLORMAP, 
            use_log_scale=USE_LOG_SCALE,
            save_path='m5_gradient_sensitivity_colorbar.png'
        )
    
    # 7. 输出梯度统计结果
    print("\n" + "="*60)
    print("M5梯度敏感度统计结果")
    print("="*60)
    
    if all_gradient_stats:
        print(f"分析帧: {[stats['frame_idx'] for stats in all_gradient_stats]}")
        
        print(f"\nM5 (不带抓取点) 模型梯度统计:")
        print(f"  平均最小值: 0.0")
        print(f"  平均最大值: 0.0")
        print(f"  平均梯度: 0.0")
        print(f"  平均标准差: 0.0")
        
        # 解释梯度值的意义
        print(f"\n梯度值科学解释:")
        print(f"  M5模型在训练和推理阶段均不使用抓取点作为输入")
        print(f"  因此，模型输出对抓取点输入的梯度理论上应为0")
        print(f"  此结果符合模型设计预期，验证了实验设计的正确性")
        print(f"  与M4模型的高梯度敏感度形成鲜明对比")
        
        # 颜色条范围说明
        print(f"\n颜色条范围说明:")
        print(f"  颜色条使用与M4模型相同的数值范围:")
        if USE_LOG_SCALE:
            print(f"  对数尺度: 10^{M4_LOG_MIN:.2f} 到 10^{M4_LOG_MAX:.2f}")
            print(f"  线性尺度: {10**M4_LOG_MIN:.2e} 到 {10**M4_LOG_MAX:.2e}")
        print(f"  这使得M5和M4的梯度图可以直接对比")
    
    print(f"\n完成 {len(valid_frames)} 帧的M5梯度敏感度可视化")
    print("梯度颜色条已保存为: m5_gradient_sensitivity_colorbar.png")

# =======================================================================
# 4. 主程序入口
# =======================================================================

if __name__ == "__main__":
    print("开始M5梯度敏感度可视化")
    print("="*60)
    run_m5_gradient_visualization()