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
from matplotlib.colors import Normalize
import matplotlib

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong']
matplotlib.rcParams['axes.unicode_minus'] = False

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

def compute_cd_percentage_heatmap(pred_points, gt_points, template_diagonal, colormap='viridis', max_percentage=30.0):
    """
    计算倒角距离（CD）并将其转换为模板包围盒体对角线的百分比
    然后生成热力图数据
    
    参数:
        pred_points: 预测点云 [N, 3]
        gt_points: 真实点云 [M, 3]
        template_diagonal: 模板包围盒体对角线长度
        colormap: 颜色映射名称
        max_percentage: 最大百分比，用于颜色归一化
    
    返回:
        avg_cd: 平均倒角距离
        pointwise_dist: 预测点到最近真实点的距离 [N]
        pointwise_percentage: 距离占模板对角线的百分比 [N]
        heatmap_colors: 热力图颜色 [N, 3]
    """
    # 使用KDTree加速最近邻搜索
    tree_gt = cKDTree(gt_points)
    distances, _ = tree_gt.query(pred_points)
    
    # 计算平均倒角距离
    tree_pred = cKDTree(pred_points)
    distances_gt_to_pred, _ = tree_pred.query(gt_points)
    avg_cd = 0.5 * (np.mean(distances) + np.mean(distances_gt_to_pred))
    
    # 将距离转换为模板对角线的百分比
    percentages = distances / template_diagonal * 100  # 转换为百分比
    
    # 归一化百分比以生成热力图颜色
    # 使用传入的max_percentage参数
    norm_percentages = np.clip(percentages / max_percentage, 0, 1)
    
    # 使用指定的颜色映射
    cmap = cm.get_cmap(colormap)
    heatmap_colors = cmap(norm_percentages)[:, :3]  # 只取RGB，忽略alpha通道
    
    return avg_cd, distances, percentages, heatmap_colors

def load_point_cloud(file_path, n_points=2048):
    """
    加载点云文件并采样到固定点数
    """
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points, dtype=np.float32)
    
    if len(points) == 0:
        return np.zeros((n_points, 3), dtype=np.float32)
    
    # 随机采样到固定点数
    if len(points) > n_points:
        indices = np.random.choice(len(points), n_points, replace=False)
        points = points[indices]
    elif len(points) < n_points:
        # 重复采样
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
    
    # 使用正则表达式匹配所有坐标
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

def compute_template_diagonal(template_path):
    """
    计算模板包围盒体对角线长度
    
    参数:
        template_path: 模板OBJ文件路径
    
    返回:
        diagonal_length: 包围盒体对角线长度
    """
    if not os.path.exists(template_path):
        print(f"错误: 模板文件不存在: {template_path}")
        return None
    
    # 加载模板
    mesh = trimesh.load(template_path)
    vertices = mesh.vertices.astype(np.float32)
    
    if len(vertices) == 0:
        print(f"错误: 模板文件为空: {template_path}")
        return None
    
    # 计算包围盒
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    
    # 计算对角线长度
    diagonal_length = np.linalg.norm(max_coords - min_coords)
    
    print(f"模板包围盒:")
    print(f"  最小坐标: {min_coords}")
    print(f"  最大坐标: {max_coords}")
    print(f"  对角线长度: {diagonal_length:.6f}")
    
    return diagonal_length

def create_percentage_colorbar(max_percentage=30.0, colormap='viridis', save_path='percentage_colorbar.png'):
    """
    创建基于百分比的颜色条图例（竖版）
    
    参数:
        max_percentage: 最大百分比
        colormap: 颜色映射名称
        save_path: 保存路径
    """
    # 修改图形尺寸，使其更适合竖版颜色条
    fig, ax = plt.subplots(figsize=(1.2, 6))
    fig.subplots_adjust(left=0.3, right=0.5)
    
    cmap = cm.get_cmap(colormap)
    norm = Normalize(vmin=0, vmax=max_percentage)
    
    # 将orientation参数从'horizontal'改为'vertical'
    cb = ColorbarBase(ax, cmap=cmap, norm=norm, orientation='vertical')
    cb.set_label('CD as % of Template\nBounding Box Diagonal', fontsize=10, fontweight='bold')
    
    # 设置刻度
    ticks = np.linspace(0, max_percentage, 7)  # 7个刻度
    cb.set_ticks(ticks)
    cb.set_ticklabels([f'{tick:.1f}%' for tick in ticks], fontsize=8)
    
    # 设置标题
    title = f'Color Scale\n0% to {max_percentage:.1f}%\nof Template Diagonal'
    ax.set_title(title, fontsize=9, fontweight='bold', pad=10)
    
    # 保存颜色条
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"竖版百分比颜色条已保存至: {save_path}")
    
    plt.close()
# =======================================================================
# 3. 基于CD百分比热力图的可视化函数
# =======================================================================

def run_cd_percentage_visualization():
    """
    基于倒角距离（CD）百分比热力图的可视化
    将CD值表示为模板包围盒体对角线的百分比
    """
    # 配置参数
    DATA_PATH = r"D:\Cu_BEM_2\tshirt-data\tshirt_kalman_3\dataset_arm"
    MODEL_PATH = "checkpoints/best_model_simplified.pth"
    TEMPLATE_PATH = r"D:\Cu_BEM_2\tshirt-data\tshirt_kalman_3\tshirt_mech00001.obj"
    GRASP_FILE = r"D:\Cu_BEM_2\tshirt-data\tshirt_kalman_3\vertex_237_trajectory.txt"
    
    # 在这里直接指定要可视化的帧索引
    TARGET_FRAMES = [100, 200,250, 300,350, 400,450, 500,550,600,650,700]  # 示例帧
    
    # 可视化参数
    COLORMAP = 'viridis'  # 使用viridis颜色映射
    
    # 颜色条最大百分比 - 现在可以调整这个值来改变颜色分布
    MAX_PERCENTAGE = 5  # 默认30%，您可以尝试调整为10.0、50.0等不同值
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print("="*60)
    print("基于倒角距离（CD）百分比热力图的可视化")
    print("="*60)
    print(f"当前颜色条范围: 0.0% - {MAX_PERCENTAGE:.1f}% of template diagonal")
    print("提示: 修改MAX_PERCENTAGE变量可以调整颜色分布")
    print("="*60)
    
    # 1. 计算模板包围盒体对角线
    print("\n1. 计算模板包围盒体对角线...")
    template_diagonal = compute_template_diagonal(TEMPLATE_PATH)
    if template_diagonal is None:
        print("错误: 无法计算模板对角线长度")
        return
    
    print(f"   模板对角线长度: {template_diagonal:.6f}")
    print(f"   颜色条范围: 0.0% - {MAX_PERCENTAGE:.1f}% of template diagonal")
    
    # 2. 生成百分比颜色条
    print("\n2. 生成百分比颜色条...")
    create_percentage_colorbar(MAX_PERCENTAGE, colormap=COLORMAP, save_path='percentage_colorbar.png')
    
    # 3. 加载模型
    print("\n3. 加载模型...")
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 模型文件不存在: {MODEL_PATH}")
        return
    
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    state_dict = checkpoint['model_state_dict']
    
    # 初始化模型
    model = RecursiveTrackingNet(
        obs_latent_dim=64,
        grasp_latent_dim=16,
        template_path=TEMPLATE_PATH
    ).to(device)
    
    # 修复状态字典
    keys_to_remove = []
    for key in state_dict.keys():
        if 'current_template' in key or ('template' in key and 'templates' not in key):
            keys_to_remove.append(key)
    
    for key in keys_to_remove:
        if key in state_dict:
            del state_dict[key]
    
    # 加载状态字典
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"   模型加载成功: {MODEL_PATH}")
    
    # 4. 加载抓取点轨迹
    print("\n4. 加载抓取点轨迹...")
    grasp_trajectory = load_grasp_trajectory(GRASP_FILE)
    if grasp_trajectory is not None:
        print(f"   已加载 {len(grasp_trajectory)} 个抓取点")
    else:
        print(f"   警告: 无法加载抓取点轨迹，将使用零向量")
    
    # 5. 获取点云文件列表
    print("\n5. 扫描点云文件...")
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
    
    print(f"\n将可视化以下帧: {valid_frames}")
    
    # 6. 递归计算并可视化每一帧
    # 初始化状态
    h_prev = None
    h_prev_prev = None
    
    # 存储所有帧的CD结果
    cd_results = []
    all_percentages = []  # 收集所有百分比值，用于统计
    
    for frame_idx in range(max(valid_frames) + 1):
        if frame_idx % 50 == 0:
            print(f"  处理进度: {frame_idx}/{max(valid_frames)}")
        
        # 加载当前帧点云
        file_path = files[frame_idx]
        input_points = load_point_cloud(file_path, n_points=2048)
        
        # 获取当前抓取点
        if grasp_trajectory is not None and frame_idx < len(grasp_trajectory):
            grasp_point = grasp_trajectory[frame_idx]
        else:
            grasp_point = np.zeros(3, dtype=np.float32)
        
        # 转换为张量
        input_tensor = torch.from_numpy(input_points).float().unsqueeze(0).to(device)
        grasp_tensor = torch.from_numpy(grasp_point).float().to(device)
        
        # 初始化隐状态
        if h_prev is None:
            h_prev = torch.zeros(1, 64).to(device)
        
        # 模型推理
        with torch.no_grad():
            pred_points, h_next = model(
                input_tensor, grasp_tensor, h_prev, h_prev_prev
            )
        
        # 更新隐状态
        h_prev_prev = h_prev
        h_prev = h_next
        
        # 如果当前帧是需要可视化的帧
        if frame_idx in valid_frames:
            print(f"\n=== 可视化帧 {frame_idx} ===")
            print(f"    文件: {os.path.basename(file_path)}")
            print(f"    抓取点: {grasp_point}")
            
            # 转换为numpy
            pred_np = pred_points[0].cpu().numpy()
            
            # 计算倒角距离和生成百分比热力图
            # 注意：现在传入了MAX_PERCENTAGE参数
            avg_cd, pointwise_dist, pointwise_percentage, heatmap_colors = compute_cd_percentage_heatmap(
                pred_np, input_points, template_diagonal, 
                colormap=COLORMAP, max_percentage=MAX_PERCENTAGE
            )
            
            # 收集所有百分比值
            all_percentages.extend(pointwise_percentage)
            
            # 计算平均百分比
            avg_percentage = np.mean(pointwise_percentage)
            
            cd_results.append((frame_idx, avg_cd, avg_percentage))
            print(f"    平均倒角距离: {avg_cd:.6f}")
            print(f"    平均百分比: {avg_percentage:.2f}% of template diagonal")
            print(f"    最大点距离: {np.max(pointwise_dist):.6f}")
            print(f"    最大百分比: {np.max(pointwise_percentage):.2f}%")
            print(f"    颜色条范围: 0.0% - {MAX_PERCENTAGE:.1f}%")
            
            # 分析颜色分布
            color_distribution = np.histogram(pointwise_percentage, bins=5, range=(0, MAX_PERCENTAGE))[0]
            print(f"    颜色分布统计:")
            for i in range(5):
                bin_start = i * MAX_PERCENTAGE / 5
                bin_end = (i + 1) * MAX_PERCENTAGE / 5
                count = color_distribution[i]
                percentage = count / len(pointwise_percentage) * 100
                print(f"      距离 {bin_start:.1f}%-{bin_end:.1f}%: {count} 点 ({percentage:.1f}%)")
            
            # 创建Open3D点云对象
            geometries = []
            
            # 1. 预测点云 (百分比热力图着色) - 居中显示
            pred_pcd = o3d.geometry.PointCloud()
            pred_pcd.points = o3d.utility.Vector3dVector(pred_np)
            pred_pcd.colors = o3d.utility.Vector3dVector(heatmap_colors)
            geometries.append(pred_pcd)
            
            # 2. 抓取点标记 (红色球体)
            grasp_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            grasp_sphere.compute_vertex_normals()
            grasp_sphere.paint_uniform_color([1.0, 0.0, 0.0])  # 红色
            grasp_sphere.translate(grasp_point)
            geometries.append(grasp_sphere)
            
            # 3. 坐标轴
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
            geometries.append(coord_frame)
            
            # 显示
            print("    正在显示可视化窗口... (关闭窗口继续)")
            o3d.visualization.draw_geometries(
                geometries,
                window_name=f"Frame {frame_idx} | CD: {avg_cd:.4f} | Avg: {avg_percentage:.1f}% | Range: 0-{MAX_PERCENTAGE:.1f}%",
                width=1200,
                height=800
            )
    
    # 7. 输出CD统计结果
    print("\n" + "="*60)
    print("倒角距离（CD）百分比统计结果")
    print("="*60)
    
    if cd_results and all_percentages:
        cd_values = [cd for _, cd, _ in cd_results]
        avg_percentages = [avg_p for _, _, avg_p in cd_results]
        all_percentages = np.array(all_percentages)
        frame_indices = [frame for frame, _, _ in cd_results]
        
        print(f"可视化帧: {frame_indices}")
        print(f"模板对角线长度: {template_diagonal:.6f}")
        print(f"平均CD: {np.mean(cd_values):.6f}")
        print(f"平均CD百分比: {np.mean(avg_percentages):.2f}% of template diagonal")
        print(f"最大CD: {np.max(cd_values):.6f} (帧 {frame_indices[np.argmax(cd_values)]})")
        print(f"最小CD: {np.min(cd_values):.6f} (帧 {frame_indices[np.argmin(cd_values)]})")
        print(f"颜色条范围: 0-{MAX_PERCENTAGE:.1f}%")
        print(f"\n所有帧百分比统计:")
        print(f"  平均百分比: {np.mean(all_percentages):.2f}%")
        print(f"  最大百分比: {np.max(all_percentages):.2f}%")
        print(f"  95%分位数: {np.percentile(all_percentages, 95):.2f}%")
        print(f"  99%分位数: {np.percentile(all_percentages, 99):.2f}%")
        
        # 建议颜色条范围
        suggested_max = np.percentile(all_percentages, 99)
        print(f"\n建议颜色条范围: 0.0% - {suggested_max:.1f}%")
        print("如果当前颜色条范围不合适，可以修改MAX_PERCENTAGE变量重新运行")
    
    print(f"\n完成 {len(valid_frames)} 帧的可视化")
    print("百分比颜色条已保存为: percentage_colorbar.png")

# =======================================================================
# 4. 主程序入口
# =======================================================================

if __name__ == "__main__":
    print("开始基于倒角距离（CD）百分比热力图的可视化")
    print("="*60)
    run_cd_percentage_visualization()