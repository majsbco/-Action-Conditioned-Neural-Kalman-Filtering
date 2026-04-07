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

def compute_cd_heatmap(pred_points, gt_points, max_dist=0.05):
    """
    计算倒角距离（CD）并生成热力图数据
    
    参数:
        pred_points: 预测点云 [N, 3]
        gt_points: 真实点云 [M, 3]
        max_dist: 热力图最大距离，用于归一化
    
    返回:
        avg_cd: 平均倒角距离
        pointwise_dist: 预测点到最近真实点的距离 [N]
        heatmap_colors: 热力图颜色 [N, 3]
    """
    # 使用KDTree加速最近邻搜索
    tree_gt = cKDTree(gt_points)
    distances, _ = tree_gt.query(pred_points)
    
    # 计算平均倒角距离
    tree_pred = cKDTree(pred_points)
    distances_gt_to_pred, _ = tree_pred.query(gt_points)
    avg_cd = 0.5 * (np.mean(distances) + np.mean(distances_gt_to_pred))
    
    # 归一化距离以生成热力图颜色
    norm_dist = np.clip(distances / max_dist, 0, 1)
    
    # 使用matplotlib的jet颜色映射
    cmap = cm.get_cmap('jet')
    heatmap_colors = cmap(norm_dist)[:, :3]  # 只取RGB，忽略alpha通道
    
    return avg_cd, distances, heatmap_colors

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

# =======================================================================
# 3. 基于CD热力图的可视化函数
# =======================================================================

def run_cd_heatmap_visualization():
    """
    基于倒角距离（CD）热力图的可视化
    不需要终端交互，直接在代码中指定帧索引
    """
    # 配置参数
    DATA_PATH = r"D:\Cu_BEM_2\tshirt-data\tshirt_kalman_3\dataset_arm"
    MODEL_PATH = "checkpoints/best_model_simplified.pth"
    TEMPLATE_PATH = r"D:\Cu_BEM_2\tshirt-data\tshirt_kalman_3\tshirt_mech00001.obj"
    GRASP_FILE = r"D:\Cu_BEM_2\tshirt-data\tshirt_kalman_3\vertex_237_trajectory.txt"
    
    # 在这里直接指定要可视化的帧索引
    TARGET_FRAMES = [10, 50, 100, 200, 300, 400, 500]  # 示例帧
    
    # 热力图参数
    HEATMAP_MAX_DIST = 0.05  # 最大距离，用于颜色映射归一化
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print("="*60)
    print("基于倒角距离（CD）热力图的可视化")
    print("="*60)
    
    # 1. 加载模型
    print("\n1. 加载模型...")
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
    
    # 2. 加载抓取点轨迹
    print("\n2. 加载抓取点轨迹...")
    grasp_trajectory = load_grasp_trajectory(GRASP_FILE)
    if grasp_trajectory is not None:
        print(f"   已加载 {len(grasp_trajectory)} 个抓取点")
    else:
        print(f"   警告: 无法加载抓取点轨迹，将使用零向量")
    
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
    
    print(f"\n将可视化以下帧: {valid_frames}")
    
    # 4. 递归计算并可视化每一帧
    # 初始化状态
    h_prev = None
    h_prev_prev = None
    
    # 存储所有帧的CD结果
    cd_results = []
    
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
            
            # 计算倒角距离和生成热力图
            avg_cd, pointwise_dist, heatmap_colors = compute_cd_heatmap(
                pred_np, input_points, max_dist=HEATMAP_MAX_DIST
            )
            
            cd_results.append((frame_idx, avg_cd))
            print(f"    平均倒角距离: {avg_cd:.6f}")
            print(f"    最大点距离: {np.max(pointwise_dist):.6f}")
            print(f"    平均点距离: {np.mean(pointwise_dist):.6f}")
            
            # 创建Open3D点云对象
            # 输入点云 (蓝色)
            input_pcd = o3d.geometry.PointCloud()
            input_pcd.points = o3d.utility.Vector3dVector(input_points)
            input_pcd.paint_uniform_color([0.2, 0.4, 0.8])  # 蓝色
            input_pcd.translate([-0.4, 0, 0])  # 左移
            
            # 预测点云 (热力图着色)
            pred_pcd = o3d.geometry.PointCloud()
            pred_pcd.points = o3d.utility.Vector3dVector(pred_np)
            pred_pcd.colors = o3d.utility.Vector3dVector(heatmap_colors)
            pred_pcd.translate([0.4, 0, 0])  # 右移
            
            # 抓取点标记 (红色球体)
            grasp_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            grasp_sphere.compute_vertex_normals()
            grasp_sphere.paint_uniform_color([1.0, 0.0, 0.0])  # 红色
            grasp_sphere.translate(grasp_point + [0.4, 0, 0])  # 与预测点云对齐
            
            # 坐标轴
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            
            # 显示
            print("    正在显示可视化窗口... (关闭窗口继续)")
            o3d.visualization.draw_geometries(
                [input_pcd, pred_pcd, grasp_sphere, coord_frame],
                window_name=f"帧 {frame_idx} | CD: {avg_cd:.4f}",
                width=1200,
                height=800
            )
    
    # 5. 输出CD统计结果
    print("\n" + "="*60)
    print("倒角距离（CD）统计结果")
    print("="*60)
    
    if cd_results:
        cd_values = [cd for _, cd in cd_results]
        frame_indices = [frame for frame, _ in cd_results]
        
        print(f"可视化帧: {frame_indices}")
        print(f"平均CD: {np.mean(cd_values):.6f}")
        print(f"最大CD: {np.max(cd_values):.6f} (帧 {frame_indices[np.argmax(cd_values)]})")
        print(f"最小CD: {np.min(cd_values):.6f} (帧 {frame_indices[np.argmin(cd_values)]})")
        
        # 绘制CD变化图
        plt.figure(figsize=(10, 6))
        plt.plot(frame_indices, cd_values, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('帧索引', fontsize=12)
        plt.ylabel('倒角距离 (CD)', fontsize=12)
        plt.title('各帧倒角距离变化', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 在图上标注每个点的值
        for i, (frame, cd) in enumerate(cd_results):
            plt.annotate(f'{cd:.4f}', 
                        xy=(frame, cd), 
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('cd_statistics.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("\nCD统计图已保存为: cd_statistics.png")
    
    print(f"\n完成 {len(valid_frames)} 帧的可视化")

# =======================================================================
# 4. 主程序入口
# =======================================================================

if __name__ == "__main__":
    print("开始基于倒角距离（CD）热力图的可视化")
    print("="*60)
    run_cd_heatmap_visualization()