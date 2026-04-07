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
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

# =======================================================================
# 1. 模型定义 (基于训练脚本的KalmanTracker，不带抓取点)
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
            nn.Linear(512, 256), nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, latent_dim)
        )

    def forward(self, x):
        x = x.transpose(2, 1)
        x = self.mlp(x)
        x = torch.max(x, 2)[0]
        return self.fc(x)

class KalmanTracker(nn.Module):
    def __init__(self, template_path, latent_dim=64):
        super(KalmanTracker, self).__init__()
        
        # 加载模板
        mesh = trimesh.load(template_path, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = list(mesh.geometry.values())[0]
        
        pts = np.array(mesh.vertices, dtype=np.float32)
        # 模板也需要初始化在原点 (减去自身中心)
        self.v_mean = np.mean(pts, axis=0)
        pts_centered = pts - self.v_mean
        
        self.register_buffer('template', torch.from_numpy(pts_centered).float())
        self.num_pts = pts_centered.shape[0]

        self.obs_encoder = PointNetEncoder(latent_dim)
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

def compute_cd_percentage_heatmap(pred_points, gt_points, template_diagonal, colormap='viridis', max_percentage=30.0):
    """
    计算倒角距离（CD）并将其转换为模板包围盒体对角线的百分比
    然后生成热力图数据
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

def compute_template_diagonal(template_path):
    """
    计算模板包围盒体对角线长度
    """
    if not os.path.exists(template_path):
        print(f"错误: 模板文件不存在: {template_path}")
        return None
    
    # 加载模板
    mesh = trimesh.load(template_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = list(mesh.geometry.values())[0]
    
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

def create_percentage_colorbar(max_percentage=30.0, colormap='viridis', save_path='percentage_colorbar_no_grasp.png'):
    """
    创建基于百分比的颜色条图例（竖版）
    """
    # 修改图形尺寸，使其更适合竖版颜色条
    fig, ax = plt.subplots(figsize=(1.2, 6))
    fig.subplots_adjust(left=0.3, right=0.5)
    
    cmap = cm.get_cmap(colormap)
    norm = Normalize(vmin=0, vmax=max_percentage)
    
    # 将orientation参数从'horizontal'改为'vertical'
    cb = ColorbarBase(ax, cmap=cmap, norm=norm, orientation='vertical')
    cb.set_label('CD as % of Template\nBounding Box Diagonal', fontsize=14, fontweight='bold')
    
    # 设置刻度
    ticks = np.linspace(0, max_percentage, 7)  # 7个刻度
    cb.set_ticks(ticks)
    cb.set_ticklabels([f'{tick:.1f}%' for tick in ticks], fontsize=12)
    
    # 设置标题
    # title = f'Color Scale\n0% to {max_percentage:.1f}%\nof Template Diagonal'
    # ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    
    # 保存颜色条
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"竖版百分比颜色条已保存至: {save_path}")
    
    plt.close()

# =======================================================================
# 3. 基于CD百分比热力图的可视化函数 (不带抓取点)
# =======================================================================

def run_cd_percentage_visualization_no_grasp():
    """
    基于倒角距离（CD）百分比热力图的可视化
    使用不带抓取点的KalmanTracker模型
    """
    # 配置参数 (与训练脚本保持一致)
    BASE = r"D:\Cu_BEM_2\tshirt-data\tshirt_kalman_3"
    TEMPLATE_PATH = os.path.join(BASE, "tshirt_mech00001.obj")
    DATA_PATH = os.path.join(BASE, "dataset_arm")
    MODEL_PATH = "checkpoints/kalman_best_aligned.pth"  # 使用训练脚本中的模型路径
    
    # 在这里直接指定要可视化的帧索引
    TARGET_FRAMES = [100, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]
    
    # 可视化参数
    COLORMAP = 'viridis'  # 使用viridis颜色映射
    
    # 颜色条最大百分比
    MAX_PERCENTAGE = 5.0  # 与带抓取点的可视化保持一致
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print("="*60)
    print("基于倒角距离（CD）百分比热力图的可视化 (不带抓取点)")
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
    create_percentage_colorbar(MAX_PERCENTAGE, colormap=COLORMAP, save_path='percentage_colorbar_no_grasp.png')
    
    # 3. 加载模型
    print("\n3. 加载模型...")
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 模型文件不存在: {MODEL_PATH}")
        return
    
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    state_dict = checkpoint['model_state_dict']
    
    # 初始化模型
    model = KalmanTracker(TEMPLATE_PATH).to(device)
    
    # 加载状态字典
    model.load_state_dict(state_dict)
    model.eval()
    print(f"   模型加载成功: {MODEL_PATH}")
    
    # 4. 获取点云文件列表
    print("\n4. 扫描点云文件...")
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
    
    # 5. 递归计算并可视化每一帧
    # 初始化状态
    h_t = None
    h_prev = None
    
    # 存储所有帧的CD结果
    cd_results = []
    all_percentages = []  # 收集所有百分比值，用于统计
    
    for frame_idx in range(max(valid_frames) + 1):
        if frame_idx % 50 == 0:
            print(f"  处理进度: {frame_idx}/{max(valid_frames)}")
        
        # 加载当前帧点云并中心化
        file_path = files[frame_idx]
        obs_norm, _ = load_point_cloud_and_center(file_path, n_points=2048)
        
        # 加载目标帧点云并中心化
        # 根据训练脚本，使用下一帧作为goal
        goal_idx = min(frame_idx + 1, total_frames - 1)
        goal_file_path = files[goal_idx]
        goal_norm, _ = load_point_cloud_and_center(goal_file_path, n_points=2048)
        
        # 转换为张量
        obs_tensor = torch.from_numpy(obs_norm).float().unsqueeze(0).to(device)
        goal_tensor = torch.from_numpy(goal_norm).float().unsqueeze(0).to(device)
        
        # 模型推理
        with torch.no_grad():
            pred_points, h_next, offsets = model(
                obs_tensor, goal_tensor, h_t, h_prev
            )
        
        # 更新状态
        h_prev = h_t.detach() if h_t is not None else None
        h_t = h_next
        
        # 如果当前帧是需要可视化的帧
        if frame_idx in valid_frames:
            print(f"\n=== 可视化帧 {frame_idx} ===")
            print(f"    文件: {os.path.basename(file_path)}")
            print(f"    目标帧: {os.path.basename(goal_file_path)} (帧 {goal_idx})")
            
            # 转换为numpy
            pred_np = pred_points[0].cpu().numpy()
            
            # 计算倒角距离和生成百分比热力图
            avg_cd, pointwise_dist, pointwise_percentage, heatmap_colors = compute_cd_percentage_heatmap(
                pred_np, obs_norm, template_diagonal, 
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
            
            # 2. 坐标轴
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
            geometries.append(coord_frame)
            
            # 显示
            print("    正在显示可视化窗口... (关闭窗口继续)")
            o3d.visualization.draw_geometries(
                geometries,
                window_name=f"Frame {frame_idx} (No Grasp) | CD: {avg_cd:.4f} | Avg: {avg_percentage:.1f}% | Range: 0-{MAX_PERCENTAGE:.1f}%",
                width=1200,
                height=800
            )
    
    # 6. 输出CD统计结果
    print("\n" + "="*60)
    print("倒角距离（CD）百分比统计结果 (不带抓取点)")
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
    print("百分比颜色条已保存为: percentage_colorbar_no_grasp.png")

# =======================================================================
# 4. 主程序入口
# =======================================================================

if __name__ == "__main__":
    print("开始基于倒角距离（CD）百分比热力图的可视化 (不带抓取点)")
    print("="*60)
    run_cd_percentage_visualization_no_grasp()