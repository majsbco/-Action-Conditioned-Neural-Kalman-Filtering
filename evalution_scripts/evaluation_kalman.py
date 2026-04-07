import torch
import torch.nn as nn
import numpy as np
import os
import glob
import re
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import open3d as o3d
from scipy.spatial import cKDTree
import trimesh  # 用于加载OBJ文件

# =======================================================================
# 1. 模型定义 (与训练脚本一致)
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
        
        # 加载模板
        self.template_dict = template_dict or {}
        self.templates = {}
        self.template_names = ['flat', 'folded']
        
        if not self.template_dict:
            print(f"[模型初始化] 警告: 未提供模板字典，将创建默认的球面模板。")
            default_template = self._generate_template_points(M)
            self.templates['default'] = default_template
            self.template_names = ['default']
        else:
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
        
        self.current_template_name = 'flat'
        self.current_template = self.templates['flat']
        print(f"[模型初始化] 模板库加载完成: {list(self.templates.keys())}")
        
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
        print(f"  -> 已加载顶点数: {len(vertices)}")

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

    def select_template_by_frame(self, frame_idx):
        if 250 <= frame_idx <= 480:
            selected_name = 'folded'
        else:
            selected_name = 'flat'

        if selected_name != self.current_template_name:
            self.current_template_name = selected_name
            selected_template_tensor = self.templates[selected_name]
            model_device = next(self.parameters()).device
            self.current_template = selected_template_tensor.to(model_device)
        else:
            model_device = next(self.parameters()).device
            if self.current_template.device != model_device:
                self.current_template = self.current_template.to(model_device)
        
        return self.current_template, self.current_template_name

    def forward(self, o_t, h_prev, h_prev_prev=None, current_frame_idx=0):
        B = o_t.size(0)
        z_t = self.enc_obs(o_t)
        
        if h_prev_prev is not None:
            velocity = h_prev - h_prev_prev
            h_inertial = h_prev + self.damping * velocity
        else:
            h_inertial = h_prev
            
        h_gru = self.f_gru(z_t, h_prev)
        
        gate_input = torch.cat([h_inertial, z_t], dim=-1)
        alpha = self.gate(gate_input)
        alpha_scaled = 0.7 + 0.3 * alpha
        h_t = (1 - alpha_scaled) * h_inertial + alpha_scaled * h_gru
        
        selected_template, used_tmpl_name = self.select_template_by_frame(current_frame_idx)
        
        z_ext = h_t.unsqueeze(1).repeat(1, self.M, 1)
        template_ext = selected_template.unsqueeze(0).repeat(B, 1, 1)
        p_hat = self.decoder(torch.cat([template_ext, z_ext], dim=-1))
        
        return p_hat, h_t, z_t, used_tmpl_name

# =======================================================================
# 2. 二次指数平滑滤波器
# =======================================================================

class DoubleExponentialSmoothing:
    def __init__(self, dim, alpha=0.5, beta=0.5):
        self.alpha = alpha
        self.beta = beta
        self.dim = dim
        self.S = None
        self.B = None
        self.initialized = False

    def update(self, observation):
        if not self.initialized:
            self.S = observation.copy().astype(np.float32)
            self.B = np.zeros(self.dim, dtype=np.float32)
            self.initialized = True
            return self.S.copy()
        S_prev = self.S.copy()
        self.S = self.alpha * observation + (1 - self.alpha) * (self.S + self.B)
        self.B = self.beta * (self.S - S_prev) + (1 - self.beta) * self.B
        return self.S.copy().astype(np.float32)

# =======================================================================
# 3. 平滑损失函数 (从训练脚本添加)
# =======================================================================

def smoothness_loss(h_current, h_previous, lambda_smooth=0.5):
    """
    平滑度损失函数 - 惩罚相邻帧潜状态之间的剧烈变化
    与训练脚本中的定义一致
    
    Args:
        h_current: 当前帧的GRU隐状态 [B, latent_dim]
        h_previous: 前一帧的GRU隐状态 [B, latent_dim]
        lambda_smooth: 平滑损失权重，默认0.5
    
    Returns:
        平滑度损失值
    """
    if h_previous is None:
        return torch.tensor(0.0, device=h_current.device)
    return lambda_smooth * torch.mean((h_current - h_previous)**2)

# =======================================================================
# 4. 简化版可视化脚本
# =======================================================================

def run_simple_visualization():
    """
    简化版可视化脚本，类似于baseline的可视化
    显示原始输入、模型输出和平滑后的输出
    同时计算并显示平滑损失
    """
    # 配置参数
    DATA_PATH = r"D:\Cu_BEM_2\tshirt-data\tshirt_kalman_3\dataset_arm"
    MODEL_PATH = "checkpoints/best_model_v2_dim64_DynamicTemplate_smooth.pth"
    TARGET_FRAMES = [10, 50, 100, 150, 200, 300, 400, 500]  # 要显示的帧
    
    # 从checkpoint中获取平滑损失权重
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    lambda_smooth = checkpoint.get('smoothness_weight', 0.5)
    
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"平滑损失权重: λ={lambda_smooth}")
    
    # 从checkpoint中获取模板字典
    if 'template_dict' in checkpoint and checkpoint['template_dict'] is not None:
        TEMPLATE_DICT = checkpoint['template_dict']
        print(f"模板字典: {list(TEMPLATE_DICT.keys())}")
    else:
        # 如果checkpoint中没有模板字典，使用默认的
        TEMPLATE_DICT = {
            'flat': r"D:\Cu_BEM_2\tshirt-data\tshirt_kalman_3\tshirt_mech00001.obj",
            'folded': r"D:\Cu_BEM_2\tshirt-data\tshirt_kalman_3\tshirt_mech00250.obj",
        }
        print(f"使用默认模板字典: {list(TEMPLATE_DICT.keys())}")
    
    # 加载模型
    model = RecursiveTrackingNet(
        latent_dim=64, 
        template_dict=TEMPLATE_DICT
    ).to(device)
    
    # 修复状态字典
    state_dict = checkpoint['model_state_dict']
    keys_to_remove = []
    for key in state_dict.keys():
        if 'current_template' in key or ('template' in key and 'templates' not in key):
            keys_to_remove.append(key)
    
    for key in keys_to_remove:
        if key in state_dict:
            del state_dict[key]
            print(f"已删除意外键: {key}")
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # 初始化二次平滑滤波器
    smoother = DoubleExponentialSmoothing(dim=64, alpha=0.5, beta=0.5)
    
    # 获取并排序文件
    files = sorted(glob.glob(os.path.join(DATA_PATH, "occ_frame_*.ply")),
                   key=lambda x: int(re.search(r"occ_frame_(\d+)", x).group(1)))
    
    print(f"开始推理，将显示帧: {TARGET_FRAMES}")
    
    # 存储平滑损失历史
    smooth_loss_history = []
    
    # 初始化GRU状态
    h_t = torch.zeros(1, 64).to(device)
    h_prev = None
    h_prev_prev = None
    
    with torch.no_grad():
        for frame_idx, f_path in enumerate(files):
            frame_id = int(re.search(r"occ_frame_(\d+)", f_path).group(1))
            
            # 读取点云
            pcd_in = o3d.io.read_point_cloud(f_path)
            pts = np.asarray(pcd_in.points, dtype=np.float32)
            
            # 采样到固定点数
            n_points = 2048
            if len(pts) > n_points:
                pts = pts[np.random.choice(len(pts), n_points, replace=False)]
            
            in_tensor = torch.from_numpy(pts).float().unsqueeze(0).to(device)
            
            # 模型推理
            recon_raw, h_next, z_t, used_tmpl = model(in_tensor, h_t, h_prev, current_frame_idx=frame_idx)
            
            # 计算平滑损失
            loss_smooth = smoothness_loss(h_next, h_prev, lambda_smooth)
            smooth_loss_history.append((frame_idx, loss_smooth.item()))
            
            # 二次平滑
            h_gru_np = h_next.squeeze(0).cpu().numpy()
            h_smooth_np = smoother.update(h_gru_np)
            
            # 用平滑后的潜状态重构点云
            h_smooth = torch.from_numpy(h_smooth_np).unsqueeze(0).to(device)
            z_ext = h_smooth.unsqueeze(1).repeat(1, model.M, 1)
            template_ext = model.current_template.unsqueeze(0)
            smooth_input = torch.cat([template_ext, z_ext], dim=-1)
            recon_smooth = model.decoder(smooth_input)
            
            # 只显示目标帧
            if frame_id in TARGET_FRAMES:
                print(f"\n=== 显示帧 {frame_id} ===")
                print(f"模板: {used_tmpl}, 平滑损失: {loss_smooth.item():.6f}")
                
                # 创建Open3D点云对象
                def make_o3d(points, color, offset):
                    p = o3d.geometry.PointCloud()
                    p.points = o3d.utility.Vector3dVector(points)
                    p.paint_uniform_color(color)
                    p.translate([offset, 0, 0])
                    return p
                
                # 创建三个点云：输入、模型输出、平滑后输出
                vis_in = make_o3d(pts, [0.2, 0.4, 0.8], -0.6)      # 蓝色：输入
                vis_raw = make_o3d(recon_raw[0].cpu().numpy(), [0.8, 0.2, 0.2], 0)  # 红色：模型输出
                vis_smooth = make_o3d(recon_smooth[0].cpu().numpy(), [0.2, 0.8, 0.2], 0.6)  # 绿色：平滑后输出
                
                # 添加坐标轴
                coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                
                # 显示
                print(f"关闭窗口以继续...")
                o3d.visualization.draw_geometries(
                    [vis_in, vis_raw, vis_smooth, coord], 
                    window_name=f"Frame {frame_id}: Input(蓝) | Model Output(红) | Smoothed(绿)"
                )
            
            # 更新GRU状态
            h_prev_prev = h_prev
            h_prev = h_t.detach()
            h_t = h_next
            
            # 每100帧打印进度
            if frame_idx % 100 == 0:
                print(f"处理进度: {frame_idx}/{len(files)}")
    
    # 打印平滑损失统计
    if smooth_loss_history:
        losses = [loss for _, loss in smooth_loss_history]
        print(f"\n=== 平滑损失统计 ===")
        print(f"平均平滑损失: {np.mean(losses):.6f}")
        print(f"最大平滑损失: {np.max(losses):.6f}")
        print(f"最小平滑损失: {np.min(losses):.6f}")
        
        # 绘制平滑损失曲线
        plt.figure(figsize=(12, 6))
        frames = [frame for frame, _ in smooth_loss_history]
        losses = [loss for _, loss in smooth_loss_history]
        
        plt.plot(frames, losses, 'b-', linewidth=1.5, alpha=0.8)
        plt.title(f'Smoothness Loss History (λ={lambda_smooth:.2f})')
        plt.xlabel('Frame Index')
        plt.ylabel('Smoothness Loss')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存图表
        filename = f"smooth_loss_history_lambda_{lambda_smooth:.2f}.png"
        plt.savefig(filename, dpi=150)
        print(f"平滑损失历史图表已保存到: {filename}")
        plt.show()

# =======================================================================
# 5. 主程序入口
# =======================================================================
if __name__ == "__main__":
    print("开始运行简化版可视化脚本")
    print("="*60)
    run_simple_visualization()