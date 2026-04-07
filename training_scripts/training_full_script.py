import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import glob
import re
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import sys
import open3d as o3d

# =======================================================================
# 1. 模型组件 (Model Components)
# =======================================================================

def generate_template_points(M=2048):
    """在单位球体上生成一组固定数量的模板点 (fixed set of template points)。"""
    np.random.seed(42) 
    # 生成点并归一化到单位球体
    coords = np.random.randn(M, 3)
    radii = np.linalg.norm(coords, axis=1, keepdims=True)
    template_points = coords / radii
    return torch.from_numpy(template_points).float()

class Encoder(nn.Module):
    def __init__(self, latent_dim=32):
        super(Encoder, self).__init__()
        # 类似 PointNet 的结构用于特征提取
        self.mlp_local = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.ReLU(),
            nn.Conv1d(128, 256, 1)
        )
        self.fc_global = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
    def forward(self, x):
        # 转换为 (B, 3, N) 以适应 Conv1d
        x = x.transpose(2, 1) 
        local_features = self.mlp_local(x)
        # 全局最大池化 (Global Max Pooling) 得到全局特征向量
        global_feature, _ = torch.max(local_features, dim=2, keepdim=False) 
        z = self.fc_global(global_feature)
        return z

class Decoder(nn.Module):
    def __init__(self, template_points, latent_dim=32):
        super(Decoder, self).__init__()
        self.template_points = template_points 
        self.M = template_points.shape[0] 
        # 解码器输入：模板点 Q (3D) 和潜在代码 z (latent_dim)
        self.mlp_reconstruction = nn.Sequential(
            nn.Linear(3 + latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 3) # 输出 3D 坐标 (预测点云 P_hat)
        )
    def forward(self, z):
        B = z.size(0)
        # 广播潜在向量 z: (B, 32) -> (B, M, 32)
        z_broadcast = z.unsqueeze(1).repeat(1, self.M, 1)
        # 广播模板点 Q: (M, 3) -> (B, M, 3)
        q_broadcast = self.template_points.to(z.device).unsqueeze(0).repeat(B, 1, 1)
        # 拼接特征: (B, M, 35)
        input_features = torch.cat([q_broadcast, z_broadcast], dim=2)
        # 重建点云
        p_hat = self.mlp_reconstruction(input_features)
        return p_hat

class MeshReconstructionNet(nn.Module):
    def __init__(self, M=2048, latent_dim=32):
        super(MeshReconstructionNet, self).__init__()
        template_points = generate_template_points(M)
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(template_points, latent_dim)
    def forward(self, O):
        # O: 遮挡点云 (Input)
        z = self.encoder(O) # 全局特征向量
        p_hat = self.decoder(z) # 重建点云
        return p_hat

# =======================================================================
# 2. Chamfer 损失函数 (Chamfer Loss Function)
# =======================================================================

def chamfer_loss(pred, gt):
    """
    计算两个点云之间的 Chamfer 距离。
    pred: (B, N, 3), gt: (B, M, 3)
    """
    # 计算点对之间的距离 (B, N, M)
    dist = torch.cdist(pred, gt)
    
    # 1. (Pred -> GT) 每个预测点到最近的 GT 点的最小距离
    min_dist_pred_to_gt, _ = torch.min(dist, dim=2)
    
    # 2. (GT -> Pred) 每个 GT 点到最近的预测点的最小距离
    min_dist_gt_to_pred, _ = torch.min(dist, dim=1)
    
    # CD 损失是平方最小距离的均值之和
    cd_loss = torch.mean(min_dist_pred_to_gt**2) + torch.mean(min_dist_gt_to_pred**2)
    return cd_loss

# =======================================================================
# 3. 数据集类 (Dataset Class) 
# =======================================================================

class PointcloudDataset(Dataset):
    """
    数据集类，加载遮挡点云 (.ply) 并从完整网格 (.obj) 中采样地面真值点云。
    """
    def __init__(self, file_paths, N_input, M_output): 
        self.file_paths = file_paths # 存储 (occ_path, full_path, time_index)
        self.N = N_input # 输入点云大小 (例如: 3000)
        self.M = M_output # 输出点云大小 (例如: 2048)
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # 解包路径和索引
        occ_file_path, full_file_path, time_index = self.file_paths[idx]

        # 1. 加载遮挡点云 (.ply) (输入 O)
        occ_pcd_o3d = o3d.io.read_point_cloud(occ_file_path)
        occ_pcd_raw = np.asarray(occ_pcd_o3d.points)
        num_points_raw = occ_pcd_raw.shape[0]
        
        # --- 实现固定大小 N 的采样和填充 ---
        if num_points_raw == 0:
            occ_pcd = torch.zeros((self.N, 3)).float()
        elif num_points_raw >= self.N:
            # 采样 N 个点 (无放回)
            choice = np.random.choice(num_points_raw, self.N, replace=False)
            occ_pcd_fixed = occ_pcd_raw[choice, :]
            occ_pcd = torch.from_numpy(occ_pcd_fixed).float()
        else:
            # 填充/过采样到 N 个点 (有放回)
            choice = np.random.choice(num_points_raw, self.N, replace=True)
            occ_pcd_fixed = occ_pcd_raw[choice, :]
            occ_pcd = torch.from_numpy(occ_pcd_fixed).float()
        
        # 2. 加载完整网格 (.obj) 并采样 M 个点作为地面真值 (P_GT)
        full_mesh = o3d.io.read_triangle_mesh(full_file_path)
        
        # 确保网格加载成功并包含顶点
        if not full_mesh.has_vertices():
            print(f"警告: 网格文件 {full_file_path} 加载成功但没有顶点。返回零张量。")
            full_pcd_gt = torch.zeros((self.M, 3)).float()
        else:
            # 从网格表面均匀采样 M 个点作为地面真值
            full_pcd_o3d = full_mesh.sample_points_uniformly(number_of_points=self.M)
            full_pcd_gt = torch.from_numpy(np.asarray(full_pcd_o3d.points)).float()
        
        # 返回 occ_pcd (N, 3), full_pcd_gt (M, 3), time_index 
        return occ_pcd, full_pcd_gt, time_index 

# =======================================================================
# 4. 训练函数 (Training Function) 
# =======================================================================

def train_model():
    
    # --- 配置 (Configuration) ---
    PROJECT_ROOT = "Cu_BEM_2/tshirt-data/tshirt_out_3"
    DATASET_ROOT = os.path.join(PROJECT_ROOT, "dataset_random")
    WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "trained_full_random_weights.pth")
    LOSS_HISTORY_PATH = os.path.join(PROJECT_ROOT, "loss_full_random_history.csv")
    
    N_INPUT_POINTS = 3000
    M_OUTPUT_POINTS = 2048
    BATCH_SIZE = 8
    NUM_EPOCHS = 500
    LEARNING_RATE = 0.001
    
    # --- 验证和早停配置 (Validation and Early Stopping Config) ---
    VALIDATION_RATIO = 0.1 
    PATIENCE = 30

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- 启动训练，设备: {device.type} ---")
    print(f"训练权重将保存到: {WEIGHTS_PATH}")
    print(f"损失历史将保存到: {LOSS_HISTORY_PATH}")

    # 1. 数据加载和分割 (Data Loading and Splitting)
    
    # 查找所有遮挡点云文件 (.ply)
    occluded_files = glob.glob(os.path.join(DATASET_ROOT, "occ_frame_*.ply"))
    
    FILE_PATHS = [] # 存储 (occ_path, full_path, time_index) 元组
    
    if not occluded_files:
        raise FileNotFoundError(f"错误: 在 {DATASET_ROOT} 中没有找到 'occ_frame_*.ply' 文件。")
        
    print(f"找到 {len(occluded_files)} 个遮挡点云文件。开始匹配完整网格...")
    
    files_with_regex_fail = 0 
    files_missing_gt = 0

    for occ_path in occluded_files:
        file_name = os.path.basename(occ_path)
        
        match = re.search(r"occ_frame_(\d+)", file_name)
        
        if match:
            time_index = int(match.group(1))
            
            # *** 搜索对应的 .obj 完整网格文件 ***
            full_file_file = f"full_frame_{time_index}.obj"
            full_file_path = os.path.join(DATASET_ROOT, full_file_file)
            
            if os.path.exists(full_file_path):
                # 匹配成功，添加到训练/验证列表
                FILE_PATHS.append((occ_path, full_file_path, time_index))
            else:
                files_missing_gt += 1
                
        else:
            files_with_regex_fail += 1

            
    if not FILE_PATHS:
        if files_with_regex_fail > 0 or files_missing_gt > 0:
            print(f"--- 警告: {files_with_regex_fail} 个输入文件因 'occ_frame_X' 命名不匹配而被跳过，{files_missing_gt} 个文件因缺少对应的 GT 文件而被跳过。 ---")
            
        raise FileNotFoundError(f"错误: 未找到匹配的 (遮挡点云, 完整网格) 数据对。请检查数据命名和路径。")

    print(f"找到的总匹配数据对: {len(FILE_PATHS)}")
    
    # 随机打乱文件路径
    np.random.shuffle(FILE_PATHS)
    
    # 分割验证集和训练集
    VAL_SIZE = int(len(FILE_PATHS) * VALIDATION_RATIO)
    train_paths = FILE_PATHS[VAL_SIZE:]
    val_paths = FILE_PATHS[:VAL_SIZE]

    print(f"数据分割: 训练集 {len(train_paths)} 对 | 验证集 {len(val_paths)} 对。")
    
    # 初始化数据集和数据加载器
    train_dataset = PointcloudDataset(train_paths, N_INPUT_POINTS, M_OUTPUT_POINTS)
    val_dataset = PointcloudDataset(val_paths, N_INPUT_POINTS, M_OUTPUT_POINTS)
    
    # num_workers=0 是在 Windows 上确保兼容性的常见做法
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 2. 模型、优化器和损失初始化 (Model, Optimizer, and Loss Initialization)
    model = MeshReconstructionNet(M=M_OUTPUT_POINTS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 损失历史初始化 ---
    loss_history = [] 

    # 3. 带有验证和早停的训练循环 (Training Loop with Validation and Early Stopping)
    best_val_loss = float('inf')
    patience_counter = 0 # 早停计数器
    
    for epoch in range(NUM_EPOCHS):
        
        # --- 训练阶段 ---
        model.train() 
        total_train_loss = 0.0
        
        # 核心修改 1: 接收地面真值 full_pcd_gt
        for batch_idx, (occluded_pcd, full_pcd_gt, _) in enumerate(train_loader): 
            
            occluded_pcd = occluded_pcd.to(device)
            full_pcd_gt = full_pcd_gt.to(device) # 将地面真值移动到设备
            
            optimizer.zero_grad()
            # 输入仍然是 occluded_pcd
            reconstructed_pcd = model(occluded_pcd)
            
            # *** 核心修改 2: 训练损失计算针对 full_pcd_gt (地面真值) ***
            loss = chamfer_loss(reconstructed_pcd, full_pcd_gt)
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)

        # --- 验证阶段 ---
        model.eval() # 切换到评估模式
        total_val_loss = 0.0
        with torch.no_grad(): # 验证期间禁用梯度计算
            # 核心修改 3: 接收地面真值 full_pcd_gt
            for occluded_pcd, full_pcd_gt, _ in val_loader: 
                occluded_pcd = occluded_pcd.to(device)
                full_pcd_gt = full_pcd_gt.to(device) # 将地面真值移动到设备
                
                # 输入仍然是 occluded_pcd
                reconstructed_pcd = model(occluded_pcd)
                
                # *** 核心修改 4: 验证损失计算针对 full_pcd_gt (地面真值) ***
                loss = chamfer_loss(reconstructed_pcd, full_pcd_gt)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        
        current_epoch = epoch + 1
        # 更新打印信息，显示损失是对 GT 计算的
        print(f"Epoch {current_epoch}/{NUM_EPOCHS} | 训练损失 (对 GT): {avg_train_loss:.6f} | 验证损失 (对 GT): {avg_val_loss:.6f}")

        # --- 损失记录 ---
        loss_history.append([current_epoch, avg_train_loss, avg_val_loss])


        # --- 4. 检查点保存和早停 ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0 # 重置耐心计数器
            # 保存最佳权重 (基于验证损失)
            torch.save(model.state_dict(), WEIGHTS_PATH)
            print(f"--- 模型已保存: 在 Epoch {current_epoch} 达到新的最佳验证损失 ({best_val_loss:.6f}) ---")
        else:
            patience_counter += 1 # 增加耐心计数器
            print(f"验证损失未改善。耐心 {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print(f"\n--- 早停触发！验证损失在 {PATIENCE} 个 Epoch 内没有改善。 ---")
            # 停止训练循环
            break 

    # --- 训练完成后: 将损失历史保存到 CSV ---
    df_loss = pd.DataFrame(loss_history, columns=['Epoch', 'Training Loss', 'Validation Loss'])
    df_loss.to_csv(LOSS_HISTORY_PATH, index=False)
    print(f"\n--- 损失历史已保存到 {LOSS_HISTORY_PATH} ---")

    print("\n--- 训练完成 ---")


if __name__ == '__main__':
    # 检查是否安装了 pandas
    try:
        import pandas as pd
    except ImportError:
        print("错误: 运行此脚本需要 'pandas'。请运行 pip install pandas")
        sys.exit(1)
    
    # 检查是否安装了 open3d
    try:
        import open3d as o3d
    except ImportError:
        print("错误: 运行此脚本需要 'open3d'。请运行 pip install open3d")
        sys.exit(1)
    
    train_model()