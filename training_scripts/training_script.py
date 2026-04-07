import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import glob
import re
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import open3d as o3d

# =======================================================================
# 1. 模型组件 (Model Components)
# =======================================================================

def generate_template_points(M=2048):
    """在单位球体上生成一组固定数量的模板点。"""
    np.random.seed(42) 
    coords = np.random.randn(M, 3)
    radii = np.linalg.norm(coords, axis=1, keepdims=True)
    template_points = coords / radii
    return torch.from_numpy(template_points).float()

class Encoder(nn.Module):
    def __init__(self, latent_dim=32):
        super(Encoder, self).__init__()
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
        x = x.transpose(2, 1) 
        local_features = self.mlp_local(x)
        global_feature, _ = torch.max(local_features, dim=2, keepdim=False) 
        z = self.fc_global(global_feature)
        return z

class Decoder(nn.Module):
    def __init__(self, template_points, latent_dim=32):
        super(Decoder, self).__init__()
        self.template_points = template_points 
        self.M = template_points.shape[0] 
        self.mlp_reconstruction = nn.Sequential(
            nn.Linear(3 + latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 3) 
        )
    def forward(self, z):
        B = z.size(0)
        z_broadcast = z.unsqueeze(1).repeat(1, self.M, 1)
        q_broadcast = self.template_points.to(z.device).unsqueeze(0).repeat(B, 1, 1)
        input_features = torch.cat([q_broadcast, z_broadcast], dim=2)
        p_hat = self.mlp_reconstruction(input_features)
        return p_hat

class MeshReconstructionNet(nn.Module):
    def __init__(self, M=2048, latent_dim=32):
        super(MeshReconstructionNet, self).__init__()
        template_points = generate_template_points(M)
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(template_points, latent_dim)
    def forward(self, O):
        z = self.encoder(O) 
        p_hat = self.decoder(z) 
        return p_hat, z 

# =======================================================================
# 2. 损失函数 (Loss Functions)
# =======================================================================

def chamfer_loss(pred, gt):
    dist = torch.cdist(pred, gt)
    min_dist_pred_to_gt, _ = torch.min(dist, dim=2)
    min_dist_gt_to_pred, _ = torch.min(dist, dim=1)
    cd_loss = torch.mean(min_dist_pred_to_gt**2) + torch.mean(min_dist_gt_to_pred**2)
    return cd_loss

def compute_smoothness_loss(latent_vectors, time_indices):
    if latent_vectors.size(0) < 2:
        return torch.tensor(0.0, device=latent_vectors.device)
    
    sorted_indices = torch.argsort(time_indices)
    z_sorted = latent_vectors[sorted_indices]
    diff = z_sorted[1:] - z_sorted[:-1]
    smooth_loss = torch.mean(torch.norm(diff, p=2, dim=-1))
    return smooth_loss

# =======================================================================
# 3. 数据集类 (Dataset Class)
# =======================================================================

class PointcloudDataset(Dataset):
    def __init__(self, file_paths, N_input, M_output): 
        self.file_paths = file_paths 
        self.N = N_input 
        self.M = M_output 
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        occ_file_path, full_file_path, time_index = self.file_paths[idx]
        
        # 读取遮挡点云
        occ_pcd_o3d = o3d.io.read_point_cloud(occ_file_path)
        occ_pcd_raw = np.asarray(occ_pcd_o3d.points)
        num_points_raw = occ_pcd_raw.shape[0]
        
        if num_points_raw == 0:
            occ_pcd = torch.zeros((self.N, 3)).float()
        else:
            choice = np.random.choice(num_points_raw, self.N, replace=(num_points_raw < self.N))
            occ_pcd = torch.from_numpy(occ_pcd_raw[choice, :]).float()
        
        full_pcd_o3d = o3d.io.read_point_cloud(full_file_path)
        full_pcd_raw = np.asarray(full_pcd_o3d.points)
        num_full_raw = full_pcd_raw.shape[0]

        if num_full_raw == 0:
            full_pcd_gt = torch.zeros((self.M, 3)).float()
        else:
            # 对 GT 进行采样以固定点数计算评价指标（如有需要）
            choice_full = np.random.choice(num_full_raw, self.M, replace=(num_full_raw < self.M))
            full_pcd_gt = torch.from_numpy(full_pcd_raw[choice_full, :]).float()
        
        return occ_pcd, full_pcd_gt, time_index 

# =======================================================================
# 4. 训练函数 (Training Function)
# =======================================================================

def train_model():
    # 使用 Windows 原始路径字符串
    PROJECT_ROOT = r"Cu_BEM_2\tshirt-data\tshirt_out_3"
    DATASET_ROOT = os.path.join(PROJECT_ROOT, "dataset_arm")
    WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "trained_arm_weights.pth")
    LOSS_HISTORY_PATH = os.path.join(PROJECT_ROOT, "loss_arm_history.csv")
    
    # 基础超参数
    N_INPUT_POINTS = 3000
    M_OUTPUT_POINTS = 2048
    BATCH_SIZE = 8
    NUM_EPOCHS = 500
    LEARNING_RATE = 0.001
    SMOOTH_WEIGHT = 0.1 
    VALIDATION_RATIO = 0.1 
    PATIENCE = 30

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- 启动训练，设备: {device} ---")

    # 数据查找逻辑：修改 full_frame 后缀为 .ply
    print(f"正在扫描: {DATASET_ROOT}")
    occluded_files = glob.glob(os.path.join(DATASET_ROOT, "occ_frame_*.ply"))
    
    FILE_PATHS = [] 
    for occ_path in occluded_files:
        file_name = os.path.basename(occ_path)
        match = re.search(r"occ_frame_(\d+)", file_name)
        if match:
            time_idx = int(match.group(1))
            # 修改此处：后缀由 .obj 改为 .ply
            full_path = os.path.join(DATASET_ROOT, f"full_frame_{time_idx}.ply")
            if os.path.exists(full_path):
                FILE_PATHS.append((occ_path, full_path, time_idx))

    # 鲁棒性检查
    if len(FILE_PATHS) == 0:
        print("错误: 未找到匹配的文件对。请确保 occ_frame_X.ply 和 full_frame_X.ply 同时存在。")
        return
    else:
        print(f"成功匹配数据对数量: {len(FILE_PATHS)}")

    # 划分数据集
    np.random.shuffle(FILE_PATHS)
    VAL_SIZE = max(1, int(len(FILE_PATHS) * VALIDATION_RATIO))
    train_dataset = PointcloudDataset(FILE_PATHS[VAL_SIZE:], N_INPUT_POINTS, M_OUTPUT_POINTS)
    val_dataset = PointcloudDataset(FILE_PATHS[:VAL_SIZE], N_INPUT_POINTS, M_OUTPUT_POINTS)
    
    # drop_last=True 能够避免 BatchSize=1 时计算平滑损失报错
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    model = MeshReconstructionNet(M=M_OUTPUT_POINTS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    loss_history = [] 
    best_val_loss = float('inf')
    patience_counter = 0 
    
    for epoch in range(NUM_EPOCHS):
        # 训练阶段
        model.train() 
        total_train_loss = 0.0
        
        for occluded_pcd, _, time_idx in train_loader: 
            occluded_pcd = occluded_pcd.to(device)
            time_idx = time_idx.to(device)
            
            optimizer.zero_grad()
            reconstructed_pcd, latent_z = model(occluded_pcd)
            
            cd_loss = chamfer_loss(reconstructed_pcd, occluded_pcd)
            smooth_loss = compute_smoothness_loss(latent_z, time_idx)
            
            loss = cd_loss + (SMOOTH_WEIGHT * smooth_loss)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)

        # 验证阶段
        model.eval() 
        total_val_loss = 0.0
        with torch.no_grad():
            for o_pcd, _, t_idx in val_loader: 
                o_pcd, t_idx = o_pcd.to(device), t_idx.to(device)
                rec_pcd, l_z = model(o_pcd)
                v_cd = chamfer_loss(rec_pcd, o_pcd)
                v_sm = compute_smoothness_loss(l_z, t_idx)
                total_val_loss += (v_cd + SMOOTH_WEIGHT * v_sm).item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        loss_history.append([epoch+1, avg_train_loss, avg_val_loss])

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0 
            torch.save(model.state_dict(), WEIGHTS_PATH)
            print(f"--- 已保存最佳模型 (Val Loss: {best_val_loss:.6f}) ---")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"触发早停，训练结束。")
                break 

    df_loss = pd.DataFrame(loss_history, columns=['Epoch', 'TrainLoss', 'ValLoss'])
    df_loss.to_csv(LOSS_HISTORY_PATH, index=False)

if __name__ == '__main__':
    train_model()