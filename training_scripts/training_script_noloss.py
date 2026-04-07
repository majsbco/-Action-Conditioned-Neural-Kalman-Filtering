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
        # x: (B, N, 3) -> (B, 3, N)
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
        return p_hat

# =======================================================================
# 2. 损失函数 (Loss Functions)
# =======================================================================

def chamfer_loss(pred, gt):
    """计算两个点云之间的 Chamfer 距离。"""
    # pred: 模型输出, gt: 这里的 gt 即为输入的被遮挡点云 (Self-fitting)
    dist = torch.cdist(pred, gt)
    min_dist_pred_to_gt, _ = torch.min(dist, dim=2)
    min_dist_gt_to_pred, _ = torch.min(dist, dim=1)
    cd_loss = torch.mean(min_dist_pred_to_gt**2) + torch.mean(min_dist_gt_to_pred**2)
    return cd_loss

# =======================================================================
# 3. 数据集类 (Dataset Class)
# =======================================================================

class OccDataset(Dataset):
    def __init__(self, file_paths, N_input): 
        self.file_paths = file_paths 
        self.N = N_input 
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        occ_file_path = self.file_paths[idx]
        occ_pcd_o3d = o3d.io.read_point_cloud(occ_file_path)
        occ_pcd_raw = np.asarray(occ_pcd_o3d.points)
        num_points_raw = occ_pcd_raw.shape[0]
        
        if num_points_raw == 0:
            occ_pcd = torch.zeros((self.N, 3)).float()
        else:
            choice = np.random.choice(num_points_raw, self.N, replace=(num_points_raw < self.N))
            occ_pcd = torch.from_numpy(occ_pcd_raw[choice, :]).float()
        
        return occ_pcd

# =======================================================================
# 4. 训练函数 (Training Function) 
# =======================================================================

def train_model():
    # --- 配置 ---
    PROJECT_ROOT = "Cu_BEM_2/tshirt-data/tshirt_out_3"
    DATASET_ROOT = os.path.join(PROJECT_ROOT, "dataset_arm")
    WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "trained_armloss_weights.pth")
    LOSS_HISTORY_PATH = os.path.join(PROJECT_ROOT, "arm_history.csv")
    
    N_INPUT_POINTS = 3000
    M_OUTPUT_POINTS = 2048 
    BATCH_SIZE = 8
    NUM_EPOCHS = 500
    LEARNING_RATE = 0.001
    VALIDATION_RATIO = 0.1 
    PATIENCE = 20

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- 启动[纯净自拟合]训练 (无平滑损失)，设备: {device.type} ---")

    # 1. 数据查找
    occluded_files = glob.glob(os.path.join(DATASET_ROOT, "occ_frame_*.ply"))
    
    if not occluded_files:
        print(f"错误: 在 {DATASET_ROOT} 未找到符合条件的 .ply 文件")
        return

    # 2. 划分数据集
    np.random.shuffle(occluded_files)
    VAL_SIZE = int(len(occluded_files) * VALIDATION_RATIO)
    train_files = occluded_files[VAL_SIZE:]
    val_files = occluded_files[:VAL_SIZE]

    train_dataset = OccDataset(train_files, N_INPUT_POINTS)
    val_dataset = OccDataset(val_files, N_INPUT_POINTS)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. 初始化模型
    model = MeshReconstructionNet(M=M_OUTPUT_POINTS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    loss_history = [] 
    best_val_loss = float('inf')
    patience_counter = 0 
    
    # 4. 循环训练
    for epoch in range(NUM_EPOCHS):
        model.train() 
        total_train_loss = 0.0
        
        for occluded_pcd in train_loader: 
            occluded_pcd = occluded_pcd.to(device)
            
            optimizer.zero_grad()
            reconstructed_pcd = model(occluded_pcd)
            
            # 仅计算重建损失 (Chamfer Loss)
            loss = chamfer_loss(reconstructed_pcd, occluded_pcd)
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)

        # --- 验证阶段 ---
        model.eval() 
        total_val_loss = 0.0
        with torch.no_grad():
            for occluded_pcd in val_loader: 
                occluded_pcd = occluded_pcd.to(device)
                reconstructed_pcd = model(occluded_pcd)
                v_loss = chamfer_loss(reconstructed_pcd, occluded_pcd)
                total_val_loss += v_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train CD Loss: {avg_train_loss:.6f} | Val CD Loss: {avg_val_loss:.6f}")

        loss_history.append([epoch+1, avg_train_loss, avg_val_loss])

        # 保存最优模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0 
            torch.save(model.state_dict(), WEIGHTS_PATH)
            print(f"--- 状态: 已保存最佳模型 (Val Loss: {best_val_loss:.6f}) ---")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("触发早停。")
                break 

    # 保存训练曲线
    pd.DataFrame(loss_history, columns=['Epoch', 'TrainLoss', 'ValLoss']).to_csv(LOSS_HISTORY_PATH, index=False)
    print(f"训练结束。模型: {WEIGHTS_PATH}")

if __name__ == '__main__':
    train_model()