# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
import open3d as o3d
import re
import sys

# === 配置路径 ===
OBJ_INPUT_DIR = "D:/Cu_BEM_2/tshirt-data/tshirt_out_3"
DATASET_OUTPUT_DIR = "D:/Cu_BEM_2/tshirt-data/tshirt_out_3/dataset_mobile"
TARGET_POINT_COUNT = 3000

# === 遮挡控制参数 ===
# 1. 平均遮挡比例 (0.0 ~ 1.0)
MEAN_OCCLUSION_RATIO = 0.3

# 2. 正态分布的抖动程度 (Sigma)
# 设置为 0.1 表示遮挡位置会在目标比例附近小幅波动
SIGMA_RATIO = 0.1 

def get_random_unit_vector():
    """生成球面均匀分布的随机单位法向量"""
    phi = np.random.uniform(0, 2 * np.pi)
    costheta = np.random.uniform(-1, 1)
    theta = np.arccos(costheta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.array([x, y, z])

def apply_custom_normal_occlusion(points, mean_ratio, sigma_ratio):
    """
    结合目标比例和正态分布的遮挡算法
    """
    if points.shape[0] == 0:
        return points

    # 1. 随机生成平面的法向量 (方向全随机)
    normal = get_random_unit_vector()
    
    # 2. 计算物体在该方向上的投影范围
    projections = np.dot(points, normal)
    proj_min, proj_max = np.min(projections), np.max(projections)
    thickness = proj_max - proj_min
    
    # 3. 计算基于目标比例的基础切割深度
    # mean_ratio = 0 对应 proj_min (不遮挡), 1 对应 proj_max (全遮挡)
    base_threshold = proj_min + (thickness * mean_ratio)
    
    # 4. 在基础深度上施加正态分布抖动
    # 标准差由物体在该方向的厚度决定
    jitter = np.random.normal(loc=0.0, scale=thickness * sigma_ratio)
    final_threshold = base_threshold + jitter
    
    # 5. 执行裁剪 (保留 threshold 以上的点)
    keep_indices = np.where(projections >= final_threshold)[0]
    
    # 鲁棒性检查：防止切空或切得太多导致训练无法进行
    # 如果剩余点少于 10% 或多于 95%，则回归到基础比例切割
    if len(keep_indices) < TARGET_POINT_COUNT * 0.1 or len(keep_indices) > TARGET_POINT_COUNT * 0.95:
        keep_indices = np.where(projections >= base_threshold)[0]
        
    return points[keep_indices]

def extract_index_from_filename(filename):
    """
    匹配 'tshirt_mech' 后面的数字串 (例如 tshirt_mech00001.obj -> 1)
    """
    match = re.search(r'tshirt_mech(\d+)\.obj', filename)
    if match:
        return int(match.group(1))
    return -1

def generate_dataset():
    """主逻辑：遍历 OBJ，采样，生成正态分布遮挡点云"""
    
    if not os.path.exists(OBJ_INPUT_DIR):
        print(f"错误: 输入目录 {OBJ_INPUT_DIR} 不存在")
        sys.exit(1)

    os.makedirs(DATASET_OUTPUT_DIR, exist_ok=True)

    # 修改匹配规则为最新的 tshirt_mech*.obj
    obj_files = glob.glob(os.path.join(OBJ_INPUT_DIR, 'tshirt_mech*.obj'))
    
    # 提取索引并过滤无效文件
    file_indices = [(p, extract_index_from_filename(os.path.basename(p))) for p in obj_files]
    sorted_files = sorted([f for f in file_indices if f[1] != -1], key=lambda x: x[1])

    if not sorted_files:
        print(f"未能在 {OBJ_INPUT_DIR} 中找到匹配 'tshirt_mech*.obj' 的文件。")
        return

    saved_full_meshes = set()
    total_processed = 0

    print(f"开始处理数据集...")
    print(f"配置设定: 平均遮挡比例={MEAN_OCCLUSION_RATIO}, 随机波动幅度={SIGMA_RATIO}")

    for obj_path, time_index in sorted_files:
        try:
            # 1. 读取网格
            mesh = o3d.io.read_triangle_mesh(obj_path)
            if not mesh.has_vertices(): 
                print(f"警告: 跳过空网格 {obj_path}")
                continue

            # 2. 保存 Ground Truth (完整物体，带面信息)
            full_name = f"full_frame_{time_index}.obj"
            if full_name not in saved_full_meshes:
                o3d.io.write_triangle_mesh(os.path.join(DATASET_OUTPUT_DIR, full_name), mesh, write_ascii=True)
                saved_full_meshes.add(full_name)

            # 3. 均匀采样点云作为基础
            full_pcd_o3d = mesh.sample_points_uniformly(number_of_points=TARGET_POINT_COUNT)
            full_pcd_data = np.asarray(full_pcd_o3d.points)

            # 4. 执行正态分布遮挡算法
            occluded_data = apply_custom_normal_occlusion(
                full_pcd_data, 
                MEAN_OCCLUSION_RATIO,
                SIGMA_RATIO
            )
            
            # 5. 保存遮挡后的点云为 PLY
            out_ply_name = f"occ_frame_{time_index}.ply"
            pcd_save = o3d.geometry.PointCloud()
            pcd_save.points = o3d.utility.Vector3dVector(occluded_data)
            o3d.io.write_point_cloud(os.path.join(DATASET_OUTPUT_DIR, out_ply_name), pcd_save, write_ascii=True)
            
            total_processed += 1
            if total_processed % 50 == 0:
                print(f"已成功处理并保存: {total_processed} 帧...")

        except Exception as e:
            print(f"处理文件 {obj_path} 时失败: {e}")

    print("\n=======================================================")
    print(f"任务完成！")
    print(f"成功生成 {total_processed} 个遮挡点云 (occ_frame_X.ply)")
    print(f"成功保存 {len(saved_full_meshes)} 个完整网格 (full_frame_X.obj)")
    print(f"所有文件位于: {DATASET_OUTPUT_DIR}")
    print("=======================================================")

if __name__ == "__main__":
    generate_dataset()