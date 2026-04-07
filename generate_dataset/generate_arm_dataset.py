import os
import glob
import numpy as np
import open3d as o3d
import re

# ================= Configuration =================
OBJ_INPUT_DIR = "D:/Cu_BEM_2/tshirt-data/tshirt_kalman_3"
DATASET_OUTPUT_DIR = "D:/Cu_BEM_2/tshirt-data/tshirt_kalman_3/dataset_arm"

# 采样点数
TARGET_POINT_COUNT = 5000 

# --- Arm Parameters (手臂遮挡圆柱体) ---
BASE_ARM_RADIUS = 6.0  
STEP_SMOOTHING = 0.7 
X_FREQ = 3.5  
Z_FREQ = 2.7 

# --- View Clipping Parameters (平面剪裁) ---
# 剪裁深度比例：0.5 表示从中间切开，0.3 表示保留较浅的前部，0.7 表示保留较多
# 因为衣服有厚度，通常 0.4~0.5 就能完美去掉背面并保留完整的正面
CLIPPING_RATIO = 0.45 

# ================= Helper Functions =================

def extract_index_from_filename(filename):
    match = re.search(r'tshirt_mech(\d+)\.obj', filename)
    return int(match.group(1)) if match else -1

def apply_plane_clipping(points, ratio=0.5):
    """
    简单的平面剪裁：根据 Z 轴范围，只保留靠近相机（Z 较大）的前半部分
    """
    if len(points) == 0: return points
    
    z_vals = points[:, 2]
    z_min, z_max = np.min(z_vals), np.max(z_vals)
    
    # 计算剪裁阈值（假设正前方是 Z 轴增大方向）
    # 阈值 = 最小值 + 跨度 * (1 - ratio)
    # 这样 ratio 越大，保留的越多
    threshold = z_min + (z_max - z_min) * (1 - ratio)
    
    mask = z_vals > threshold
    return points[mask]

def get_adaptive_arm_pos(points, norm_t, prev_pos=None):
    if len(points) == 0: return np.array([0, 0])
    p_min, p_max = np.min(points, axis=0), np.max(points, axis=0)
    center = np.mean(points, axis=0)
    
    width_x = p_max[0] - p_min[0]
    height_z = p_max[2] - p_min[2]

    range_x, range_z = width_x * 0.4, height_z * 0.2
    target_x = center[0] + np.sin(norm_t * np.pi * X_FREQ) * range_x
    target_z = p_max[2] - (np.abs(np.cos(norm_t * np.pi * Z_FREQ)) * range_z)
    
    noise = np.random.uniform(-3.0, 3.0, size=2)
    current_target = np.array([target_x + noise[0], target_z + noise[1]])

    if prev_pos is None: return current_target
    return prev_pos * STEP_SMOOTHING + current_target * (1 - STEP_SMOOTHING)

def apply_arm_occlusion(points, arm_pos_xz, radius):
    if len(points) == 0: return points
    dists_sq = np.sum((points[:, [0, 2]] - arm_pos_xz)**2, axis=1)
    mask = dists_sq < (radius**2)
    return points[~mask]

# ================= Main Process =================

def run_simulation():
    if not os.path.exists(DATASET_OUTPUT_DIR):
        os.makedirs(DATASET_OUTPUT_DIR)

    obj_files = sorted(glob.glob(os.path.join(OBJ_INPUT_DIR, 'tshirt_mech*.obj')), 
                       key=lambda x: extract_index_from_filename(os.path.basename(x)))

    if not obj_files:
        print("未找到文件")
        return

    last_arm_pos = None 

    for i, obj_path in enumerate(obj_files):
        time_idx = extract_index_from_filename(os.path.basename(obj_path))
        norm_t = i / (len(obj_files) - 1) if len(obj_files) > 1 else 0

        # 1. 加载并采样
        mesh = o3d.io.read_triangle_mesh(obj_path)
        full_pcd = mesh.sample_points_uniformly(number_of_points=TARGET_POINT_COUNT)
        all_points = np.asarray(full_pcd.points)
        
        # 2. 【核心修改】平面剪裁 (保留正面)
        visible_points = apply_plane_clipping(all_points, ratio=CLIPPING_RATIO)

        # 3. 动态手臂遮挡
        current_arm_pos = get_adaptive_arm_pos(visible_points, norm_t, last_arm_pos)
        last_arm_pos = current_arm_pos 

        # 执行手臂圆柱体切割
        occ_points_arr = apply_arm_occlusion(visible_points, current_arm_pos, BASE_ARM_RADIUS)

        # 4. 保存
        full_path = os.path.join(DATASET_OUTPUT_DIR, f"full_frame_{time_idx}.ply")
        o3d.io.write_point_cloud(full_path, full_pcd)

        final_occ_pcd = o3d.geometry.PointCloud()
        final_occ_pcd.points = o3d.utility.Vector3dVector(occ_points_arr)
        occ_path = os.path.join(DATASET_OUTPUT_DIR, f"occ_frame_{time_idx}.ply")
        o3d.io.write_point_cloud(occ_path, final_occ_pcd)

        if time_idx % 20 == 0:
            print(f"Frame {time_idx:04d} | 原始: {len(all_points)} | 剪裁后: {len(occ_points_arr)}")

    print(f"处理完成！使用了平面剪裁 (Ratio: {CLIPPING_RATIO})。")

if __name__ == "__main__":
    run_simulation()