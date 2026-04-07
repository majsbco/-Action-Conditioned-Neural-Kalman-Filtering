import os
import glob
import numpy as np
import open3d as o3d
import re
import sys

# ================= 配置参数 =================
OBJ_INPUT_DIR = "D:/Cu_BEM_2/tshirt-data/tshirt_out_3"
DATASET_OUTPUT_DIR = "D:/Cu_BEM_2/tshirt-data/tshirt_out_3/dataset_continuous"

# 采样参数
TARGET_POINT_COUNT = 3000 
NUM_OCCLUSIONS_PER_FRAME = 1

MIN_CLIP_RATIO = 0.2
MAX_CLIP_RATIO = 0.4

# ================= 逻辑函数 =================

def extract_index_from_filename(filename):
    """
    从 tshirt_mechXXXXX.obj 文件名中提取时间索引。
    """
    match = re.search(r'tshirt_mech(\d+)\.obj', filename)
    if match:
        return int(match.group(1))
    return -1

def get_smooth_occlusion_params(norm_t):
    """
    根据归一化时间进度 (0.0 - 1.0) 生成平滑变化的平面参数。
    """
    # 1. 法向量平滑旋转
    # 让法向量在 X-Y 平面上缓慢摆动，并在 Z 轴保持一定倾斜
    angle = norm_t * np.pi  # 随时间从 0 转到 180 度
    nx = np.cos(angle)
    ny = np.sin(angle)
    nz = 0.4  # 固定一个向下/向上的倾斜度，使遮挡看起来更自然
    normal = np.array([nx, ny, nz])
    normal /= np.linalg.norm(normal)

    # 2. 遮挡比例平滑线性增长 (0.2 -> 0.3)
    clip_ratio = MIN_CLIP_RATIO + norm_t * (MAX_CLIP_RATIO - MIN_CLIP_RATIO)

    return normal, clip_ratio

def simulate_smooth_occlusion(points, normal, clip_ratio):
    """
    根据给定的法线和比例进行连续裁剪。
    通过距离分布排序，确保遮挡边缘随法线平滑移动。
    """
    if points.shape[0] == 0:
        return points

    # 计算点在法向量方向上的投影距离
    # 我们不需要随机 plane_point，因为排序后的 clip_ratio 会自动决定裁剪面的位置
    distances = np.dot(points, normal)
    
    # 排序
    sorted_indices = np.argsort(distances)
    
    # 计算需要保留的点数
    num_points_to_keep = int(points.shape[0] * (1.0 - clip_ratio))
    
    # 保留投影值较大的部分 (遮挡物从一侧慢慢推进)
    keep_indices = sorted_indices[-num_points_to_keep:]
    occluded_points = points[keep_indices]
    
    return occluded_points

# ================= 主处理流程 =================

def generate_dataset():
    if not os.path.exists(OBJ_INPUT_DIR):
        print(f"错误: 输入文件夹不存在: {OBJ_INPUT_DIR}")
        sys.exit(1)

    if not os.path.exists(DATASET_OUTPUT_DIR):
        os.makedirs(DATASET_OUTPUT_DIR)

    # 匹配 tshirt_mech*.obj
    obj_files = glob.glob(os.path.join(OBJ_INPUT_DIR, 'tshirt_mech*.obj'))
    
    # 预处理：提取索引并排序，确保平滑插值的逻辑顺序正确
    file_info = []
    for p in obj_files:
        idx = extract_index_from_filename(os.path.basename(p))
        if idx != -1:
            file_info.append((p, idx))
    
    # 按帧序号排序
    file_info.sort(key=lambda x: x[1])

    if not file_info:
        print("未找到符合格式的文件。")
        return

    # 获取时间跨度用于归一化
    min_idx = file_info[0][1]
    max_idx = file_info[-1][1]
    index_range = max_idx - min_idx if max_idx != min_idx else 1

    print(f"开始生成连续遮挡。帧范围: {min_idx} 到 {max_idx}")
    print(f"遮挡比例: {MIN_CLIP_RATIO} -> {MAX_CLIP_RATIO}\n")

    saved_full_meshes = set()

    for obj_path, time_index in file_info:
        # 计算当前帧的归一化进度 (0.0 ~ 1.0)
        norm_t = (time_index - min_idx) / index_range
        
        # 获取该时刻的连续参数
        normal, current_ratio = get_smooth_occlusion_params(norm_t)

        try:
            # 读取网格
            mesh = o3d.io.read_triangle_mesh(obj_path)
            
            # 保存 Ground Truth 网格
            full_mesh_name = f"full_frame_{time_index}.obj"
            if full_mesh_name not in saved_full_meshes:
                o3d.io.write_triangle_mesh(os.path.join(DATASET_OUTPUT_DIR, full_mesh_name), mesh, write_ascii=True)
                saved_full_meshes.add(full_mesh_name)

            # 统一采样点云
            full_pcd = mesh.sample_points_uniformly(number_of_points=TARGET_POINT_COUNT)
            points = np.asarray(full_pcd.points)

            # 生成平滑遮挡
            occluded_points = simulate_smooth_occlusion(points, normal, current_ratio)

            # 保存遮挡 PLY
            occ_filename = f"occ_frame_{time_index}.ply"
            occ_pcd = o3d.geometry.PointCloud()
            occ_pcd.points = o3d.utility.Vector3dVector(occluded_points)
            o3d.io.write_point_cloud(os.path.join(DATASET_OUTPUT_DIR, occ_filename), occ_pcd, write_ascii=True)

            if time_index % 20 == 0:
                print(f"进度: {time_index} | 比例: {current_ratio:.3f} | 法向: {normal.round(2)}")

        except Exception as e:
            print(f"处理帧 {time_index} 出错: {e}")

    print("\n[完成] 连续平滑遮挡数据集生成完毕。")

if __name__ == "__main__":
    generate_dataset()