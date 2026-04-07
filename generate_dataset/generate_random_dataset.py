import os
import glob
import numpy as np
import open3d as o3d
import re
import sys

OBJ_INPUT_DIR = "D:/Cu_BEM_2/tshirt-data/tshirt_out_3"
DATASET_OUTPUT_DIR = "D:/Cu_BEM_2/tshirt-data/tshirt_out_3/dataset_random"

# 采样参数
TARGET_POINT_COUNT = 3000  # 用于生成遮挡点云的基础点数
NUM_OCCLUSIONS_PER_FRAME = 1 # 为每帧生成多少种不同的遮挡版本


def simulate_plane_occlusion(point_cloud_data: np.ndarray, clip_ratio_range=(0.2, 0.5)):
    """
    通过随机平面裁剪来模拟遮挡效果。
    """
    points = point_cloud_data.copy()
    
    if points.shape[0] == 0:
        return points

    # 1. 确定裁剪比例
    clip_ratio = np.random.uniform(clip_ratio_range[0], clip_ratio_range[1])

    # 2. 随机生成一个裁剪平面
    normal = np.random.rand(3) - 0.5
    normal /= np.linalg.norm(normal)
    
    # 随机中心点 (Point on Plane)
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    plane_point = np.random.uniform(min_coords, max_coords)

    # 3. 计算所有点到平面的距离
    distances = np.dot(points - plane_point, normal)
    
    # 4. 根据裁剪比例确定裁剪阈值
    sorted_indices = np.argsort(distances)
    num_points_to_keep = int(points.shape[0] * (1.0 - clip_ratio))
    
    # 5. 应用裁剪
    keep_indices = sorted_indices[-num_points_to_keep:]
    occluded_points = points[keep_indices]
    
    return occluded_points

def extract_index_from_filename(filename):
    """
    从 tshirt_mechXXXXX.obj 文件名中提取时间索引 (例如 tshirt_mech00001.obj -> 1)。
    """
    # 匹配 'tshirt_mech' 后面的数字串，并允许数字前有零
    match = re.search(r'tshirt_mech(\d+)\.obj', filename)
    if match:
        # 转换为整数，int() 会自动处理前导零
        return int(match.group(1))
    return -1

# =======================================================================
# 批量处理与保存逻辑
# =======================================================================

def generate_dataset():
    """主逻辑：遍历 OBJ，采样，生成遮挡点云，并批量保存到 PLY 文件。"""
    
    # --- 路径检查 ---
    if not os.path.exists(OBJ_INPUT_DIR):
        print(f"错误: OBJ 输入文件夹不存在: {OBJ_INPUT_DIR}")
        sys.exit(1)

    if not os.path.exists(DATASET_OUTPUT_DIR):
        try:
            os.makedirs(DATASET_OUTPUT_DIR)
            print(f"已创建输出文件夹: {DATASET_OUTPUT_DIR}")
        except Exception as e:
            print(f"严重错误: 无法创建输出文件夹 {DATASET_OUTPUT_DIR}。请检查权限。错误: {e}")
            sys.exit(1)

    # --- 文件匹配检查：匹配所有 'tshirt_mech*.obj' 文件 ---
    obj_files = glob.glob(os.path.join(OBJ_INPUT_DIR, 'tshirt_mech*.obj')) 
    
    if not obj_files:
        print(f"未在 {OBJ_INPUT_DIR} 找到任何 'tshirt_mech*.obj' 文件。请检查文件命名。")
        return

    print(f"找到 {len(obj_files)} 个 OBJ 文件。目标采样点数: {TARGET_POINT_COUNT}，每帧遮挡数: {NUM_OCCLUSIONS_PER_FRAME}\n")

    # 用于记录已保存的完整网格文件名
    saved_full_meshes = set()
    total_processed = 0

    for obj_path in obj_files:
        file_name = os.path.basename(obj_path)
        time_index = extract_index_from_filename(file_name)

        if time_index == -1:
             print(f"警告: 文件名 {file_name} 不符合 'tshirt_mechXXXXX.obj' 格式，跳过。")
             continue

        print(f"--- 正在处理文件: {file_name} (Index: {time_index}) ---")

        try:
            # 1. 读取网格 (包含顶点和面信息)
            mesh = o3d.io.read_triangle_mesh(obj_path)
            if not mesh.has_vertices() or not mesh.has_triangles():
                print(f"警告: 文件 {file_name} 不是有效的网格或为空，跳过。")
                continue

            # 2. 实时保存完整网格（作为 Mesh Ground Truth）
            # !!! 关键修改: 使用索引 time_index 命名为 full_frame_X.obj，确保与遮挡点云文件名可匹配。
            full_mesh_filename = f"full_frame_{time_index}.obj"
            
            if full_mesh_filename not in saved_full_meshes:
                full_mesh_path = os.path.join(DATASET_OUTPUT_DIR, full_mesh_filename)
                
                # 使用 write_triangle_mesh 保存原始 mesh 对象，保留面信息
                o3d.io.write_triangle_mesh(full_mesh_path, mesh, write_ascii=True)
                saved_full_meshes.add(full_mesh_filename)
                print(f"   [保存成功] 完整网格 (带面信息) 到 {full_mesh_filename}")


            # 3. 统一采样点云 (用于生成 occluded input data)
            full_pcd_o3d = mesh.sample_points_uniformly(number_of_points=TARGET_POINT_COUNT)
            full_pcd_data = np.asarray(full_pcd_o3d.points)
            
            if full_pcd_data.shape[0] < TARGET_POINT_COUNT * 0.9:
                 print(f"警告: 实际采样点数过少 ({full_pcd_data.shape[0]}/{TARGET_POINT_COUNT})，可能网格有缺陷，跳过。")
                 continue
            
            print(f"   [成功] 采样 {full_pcd_data.shape[0]} 个点作为遮挡基础。")


            # 4. 生成并保存多种遮挡点云
            for i in range(NUM_OCCLUSIONS_PER_FRAME):
                occluded_pcd_data = simulate_plane_occlusion(full_pcd_data)
                
                # 遮挡点云输入文件的文件名，例如 occ_frame_1.ply
                occluded_pcl_filename = f"occ_frame_{time_index}.ply" 
                occluded_pcl_path = os.path.join(DATASET_OUTPUT_DIR, occluded_pcl_filename)
                
                # 创建 Open3D 点云对象
                occluded_o3d_to_save = o3d.geometry.PointCloud()
                occluded_o3d_to_save.points = o3d.utility.Vector3dVector(occluded_pcd_data)
                
                # 写入 PLY 文件 (作为点云)
                o3d.io.write_point_cloud(occluded_pcl_path, occluded_o3d_to_save, write_ascii=True)
                
                total_processed += 1
                
                # 实时反馈处理进度
                if total_processed % 100 == 0:
                     print(f"   已保存 {total_processed} 个遮挡 PLY 文件。")
            
            print(f"   该帧处理完成，共生成 {NUM_OCCLUSIONS_PER_FRAME} 个遮挡实例。")


        except Exception as e:
            # 打印出具体的错误信息
            print(f"\n！！！严重错误！！！\n处理文件 {file_name} 时发生错误。具体错误信息：{e}")
            print("请检查网格文件是否损坏或Open3D是否正确安装。")
            continue
    
    # 5. 最终总结
    total_files_generated = len(saved_full_meshes) + total_processed
    
    print("\n=======================================================")
    print(f"【处理完成】成功生成 {total_processed} 个遮挡点云文件 (occ_frame_X.ply) 和 {len(saved_full_meshes)} 个完整网格文件 (full_frame_X.obj)。")
    print(f"总文件数: {total_files_generated}")
    print(f"所有文件已保存至: {DATASET_OUTPUT_DIR}")
    print("=======================================================")

if __name__ == "__main__":
    generate_dataset()