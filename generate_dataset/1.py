import open3d as o3d
import numpy as np
import os
import glob

# 修改为你的一张 OBJ 路径
FILE_PATH = "D:/Cu_BEM_2/tshirt-data/tshirt_out_3/tshirt_mech00001.obj"

def diagnose():
    if not os.path.exists(FILE_PATH):
        print(f"Error: 找不到文件 {FILE_PATH}")
        return

    # 读取模型
    mesh = o3d.io.read_triangle_mesh(FILE_PATH)
    pcd = mesh.sample_points_uniformly(number_of_points=5000)
    points = np.asarray(pcd.points)

    if points.size == 0:
        print("Error: 模型中没有点数据！")
        return

    # 计算包围盒
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    size = max_bound - min_bound
    center = (min_bound + max_bound) / 2

    print("="*30)
    print("      点云尺寸诊断报告")
    print("="*30)
    print(f"坐标最小值 (Min):  X={min_bound[0]:.4f}, Y={min_bound[1]:.4f}, Z={min_bound[2]:.4f}")
    print(f"坐标最大值 (Max):  X={max_bound[0]:.4f}, Y={max_bound[1]:.4f}, Z={max_bound[2]:.4f}")
    print(f"物体中心 (Center): X={center[0]:.4f}, Y={center[1]:.4f}, Z={center[2]:.4f}")
    print("-" * 30)
    print(f"物体宽度 (X-Width):  {size[0]:.4f}")
    print(f"物体厚度 (Y-Depth):  {size[1]:.4f}")
    print(f"物体高度 (Z-Height): {size[2]:.4f}")
    print("-" * 30)
    
    # 建议半径：取宽度的 15%
    suggested_radius = size[0] * 0.15
    print(f"建议 ARM_RADIUS 设置为: {suggested_radius:.4f}")
    print("="*30)

    # 可视化 (如果在本地运行，会弹出一个窗口，按 'q' 退出)
    print("正在尝试打开预览窗口... (若无反应请手动关闭)")
    # 创建一个坐标系，用来参考大小 (红色=X, 绿色=Y, 蓝色=Z)
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size.max()*0.2, origin=min_bound)
    o3d.visualization.draw_geometries([pcd, coord_frame], window_name="Size Diagnosis")

if __name__ == "__main__":
    diagnose()