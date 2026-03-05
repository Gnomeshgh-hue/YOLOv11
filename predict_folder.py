import os
import cv2
import numpy as np
import sys
import glob
from ultralytics import YOLO

# 引入原项目核心处理函数
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
import config.kitti_config as cnf
from data_process.kitti_bev_utils import makeBVFeature, removePoints
from data_process.kitti_data_utils import Calibration

COLORS = {0: (0, 255, 255), 1: (255, 0, 0), 2: (0, 0, 255)}

def bev_to_lidar_3d(corners_pixels, cls_id):
    z_bottom = -1.73
    if cls_id == 1: h = 1.7
    elif cls_id == 2: h = 1.5
    else: h = 1.6
    z_top = z_bottom + h

    corners_3d = np.zeros((8, 3))
    for i in range(4):
        x_p, y_p = corners_pixels[i]
        x_l = (y_p / 608.0) * 50.0
        y_l = (x_p / 608.0) * 50.0 - 25.0
        corners_3d[i] = [x_l, y_l, z_bottom]
        corners_3d[i+4] = [x_l, y_l, z_top]
    return corners_3d

def draw_3d_box(img, pts_2d, color):
    pts = pts_2d.astype(int)
    for i in range(4):
        cv2.line(img, tuple(pts[i]), tuple(pts[(i + 1) % 4]), color, 2)
    for i in range(4, 8):
        cv2.line(img, tuple(pts[i]), tuple(pts[(i - 4 + 1) % 4 + 4]), color, 2)
    for i in range(4):
        cv2.line(img, tuple(pts[i]), tuple(pts[i + 4]), color, 2)
    cv2.line(img, tuple(pts[4]), tuple(pts[6]), color, 1)
    cv2.line(img, tuple(pts[5]), tuple(pts[7]), color, 1)

def main():
    # ==============================================================
    test_folder = "my_test_folder" 
    output_folder = "my_test_results"
    
    # 请确保这是你最好的权重路径
    model_path = "runs/obb/train2/weights/best.pt"
    # ==============================================================
    
    os.makedirs(output_folder, exist_ok=True)
    
    print("正在加载 YOLO 模型...")
    model = YOLO(model_path) 
    
    # 👇 【修改点】：去 velodyne 子文件夹里找 .bin 文件
    bin_files = glob.glob(os.path.join(test_folder, "velodyne", "*.bin"))
    
    if len(bin_files) == 0:
        print(f"⚠️ 在 {test_folder}/velodyne 里没有找到任何 .bin 文件！")
        return
        
    print(f"✅ 一共找到 {len(bin_files)} 帧测试数据，开始批量处理...\n")

    for bin_file in bin_files:
        base_name = os.path.splitext(os.path.basename(bin_file))[0]
        
        # 👇 【修改点】：去各自对应的子文件夹里找图片和标定矩阵
        img_file = os.path.join(test_folder, "image_2", f"{base_name}.png")
        calib_file = os.path.join(test_folder, "calib", f"{base_name}.txt")
        
        if not (os.path.exists(img_file) and os.path.exists(calib_file)):
            print(f"⚠️ 帧 {base_name} 缺少照片或 txt 文件，已跳过。")
            continue
            
        print(f"▶️ 正在处理帧: {base_name} ...")
        
        # 1. 点云预处理
        lidar_data = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
        lidar_data = removePoints(lidar_data, cnf.boundary)
        bev_map = makeBVFeature(lidar_data, cnf.DISCRETIZATION, cnf.boundary) 
        bev_img = np.transpose(bev_map, (1, 2, 0))
        bev_img = (bev_img * 255).astype(np.uint8)
        
        # 2. 模型推理
        results = model(bev_img, conf=0.25, verbose=False)[0] 
        
        # 3. 3D 投影
        rgb_img = cv2.imread(img_file)
        calib = Calibration(calib_file)
        
        if results.obb is not None:
            classes = results.obb.cls.cpu().numpy().astype(int)
            corners_all = results.obb.xyxyxyxy.cpu().numpy() 
            
            for i in range(len(classes)):
                cls_id = classes[i]
                color = COLORS.get(cls_id, (0, 255, 0))
                
                corners_3d = bev_to_lidar_3d(corners_all[i], cls_id)
                pts_rect = calib.project_velo_to_rect(corners_3d)
                if np.any(pts_rect[:, 2] < 0.1): continue
                pts_2d_rgb = calib.project_rect_to_image(pts_rect)
                draw_3d_box(rgb_img, pts_2d_rgb, color)
                
        # 4. 保存单张结果
        out_name = os.path.join(output_folder, f"{base_name}_result.jpg")
        cv2.imwrite(out_name, rgb_img)

    print(f"\n✨ 批量测试大功告成！所有带有 3D 框的图片都保存在了 {output_folder} 文件夹里！")

if __name__ == "__main__":
    main()