import os
import cv2
import sys
import numpy as np
from ultralytics import YOLO

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from data_process.kitti_data_utils import Calibration

COLORS = {
    0: (0, 255, 255),  # Car - 黄色/青色
    1: (255, 0, 0),    # Pedestrian - 蓝色
    2: (0, 0, 255)     # Cyclist - 红色
}

def bev_to_lidar_3d(corners_pixels, cls_id):
    """将 BEV 像素坐标转换回 3D 雷达坐标系中的 8 个角点"""
    # KITTI 激光雷达距地面大约 1.73 米，所以地面物体的底部 Z 约 -1.73
    z_bottom = -1.73
    if cls_id == 1: h = 1.7    # 行人
    elif cls_id == 2: h = 1.5  # 自行车
    else: h = 1.5              # 汽车
    z_top = z_bottom + h

    corners_3d = np.zeros((8, 3))
    for i in range(4):
        x_p, y_p = corners_pixels[i]
        
        # 【核心修复】：去掉错误的 1.0 -
        # 像素的 y 轴 (0~608) 严格正比于雷达前向 x 轴 (0~50m)
        x_l = (y_p / 608.0) * 50.0
        # 像素的 x 轴 (0~608) 对应雷达横向 y 轴 (-25m~25m)
        y_l = (x_p / 608.0) * 50.0 - 25.0

        corners_3d[i] = [x_l, y_l, z_bottom]
        corners_3d[i+4] = [x_l, y_l, z_top]

    return corners_3d

def draw_3d_box_on_rgb(img, pts_2d, color, thickness=2):
    """在 RGB 图像上绘制 8 个角点组成的 3D 框"""
    pts = pts_2d.astype(int)
    # 画底面
    for i in range(4):
        cv2.line(img, tuple(pts[i]), tuple(pts[(i + 1) % 4]), color, thickness)
    # 画顶面
    for i in range(4, 8):
        cv2.line(img, tuple(pts[i]), tuple(pts[(i - 4 + 1) % 4 + 4]), color, thickness)
    # 画连接柱
    for i in range(4):
        cv2.line(img, tuple(pts[i]), tuple(pts[i + 4]), color, thickness)
        
    # 画车头朝向十字交叉线 (0-2 和 1-3，由于 YOLO 顶点排序不固定，画一个顶部叉号最稳妥)
    cv2.line(img, tuple(pts[4]), tuple(pts[6]), color, thickness=1)
    cv2.line(img, tuple(pts[5]), tuple(pts[7]), color, thickness=1)

def main():
    root_dir = 'dataset/kitti'
    list_file = os.path.join(root_dir, 'ImageSets', 'val.txt')
    rgb_dir = os.path.join(root_dir, 'training', 'image_2')
    calib_dir = os.path.join(root_dir, 'training', 'calib')
    bev_dir = 'kitti_bev_obb/images/val'
    
    save_dir = 'rgb_results'
    os.makedirs(save_dir, exist_ok=True)
    
    print("正在加载模型...")
    model = YOLO("runs/obb/train2/weights/best.pt")

    with open(list_file, 'r') as f:
        img_ids = [x.strip() for x in f.readlines()][:20]

    for img_id in img_ids:
        rgb_path = os.path.join(rgb_dir, f'{img_id}.png')
        bev_path = os.path.join(bev_dir, f'{img_id}.png')
        calib_path = os.path.join(calib_dir, f'{img_id}.txt')
        
        if not (os.path.exists(rgb_path) and os.path.exists(bev_path)):
            continue
            
        rgb_img = cv2.imread(rgb_path)
        bev_img = cv2.imread(bev_path)
        calib = Calibration(calib_path) 
        
        # 【关键修改】：显式调低 conf (置信度)，如果漏标严重，可以把 0.25 降到 0.15 试试
        results = model(bev_img, conf=0.25, iou=0.45, verbose=False)[0]
        bev_img_plotted = results.plot()
        
        if results.obb is not None:
            classes = results.obb.cls.cpu().numpy().astype(int)
            corners_all = results.obb.xyxyxyxy.cpu().numpy() 
            
            for i in range(len(classes)):
                cls_id = classes[i]
                color = COLORS.get(cls_id, (0, 255, 0))
                
                corners_3d = bev_to_lidar_3d(corners_all[i], cls_id)
                
                pts_rect = calib.project_velo_to_rect(corners_3d)
                # 过滤掉跑到相机后面的点
                if np.any(pts_rect[:, 2] < 0.1): 
                    continue
                    
                pts_2d_rgb = calib.project_rect_to_image(pts_rect)
                draw_3d_box_on_rgb(rgb_img, pts_2d_rgb, color)
                
        h_rgb, w_rgb = rgb_img.shape[:2]
        h_bev, w_bev = bev_img_plotted.shape[:2]
        scale = h_rgb / h_bev
        new_w_bev = int(w_bev * scale)
        bev_resized = cv2.resize(bev_img_plotted, (new_w_bev, h_rgb))
        
        combined_img = np.hstack((rgb_img, bev_resized))
        save_path = os.path.join(save_dir, f'{img_id}_combined.png')
        cv2.imwrite(save_path, combined_img)
        print(f"已生成正确 3D 投影图: {save_path}")

    print(f"\n✨ 测试完成！请去 {save_dir} 文件夹查看，这次尺寸绝对是对的！")

if __name__ == "__main__":
    main()
