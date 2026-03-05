import os
import cv2
import numpy as np
import sys

# 将原项目的 src 目录加入环境变量
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import config.kitti_config as cnf
from data_process.kitti_bev_utils import makeBVFeature, removePoints, build_yolo_target, get_corners
from data_process.kitti_data_utils import Calibration
from data_process import transformation

def read_kitti_labels(label_path):
    """读取原始 KITTI 标签，过滤掉 DontCare"""
    objects = []
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split(' ')
            cls_name = parts[0]
            if cls_name == 'DontCare':
                continue
            h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
            x, y, z = float(parts[11]), float(parts[12]), float(parts[13])
            ry = float(parts[14])
            objects.append({'cls': cls_name, 'x': x, 'y': y, 'z': z, 'l': l, 'w': w, 'h': h, 'ry': ry})
    return objects

def process_dataset(split='train'):
    print(f"========== 正在处理 {split} 集 ==========")
    root_dir = 'dataset/kitti' 
    list_file = os.path.join(root_dir, 'ImageSets', f'{split}.txt')
    lidar_dir = os.path.join(root_dir, 'training', 'velodyne')
    label_dir = os.path.join(root_dir, 'training', 'label_2')
    calib_dir = os.path.join(root_dir, 'training', 'calib') # 新增标定文件路径
    
    out_img_dir = os.path.join('kitti_bev_obb', 'images', split)
    out_label_dir = os.path.join('kitti_bev_obb', 'labels', split)
    
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)

    with open(list_file, 'r') as f:
        img_ids = [x.strip() for x in f.readlines()]

    for i, img_id in enumerate(img_ids):
        lidar_path = os.path.join(lidar_dir, f'{img_id}.bin')
        calib_path = os.path.join(calib_dir, f'{img_id}.txt')
        if not os.path.exists(lidar_path) or not os.path.exists(calib_path):
            continue
            
        lidar_data = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
        
        # 造图
        lidar_data = removePoints(lidar_data, cnf.boundary)
        bev_map = makeBVFeature(lidar_data, cnf.DISCRETIZATION, cnf.boundary) 
        bev_img = np.transpose(bev_map, (1, 2, 0))
        bev_img = (bev_img * 255).astype(np.uint8)
        
        # 读取标签和标定矩阵
        calib = Calibration(calib_path)
        label_path = os.path.join(label_dir, f'{img_id}.txt')
        objects = read_kitti_labels(label_path)
        
        labels = []
        for obj in objects:
            if obj['cls'] in cnf.CLASS_NAME_TO_ID:
                cl = cnf.CLASS_NAME_TO_ID[obj['cls']]
                labels.append([cl, obj['x'], obj['y'], obj['z'], obj['h'], obj['w'], obj['l'], obj['ry']])
        
        obb_lines = []
        if len(labels) > 0:
            labels_np = np.array(labels, dtype=np.float32)
            
            # 【！！！！致命错误修复点！！！！】
            # 必须用标定矩阵把 3D 框从相机坐标系转换到雷达坐标系，然后再传给 YOLO！
            labels_np[:, 1:] = transformation.camera_to_lidar_box(labels_np[:, 1:], calib.V2C, calib.R0, calib.P)
            
            targets = build_yolo_target(labels_np)
            
            for t in targets:
                cl = int(t[0])
                x_img = t[1] * cnf.BEV_WIDTH
                y_img = t[2] * cnf.BEV_HEIGHT
                w_img = t[3] * cnf.BEV_WIDTH
                l_img = t[4] * cnf.BEV_HEIGHT
                yaw = np.arctan2(t[5], t[6])
                
                corners = get_corners(x_img, y_img, w_img, l_img, yaw)
                corners[:, 0] = np.clip(corners[:, 0] / cnf.BEV_WIDTH, 0.0, 1.0)
                corners[:, 1] = np.clip(corners[:, 1] / cnf.BEV_HEIGHT, 0.0, 1.0)
                
                flat_corners = corners.flatten()
                line = f"{cl} " + " ".join([f"{coord:.6f}" for coord in flat_corners])
                obb_lines.append(line)
        
        # 保存图片和 txt 标签
        cv2.imwrite(os.path.join(out_img_dir, f'{img_id}.png'), bev_img)
        with open(os.path.join(out_label_dir, f'{img_id}.txt'), 'w') as f:
            f.write("\n".join(obb_lines))
            
        if (i + 1) % 500 == 0:
            print(f"已处理进度: {i + 1} / {len(img_ids)}")

if __name__ == '__main__':
    process_dataset('train')
    process_dataset('val')
    print("✨ 全部转换完成！可以开始 YOLO11 的训练了！")