import cv2
import numpy as np
import os
from ultralytics import YOLO

# 定义颜色 (B, G, R) - 和你原来 kitti_config 里的一致
COLORS = {
    0: (255, 255, 0),  # Car - 青色
    1: (255, 0, 0),    # Pedestrian - 蓝色
    2: (0, 0, 255)     # Cyclist - 红色
}

def get_corners(x, y, w, l, yaw):
    """你旧代码里的角点计算函数"""
    bev_corners = np.zeros((4, 2), dtype=np.float32)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    
    # 按照原来的顺序计算 4 个角点
    bev_corners[0, 0] = x - w / 2 * cos_yaw - l / 2 * sin_yaw
    bev_corners[0, 1] = y - w / 2 * sin_yaw + l / 2 * cos_yaw
    bev_corners[1, 0] = x - w / 2 * cos_yaw + l / 2 * sin_yaw
    bev_corners[1, 1] = y - w / 2 * sin_yaw - l / 2 * cos_yaw
    bev_corners[2, 0] = x + w / 2 * cos_yaw + l / 2 * sin_yaw
    bev_corners[2, 1] = y + w / 2 * sin_yaw - l / 2 * cos_yaw
    bev_corners[3, 0] = x + w / 2 * cos_yaw - l / 2 * sin_yaw
    bev_corners[3, 1] = y + w / 2 * sin_yaw + l / 2 * cos_yaw
    return bev_corners

def draw_old_style_box(img, x, y, w, l, yaw, cls_id):
    """完美复刻你旧代码的画框逻辑"""
    color = COLORS.get(cls_id, (0, 255, 0)) # 默认绿色
    
    # 获取 4 个角点
    bev_corners = get_corners(x, y, w, l, yaw)
    corners_int = bev_corners.reshape(-1, 1, 2).astype(int)
    
    # 画矩形框的主体
    cv2.polylines(img, [corners_int], isClosed=True, color=color, thickness=2)
    
    # 画车头朝向线 (连接角点0和角点3)
    corners_flat = bev_corners.reshape(-1, 2).astype(int)
    cv2.line(img, (corners_flat[0, 0], corners_flat[0, 1]), 
                  (corners_flat[3, 0], corners_flat[3, 1]), 
                  (0, 255, 255), 2) # 用黄色画车头

def main():
    # 1. 加载你刚刚训练好的最新模型
    model = YOLO("runs/obb/train/weights/best.pt")
    
    # 2. 指定要测试的图片所在目录
    img_dir = "kitti_bev_obb/images/val"
    save_dir = "custom_results"
    os.makedirs(save_dir, exist_ok=True)
    
    # 挑选几张图来测试
    img_files = os.listdir(img_dir)[:10] 
    
    for img_file in img_files:
        img_path = os.path.join(img_dir, img_file)
        # 用 cv2 读图
        original_img = cv2.imread(img_path)
        
        # 让 YOLO11 进行推理 (不让它自己画图)
        results = model(img_path, verbose=False)
        result = results[0]
        
        # 解析 YOLO11 的 OBB 预测结果
        if result.obb is not None:
            # 提取类别、中心点、宽、长、角度
            classes = result.obb.cls.cpu().numpy().astype(int)
            xywhr = result.obb.xywhr.cpu().numpy() # [x_center, y_center, width, height, rotation]
            
            for i in range(len(classes)):
                cls_id = classes[i]
                x, y, w, h, yaw = xywhr[i]
                # 调用我们复刻的老派画法
                draw_old_style_box(original_img, x, y, w, h, yaw, cls_id)
                
        # 保存结果
        save_path = os.path.join(save_dir, img_file)
        cv2.imwrite(save_path, original_img)
        print(f"已生成复古风格测试图: {save_path}")

if __name__ == "__main__":
    main()