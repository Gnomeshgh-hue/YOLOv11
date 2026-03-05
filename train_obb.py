from ultralytics import YOLO

def main():
    # 1. 加载官方最新的 YOLO11 旋转框 (OBB) 预训练模型
    # n代表nano(最小最快)，你也可以换成 yolo11s-obb.pt(small), yolo11m-obb.pt(medium)
    model = YOLO("yolo11n-obb.pt")

    # 2. 开始训练！
    results = model.train(
        data="kitti.yaml",      # 刚才建的配置文件
        epochs=100,             # 训练轮数 (可以先设100看看效果)
        imgsz=608,              # 图像尺寸，与你原先的 BEV_WIDTH 一致
        device="0,1,2,3,4,5,6,7", # 👈 这里的魔法：直接调用你的 8 张 4090 并行训练！
        batch=128,              # 批次大小。8张4090显存巨大，设128或256都可以跑飞起
        workers=32,             # 多线程加载数据
        optimizer="AdamW",      # 推荐使用 AdamW 优化器
        lr0=0.001,              # 初始学习率
        project="Complex-YOLO-OBB", # 训练结果保存的文件夹名
        name="train_exp_1"      # 本次实验的名称
    )

if __name__ == '__main__':
    # 在 Windows 或多卡训练时，必须把训练代码放在 if __name__ == '__main__': 下面
    main()