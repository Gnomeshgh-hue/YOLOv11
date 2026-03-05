import os
import random
import glob

def main():
    # 1. 定义路径
    root_dir = 'dataset/kitti'
    velodyne_dir = os.path.join(root_dir, 'training', 'velodyne')
    imagesets_dir = os.path.join(root_dir, 'ImageSets')
    
    # 确保 ImageSets 文件夹存在
    os.makedirs(imagesets_dir, exist_ok=True)
    
    # 2. 获取所有的雷达文件并提取编号 (例如 '000123')
    bin_files = glob.glob(os.path.join(velodyne_dir, '*.bin'))
    if len(bin_files) == 0:
        print("❌ 找不到任何 .bin 文件，请检查 dataset/kitti/training/velodyne 路径！")
        return
        
    # 提取纯数字文件名
    file_ids = [os.path.splitext(os.path.basename(f))[0] for f in bin_files]
    
    # 3. 随机打乱这些编号
    print(f"✅ 一共找到 {len(file_ids)} 个数据样本。")
    random.seed(42)  # 固定随机种子，保证每次划分结果一样
    random.shuffle(file_ids)
    
    # 4. 按 8:1:1 的比例切分
    total = len(file_ids)
    train_end = int(total * 0.8)
    val_end = int(total * 0.9)
    
    train_ids = file_ids[:train_end]
    val_ids = file_ids[train_end:val_end]
    test_ids = file_ids[val_end:]
    
    # 5. 分别写入 txt 文件
    files_to_save = {
        'train.txt': train_ids,
        'val.txt': val_ids,
        'test.txt': test_ids
    }
    
    for filename, ids in files_to_save.items():
        filepath = os.path.join(imagesets_dir, filename)
        with open(filepath, 'w') as f:
            f.write('\n'.join(ids))
        print(f"📁 成功生成 {filename}，包含 {len(ids)} 个样本。")

    print("\n✨ 数据集划分大功告成！")

if __name__ == "__main__":
    main()