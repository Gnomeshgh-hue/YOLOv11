import os
import shutil

def main():
    # 1. 定义源路径和名单路径
    root_dir = 'dataset/kitti/training'
    test_txt_path = 'dataset/kitti/ImageSets/test.txt'
    
    # 2. 定义你要存放测试数据的目标文件夹
    target_dir = 'my_test_folder'
    sub_dirs = ['velodyne', 'image_2', 'calib', 'label_2']
    
    # 检查 test.txt 是否存在
    if not os.path.exists(test_txt_path):
        print("❌ 找不到 test.txt，请先运行 split_dataset.py！")
        return
        
    # 读取名单
    with open(test_txt_path, 'r') as f:
        test_ids = [line.strip() for line in f.readlines()]
        
    print(f"✅ 成功读取 test.txt，共有 {len(test_ids)} 个测试样本。")
    print(f"🚀 正在将它们提取到 {target_dir} 文件夹中...\n")
    
    # 创建目标子文件夹
    for sub in sub_dirs:
        os.makedirs(os.path.join(target_dir, sub), exist_ok=True)
        
    # 3. 按名单捞人（复制文件）
    success_count = 0
    for file_id in test_ids:
        # 定义需要复制的文件对应关系
        files_to_copy = [
            (f"velodyne/{file_id}.bin", f"velodyne/{file_id}.bin"),
            (f"image_2/{file_id}.png", f"image_2/{file_id}.png"),
            (f"calib/{file_id}.txt", f"calib/{file_id}.txt"),
            (f"label_2/{file_id}.txt", f"label_2/{file_id}.txt")
        ]
        
        all_exist = True
        for src_rel, dst_rel in files_to_copy:
            src_path = os.path.join(root_dir, src_rel)
            dst_path = os.path.join(target_dir, dst_rel)
            
            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)
            else:
                all_exist = False
                
        if all_exist:
            success_count += 1
            
        if success_count % 50 == 0 and success_count > 0:
            print(f"已提取 {success_count} / {len(test_ids)} 个样本...")

    print(f"\n✨ 提取完成！成功将 {success_count} 套测试数据放入 {target_dir}！")
    print("现在你的 my_test_folder 就是一个完美的、纯净的、模型绝对没见过的考场了！")

if __name__ == "__main__":
    main()