import re
import os

file_path = 'kitti_data_utils.py'

# 新的、兼容性最强的读取函数
new_function = """    def read_calib_file(self, filepath):
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if not line or ':' not in line: continue
                key, value = line.split(':', 1)
                try:
                    data[key] = np.array([float(x) for x in value.split()], dtype=np.float32)
                except ValueError:
                    pass
        
        # 1. 读取 P2 (如果是空的就给个默认零矩阵)
        if 'P2' in data:
            P2 = data['P2'].reshape(3, 4)
        else:
            P2 = np.zeros((3, 4), dtype=np.float32)

        # 2. 读取 R0_rect (自动处理 9个数 和 12个数 的情况)
        if 'R0_rect' in data:
            if data['R0_rect'].size == 9:
                R0 = data['R0_rect'].reshape(3, 3)
            elif data['R0_rect'].size == 12:
                # 如果是12个数(3x4)，我们只取前3列作为旋转矩阵(3x3)
                R0 = data['R0_rect'].reshape(3, 4)[:, :3]
            else:
                R0 = np.eye(3, dtype=np.float32)
        else:
            R0 = np.eye(3, dtype=np.float32)

        # 3. 读取 Tr_velo_to_cam
        if 'Tr_velo_to_cam' in data:
            Tr_velo = data['Tr_velo_to_cam'].reshape(3, 4)
        else:
            Tr_velo = np.zeros((3, 4), dtype=np.float32)

        # 返回字典 (注意键名必须匹配 __init__ 中的调用: R_rect, Tr_velo2cam)
        return {'P2': P2, 'R_rect': R0, 'Tr_velo2cam': Tr_velo}"""

with open(file_path, 'r') as f:
    content = f.read()

# 使用正则表达式替换整个函数块
# 匹配从 def read_calib_file 到下一个函数 def cart2hom 之间的内容
pattern = r"def read_calib_file\(self, filepath\):.*?def cart2hom"
new_content = re.sub(pattern, new_function + "\n\n    def cart2hom", content, flags=re.DOTALL)

# 如果上面的正则没匹配到（可能文件结构不同），尝试暴力替换
if new_content == content:
    print("正则匹配失败，尝试查找定位点替换...")
    # 尝试寻找上一版的特征
    start_marker = "def read_calib_file(self, filepath):"
    end_marker = "def cart2hom(self, pts_3d):"
    
    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker)
    
    if start_idx != -1 and end_idx != -1:
        # 保留缩进
        indent = "    "
        new_content = content[:start_idx] + new_function.strip() + "\n\n" + indent + content[end_idx:]
    else:
        print("错误：无法定位 read_calib_file 函数，请检查文件完整性。")
        exit(1)

with open(file_path, 'w') as f:
    f.write(new_content)

print("修复成功！已升级为支持 12位 R0_rect 的超级兼容模式。")
