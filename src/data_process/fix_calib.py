import kitti_data_utils
import inspect

# 读取原文件内容
with open('kitti_data_utils.py', 'r') as f:
    content = f.read()

# 新的健壮的读取函数
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
        
        return {'P2': data['P2'].reshape(3, 4),
                'R0': data['R0_rect'].reshape(3, 3),
                'Tr_velo2cam': data['Tr_velo_to_cam'].reshape(3, 4)}"""

# 替换旧函数 (通过简单的字符串定位，稍微暴力但有效)
# 我们定位原函数的特征部分
old_part_start = "    def read_calib_file(self, filepath):"
old_part_end = "'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)}"

if old_part_start in content:
    start_idx = content.find(old_part_start)
    # 找到函数结束的大概位置
    end_idx = content.find(old_part_end, start_idx) + len(old_part_end)
    
    # 执行替换
    new_content = content[:start_idx] + new_function + content[end_idx:]
    
    with open('kitti_data_utils.py', 'w') as f:
        f.write(new_content)
    print("修复成功！已将 read_calib_file 替换为智能读取模式。")
else:
    print("未找到目标函数，请手动修改。")
