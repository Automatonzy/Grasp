import os
import json
import pickle
import numpy as np
from PIL import Image
import cv2

def load_caption(caption_file):
    """从文本文件加载描述内容"""
    try:
        with open(caption_file, 'r') as f:
            caption = f.read().strip()
        return caption
    except Exception as e:
        print(f"无法从文件加载描述: {e}")
        return "A bottle"  # 默认描述

def load_trajectory_from_json(json_path):
    """从JSON文件中加载affordance点轨迹"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # 解析labelme格式的JSON
        if 'shapes' in data:
            # 提取所有标注为point的点
            points = []
            for shape in data['shapes']:
                if shape['shape_type'] == 'point':
                    # 每个point标注包含一个坐标点
                    points.append(tuple(shape['points'][0]))
            if points:
                return points
        
        # 兼容其他可能的格式
        elif 'trajectory' in data:
            return data['trajectory']
        elif 'keypoints' in data:
            return data['keypoints']
        else:
            # 如果找不到轨迹信息，创建一个默认点
            height, width = 480, 640
            return [(width//2, height//2)]
    except Exception as e:
        print(f"无法从JSON加载轨迹: {e}")
        return [(0, 0)]

# def create_bottle_pkl(base_dir, output_file,shape_name):
#     """
#     从bottle目录创建PKL文件
    
#     参数:
#         base_dir: bottle目录的基本路径
#         output_file: 输出PKL文件的路径
#     """
#     # 定义要搜索的子目录
#     subdirs = ['hand', 'no_hand']
    
#     # 初始化数据结构
#     data_dict = {
#         'img': [],         # 原始图像列表
#         'traj': [],        # affordance点轨迹
#         'caption': [],     # 图像描述
#         'image_size': [],  # 图像尺寸
#         'name': f'{shape_name}'   # 对象名称
#     }
    
#     # 遍历每个子目录
#     for subdir in subdirs:
#         dir_path = os.path.join(base_dir, subdir)
#         if not os.path.exists(dir_path):
#             print(f"警告: 目录 {dir_path} 不存在")
#             continue
        
#         # 指定目标文件路径
#         img_path = os.path.join(dir_path, f'rgb_{shape_name}.png')
#         json_path = os.path.join(dir_path, f'rgb_{shape_name}.json')
#         caption_path = os.path.join(dir_path, f'{shape_name}_caption.txt')
#         print(f"所有路径: {img_path}\n{json_path}\n{caption_path}")
#         # 检查并读取图像
#         if os.path.exists(img_path):
#             try:
#                 img = np.array(Image.open(img_path))
#                 data_dict['img'].append({subdir:img})
                
#                 # 记录图像尺寸
#                 height, width = img.shape[:2]
#                 data_dict['image_size'].append({subdir:(height, width)})
                
#                 # 读取affordance点轨迹
#                 if os.path.exists(json_path):
#                     traj = load_trajectory_from_json(json_path)
#                     data_dict['traj'].append(traj)
#                     print(f"从JSON加载了 {len(traj)} 个标注点")
#                 else:
#                     # 没有JSON文件时创建默认轨迹
#                     traj = [(width//2, height//2)]
#                     data_dict['traj'].append(traj)
#                     print(f"未找到JSON文件，使用默认轨迹点")
                
#                 # 读取描述文本
#                 if os.path.exists(caption_path) and subdir=="hand":
#                     caption = load_caption(caption_path)
#                     data_dict['caption'].append(caption)
#                     print(f"加载描述: {caption}")
#                 else:
#                     data_dict['caption'].append("A bottle")
#                     print("未找到描述文件，使用默认描述")
                
#                 print(f"已添加图像: {img_path}, 尺寸: {width}x{height}")
#             except Exception as e:
#                 print(f"处理图像时出错 {img_path}: {e}")
    
#     # 确保输出目录存在
#     os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
#     # 保存为PKL文件
#     with open(output_file, 'wb') as f:
#         pickle.dump(data_dict, f)
    
#     print(f"\nPKL文件已保存到: {output_file}")
#     print(f"包含 {len(data_dict['img'])} 张图像和对应轨迹")
    
#     return data_dict
def create_bottle_pkl(base_dir, output_file, shape_name):
    """
    从bottle目录创建PKL文件，组织同一组的hand和no_hand图像
    
    参数:
        base_dir: bottle目录的基本路径
        output_file: 输出PKL文件的路径
        shape_name: 对象形状名称
    """
    # 初始化数据结构
    data_dict = {
        'img': [],         # 原始图像组列表
        'traj': [],        # affordance点轨迹
        'caption': [],     # 图像描述
        'image_size': [],  # 图像尺寸
        'direction': [],    # 图像方向
        'name': f'{shape_name}'   # 对象名称
    }
    
    # 为每个数据条目创建一个组合字典
    img_group = {}
    size_group = {}
    trajs = []
    caption = None
    
    # 处理每个子目录
    subdirs = ['hand', 'no_hand']
    for subdir in subdirs:
        dir_path = os.path.join(base_dir, subdir)
        if not os.path.exists(dir_path):
            print(f"警告: 目录 {dir_path} 不存在")
            continue
        
        # 指定目标文件路径
        img_path = os.path.join(dir_path, f'rgb_{shape_name}.png')
        json_path = os.path.join(dir_path, f'rgb_{shape_name}.json')
        direction_path = os.path.join(dir_path, f'direction.json')
        caption_path = os.path.join(dir_path, f'{shape_name}_caption.txt')
        
        # 检查并读取图像
        if os.path.exists(img_path):
            try:
                # 读取并添加图像到当前组
                img = np.array(Image.open(img_path))
                img_group[subdir] = img
                
                # 记录图像尺寸
                height, width = img.shape[:2]
                size_group[subdir] = (height, width)
                
                # 读取affordance点轨迹
                if os.path.exists(json_path):
                    traj = load_trajectory_from_json(json_path)
                    if subdir == 'no_hand':  # 只保存no_hand版本的轨迹
                        trajs = traj
                    print(f"从JSON加载了 {len(traj)} 个标注点")
                else:
                    if subdir == 'no_hand':  # 只为no_hand创建默认轨迹
                        trajs = [(width//2, height//2)]
                    print(f"未找到JSON文件，使用默认轨迹点")
                
                # 读取方向信息
                if os.path.exists(direction_path):
                    if subdir == 'hand':
                        with open(direction_path, 'r') as f:
                            direction = json.load(f)
                        data_dict['direction'].append(direction['direction_vector'])
                        print(f"加载方向信息: {direction}")
                else:
                    if subdir == 'hand':
                        data_dict['direction'].append(None)
                        print("未找到方向文件，使用默认值")
                
                # 读取描述文本 (只读取一次)
                if caption is None and os.path.exists(caption_path):
                    caption = load_caption(caption_path)
                    print(f"加载描述: {caption}")
                
                print(f"已添加图像: {img_path}, 尺寸: {width}x{height}")
            except Exception as e:
                print(f"处理图像时出错 {img_path}: {e}")
    
    # 将完整的图像组添加到数据字典
    if img_group:
        data_dict['img'].append(img_group)
        data_dict['image_size'].append(size_group)
        
        # 添加轨迹和描述
        if trajs:
            data_dict['traj'].append(trajs)
        else:
            # 如果没有找到轨迹，添加默认轨迹
            data_dict['traj'].append([(320, 240)])  # 默认中心点
        
        # 添加方向
        # if direction:
        #     data_dict['direction'].append(direction)
        # else:
        #     data_dict['direction'].append(None)
            
        if caption:
            data_dict['caption'].append(caption)
        else:
            data_dict['caption'].append(f"A {shape_name}")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 保存为PKL文件
    with open(output_file, 'wb') as f:
        pickle.dump(data_dict, f)
    
    print(f"\nPKL文件已保存到: {output_file}")
    print(f"包含 {len(data_dict['img'])} 组图像和对应轨迹")
    
    return data_dict

if __name__ == "__main__":
    # 设置路径
    shape_name = "bottle"  # 形状名称
    cur_dir = "/home/user/project/xuhy/RAM/RAM_code_copy/assets/data/new_data/"
    output_dir = os.path.join(cur_dir, shape_name)
    output_file = os.path.join(output_dir, f"{shape_name}_new.pkl")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建PKL文件
    data = create_bottle_pkl(output_dir, output_file,shape_name)
    
    # 打印统计信息
    print("\n数据统计:")
    print(f"图像数量: {len(data['img'])}")
    if len(data['img']) > 0:
        print(f"第一张图像形状: {data['img'][0]['hand'].shape}")
    print(f"轨迹数量: {len(data['traj'])}")
    if len(data['traj']) > 0:
        print(f"第一个轨迹包含 {len(data['traj'][0])} 个点")
    print(f"方向信息数量: {len(data['direction'])}")
    if len(data['direction']) > 0:
        print(f"第一个方向信息: {data['direction'][0]}")
    if 'caption' in data and len(data['caption']) > 0:
        print(f"第一个描述: {data['caption'][0]}")
    print(data)