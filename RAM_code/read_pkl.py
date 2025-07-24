import pickle
import os
import argparse
import numpy as np
try:
    import matplotlib.pyplot as plt
    from PIL import Image
    import cv2
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False
    print("警告: 未找到可视化库 (matplotlib, PIL 或 cv2)，图像显示功能将被禁用")

def analyze_numpy_array(arr, name):
    """分析numpy数组的基本信息"""
    info = {
        "类型": type(arr),
        "形状": arr.shape if hasattr(arr, 'shape') else "无形状信息",
        "数据类型": arr.dtype if hasattr(arr, 'dtype') else "无类型信息",
        "最小值": np.min(arr) if hasattr(arr, 'min') else "无法计算",
        "最大值": np.max(arr) if hasattr(arr, 'max') else "无法计算",
        "平均值": np.mean(arr) if hasattr(arr, 'mean') else "无法计算"
    }
    print(f"\n{name} 分析:")
    for k, v in info.items():
        print(f"- {k}: {v}")
    return info

def read_pickle_file(file_path, show_images=False, save_images=False, output_dir="./output_images"):
    """
    读取.pkl文件并打印其内容
    
    参数:
        file_path: .pkl文件的路径
        show_images: 是否显示图像数据
        save_images: 是否保存图像数据
        output_dir: 图像保存目录
    """
    try:
        # 以二进制读取模式打开文件
        with open(file_path, 'rb') as f:
            # 使用pickle.load函数加载文件内容
            data = pickle.load(f)
            
        print("文件读取成功!")
        print("\n数据类型:", type(data))
        
        # 根据数据类型打印不同的信息
        if isinstance(data, dict):
            print("\n字典的键:")
            for key in data.keys():
                value = data[key]
                print(f"- {key} (类型: {type(value)})")
                
                # 分析值的类型和内容
                if isinstance(value, np.ndarray):
                    analyze_numpy_array(value, key)
                    
                    # 处理图像数据
                    if key in ['img', 'masked_img', 'mask'] and HAS_VISUALIZATION:
                        if show_images or save_images:
                            # 确保输出目录存在
                            if save_images and not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            
                            # 尝试显示/保存图像
                            try:
                                # 根据维度决定如何处理
                                if len(value.shape) == 3:  # 彩色图或带单通道的图像
                                    if value.shape[2] == 3 or value.shape[2] == 4:  # RGB或RGBA
                                        img_to_show = value
                                    elif value.shape[2] == 1:  # 单通道但有额外维度
                                        img_to_show = value[:, :, 0]
                                    else:
                                        print(f"警告: {key}的通道数({value.shape[2]})无法识别")
                                        continue
                                elif len(value.shape) == 2:  # 灰度图
                                    img_to_show = value
                                else:
                                    print(f"警告: {key}的维度({len(value.shape)})无法作为图像显示")
                                    continue
                                
                                # 显示图像
                                if show_images:
                                    plt.figure(figsize=(10, 8))
                                    if len(img_to_show.shape) == 2 or img_to_show.shape[2] == 1:
                                        plt.imshow(img_to_show, cmap='gray')
                                    else:
                                        plt.imshow(img_to_show)
                                    plt.title(f"{key} - Shape: {value.shape}")
                                    plt.axis('on')
                                    plt.show()
                                
                                # 保存图像
                                if save_images:
                                    output_path = os.path.join(output_dir, f"{key}.png")
                                    if len(img_to_show.shape) == 2:  # 灰度图
                                        cv2.imwrite(output_path, img_to_show)
                                    else:  # 彩色图
                                        cv2.imwrite(output_path, cv2.cvtColor(img_to_show, cv2.COLOR_RGB2BGR))
                                    print(f"已保存图像到: {output_path}")
                            except Exception as e:
                                print(f"显示/保存图像 {key} 时出错: {e}")
                
                # 如果是列表或字典，显示其长度和第一个元素的信息
                elif isinstance(value, list):
                    print(f"  列表长度: {len(value)}")
                    if len(value) > 0:
                        print(f"  第一个元素类型: {type(value[0])}")
                        if isinstance(value[0], np.ndarray):
                            analyze_numpy_array(value[0], f"{key}[0]")
                
                # 如果是轨迹数据，尝试分析其结构
                if key == 'traj' and isinstance(value, list):
                    print(f"\ntraj 包含 {len(value)} 个轨迹点")
                    if len(value) > 0:
                        # 显示第一个和最后一个轨迹点
                        print(f"  第一个轨迹点: {value[0]}")
                        print(f"  最后一个轨迹点: {value[-1]}")
                
                # 如果是name，显示其值
                if key == 'name':
                    print(f"\nname: {value}")
                    
        elif isinstance(data, list):
            print(f"\n列表长度: {len(data)}")
            if len(data) > 0:
                print(f"列表第一个元素类型: {type(data[0])}")
                if isinstance(data[0], dict) and len(data[0]) > 0:
                    print("第一个元素的键:", list(data[0].keys()))
        else:
            print("\n数据内容摘要:")
            print(str(data)[:200] + "..." if len(str(data)) > 200 else data)
        
        return data
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="读取pickle (.pkl) 文件")
    parser.add_argument("--file_path", "-f", help="要读取的.pkl文件的路径", 
                       default="./assets/data/droid/open_the_drawer/open_the_drawer_new.pkl")
    parser.add_argument("--show_images", "-s", action="store_true", help="显示图像数据")
    parser.add_argument("--save_images", "-si", action="store_true", help="保存图像数据")
    parser.add_argument("--output_dir", "-o", default="./output_images", help="图像保存目录")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file_path):
        print(f"错误: 文件 '{args.file_path}' 不存在。")
    else:
        data = read_pickle_file(args.file_path, args.show_images, args.save_images, args.output_dir)