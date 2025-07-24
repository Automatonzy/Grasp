import torch
from vision.featurizer import SDFeaturizer, DINOFeaturizer, CLIPFeaturizer, DINOv2Featurizer, RADIOFeaturizer, SD_DINOv2Featurizer
from vision.featurizer.utils.visualization import IMG_SIZE, Demo, visualize_max_xy, visualize_max_xy_linear, visualize_max_xy_list
from PIL import Image,ImageFilter
from torchvision.transforms import PILToTensor
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label
import json
import cv2
import os
import time

featurizers = {
    'sd': SDFeaturizer,
    'clip': CLIPFeaturizer,
    'dino': DINOFeaturizer,
    'dinov2': DINOv2Featurizer,
    'radio': RADIOFeaturizer,
    'sd_dinov2': SD_DINOv2Featurizer
}

def run_demo(src_path, tgt_path, prompt):
    file_list = [src_path, tgt_path]
    img_list = []
    ft_list = []
    for filename in file_list:
        img = Image.open(filename).convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_list.append(img)
        ft = extract_ft(img, prompt)
        ft_list.append(ft)
    
    ft = torch.cat(ft_list, dim=0)
    demo = Demo(img_list, ft, IMG_SIZE)
    demo.plot_img_pairs(fig_size=5)


#def extract_ft(img: Image.Image, prompt=None, ftype='sd'):
    '''
    preprocess of img to `img`:
    img = Image.open(filename).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    '''
    '''if img.size != (IMG_SIZE, IMG_SIZE):
        img = img.resize((IMG_SIZE, IMG_SIZE))
    img_tensor = (PILToTensor()(img) / 255.0 - 0.5) * 2 # C, H, W
    img_tensor = img_tensor.unsqueeze(0).cuda() # 1, C, H, W

    assert ftype in ['sd', 'clip', 'dino', 'dinov2', 'radio', 'sd_dinov2']
    featurizer = featurizers[ftype]()
    
    ft = featurizer.forward(
        img_tensor,
        block_index=1, # only for clip & dino
        prompt=prompt, # only for sd
        ensemble_size=2 # only for sd
    )
    return ft'''
def extract_ft1(img: Image.Image, prompt=None, ftype='sd', featurizer=None):

    if img.size != (IMG_SIZE, IMG_SIZE):
        img = img.resize((IMG_SIZE, IMG_SIZE))
    img_tensor = (PILToTensor()(img) / 255.0 - 0.5) * 2  # C, H, W
    img_tensor = img_tensor.unsqueeze(0).cuda()  # 1, C, H, W

    assert ftype in ['sd', 'clip', 'dino', 'dinov2', 'radio', 'sd_dinov2']

    # ✅ 不再每次新建 featurizer 实例
    if featurizer is None:
        featurizer = featurizers[ftype]()

    ft = featurizer.forward(
        img_tensor,
        block_index=1,     # only for clip & dino
        prompt=prompt,     # only for sd
        ensemble_size=2    # only for sd
    )
    return ft
class CoordinateMapper:
    def __init__(self, bbox, ratio, pad_x, pad_y, output_size):
        """
        坐标映射器：
        - bbox: (x, y, w, h) 原始图中裁剪区域
        - ratio: 缩放比例（裁剪后缩放前）
        - pad_x, pad_y: 缩放后居中填充的边距
        - output_size: resize 的目标尺寸（默认 224x224）
        """
        self.x, self.y, self.w, self.h = bbox
        self.ratio = ratio
        self.pad_x = pad_x
        self.pad_y = pad_y
        self.output_size = output_size

    def input_to_output(self, point):
        """
        将原图坐标映射到 resize+padding 后的特征图坐标
        """
        px, py = point
        x_scaled = (px - self.x) * self.ratio + self.pad_x
        y_scaled = (py - self.y) * self.ratio + self.pad_y
        return (x_scaled, y_scaled)

    def output_to_input(self, point):
        """
        将 resize+padding 后的图像坐标映射回原图
        """
        px, py = point
        x_orig = (px - self.pad_x) / self.ratio + self.x
        y_orig = (py - self.pad_y) / self.ratio + self.y
        return (x_orig, y_orig)
    def to_dict(self):
        return {
            "bbox": (self.x, self.y, self.w, self.h),
            "ratio": self.ratio,
            "pad_x": self.pad_x,
            "pad_y": self.pad_y,
            "output_size": self.output_size
        }

    @staticmethod
    def from_dict(d):
        return CoordinateMapper(
            bbox=tuple(d["bbox"]),
            ratio=d["ratio"],
            pad_x=d["pad_x"],
            pad_y=d["pad_y"],
            output_size=tuple(d["output_size"])
        )
def resize_masked_object_letterbox(img: Image.Image, mask: np.ndarray, output_size=(IMG_SIZE, IMG_SIZE), fill_value=255):
    """
    对 mask 掩膜区域裁剪后，等比 resize 到 output_size，并居中 padding。

    返回：
    - new_img: 处理后的图像
    - mapper: CoordinateMapper 坐标映射器
    """
    mask = mask.astype(np.uint8)
    if mask.ndim == 3:
        mask = mask[..., 0]  # 取第一个通道，确保单通道
    bbox = cv2.boundingRect(mask.astype(np.uint8))  # (x, y, w, h)
    x, y, w, h = bbox
    cropped_img = img.crop((x, y, x + w, y + h))

    ratio = min(output_size[0] / w, output_size[1] / h)
    new_w, new_h = int(w * ratio), int(h * ratio)
    resized_obj = cropped_img.resize((new_w, new_h), resample=Image.BILINEAR)

    new_img = Image.new("RGB", output_size, color=(fill_value, fill_value, fill_value))
    pad_x = (output_size[0] - new_w) // 2
    pad_y = (output_size[1] - new_h) // 2
    new_img.paste(resized_obj, (pad_x, pad_y))

    mapper = CoordinateMapper((x, y, w, h), ratio, pad_x, pad_y, output_size)
    return new_img, mapper

def extract_ft(img: Image.Image,
               prompt=None,
               ftype='sd',
               featurizer=None,
               mask: np.ndarray = None,
               output_size=(IMG_SIZE, IMG_SIZE)):
    """
    提取图像特征（支持 mask 裁剪 + 坐标映射器返回）

    返回：
    - ft: 特征张量 [1, C, H, W]
    - mapper: CoordinateMapper（若用 mask 启用）
    """
    mapper = None

    if mask is not None:
        img, mapper = resize_masked_object_letterbox(img, mask, output_size=output_size)
    elif img.size != output_size:
        img = img.resize(output_size, resample=Image.BILINEAR)
    img = img.convert("L").convert("RGB")  # 转灰度再转回RGB
    img = img.filter(ImageFilter.EDGE_ENHANCE)  # 加强边缘
    img_tensor = (PILToTensor()(img) / 255.0 - 0.5) * 2
    img_tensor = img_tensor.unsqueeze(0).cuda()

    assert ftype in ['sd', 'clip', 'dino', 'dinov2', 'radio', 'sd_dinov2'], f"Unsupported type: {ftype}"

    if featurizer is None:
        featurizer = featurizers[ftype]()
    ft = featurizer.forward(
        img_tensor,
        block_index=1,     # only for clip & dino
        prompt="",     # only for sd
        ensemble_size=2    # only for sd
    )
    return ft, mapper

def match_fts(src_ft, tgt_ft, pos, save_root=None):
    num_channel = src_ft.size(1)
    src_ft = nn.Upsample(size=(IMG_SIZE, IMG_SIZE), mode='bilinear')(src_ft)
    tgt_ft = nn.Upsample(size=(IMG_SIZE, IMG_SIZE), mode='bilinear')(tgt_ft)
    x, y = pos[0], pos[1]
    # interpolation from src_ft
    # src_vec = src_ft[0, :, int(y), int(x)].view(1, num_channel)  # 1, C
    x_norm = 2 * x / (IMG_SIZE - 1) - 1
    y_norm = 2 * y / (IMG_SIZE - 1) - 1
    src_vec = torch.nn.functional.grid_sample(src_ft, torch.tensor([[[[x_norm, y_norm]]]]).float().cuda(), align_corners=True).squeeze(2).squeeze(2)
    tgt_vecs = tgt_ft.view(1, num_channel, -1) # 1, C, H*W
    src_vec = F.normalize(src_vec) # 1, C
    tgt_vecs = F.normalize(tgt_vecs) # 1, C, HW
    cos_map = torch.matmul(src_vec, tgt_vecs).view(1, IMG_SIZE, IMG_SIZE).cpu().numpy() # 1, H, W

    return cos_map
'''def match_fts(src_ft, tgt_ft, pos):
    """
    计算 src_ft 的 pos 点 与 tgt_ft 所有点的 cosine 相似度图（预上采样 + 归一化）

    参数：
    - src_ft: Tensor, [1, C, H, W] 已上采样、已归一化
    - tgt_ft: Tensor, [1, C, H, W] 已上采样、已归一化
    - pos: (x, y) 点坐标（在 feature map 尺寸下）

    返回：
    - cos_map: 相似度热图，大小为 (1, H, W)，类型为 numpy
    """
    B, C, H, W = src_ft.shape
    x, y = pos
    x_norm = 2 * x / (W - 1) - 1
    y_norm = 2 * y / (H - 1) - 1
    grid = torch.tensor([[[[x_norm, y_norm]]]], dtype=torch.float32).to(src_ft.device)

    src_vec = F.grid_sample(src_ft, grid, align_corners=True).squeeze(2).squeeze(2)  # [1, C]

    tgt_vecs = tgt_ft.view(1, C, -1)                     # [1, C, H*W]
    cos_map = torch.matmul(src_vec, tgt_vecs)            # [1, 1, H*W]
    cos_map = cos_map.view(1, H, W).cpu().numpy()        # [1, H, W]

    return cos_map'''
def sample_highest(cos_map: np.ndarray):
    max_yx = np.unravel_index(cos_map[0].argmax(), cos_map[0].shape)
    max_xy = (max_yx[1], max_yx[0])
    return max_xy, cos_map[0][max_yx]

def sample_region(cos_map: np.ndarray, cos_threshold=0.9, size_threshold=1000):
    '''sample regions with high confidence from cos_map'''
    high_confidence = cos_map > cos_threshold
    labeled, num_features = label(high_confidence)
    region_map = np.zeros_like(high_confidence)
    for i in range(1, num_features):
        region_mask = labeled == i
        if np.sum(region_mask) > size_threshold:
            region_map += i * region_mask
    return region_map

def sample_points_from_best_region(cos_map: np.ndarray, best_region_map, topk=10, cos_threshold=0.9):
    '''sample pixel points with highest confidences from the best region'''
    best_region_mask = best_region_map == 1
    cos_map = cos_map * best_region_mask
    cos_map[cos_map < cos_threshold] = 0
    max_idx = np.argsort(cos_map, axis=None)[-topk:]
    max_yx = np.unravel_index(max_idx, cos_map.shape)
    return max_yx # (vec_0, vec_y, vec_x)


def fit_linear_ransac(points, threshold=10, min_samples=2, max_trials=1000):
    '''fit a line to points using RANSAC'''
    if min_samples >= len(points):
        min_samples = len(points) // 2
    best_inliers = []
    best_line = None
    for _ in range(max_trials):
        sample = points[np.random.choice(len(points), min_samples, replace=False)]
        line = np.polyfit(sample[:, 0], sample[:, 1], 1)
        inliers = np.abs(points[:, 1] - (line[0] * points[:, 0] + line[1])) < threshold
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_line = line
    print("Number of best inliers:", np.sum(best_inliers))
    inlier_points = points[best_inliers]
    # turn best_line into direction
    best_line = np.array([1, best_line[0]])
    best_line = best_line / np.linalg.norm(best_line)
    # determine the sign of best_line
    positive_value = 0
    for idx in range(inlier_points.shape[0]-1):
        positive_value += np.dot(inlier_points[idx+1] - inlier_points[idx], best_line)
    if positive_value < 0:
        best_line = -best_line
    # end_start_vec = points[-1] - points[0]
    # if np.dot(best_line, end_start_vec) < 0:
    #     best_line = -best_line
    return inlier_points, best_line


def horizontal_flip_augmentation(src_img_PIL, src_pos_list):
    size = src_img_PIL.size
    augmented_img_PILs = [src_img_PIL]
    augmented_pos_lists = [src_pos_list]
    flipped_img_PIL = src_img_PIL.transpose(Image.FLIP_LEFT_RIGHT)
    flipped_pos_list = []
    for pos in src_pos_list:
        flipped_pos_list.append((size[0] - pos[0], pos[1]))
    augmented_img_PILs.append(flipped_img_PIL)
    augmented_pos_lists.append(flipped_pos_list)
    return augmented_img_PILs, augmented_pos_lists

def choose_from_augmentation(augmented_img_PILs, augmented_pos_lists, tgt_img_PIL, prompt, ftype='sd'):
    max_cos = -1e6
    max_idx = -1
    tgt_ft = extract_ft(tgt_img_PIL, prompt=prompt, ftype=ftype)
    for idx in range(len(augmented_img_PILs)):
        src_ft = extract_ft(augmented_img_PILs[idx], prompt=prompt, ftype=ftype)
        cos_map = match_fts(src_ft, tgt_ft, augmented_pos_lists[idx][0])
        _, cos = sample_highest(cos_map)
        if cos > max_cos:
            max_cos = cos
            max_idx = idx
    return augmented_img_PILs[max_idx], augmented_pos_lists[max_idx]
        
def transfer_multiple_contacts(src_img_PIL, tgt_img_PIL, prompt, src_pos_list, save_root=None, ftype='sd'):
    """
    将源图像上的多个接触点映射到目标图像上
    
    参数:
        src_img_PIL: 源图像
        tgt_img_PIL: 目标图像
        prompt: 文本提示，用于引导特征提取
        src_pos_list: 源图像上的接触点坐标列表
        save_root: 可视化结果保存路径
        ftype: 特征提取器类型，默认'sd'
        
    返回:
        contact_points: 目标图像上所有对应点的坐标列表
        similarity_scores: 每个对应点的相似度分数
    """
    # 提取特征
    src_ft = extract_ft(src_img_PIL, prompt=prompt, ftype=ftype)
    tgt_ft = extract_ft(tgt_img_PIL, prompt=prompt, ftype=ftype)
    
    # 存储结果
    contact_points = []
    similarity_scores = []
    cos_maps = []
    
    # 为每个源接触点找到目标图像上的对应点
    for i, src_pos in enumerate(src_pos_list):
        # 计算特征匹配图
        cos_map = match_fts(src_ft, tgt_ft, src_pos)
        cos_maps.append(cos_map)
        
        # 找到最佳匹配点
        max_xy, max_sim = sample_highest(cos_map)
        
        # 将坐标从特征尺寸转换到原始图像尺寸
        contact_point = (
            int(max_xy[0] * tgt_img_PIL.size[0] / IMG_SIZE), 
            int(max_xy[1] * tgt_img_PIL.size[1] / IMG_SIZE)
        )
        
        # 保存结果
        contact_points.append(contact_point)
        similarity_scores.append(max_sim)
        
        # 输出调试信息
        print(f'接触点 {i+1}: 源点={src_pos}, 目标点={contact_point}, 相似度={max_sim:.4f}')
    
    # 可视化结果
    if save_root:
        # 可视化每个单独的点对应关系
        print(f'可视化每个接触点的对应关系...{save_root}')
        for i, (src_pos, contact_point, cos_map) in enumerate(zip(src_pos_list, contact_points, cos_maps)):
            visualize_max_xy(
                save_root, 
                src_pos, 
                contact_point, 
                src_img_PIL, 
                tgt_img_PIL, 
                heatmap=cos_map[0],
                filename=f'contact_point_{i+1}'
            )
        
        # 可视化所有点的对应关系
        src_pos_list_np = np.array(src_pos_list)
        contact_points_np = np.array(contact_points)
        visualize_max_xy_list(
            save_root, 
            src_pos_list_np, 
            contact_points_np, 
            src_img_PIL, 
            tgt_img_PIL, 
            filename='all_contact_points'
        )
    
    return contact_points, similarity_scores
'''def transfer_affordance(src_img_PIL, tgt_img_PIL, prompt, src_pos_list, save_root=None, ftype='sd'):
    
    print("原图像尺寸:", src_img_PIL.size)
    print("目标图像尺寸:", tgt_img_PIL.size)
    
    ori_cos_map = None
    ori_xy_list = []
    max_xy_list = []
    cos_maps = []
    src_ft= extract_ft(src_img_PIL, prompt=prompt, ftype=ftype)
    tgt_ft = extract_ft(tgt_img_PIL, prompt=prompt, ftype=ftype)
    
    for scaled_src_pos in src_pos_list:
        cos_map = match_fts(src_ft, tgt_ft, scaled_src_pos)
        if ori_cos_map is None:
            ori_cos_map = cos_map
        cos_maps.append(cos_map)
        
        # 4. 获取最佳匹配点并转换回原始尺寸
        max_xy, _ = sample_highest(cos_map)
        max_xy = (max_xy[0] * tgt_img_PIL.size[0] / IMG_SIZE, 
                 max_xy[1] * tgt_img_PIL.size[1] / IMG_SIZE)
        max_xy_list.append(max_xy)
        ori_xy=(scaled_src_pos[0] * src_img_PIL.size[0] / IMG_SIZE,
                 scaled_src_pos[1] * src_img_PIL.size[1] / IMG_SIZE)
        ori_xy_list.append(ori_xy)
    
    # ori_xy_list_np = np.array(src_pos_list)  # 原始坐标
    ori_xy_list_np = np.array(ori_xy_list)  # 原始坐标(已转换回原始尺寸)
    max_xy_list_np = np.array(max_xy_list)    # 目标坐标(已转换回原始尺寸)
    
    print('源图像接触点:', ori_xy_list_np)
    print('目标图像接触点:', max_xy_list_np)
    
    
    if save_root:
        #将目标图像接触点和原图像接触点对应存入json文件
        with open(f'{save_root}/contact_points.json', 'w') as f:
            json.dump({
                'source_contact_points': ori_xy_list_np.tolist(),
                'target_contact_points': max_xy_list_np.tolist()
            }, f, indent=4)
        print(f'接触点已保存到 {save_root}/contact_points.json')

        print(f'可视化每个接触点的对应关系...{save_root}')
        
        # 使用原始尺度的坐标进行可视化
        for i, (src_pos, tgt_pos, cos_map) in enumerate(zip(src_pos_list, max_xy_list, cos_maps)):
            visualize_max_xy(
                save_root, 
                src_pos,  # 原始坐标
                tgt_pos,  # 已转换回原始尺寸的坐标
                src_img_PIL, 
                tgt_img_PIL, 
                heatmap=cos_map[0],
                filename=f'contact_point_{i+1}'
            )
        
        visualize_max_xy_list(
            save_root, 
            ori_xy_list_np,
            max_xy_list_np, 
            src_img_PIL, 
            tgt_img_PIL, 
            filename='all_contact_points'
        )
        
    
    return max_xy_list_np, max_xy_list[1:] if len(max_xy_list) > 1 else 0'''
'''def transfer_affordance(src_img_PIL, tgt_img_PIL, prompt, src_pos_list, save_root=None, ftype='sd',
                        src_mask=None, tgt_mask=None, featurizer=None):
    """
    迁移源图像上的接触点到目标图像。

    参数：
    - src_img_PIL: 源图像
    - tgt_img_PIL: 目标图像
    - src_pos_list: 源图像中的接触点（原图坐标）
    - src_mask: 源图像的mask（用于裁剪/统一尺度）
    - tgt_mask: 目标图像的mask
    """
    src_ft, src_mapper = extract_ft(src_img_PIL, prompt=prompt, ftype=ftype, mask=src_mask, featurizer=featurizer)
    tgt_ft, tgt_mapper = extract_ft(tgt_img_PIL, prompt=prompt, ftype=ftype, mask=tgt_mask, featurizer=featurizer)
    ori_xy_list = []
    max_xy_list = []
    cos_maps = []

    src_pos_list_scaled = [src_mapper.input_to_output(p) for p in src_pos_list]

    for i, src_pos_scaled in enumerate(src_pos_list_scaled):
        cos_map = match_fts(src_ft, tgt_ft, src_pos_scaled)
        cos_maps.append(cos_map)

        max_xy, _ = sample_highest(cos_map)
        tgt_point_orig = tgt_mapper.output_to_input(max_xy)
        src_point_orig = src_mapper.output_to_input(src_pos_scaled)

        ori_xy_list.append(src_point_orig)
        max_xy_list.append(tgt_point_orig)

    ori_xy_list_np = np.array(ori_xy_list)
    max_xy_list_np = np.array(max_xy_list)

    if save_root:
        os.makedirs(save_root, exist_ok=True)
        with open(f'{save_root}/contact_points.json', 'w') as f:
            json.dump({
                'source_contact_points': ori_xy_list_np.tolist(),
                'target_contact_points': max_xy_list_np.tolist()
            }, f, indent=4)

        for i, (src_pos, tgt_pos, cos_map) in enumerate(zip(ori_xy_list_np, max_xy_list_np, cos_maps)):
            visualize_max_xy(
                save_root,
                src_pos,
                tgt_pos,
                src_img_PIL,
                tgt_img_PIL,
                heatmap=cos_map[0],
                filename=f'contact_point_{i + 1}'
            )
        visualize_max_xy_list(
            save_root,
            ori_xy_list_np,
            max_xy_list_np,
            src_img_PIL,
            tgt_img_PIL,
            filename='all_contact_points'
        )

    post_contact_dir = max_xy_list[1:] if len(max_xy_list) > 1 else 0
    return max_xy_list_np, post_contact_dir'''
#加速
def transfer_affordance(src_img_PIL, tgt_img_PIL, prompt, src_pos_list, save_root=None, ftype='sd',
                        src_mask=None, tgt_mask=None, featurizer=None, src_ft=None,src_mapper=None):
    """
    迁移源图像上的接触点到目标图像（优化版）

    - 避免冗余上采样
    - 避免重复 normalize
    """
    start=time.time()
    if src_ft is None or src_mapper is None:
        src_ft, src_mapper = extract_ft(src_img_PIL, prompt=prompt, ftype=ftype, mask=src_mask, featurizer=featurizer)
    else:
        src_mapper = CoordinateMapper.from_dict(src_mapper)
    tgt_ft, tgt_mapper = extract_ft(tgt_img_PIL, prompt=prompt, ftype=ftype, mask=tgt_mask, featurizer=featurizer)
    # 统一上采样 + normalize
    src_ft = F.interpolate(src_ft, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=True)
    tgt_ft = F.interpolate(tgt_ft, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=True)

    src_ft = F.normalize(src_ft, dim=1)
    tgt_ft = F.normalize(tgt_ft, dim=1)

    # 坐标映射
    src_pos_list_scaled = [src_mapper.input_to_output(p) for p in src_pos_list]
    ori_xy_list = []
    max_xy_list = []
    cos_maps = []

    for i, src_pos_scaled in enumerate(src_pos_list_scaled):
        cos_map = match_fts(src_ft, tgt_ft, src_pos_scaled)
        cos_maps.append(cos_map)

        max_xy, _ = sample_highest(cos_map)
        tgt_point_orig = tgt_mapper.output_to_input(max_xy)
        src_point_orig = src_mapper.output_to_input(src_pos_scaled)

        ori_xy_list.append(src_point_orig)
        max_xy_list.append(tgt_point_orig)

    ori_xy_list_np = np.array(ori_xy_list)
    max_xy_list_np = np.array(max_xy_list)
    end=time.time()
    print(end-start)
    if save_root:
        os.makedirs(save_root, exist_ok=True)
        with open(f'{save_root}/contact_points.json', 'w') as f:
            json.dump({
                'source_contact_points': ori_xy_list_np.tolist(),
                'target_contact_points': max_xy_list_np.tolist()
            }, f, indent=4)

        for i, (src_pos, tgt_pos, cos_map) in enumerate(zip(ori_xy_list_np, max_xy_list_np, cos_maps)):
            visualize_max_xy(
                save_root,
                src_pos,
                tgt_pos,
                src_img_PIL,
                tgt_img_PIL,
                heatmap=cos_map[0],
                filename=f'contact_point_{i + 1}'
            )
        visualize_max_xy_list(
            save_root,
            ori_xy_list_np,
            max_xy_list_np,
            src_img_PIL,
            tgt_img_PIL,
            filename='all_contact_points'
        )

    post_contact_dir = max_xy_list[1:] if len(max_xy_list) > 1 else 0
    return max_xy_list_np, post_contact_dir

def transfer_affordance_w_mask(src_img_PIL, tgt_img_PIL, tgt_mask, prompt, src_pos_list, save_root=None, ftype='sd'):
    mask = torch.from_numpy(tgt_mask[...,0]).cuda() # h,w
    resized_mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(IMG_SIZE, IMG_SIZE), mode='nearest').squeeze().cuda() # h,w -> 1,IMG_SIZE,IMG_SIZE
    resized_mask = resized_mask.cpu().numpy()
    ori_cos_map = None
    max_xy_list = []
    src_ft = extract_ft(src_img_PIL, prompt=prompt, ftype=ftype)
    tgt_ft = extract_ft(tgt_img_PIL, prompt=prompt, ftype=ftype)
    match_fts(src_ft, tgt_ft, src_pos_list[0], save_root)
    for src_pos_id, src_pos in enumerate(src_pos_list):
        cos_map = match_fts(src_ft, tgt_ft, src_pos) # 1,IMG_SIZE,IMG_SIZE
        if src_pos_id == 0:
            cos_map = cos_map * resized_mask
        if ori_cos_map is None:
            ori_cos_map = cos_map
        max_xy, _ = sample_highest(cos_map)
        max_xy = (max_xy[0] * tgt_img_PIL.size[0] / IMG_SIZE, max_xy[1] * tgt_img_PIL.size[1] / IMG_SIZE)
        max_xy_list.append(max_xy)
    src_pos_list_np = np.array(src_pos_list)
    max_xy_list_np = np.array(max_xy_list)
    src_pos_inliers, src_best_line = fit_linear_ransac(src_pos_list_np, threshold=5, min_samples=10)
    max_xy_inliers, tgt_best_line = fit_linear_ransac(max_xy_list_np, threshold=5, min_samples=10)
    # src_best_line and tgt_best_line should be in the same direction (under similar viewpoints)
    if np.dot(src_best_line, tgt_best_line) < 0:
        tgt_best_line = -tgt_best_line
    contact_point = (int(max_xy_list[0][0]), int(max_xy_list[0][1]))
    if save_root:
        print('src & tgt best lines:\n', src_best_line, tgt_best_line)
        visualize_max_xy(save_root, src_pos_list[0], contact_point, src_img_PIL, tgt_img_PIL, heatmap=ori_cos_map[0])
        visualize_max_xy_list(save_root, src_pos_list_np, max_xy_list_np, src_img_PIL, tgt_img_PIL, filename='max_xy_list_all')
        visualize_max_xy_list(save_root, src_pos_inliers, max_xy_inliers, src_img_PIL, tgt_img_PIL)
        visualize_max_xy_linear(save_root, src_pos_list[0], src_best_line, contact_point, tgt_best_line, src_img_PIL, tgt_img_PIL)
    return contact_point, tgt_best_line

'''make sure the contact point is within the mask'''
def transfer_affordance_tt(src_img_PIL, tgt_img_PIL, tgt_mask, prompt, src_pos_list, save_root=None, ftype='sd'):
    mask = torch.from_numpy(tgt_mask[...,0]).cuda() # h,w
    resized_mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(IMG_SIZE, IMG_SIZE), mode='nearest').squeeze().cuda() # h,w -> 1,IMG_SIZE,IMG_SIZE
    resized_mask = resized_mask.cpu().numpy()
    ori_cos_map = None
    max_xy_list = []
    src_ft = extract_ft(src_img_PIL, prompt=prompt, ftype=ftype)
    tgt_ft = extract_ft(tgt_img_PIL, prompt=prompt, ftype=ftype)
    match_fts(src_ft, tgt_ft, src_pos_list[0], save_root)
    for src_pos_id, src_pos in enumerate(src_pos_list):
        cos_map = match_fts(src_ft, tgt_ft, src_pos) # 1,IMG_SIZE,IMG_SIZE
        if src_pos_id == 0:
            cos_map = cos_map * resized_mask
        if ori_cos_map is None:
            ori_cos_map = cos_map
        max_xy, _ = sample_highest(cos_map)
        max_xy = (max_xy[0] * tgt_img_PIL.size[0] / IMG_SIZE, max_xy[1] * tgt_img_PIL.size[1] / IMG_SIZE)
        max_xy_list.append(max_xy)
    src_pos_list_np = np.array(src_pos_list)
    max_xy_list_np = np.array(max_xy_list)
    src_pos_inliers, src_best_line = fit_linear_ransac(src_pos_list_np, threshold=5, min_samples=10)
    max_xy_inliers, tgt_best_line = fit_linear_ransac(max_xy_list_np, threshold=5, min_samples=10)
    # tgt_best_line should be upwards
    if tgt_best_line[1] > 0:
        tgt_best_line = -tgt_best_line
    contact_point = (int(max_xy_list[0][0]), int(max_xy_list[0][1]))
    if save_root:
        print('src & tgt best lines:\n', src_best_line, tgt_best_line)
        visualize_max_xy(save_root, src_pos_list[0], contact_point, src_img_PIL, tgt_img_PIL, heatmap=ori_cos_map[0])
        visualize_max_xy_list(save_root, src_pos_list_np, max_xy_list_np, src_img_PIL, tgt_img_PIL, filename='max_xy_list_all')
        visualize_max_xy_list(save_root, src_pos_inliers, max_xy_inliers, src_img_PIL, tgt_img_PIL)
        visualize_max_xy_linear(save_root, src_pos_list[0], src_best_line, contact_point, tgt_best_line, src_img_PIL, tgt_img_PIL)
    return contact_point, tgt_best_line
    