import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
import mediapipe as mp
import os
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.neighbors import BallTree
import sys
import torch
from PIL import Image
import time
import json
from scipy.spatial.distance import cdist
import matplotlib
from scipy.spatial.transform import Rotation as R
import scipy.io as scio  # 在文件顶部确保引入该包
import random
matplotlib.use('svg')  # NOTE: fix backend error while GPU is in use



# 添加根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
os.environ["PYTHONPATH"] = root_dir
sys.path.insert(0, root_dir)
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# 从其他模块导入功能
from run_realworld.env import MiniEnv
from run_realworld.utils import read_yaml_config
from vision.GroundedSAM.grounded_sam_utils import prepare_gsam_model, inference_one_image
from vision.featurizer.run_featurizer import transfer_affordance
from vision.featurizer.utils.visualization import IMG_SIZE
from subset_retrieval.subset_retrieve_pipeline_while import SubsetRetrievePipeline, visualize_mask_and_trajectory
from vision.featurizer import SDFeaturizer, DINOFeaturizer, CLIPFeaturizer, DINOv2Featurizer, RADIOFeaturizer, SD_DINOv2Featurizer
# 全局模型变量
grounded_dino_model = None
sam_predictor = None
subset_retrieve_pipeline = None
def get_next_indexed_folder(base_dir):
    existing = [int(name) for name in os.listdir(base_dir) if name.isdigit()]
    next_idx = max(existing) + 1 if existing else 0
    folder = os.path.join(base_dir, str(next_idx))
    os.makedirs(folder, exist_ok=True)

    # 创建空白 json 文件
    empty_json_path = os.path.join(folder, "empty_record.json")
    with open(empty_json_path, "w") as f:
        json.dump({}, f, indent=2)
    print(f"✅ 创建空白 JSON 文件: {empty_json_path}")

    return folder
def init_models(config_path, save_root):
    """初始化所有模型并加载配置"""
    global grounded_dino_model, sam_predictor, subset_retrieve_pipeline,featurizer
    featurizers = {
    'sd': SDFeaturizer,
    'clip': CLIPFeaturizer,
    'dino': DINOFeaturizer,
    'dinov2': DINOv2Featurizer,
    'radio': RADIOFeaturizer,
    'sd_dinov2': SD_DINOv2Featurizer
}
    featurizer = featurizers['sd']()
    # 读取配置文件
    cfgs = read_yaml_config(f"run_realworld/{config_path}")
    
    # 初始化Grounded-SAM模型
    print("初始化Grounded-SAM模型...")
    grounded_dino_model, sam_predictor = prepare_gsam_model(device="cuda")
    
    # 初始化检索管道
    print("初始化SubsetRetrievePipeline...")
    subset_retrieve_pipeline = SubsetRetrievePipeline(
        subset_dir="assets/data",
        save_root=save_root,
        lang_mode='clip',
        topk=5, 
        crop=False,
        data_source=cfgs.get("DATA_SOURCE", "database"),
    )
    
    return cfgs

# 第一部分：图像捕获和碰撞点检测
'''def capture_and_detect():
    """捕获图像并检测手部碰撞点"""
    def draw_landmarks(image, landmarks_list):
        for landmarks in landmarks_list:
            for i, landmark in enumerate(landmarks):
                cv2.circle(image, (int(landmark[0]), int(landmark[1])), 5, (0, 255, 0), -1)
                cv2.putText(image, str(i), (int(landmark[0]), int(landmark[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    def _get_rotation_between(v1, v2):
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        axis = np.cross(v1, v2)
        if np.allclose(axis, [0, 0, 0]):
            return np.eye(3)
        angle = np.arccos(np.dot(v1, v2))
        return o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)

    def capture_frame():
        """使用RealSense相机捕获帧"""
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        try:
            pipeline.start(config)
        except RuntimeError as e:
            print("\u9519\u8bef\u89e3\u51b3\u65b9\u6848\u5efa\u8bae:")
            print("1. \u5c1d\u8bd5\u66f4\u6362USB\u63a5\u53e3\uff08\u4f18\u5148USB3.0\uff09")
            print("2. \u68c0\u67e5\u76f8\u673a\u56fa\u4ef6\u7248\u672c\u662f\u5426\u4e3a\u6700\u65b0")
            print("3. \u964d\u4f4e\u5206\u8fa8\u7387\u914d\u7f6e")
            raise e
        try:
            profile = pipeline.get_active_profile()
            intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            fx, fy, cx, cy = intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy
            align = rs.align(rs.stream.color)
            for _ in range(20):
                frames = pipeline.wait_for_frames()
                frames = align.process(frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                raise Exception("Failed to capture frames")
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            return color_image, depth_image, fx, fy, cx, cy
        finally:
            pipeline.stop()

    def detect_hand_with_mediapipe(color_image):
        """使用MediaPipe检测手部，并选取面积最大的手进行处理"""
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=True, max_num_hands=5,
                            min_detection_confidence=0.5, min_tracking_confidence=0.2)
        results = hands.process(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
        
        mask = np.zeros(color_image.shape[:2], dtype=np.uint8)
        max_area = -1
        selected_landmarks = None
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 提取每个关键点的像素坐标
                landmarks = [(min(max(lm.x * color_image.shape[1], 0), color_image.shape[1] - 1),
                            min(max(lm.y * color_image.shape[0], 0), color_image.shape[0] - 1))
                            for lm in hand_landmarks.landmark]

                x_coords = [p[0] for p in landmarks]
                y_coords = [p[1] for p in landmarks]
                box_width = max(x_coords) - min(x_coords)
                box_height = max(y_coords) - min(y_coords)
                area = box_width * box_height

                if area > max_area:
                    max_area = area
                    selected_landmarks = landmarks

            if selected_landmarks is not None:
                # 选出最大面积手后绘制其mask
                x_coords = [p[0] for p in selected_landmarks]
                y_coords = [p[1] for p in selected_landmarks]
                box_width = max(x_coords) - min(x_coords)
                box_height = max(y_coords) - min(y_coords)
                x_min = max(0, int(min(x_coords) - box_width * 0.1))
                y_min = max(0, int(min(y_coords) - box_height * 0.1))
                x_max = min(color_image.shape[1] - 1, int(max(x_coords) + box_width * 0.1))
                y_max = min(color_image.shape[0] - 1, int(max(y_coords) + box_height * 0.1))
                cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 255, -1)

        segmented = cv2.bitwise_and(color_image, color_image, mask=mask)
        return mask, segmented, [selected_landmarks] if selected_landmarks is not None else []

    def get_3d_point(landmark, depth_image, fx, fy, cx, cy):
        """将2D点转换为3D点"""
        u, v = int(landmark[0]), int(landmark[1])
        depth = depth_image[v, u]
        if depth == 0: 
            return None
        z = depth / 1000.0
        return np.array([(u - cx) * z / fx, (v - cy) * z / fy, z])

    # 主捕获和检测逻辑
    print("正在捕获图像并检测手部...")
    color_image, depth_image, fx, fy, cx, cy = capture_frame()
    original_color = color_image.copy()
    color_image = cv2.GaussianBlur(color_image, (5, 5), 0)
    mask, segmented, landmarks_list = detect_hand_with_mediapipe(color_image)
    
    contact_point_2d = None
    if landmarks_list:
        try:
            idx_proximal, idx_tip = 7, 8
            p1, p2 = landmarks_list[0][idx_proximal], landmarks_list[0][idx_tip]
            point1 = get_3d_point(p1, depth_image, fx, fy, cx, cy)
            point2 = get_3d_point(p2, depth_image, fx, fy, cx, cy)
            
            if point1 is None or point2 is None:
                print("无法获取指尖3D点")
                return color_image, depth_image, fx, fy, cx, cy, None
            
            initial_dir = point2 - point1
            points = np.stack([np.asarray([(u - cx) * depth_image[v, u] / fx / 1000.0,
                                          (v - cy) * depth_image[v, u] / fy / 1000.0,
                                          depth_image[v, u] / 1000.0])
                              for v in range(depth_image.shape[0])
                              for u in range(depth_image.shape[1])
                              if depth_image[v, u] > 0 and depth_image[v, u] / 1000.0 <= 2], axis=0)
            
            tree = BallTree(points, leaf_size=2)
            hand_indices = tree.query_radius([point1, point2], r=0.01)
            selected_points = points[np.unique(np.concatenate(hand_indices))]
            selected_points = np.vstack([selected_points, point1, point2])

            pca = PCA(n_components=3).fit(selected_points)
            if np.dot(pca.components_[0], initial_dir) < 0:
                pca.components_[0] *= -1
            direction = pca.components_[0]
            direction=initial_dir
            hand_mask = np.zeros(len(points), dtype=bool)
            hand_mask[np.unique(np.concatenate(hand_indices))] = True

            non_hand_points = points[~hand_mask]
            vectors = non_hand_points - point2
            distances = np.linalg.norm(vectors, axis=1)
            valid = distances > 0.01
            vectors = vectors[valid]
            distances = distances[valid]
            non_hand_points = non_hand_points[valid]

            cos_angles = np.dot(vectors, direction) / distances
            in_line = cos_angles > np.cos(np.deg2rad(10))
            if np.any(in_line):
                candidates = non_hand_points[in_line]
                candidate_distances = distances[in_line]
                contact_point_3d = candidates[np.argmin(candidate_distances)]
                
                # 将3D点转换回2D像素坐标
                u = int((contact_point_3d[0] * fx) / contact_point_3d[2] + cx)
                v = int((contact_point_3d[1] * fy) / contact_point_3d[2] + cy)
                contact_point_2d = [u, v]
                print(f"检测到碰撞点: ({u}, {v})")
            else:
                print("未找到碰撞点")
        except Exception as e:
            print(f"分析失败: {e}")
    else:print("未检测到手部")
    return original_color, depth_image, fx, fy, cx, cy, contact_point_2d'''
def capture_and_detect(save_root="debug_output"):
    """
    捕获图像并检测手部接触点，保存图像、相机参数、点云与射线模型
    同时执行射线与点云的碰撞检测并输出接触点
    """
    import os
    import json
    import cv2
    import numpy as np
    import open3d as o3d
    import mediapipe as mp
    import pyrealsense2 as rs
    from sklearn.neighbors import BallTree

    os.makedirs(save_root, exist_ok=True)

    def get_3d_point(u, v, depth_image, fx, fy, cx, cy):
        if not (0 <= u < depth_image.shape[1] and 0 <= v < depth_image.shape[0]):
            return None
        d = depth_image[v, u]
        if d == 0:
            return None
        z = d / 1000.0
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return np.array([x, y, z])

    # ========== 1. 捕获图像 ==========
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    for _ in range(20):
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)

    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    pipeline.stop()

    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    # ========== 2. 获取相机内参 ==========
    intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    fx, fy, cx, cy = intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy

    cv2.imwrite(os.path.join(save_root, "captured_color.png"), color_image)
    cv2.imwrite(os.path.join(save_root, "captured_depth.png"), depth_image)
    with open(os.path.join(save_root, "camera_params.json"), 'w') as f:
        json.dump({"width": color_image.shape[1], "height": color_image.shape[0],
                   "fx": fx, "fy": fy, "cx": cx, "cy": cy}, f, indent=2)
    # ========== 额外：保存 meta.mat ==========
    meta_path = os.path.join(save_root, "meta.mat")
    meta = {
        'intrinsic_matrix': np.array([
            [fx,   0, cx],
            [0,   fy, cy],
            [0,    0,  1]
        ]),
        'factor_depth': np.array([1000.0])  # GraspNet 默认深度单位转换（mm → m）
    }
    scio.savemat(meta_path, meta)
    print(f"✅ 已保存 GraspNet 格式的 meta.mat 到: {meta_path}")

    # ========== 3. 构建点云（不做下采样） ==========
    color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    rgb_o3d = o3d.geometry.Image(color_image_rgb)
    depth_o3d = o3d.geometry.Image(depth_image)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d, depth_o3d, depth_scale=1000.0, convert_rgb_to_intensity=False)

    intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(
        color_image.shape[1], color_image.shape[0], fx, fy, cx, cy)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic_o3d)
    o3d.io.write_point_cloud(os.path.join(save_root, "scene_pointcloud.ply"), pcd)

    # ========== 4. 检测手部关键点 ==========
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
    results = hands.process(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
    hands.close()

    if not results.multi_hand_landmarks:
        print("未检测到手部")
        return color_image, depth_image, fx, fy, cx, cy, None

    hand_landmarks = results.multi_hand_landmarks[0]
    landmark_px = [(int(lm.x * color_image.shape[1]), int(lm.y * color_image.shape[0])) for lm in hand_landmarks.landmark]
    # ========== 4.5 使用手部框过滤点云 ==========
    x_coords = [u for u, v in landmark_px]
    y_coords = [v for u, v in landmark_px]
    x_min, x_max = max(0, int(min(x_coords))), min(color_image.shape[1]-1, int(max(x_coords)))
    y_min, y_max = max(0, int(min(y_coords))), min(color_image.shape[0]-1, int(max(y_coords)))

    # 放大比例（可调节）
    scale = 1.4
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    box_width = (x_max - x_min) * scale
    box_height = (y_max - y_min) * scale

    # 计算新的放大边界框
    x_min = max(0, int(x_center - box_width / 2))
    x_max = min(color_image.shape[1] - 1, int(x_center + box_width / 2))
    y_min = max(0, int(y_center - box_height / 2))
    y_max = min(color_image.shape[0] - 1, int(y_center + box_height / 2))

    # 可视化手部框
    '''color_image = cv2.rectangle(color_image.copy(), (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.imwrite(os.path.join(save_root, "color_with_hand_box.png"), color_image)'''
    # 将图像坐标映射为 mask
    mask = np.zeros(depth_image.shape, dtype=np.uint8)
    cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 1, -1)

    # 创建手部掩码点云索引
    valid_indices = []
    for i, (x, y, z) in enumerate(np.asarray(pcd.points)):
        u = int((x * fx) / z + cx)
        v = int((y * fy) / z + cy)
        if not (0 <= u < depth_image.shape[1] and 0 <= v < depth_image.shape[0]):
            continue
        if mask[v, u] == 0:
            valid_indices.append(i)

    # 筛选点云
    pcd = pcd.select_by_index(valid_indices)
    u1, v1 = landmark_px[6]  # 掌基
    u2, v2 = landmark_px[7]  # 指尖
    p1 = get_3d_point(u1, v1, depth_image, fx, fy, cx, cy)
    p2 = get_3d_point(u2, v2, depth_image, fx, fy, cx, cy)
    if p1 is None or p2 is None:
        print("无法获取指尖3D点")
        return color_image, depth_image, fx, fy, cx, cy, None

    # ========== 5. 构建射线 ==========
    ray_dir = p2 - p1
    ray_dir = ray_dir / np.linalg.norm(ray_dir)
    ray_end = p2 + ray_dir * 1.0  # 1m
    ray_line = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector([p2, ray_end]),
        lines=o3d.utility.Vector2iVector([[0, 1]])
    )
    ray_line.colors = o3d.utility.Vector3dVector([[1, 0, 0]])
    o3d.io.write_line_set(os.path.join(save_root, "ray_line.ply"), ray_line)

    # ========== 6. 射线碰撞检测（新逻辑） ==========
    xyz = np.asarray(pcd.points)
    if xyz.shape[0] == 0:
        print("点云为空")
        return color_image, depth_image, fx, fy, cx, cy, None

    vecs = xyz - p2
    max_tip_distance = 0.8
    proj_lens = np.dot(vecs, ray_dir)
    dists_to_tip = np.linalg.norm(xyz - p2, axis=1)
    valid_mask = (proj_lens > 0) & (dists_to_tip <= max_tip_distance)
    proj_points = p2 + np.outer(proj_lens, ray_dir)
    dists_to_ray = np.linalg.norm(xyz - proj_points, axis=1)
    dists_to_tip = np.linalg.norm(xyz - p2, axis=1)
    def refine_point_in_3d(pcd_xyz, center_point, radius=0.03):
        """
        在点云中搜索与中心点相距小于 radius 的点，计算质心
        """
        dists = np.linalg.norm(pcd_xyz - center_point, axis=1)
        neighbor_indices = np.where(dists < radius)[0]
        if len(neighbor_indices) == 0:
            print("⚠️ 没有找到3D邻域点，保留原始接触点")
            return center_point
        neighbor_points = pcd_xyz[neighbor_indices]
        refined_point = np.mean(neighbor_points, axis=0)
        return refined_point

    # 先选出射线距离 < 3cm 的所有点，再从中取距离指尖最近的
    candidate_indices = np.where((dists_to_ray < 0.02)&valid_mask)[0]
    if len(candidate_indices) > 0:
        best_idx = candidate_indices[np.argmin(dists_to_tip[candidate_indices])]
        contact_point_3d = xyz[best_idx]
        contact_point_3d = refine_point_in_3d(xyz, contact_point_3d, radius=0.03)
        contact_ray_dist = dists_to_ray[best_idx]
        contact_tip_dist = dists_to_tip[best_idx]

        u = int((contact_point_3d[0] * fx) / contact_point_3d[2] + cx)
        v = int((contact_point_3d[1] * fy) / contact_point_3d[2] + cy)

        print(f"✅ 接触点像素坐标: ({u}, {v})")
        print(f"📏 射线距离: {contact_ray_dist:.4f} m")
        print(f"📏 指尖距离: {contact_tip_dist:.4f} m")

        return color_image, depth_image, fx, fy, cx, cy, (u, v)
    else:
        print("未找到接触点")
        return color_image, depth_image, fx, fy, cx, cy, None

# 第二部分：Affordance Transfer和3D投影
def process_affordance(
    color_image, depth_image_mm, fx, fy, cx, cy, contact_point_2d, 
    cfgs, save_root, use_retrieve=True,
    grounded_dino_model=None, sam_predictor=None, subset_retrieve_pipeline=None
):
    """处理affordance transfer和3D投影（使用预加载的模型）"""
    # 设置随机种子
    random.seed(cfgs.get('seed', 100))
    np.random.seed(cfgs.get('seed', 100))
    torch.manual_seed(cfgs.get('seed', 100))
    torch.cuda.manual_seed(cfgs.get('seed', 100))
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    
    # 将深度图转换为米
    depth_image = depth_image_mm.astype(np.float32) / 1000.0
    
    # 准备目标图像
    tgt_img_PIL = Image.fromarray(color_image).convert('RGB')
    tgt_img_PIL.save(f"{save_root}/tgt_img.png")
    
    ########################## 新增的可视化代码 ##########################
    # 在原始图像上可视化接触点
    if contact_point_2d is not None:
        vis_img = color_image.copy()
        u, v = int(contact_point_2d[0]), int(contact_point_2d[1])
        # 绘制点
        cv2.circle(vis_img, (u, v), 8, (0, 255, 0), -1)  # 绿色点
        # 绘制文字标注
        cv2.putText(vis_img, "Contact Point", (u+10, v), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 保存可视化结果
        vis_path = os.path.join(save_root, "initial_contact_point.png")
        cv2.imwrite(vis_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
        print(f"✅ 保存初始接触点可视化图像到 {vis_path}")
    else:
        print("⚠️ 未检测到接触点，跳过可视化")
    
    # 使用检测到的碰撞点作为点提示
    if contact_point_2d is None:
        print("警告：未检测到碰撞点，使用默认点提示")
        contact_point_2d = [320,240]  # 默认点
    
    # 使用点提示分割目标物体
    print(f"使用点提示 {contact_point_2d} 分割目标物体...")
    tgt_masks = inference_one_image(
        np.array(tgt_img_PIL), 
        grounded_dino_model, 
        sam_predictor, 
        box_threshold=None,
        text_threshold=None,
        text_prompt=None,
        device="cuda",
        point_prompt=contact_point_2d
    ).cpu().numpy()
    
    tgt_mask = np.repeat(tgt_masks[0,0][:, :, np.newaxis], 3, axis=2).astype(np.uint8)
    tgt_img_masked = np.array(tgt_img_PIL) * tgt_mask + 255 * (1 - tgt_mask)
    tgt_img_PIL = Image.fromarray(tgt_img_masked).convert('RGB')
    tgt_img_PIL.save(f"{save_root}/tgt_img_masked.png")

    white_img = np.ones((480, 640, 3), dtype=np.uint8) * 255

    # 转为 PIL 图像并保存
    white_img_pil = Image.fromarray(white_img)
    white_img_pil.save("work_space_mask.png")
    # 创建查询掩码
    query_mask = (tgt_mask[:, :, 0] > 0).astype(np.uint8) * 255
    
    ####################### SOURCE DEMONSTRATION ########################
    if not use_retrieve:
        # 不使用检索，直接加载演示数据
        data_dict = np.load("run_realworld/real_data/demonstration/data.pkl", allow_pickle=True)
        traj = data_dict['traj']
        src_img_np = data_dict['masked_img']
        src_img_PIL = Image.fromarray(src_img_np).convert('RGB')
        src_img_PIL.save(f"{save_root}/src_img.png")
        mask = None
        direction = None
    else:
        # 使用检索获取源演示
        print("检索源演示数据...")
        _, top1_retrieved_data_dict = subset_retrieve_pipeline.retrieve(
            cfgs['instruction'], 
            np.array(tgt_img_PIL)
        )
        
        traj = top1_retrieved_data_dict['traj']
        src_img_np = top1_retrieved_data_dict['masked_img']
        src_img_PIL = Image.fromarray(src_img_np).convert('RGB')
        mask = top1_retrieved_data_dict['mask']
        direction = top1_retrieved_data_dict['direction']
        print(f"检索到的方向: {repr(direction)}")
        print(f"检索到的描述: {top1_retrieved_data_dict['caption']}")

    # 可视化源图像和轨迹
    visualize_mask_and_trajectory(src_img_np, mask, traj, save_path=os.path.join(save_root, "src_img_traj.png"))
    
    # 缩放轨迹到IMG_SIZE
    tgt_mask = tgt_masks[0, 0]
    src_pos_list = []
    for xy in traj:
        src_pos_list.append((xy[0], xy[1]))
    
    # Affordance Transfer
    print("执行Affordance Transfer...")
    contact_points_traj, post_contact_dir = transfer_affordance(
        src_img_PIL,
        tgt_img_PIL,
        cfgs['prompt'],
        src_pos_list,
        save_root=save_root,
        ftype='sd',
        src_mask=mask,
        tgt_mask=tgt_mask
    )
    print(f"转换后的接触点: {contact_points_traj}")
    ########################## 3D 投影部分 ##########################
    print("执行3D投影...")
    object_mask = query_mask
    object_mask_resized = np.array(Image.fromarray(object_mask).resize(
        (depth_image.shape[1], depth_image.shape[0]), 
        resample=Image.NEAREST
    ))
    
    # 处理像素点
    points_3d_list = []
    adjusted_pixel_list = []

    for i, (u, v) in enumerate(contact_points_traj):
        u = int(round(u))
        v = int(round(v))
        valid_point = True

        if not (0 <= u < color_image.shape[1] and 0 <= v < color_image.shape[0]):
            print(f"[Info] Point ({u},{v}) out of bounds, searching nearest valid point on object...")
            valid_point = False
        elif depth_image_mm[v, u] == 0 or object_mask_resized[v, u] == 0:
            print(f"[Info] Depth at ({u},{v}) is 0, searching nearest valid point on object...")
            valid_point = False

        if not valid_point:
            # 搜索有效点
            if object_mask_resized.ndim == 3:
                object_mask_resized = object_mask_resized[:, :, 0]
            valid_mask = (depth_image_mm > 0) & (object_mask_resized > 0)
            valid_coords = np.argwhere(valid_mask)
            if valid_coords.shape[0] == 0:
                print(f"[Error] No valid depth points on object for point {i}. Skipping.")
                continue
            distances = cdist(np.array([[v, u]]), valid_coords)
            nearest_idx = np.argmin(distances)
            v_alt, u_alt = valid_coords[nearest_idx]
            z = depth_image[v_alt, u_alt]  # 注意：深度图已转换为米
            u, v = int(u_alt), int(v_alt)
            print(f"[Info] Replaced with ({u},{v}), depth = {z:.3f} m")
        else:
            z = depth_image[v, u]  # 深度单位为米

        # 投影到相机坐标系
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        points_3d_list.append([x, y, z])
        adjusted_pixel_list.append([u, v])

    # 处理方向
    if isinstance(direction, list):
        if all(isinstance(v, (int, float)) for v in direction) and len(direction) == 4:
            direction = [direction] * len(points_3d_list)
        elif all(isinstance(v, list) and len(v) == 4 for v in direction):
            assert len(direction) == len(points_3d_list), "四元数数量与接触点不一致"
        else:
            raise ValueError("direction格式非法")
    else:
        raise TypeError(f"direction类型非法: {type(direction)}")

    seven_dof_list = []
    for i in range(len(points_3d_list)):
        xyz = points_3d_list[i]
        quat = direction[i]  # [qx, qy, qz, qw]
        pose_7d = xyz + quat
        seven_dof_list.append(pose_7d)
    # ---------------- 保存JSON ----------------
    seven_dof_list_cleaned = [[float(x) for x in pose] for pose in seven_dof_list]
    adjusted_pixel_list_cleaned = [[int(x) for x in pt] for pt in adjusted_pixel_list]

    os.makedirs(save_root, exist_ok=True)
    
    path = os.path.join(save_root, "contact_points_3d.json")
    with open(path, "w") as f:
        json.dump(seven_dof_list_cleaned, f, indent=2)
        print(f"✅ contact_points_3d 已保存到: {path}")
    with open(os.path.join(save_root, "contact_points_adjusted.json"), "w") as f:
        json.dump(adjusted_pixel_list_cleaned, f, indent=2)

    print(f"✅ 保存3D接触点到 {save_root}/contact_points_3d.json")
    print(f"✅ 保存调整后的像素坐标到 {save_root}/contact_points_adjusted.json")
    print("====== 处理完成 ======")

    # ---------------- 变换为真实世界坐标 ----------------
    pose_7d = seven_dof_list[0]
    position = seven_tuple_to_transform(pose_7d)

    T_left = np.array([
        [-0.998944, 0.020487, -0.041116, 0.401694357],
        [0.042548, 0.075189, -0.996261, 0.802152163],
        [ -0.017319, -0.996959, -0.075982, 0.379463994],
        [0, 0, 0, 1]
    ])
    R_z_minus_90 = np.array([
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 1]
    ])
    T_z_rot = np.eye(4)
    T_z_rot[:3, :3] = R_z_minus_90

    position = T_left @ position @ T_z_rot  # ← 加这一项影响的是 gripper 自身绕 Z 轴的旋转
    position = torch.from_numpy(position).float()
    
    # 转换为7DOF姿态
    result = matrix_to_pose7(position)
    
    # 保存结果
    json_path = os.path.join(save_root, "grasp_pose.json")
    with open(json_path, "w") as f:
        json.dump(result.tolist(), f, indent=4)
    
    print(f"✅ 抓取姿态已保存到: {json_path}")
    return result

def seven_tuple_to_transform(pose_7d):
    x, y, z, qx, qy, qz, qw = pose_7d
    R_mat = R.from_quat([qx, qy, qz, qw]).as_matrix()  # [3, 3]
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = [x, y, z]
    return T

def matrix_to_pose7(matrix):
    """将 4x4 齐次变换矩阵转为 [x, y, z, qx, qy, qz, qw] 七元组"""
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.cpu().numpy()

    trans = matrix[:3, 3]
    quat = R.from_matrix(matrix[:3, :3]).as_quat()  # x, y, z, w
    return np.concatenate([trans, quat])

def main_loop(cfgs, save_root_base):
    """主循环处理函数"""
    global grounded_dino_model, sam_predictor, subset_retrieve_pipeline
    
    print("\n===============================")
    print("系统初始化完成，等待用户输入...")
    print("按 's' 开始捕获图像并处理")
    print("按 'q' 退出程序")
    print("===============================\n")
    
    while True:
        user_input = input("> ").strip().lower()
        
        if user_input == 'q':
            print("退出程序")
            break
            
        if user_input != 's':
            print("无效输入，请按 's' 或 'q'")
            continue
            
        save_root = get_next_indexed_folder(save_root_base)
        print(f"✅ 创建结果目录: {save_root}")
        
        # 步骤1: 捕获图像并检测碰撞点
        print("\n===== 步骤1: 捕获图像并检测手部碰撞点 =====")
        color_img, depth_img, fx, fy, cx, cy, contact_point = capture_and_detect(save_root)

        '''# 如果未检测到接触点，自动删除该轮文件夹并跳过
        if contact_point is None:
            print("❌ 未检测到接触点，删除此次结果目录并跳过")
            try:
                import shutil
                shutil.rmtree(save_root)
                print(f"🗑️ 已删除目录: {save_root}")
            except Exception as e:
                print(f"⚠️ 删除失败: {e}")
            continue'''

        # 保存原始数据用于调试
        Image.fromarray(color_img).save(f"{save_root}/captured_color.png")
        Image.fromarray(depth_img).save(f"{save_root}/captured_depth.png")
        print(f"✅ 保存原始彩色图像到: {save_root}/captured_color.png")
        print(f"✅ 保存原始深度图像到: {save_root}/captured_depth.png")
        
        # 步骤2: 处理affordance transfer和3D投影
        print("\n===== 步骤2: Affordance Transfer 和 3D投影 =====")
        result = process_affordance(
            color_image=color_img,
            depth_image_mm=depth_img,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            contact_point_2d=contact_point,
            cfgs=cfgs,
            save_root=save_root,
            use_retrieve=True,
            grounded_dino_model=grounded_dino_model,
            sam_predictor=sam_predictor,
            subset_retrieve_pipeline=subset_retrieve_pipeline
        )

        print("\n===== 最终结果 =====")
        print(f"抓取姿态 (7DOF): {result}")
        print(f"本次处理结果已保存到: {save_root}")
        print("=" * 50)
        print("等待下一次输入...")

def main():
    """主函数"""
    # 配置参数
    config_path = "configs/object_grasp.yaml" 
    save_root_base = "run_realworld/gym_outputs/object_grasp"
    os.makedirs(save_root_base, exist_ok=True)
    
    # 初始化模型和配置
    print("===== 初始化模型 =====")
    cfgs = init_models(config_path, save_root_base)
    
    # 进入主循环
    main_loop(cfgs, save_root_base)

if __name__ == "__main__":
    main()