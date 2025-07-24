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
from subset_retrieval.subset_retrieve_pipeline_once import SubsetRetrievePipeline, visualize_mask_and_trajectory

# 第一部分：图像捕获和碰撞点检测
def capture_and_detect():
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

    '''def detect_hand_with_mediapipe(color_image):
        """使用MediaPipe检测手部"""
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                               min_detection_confidence=0.5, min_tracking_confidence=0.2)
        results = hands.process(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
        mask = np.zeros(color_image.shape[:2], dtype=np.uint8)
        landmarks_list = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = [(min(max(lm.x*color_image.shape[1], 0), color_image.shape[1]-1),
                              min(max(lm.y*color_image.shape[0], 0), color_image.shape[0]-1))
                             for lm in hand_landmarks.landmark]
                landmarks_list.append(landmarks)
                x_coords = [p[0] for p in landmarks]
                y_coords = [p[1] for p in landmarks]
                box_width = max(x_coords) - min(x_coords)
                box_height = max(y_coords) - min(y_coords)
                x_min = max(0, int(min(x_coords) - box_width*0.1))
                y_min = max(0, int(min(y_coords) - box_height*0.1))
                x_max = min(color_image.shape[1]-1, int(max(x_coords) + box_width*0.1))
                y_max = min(color_image.shape[0]-1, int(max(y_coords) + box_height*0.1))
                cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 255, -1)
        segmented = cv2.bitwise_and(color_image, color_image, mask=mask)
        return mask, segmented, landmarks_list'''
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
    
    return original_color, depth_image, fx, fy, cx, cy, contact_point_2d

# 第二部分：Affordance Transfer和3D投影
def process_affordance(color_image, depth_image_mm, fx, fy, cx, cy, contact_point_2d, cfgs, save_root, use_retrieve=True):
    """处理affordance transfer和3D投影"""
    # 设置随机种子
    random.seed(cfgs.get('seed', 100))
    np.random.seed(cfgs.get('seed', 100))
    torch.manual_seed(cfgs.get('seed', 100))
    torch.cuda.manual_seed(cfgs.get('seed', 100))
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    # 准备模型
    print("准备Grounded-SAM模型...")
    grounded_dino_model, sam_predictor = prepare_gsam_model(device="cuda")
    
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
        subset_retrieve_pipeline = SubsetRetrievePipeline(
            subset_dir="assets/data",
            save_root=save_root,
            lang_mode='clip',
            topk=5, 
            crop=False,
            data_source=cfgs.get("DATA_SOURCE", "new_data"),
        )
        
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
    src_pos_list = []
    for xy in traj:
        src_pos_list.append((xy[0] * IMG_SIZE / src_img_PIL.size[0], xy[1] * IMG_SIZE / src_img_PIL.size[1]))
    
    # Affordance Transfer
    print("执行Affordance Transfer...")
    contact_points_traj, post_contact_dir = transfer_affordance(
        src_img_PIL, tgt_img_PIL, cfgs['prompt'], src_pos_list, 
        save_root=save_root, ftype='sd'
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
        if all(isinstance(v, (int, float)) for v in direction) and len(direction) == 3:
            direction = [direction] * len(points_3d_list)
        elif all(isinstance(v, list) and len(v) == 3 for v in direction):
            assert len(direction) == len(points_3d_list), "direction list size mismatch"
        else:
            raise ValueError(f"Invalid direction format: {direction}")
    else:
        raise TypeError(f"Invalid direction type: {type(direction)}")

    # 合并为6D向量 [x, y, z, dx, dy, dz]
    six_dof_list = []
    for i in range(len(points_3d_list)):
        p3d = points_3d_list[i]
        dir_vec = direction[i]
        six_dof = list(p3d) + list(dir_vec)
        six_dof_list.append(six_dof)

    # 保存结果
    six_dof_list_py = [[float(x) for x in six_dof] for six_dof in six_dof_list]
    position=six_tuple_to_transform(six_dof_list_py[0])
    T_left = np.array([
    [ 0.98498,  -0.070744, 0.157509,  -0.039546093],
    [-0.160834,  -0.044025, 0.985999,  -0.812045048],
    [-0.062819,  -0.996522,  -0.054742, 0.464458135],
    [ 0,  0,  0,  1]
])
    #T_left=np.linalg.inv(T_left)
    position=T_left@position
    position = torch.from_numpy(position).float()
    with open(os.path.join(save_root, "contact_points_3d.json"), "w") as f:
        json.dump(six_dof_list_py, f, indent=2)
    with open(os.path.join(save_root, "contact_points_adjusted.json"), "w") as f:
        json.dump(adjusted_pixel_list, f, indent=2)

    print(f"✅ 保存3D接触点到 {save_root}/contact_points_3d.json")
    print(f"✅ 保存调整后的像素坐标到 {save_root}/contact_points_adjusted.json")
    print("====== 处理完成 ======")
    return position
def six_tuple_to_transform(six_tuple):
    x, y, z, dx, dy, dz = six_tuple

    z_axis = np.array([dx, dy, dz], dtype=np.float64)
    z_axis /= np.linalg.norm(z_axis)

    # 构造正交张开方向 x_axis（张开方向）
    up = np.array([0, 1, 0], dtype=np.float64)  # 优先从 y 构造
    if abs(np.dot(z_axis, up)) > 0.95:
        up = np.array([1, 0, 0], dtype=np.float64)

    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    R = np.column_stack([x_axis, y_axis, z_axis])  # R = [x y z]

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T

def matrix_to_pose7(matrix):
    """将 4x4 齐次变换矩阵转为 [x, y, z, qx, qy, qz, qw] 七元组"""
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.cpu().numpy()

    trans = matrix[:3, 3]
    quat = R.from_matrix(matrix[:3, :3]).as_quat()  # x, y, z, w
    return np.concatenate([trans, quat])
def main():
    """主函数"""
    # 配置参数
    config_path = "configs/object_grasp.yaml" 
    use_retrieve = True
    save_root = "run_realworld/gym_outputs/object_grasp"
    os.makedirs(save_root, exist_ok=True)
    
    # 读取配置文件
    cfgs = read_yaml_config(f"run_realworld/{config_path}")
    
    # 步骤1: 捕获图像并检测碰撞点
    color_img, depth_img, fx, fy, cx, cy, contact_point = capture_and_detect()
    
    # 保存原始数据用于调试
    Image.fromarray(color_img).save(f"{save_root}/captured_color.png")
    Image.fromarray(depth_img).save(f"{save_root}/captured_depth.png")
    
    # 步骤2: 处理affordance transfer和3D投影
    result=process_affordance(
        color_image=color_img,
        depth_image_mm=depth_img,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        contact_point_2d=contact_point,
        cfgs=cfgs,
        save_root=save_root,
        use_retrieve=use_retrieve
    )
    result = matrix_to_pose7(result)
    print(result)
    json_path = os.path.join(save_root, "grasp_pose.json")
    with open(json_path, "w") as f:
        json.dump(result.tolist(), f, indent=4)
if __name__ == "__main__":
    main()