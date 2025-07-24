#python run_realworld/run.py
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
os.environ["PYTHONPATH"] = root_dir
sys.path.insert(0, root_dir)
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
from run_realworld.env import MiniEnv
import numpy as np
from run_realworld.utils import read_yaml_config
from vision.GroundedSAM.grounded_sam_utils import prepare_gsam_model, inference_one_image
import torch
from PIL import Image
import glob, time
from vision.featurizer.run_featurizer import transfer_affordance,transfer_multiple_contacts
from vision.featurizer.utils.visualization import IMG_SIZE
from subset_retrieval.subset_retrieve_pipeline_once import SubsetRetrievePipeline,visualize_mask_and_trajectory
import argparse
import traceback
import matplotlib
matplotlib.use('svg') # NOTE: fix backend error while GPU is in use
from tqdm import tqdm
import shutil
import random
import open3d as o3d
import json
from scipy.spatial.distance import cdist
import time
from vision.featurizer import SDFeaturizer, DINOFeaturizer, CLIPFeaturizer, DINOv2Featurizer, RADIOFeaturizer, SD_DINOv2Featurizer

def backup(args, cfgs):
    shutil.copyfile(f"run_realworld/{args.config}", f"{cfgs['SAVE_ROOT']}/config.yaml")

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    cfgs = read_yaml_config(f"run_realworld/{args.config}")
    os.makedirs(cfgs['SAVE_ROOT'], exist_ok=True)
    backup(args, cfgs)
    torch.set_printoptions(precision=4, sci_mode=False)
            
    instruction = cfgs['instruction']
    obj = cfgs['obj']
    prompt = cfgs['prompt']
    data_source = cfgs.get("DATA_SOURCE", "new_data")
    save_root = cfgs['SAVE_ROOT']
    
    grounded_dino_model, sam_predictor = prepare_gsam_model(device="cuda")
    gym = MiniEnv(cfgs, grounded_dino_model, sam_predictor)
    
    subset_retrieve_pipeline = SubsetRetrievePipeline(
        subset_dir="assets/data",
        save_root=save_root,
        lang_mode='clip',
        topk=5, 
        crop=False,
        data_source=data_source,
    )
    
    rgb = Image.open("database_test/milk_carton/no_hand/rgb_milk_carton2.png")
    #rgb = Image.open("database_test/box/no_hand/rgb_box1.png")

    tgt_img_PIL = rgb
    tgt_img_PIL.save(f"{save_root}/tgt_img.png")
    rgb = np.array(rgb)
    
    #tgt_masks = inference_one_image(np.array(tgt_img_PIL), grounded_dino_model, sam_predictor, box_threshold=cfgs['box_threshold'], text_threshold=cfgs['text_threshold'], text_prompt=obj, device="cuda").cpu().numpy() # you can set point_prompt to traj[0]
    point_prompt = [291,445]  # 指定目标物体的一个点坐标（例如网球的中心点）
    #point_prompt = [320,240]  # 指定目标物体的一个点坐标（例如网球的中心点）
    tgt_masks = inference_one_image(
        np.array(tgt_img_PIL), 
        grounded_dino_model, 
        sam_predictor, 
        box_threshold=0.3,    # 点提示不需要检测框
        text_threshold=0.25,   # 点提示不需要文本阈值
        text_prompt="object",      # 禁用文本提示
        device="cuda",
        #point_prompt=point_prompt  # 传入点坐标
        point_prompt=None
    ).cpu().numpy()
    tgt_mask = np.repeat(tgt_masks[0,0][:, :, np.newaxis], 3, axis=2).astype(np.uint8)
    # if mask is false, make it white
    tgt_img_masked = np.array(tgt_img_PIL) * tgt_mask + 255 * (1 - tgt_mask)
    # tgt_img_masked, _, _ = crop_image(tgt_img_masked, tgt_mask)
    tgt_img_PIL = Image.fromarray(tgt_img_masked).convert('RGB')
    tgt_img_PIL.save(f"{save_root}/tgt_img_masked.png")
    ######## src
    ####################### SOURCE DEMONSTRATION ########################
    if not args.retrieve:
        data_dict = np.load("run_realworld/real_data/demonstration/data.pkl", allow_pickle=True)
        traj = data_dict['traj']
        src_img_np = data_dict['masked_img']
        src_img_PIL = Image.fromarray(src_img_np).convert('RGB')
        src_img_PIL.save(f"{save_root}/src_img.png")
        mask=None
        direction=None
    else:
        # use retrieval to get src_path (or src image) and src trajectory in 2d space
        _, top1_retrieved_data_dict = subset_retrieve_pipeline.retrieve(instruction, np.array(tgt_img_PIL))
        traj = top1_retrieved_data_dict['traj']
        src_img_np = top1_retrieved_data_dict['masked_img']
        src_img_PIL = Image.fromarray(src_img_np).convert('RGB')
        mask = top1_retrieved_data_dict['mask']
        query_mask = top1_retrieved_data_dict['query_mask']
        direction = top1_retrieved_data_dict['direction']
        print(repr(direction))
        print(top1_retrieved_data_dict["caption"])
    #visualize src image and trajectory
    visualize_mask_and_trajectory(src_img_np,mask, traj, save_path=os.path.join(save_root, "src_img_traj.png"))
    ####################### SOURCE DEMONSTRATION ########################

    featurizers = {
    'sd': SDFeaturizer,
    'clip': CLIPFeaturizer,
    'dino': DINOFeaturizer,
    'dinov2': DINOv2Featurizer,
    'radio': RADIOFeaturizer,
    'sd_dinov2': SD_DINOv2Featurizer
}
    # scale cropped_traj to IMG_SIZE
    tgt_mask = tgt_masks[0, 0]
    src_pos_list = []
    for xy in traj:
        src_pos_list.append((xy[0], xy[1]))
    
    while True:
        try:
            contact_points_traj, post_contact_dir = transfer_affordance(
    src_img_PIL,
    tgt_img_PIL,
    prompt,
    src_pos_list,
    save_root=save_root,
    ftype='sd',
    src_mask=mask,
    tgt_mask=tgt_mask,featurizer=featurizers['sd']()
)
            # contact_point, sim_scores = transfer_multiple_contacts(src_img_PIL, tgt_img_PIL, prompt, src_pos_list, save_root=save_root, ftype='sd')
            print("contact_point:", contact_points_traj)
            break
        except Exception as transfer_e:
            traceback.print_exc()
            print('[ERROR] in transfer_affordance:', transfer_e)

    #visualize_mask_and_trajectory(tgt_img_masked,query_mask, contact_points_traj, save_path=os.path.join(save_root, "rgb_coke_with_traj.png"))
    # contact point + post-contact direction
    # ret_dict = gym.lift_affordance(rgb, pcd, contact_point, post_contact_dir)
    
    # print("3D Affordance:\n", ret_dict)
    ########################## 3D 投影部分 ##########################

    # Step 1: 加载深度图（假设路径如下，单位为 mm，需除以1000转换为 m）
    '''depth = np.array(Image.open("database_test/box/no_hand/depth_box1.png")).astype(np.float32) / 1000.0

    # Step 2: 读取相机内参
    with open("database_test/box/no_hand/camera_params1.json", "r") as f:
        cam_intrinsics = json.load(f)
    fx = cam_intrinsics["fx"]
    fy = cam_intrinsics["fy"]
    cx = cam_intrinsics["cx"]
    cy = cam_intrinsics["cy"]
    width = cam_intrinsics["width"]
    height = cam_intrinsics["height"]

    object_mask = query_mask.astype(np.uint8)
    object_mask_resized = np.array(Image.fromarray(object_mask).resize((depth.shape[1], depth.shape[0]), resample=Image.NEAREST))
    start_time = time.time()
    # Step 4: 处理像素点
    points_3d_list = []
    adjusted_pixel_list = []

    for i, (u, v) in enumerate(contact_points_traj):
        u = int(round(u))
        v = int(round(v))
        valid_point = True

        if not (0 <= u < width and 0 <= v < height):
            print(f"[Info] Point ({u},{v}) out of bounds, searching nearest valid point on object...")
            valid_point = False
        elif depth[v, u] == 0 or object_mask_resized[v, u,0] == 0:
            print(f"[Info] Depth at ({u},{v}) is 0, searching nearest valid point on object...")
            valid_point = False

        if not valid_point:
            # 搜索 depth > 0 且在 object_mask 上的有效点
            # 若尚未转为单通道，请先加这一句
            if object_mask_resized.ndim == 3:
                object_mask_resized = object_mask_resized[:, :, 0]
            valid_mask = (depth > 0) & (object_mask_resized > 0)
            valid_coords = np.argwhere(valid_mask)
            if valid_coords.shape[0] == 0:
                print(f"[Error] No valid depth points on object for point {i}. Skipping.")
                continue
            distances = cdist(np.array([[v, u]]), valid_coords)
            nearest_idx = np.argmin(distances)
            v_alt, u_alt = valid_coords[nearest_idx]
            z = depth[v_alt, u_alt]
            u, v = int(u_alt), int(v_alt)
            print(f"[Info] Replaced with ({u},{v}), depth = {z:.3f} m")
        else:
            z = depth[v, u]

        # 投影到相机坐标系
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        points_3d_list.append([x, y, z])
        adjusted_pixel_list.append([u, v])

    # Step 6: 处理 direction（适配单个方向 or 多个方向）
    if isinstance(direction, list):
        if all(isinstance(v, (int, float)) for v in direction) and len(direction) == 3:
            direction = [direction] * len(points_3d_list)
        elif all(isinstance(v, list) and len(v) == 3 for v in direction):
            assert len(direction) == len(points_3d_list), "direction list size mismatch"
        else:
            raise ValueError(f"Invalid direction format: {direction}")
    else:
        raise TypeError(f"Invalid direction type: {type(direction)}")

    # Step 7: 合并为 6D 向量 [x, y, z, dx, dy, dz]
    six_dof_list = []
    for i in range(len(points_3d_list)):
        p3d = points_3d_list[i]
        dir_vec = direction[i]
        six_dof = list(p3d) + list(dir_vec)  # 拼接
        six_dof_list.append(six_dof)

    # Step 8: 保存结果
    six_dof_list_py = [[float(x) for x in six_dof] for six_dof in six_dof_list]
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"⏱️ 后处理耗时: {elapsed_time:.4f} 秒")
    with open(os.path.join(save_root, "contact_points_3d.json"), "w") as f:
        json.dump(six_dof_list_py, f, indent=2)
    with open(os.path.join(save_root, "contact_points_adjusted.json"), "w") as f:
        json.dump(adjusted_pixel_list, f, indent=2)

    print(f"✅ Saved 3D contact points with direction to contact_points_3d.json")
    print(f"✅ Saved adjusted pixel coordinates to contact_points_adjusted.json")'''
    print("====== DONE ======")
        

'''if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to the config file') # e.g. configs/drawer_open.yaml
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--retrieve', action='store_true')
    args = parser.parse_args()
    
    main(args)'''
if __name__ == "__main__":
    config_path = "configs/object_grasp.yaml" 
    use_retrieve = True

    from types import SimpleNamespace
    args = SimpleNamespace(
        config=config_path,
        seed=100,
        retrieve=use_retrieve
    )

    main(args)
