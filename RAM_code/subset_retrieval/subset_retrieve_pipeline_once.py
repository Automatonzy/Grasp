import os
from subset_retrieval.gpt_core.chatbot import Chatbot
import json
import numpy as np
from vision.GroundedSAM.grounded_sam_utils import prepare_gsam_model, inference_one_image, crop_image
from PIL import Image, ImageDraw, ImageFont
from vision.clip_encoder import ClipModel
import pickle
import torch
import glob
import cv2
import traceback
from vision.featurizer.run_featurizer import extract_ft, match_fts, sample_highest
from vision.featurizer.utils.correspondence import get_distance_bbnn, get_distance_imd
from einops import rearrange
from tqdm import tqdm
from vision.featurizer.run_featurizer import featurizers
import time

MAX_IMD_RANKING_NUM = 30 # change this for different levels of efficiency

def visualize_mask_and_trajectory(frame, mask, trajectory, save_path=None):
    """
    将掩码和轨迹点可视化在同一张图像上
    
    参数:
        frame: 原始图像 numpy数组 (H, W, 3)
        mask: 掩码 numpy数组 (H, W, 3) 或 (H, W, 1)
        trajectory: 轨迹点列表，每个点是(x, y)坐标
        save_path: 保存路径，如果为None则显示图像
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image, ImageDraw
    
    # 确保掩码是三通道的
    if mask.shape[-1] != 3:
        mask = np.repeat(mask, 3, axis=-1)
    
    # 创建可视化图像
    # 方法1：掩码轮廓叠加在原始图像上
    mask_binary = mask[..., 0] > 0  # 取第一个通道作为二值掩码
    
    # 找到掩码的轮廓
    import cv2
    contours, _ = cv2.findContours((mask_binary * 255).astype(np.uint8), 
                                   cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_SIMPLE)
    
    # 将原始图像转换为PIL图像以便绘制
    vis_img = Image.fromarray(frame.astype(np.uint8))
    draw = ImageDraw.Draw(vis_img)
    
    # 绘制掩码轮廓 - 绿色
    # for contour in contours:
    #     for i in range(len(contour)):
    #         if i == len(contour) - 1:
    #             draw.line([tuple(contour[i][0]), tuple(contour[0][0])], 
    #                      fill=(0, 255, 0), width=2)
    #         else:
    #             draw.line([tuple(contour[i][0]), tuple(contour[i+1][0])], 
    #                      fill=(0, 255, 0), width=2)
    
    # 绘制轨迹点
    point_size = max(5, int(min(frame.shape[0], frame.shape[1]) / 100))
    
    # 首先绘制轨迹线 - 蓝色
    if len(trajectory) > 1:
        for i in range(len(trajectory)-1):
            p1 = tuple(map(int, trajectory[i]))
            p2 = tuple(map(int, trajectory[i+1]))
            draw.line([p1, p2], fill=(0, 0, 255), width=2)
    
    # 绘制所有轨迹点 - 蓝色
    for i, point in enumerate(trajectory):
        x, y = map(int, point)
        
        # 计算点所在位置的掩码值
        in_mask = False
        if 0 <= y < mask_binary.shape[0] and 0 <= x < mask_binary.shape[1]:
            in_mask = mask_binary[y, x]
        
        # 第一个点标红，其他点标蓝，掩码外的点用黄色边框
        if i == 0:  # 起始点 - 红色
            color = (255, 0, 0)
            # border_color = (255, 255, 0) if not in_mask else None
        else:  # 其他点 - 蓝色
            color = (0, 0, 255)
            # border_color = (255, 255, 0) if not in_mask else None
        
        # 绘制点
        bbox = [(x-point_size, y-point_size), (x+point_size, y+point_size)]
        draw.ellipse(bbox, fill=color)
        
        # 如果点不在掩码内，添加黄色边框警告
        # if border_color:
        #     draw.ellipse([(x-point_size-2, y-point_size-2), 
        #                   (x+point_size+2, y+point_size+2)], 
        #                  outline=border_color, width=2)
            
            # 添加文本标签指示不在掩码内
            # draw.text((x+point_size+5, y-point_size), 
            #          f"Point {i} OUT!", fill=(255, 255, 0))
    
    # # 修改这部分代码
    # legend_text = [
    #     "Green line: Mask boundary",
    #     "Red dot: Starting trajectory point",
    #     "Blue dots: Other trajectory points",
    #     "Yellow border: Points outside mask"
    # ]
    
    # for i, text in enumerate(legend_text):
    #     draw.text((10, 10 + i*20), text, fill=(255, 255, 255), 
    #              stroke_width=2, stroke_fill=(0, 0, 0))
    
    # 保存或显示图像
    if save_path:
        vis_img.save(save_path)
        print(f"可视化图像已保存到: {save_path}")
    else:
        plt.figure(figsize=(12, 10))
        plt.imshow(np.array(vis_img))
        plt.axis('off')
        plt.title("掩码和轨迹点可视化")
        plt.show()
    
    return vis_img

def segment_images(frames, trajs, text_prompt, grounded_dino_model, sam_predictor, box_threshold=0.3, text_threshold=0.25, device="cuda"):
    masked_frames = []
    frame_masks = []

    for idx, frame in enumerate(frames):
        masks = inference_one_image(frame, grounded_dino_model, sam_predictor, box_threshold=box_threshold, text_threshold=text_threshold, text_prompt=text_prompt, device=device,point_prompt=None).cpu().numpy()
        mask = np.repeat(masks[0,0][:, :, np.newaxis], 3, axis=2).astype(np.uint8)
        
        # # save the mask and the trajs to a file
        # with open(f"{text_prompt}_mask.json", 'w') as f:
        #     json.dump({"mask": mask.tolist(), "trajs": trajs[idx].tolist()}, f)
        # if no object is detected or the contact point is not in the mask, return the original image
        if mask.sum() == 0:
            print("Mask set all to 1 because sum to 1")
            mask = np.ones_like(mask)
        else:
            # 对掩码进行膨胀操作扩展边界
            import cv2
            kernel_size = 5  # 可以根据需要调整大小
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            
            # 创建临时掩码副本进行膨胀
            dilated_mask = np.zeros_like(mask)
            for c in range(mask.shape[2]):
                dilated_mask[:,:,c] = cv2.dilate(mask[:,:,c], kernel, iterations=1)
            
            # 用膨胀后的掩码替换原掩码
            mask = dilated_mask
        
        if trajs is not None:
            # masked_frame = frame * mask + 255 * (1 - mask)
            '''vis_img = visualize_mask_and_trajectory(
            frame, 
            mask, 
            trajs[idx], 
            save_path=f"masked_{text_prompt}_traj_mask_debug.png"
        )'''
            cp = trajs[idx][0]
            # if mask[...,0][int(cp[1]), int(cp[0])] == 0:
            #     print("Mask set all to 1 because traj not in mask")
            #     mask = np.ones_like(mask)

        masked_frame = frame * mask + 255 * (1 - mask)
        masked_frames.append(masked_frame)
        frame_masks.append(mask)

        '''if True:
            tgt_img_PIL = Image.fromarray(masked_frame).convert('RGB')
            tgt_img_PIL.save(f"{text_prompt}_tgt_img_masked.png")'''

    return masked_frames, frame_masks

def concat_images_with_lines(img_list):
    # Assume all images are the same size for simplicity
    img_width, img_height = img_list[0].size
    
    # Create a new image with a white background
    total_width = img_width * 3
    total_height = img_height * 2
    new_img = Image.new('RGB', (total_width, total_height), 'white')
    
    # Font for numbering images
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=int(img_height/10))
    except IOError as ioe:
        print('=> [ERROR] in font:', ioe)
        font = ImageFont.load_default()
    
    # Draw instance to draw lines and text
    draw = ImageDraw.Draw(new_img)
    
    color_set = ['red', 'green', 'blue', 'brown', 'purple', 'orange']

    # Place images and draw red lines and numbers
    for index, img in enumerate(img_list):
        img = img.resize((img_width, img_height), Image.BILINEAR)
        x = index % 3 * img_width
        y = index // 3 * img_height
        new_img.paste(img, (x, y))
        number = str(index)
        if index == 0:
            number = 'src'
        draw.text((x + img_width/20, y + img_height/20), number, font=font, fill=color_set[index])
    
    # Draw red lines
    for i in range(1, 3):
        draw.line([(i * img_width, 0), (i * img_width, total_height)], fill='red', width=5)
    for i in range(1, 2):
        draw.line([(0, i * img_height), (total_width, i * img_height)], fill='red', width=5)
    
    return new_img


def crop_images(masked_frames, masks, trajs=None):
    cropped_frames = []
    cropped_masks = []
    cropped_trajs = []
    for idx in range(len(masked_frames)):
        masked_frame = masked_frames[idx]
        mask = masks[idx]
        traj = trajs[idx] if trajs is not None else []
        cropped_image, cropped_mask, cropped_traj = crop_image(masked_frame, mask, traj, margin=100)
        cropped_frames.append(cropped_image)
        cropped_masks.append(cropped_mask)
        cropped_trajs.append(cropped_traj)
        
    return cropped_frames, cropped_masks, cropped_trajs

def get_end_start_direction(trajs):
    dirs = []
    for traj in trajs:
        end_start_dir = traj[-1] - traj[0]
        dirs.append(end_start_dir)
    return dirs

class SubsetRetrievePipeline:
    def __init__(self, subset_dir, save_root='./', topk=5, lang_mode='clip', crop=False, data_source=None) -> None:
        self.subset_dir = subset_dir
        self.save_root = save_root
        self.topk = topk
        self.sd_featurizer = featurizers['sd']()
        self.crop = crop
        self.last_retrieved_pkl_path = None
        self.grounded_dino_model, self.sam_predictor = prepare_gsam_model(device="cuda")
        self.clip_model = ClipModel()

        # 1. construct a task list
        #task_list_droid = os.listdir(os.path.join(subset_dir, "droid"))
        #task_list_hoi4d = os.listdir(os.path.join(subset_dir, "HOI4D"))
        #task_list_customize = os.listdir(os.path.join(subset_dir, "customize"))
        task_list_ours_bottle = os.listdir(os.path.join(subset_dir, "database")) #后面会改成shape_list，现在为了一致性先用task_list
        #if data_source == "droid":
            #task_list = task_list_droid
        #elif data_source == "HOI4D":
            #task_list = task_list_hoi4d
        #elif data_source == "customize":
            #task_list = task_list_customize
        #elif 
        # data_source == "new_data":
        task_list = task_list_ours_bottle


        self.data_source = data_source

        self.task_list = [task.replace("_", " ") for task in task_list]

        self.lang_mode = lang_mode
        if lang_mode == 'gpt':
            self.gpt_chatbot = Chatbot(api_key=None) # NEED TO FILL IN YOUR API KEY LIKE 'sk-...'
        elif lang_mode == 'clip':
            pass
        else:
            raise ValueError("lang_mode should be 'gpt' or 'clip'")

    def language_retrieve(self, current_task):
        if self.lang_mode == 'gpt':
            retrieved_task = self.gpt_chatbot.task_retrieval(self.task_list, current_task)
        elif self.lang_mode == 'clip':
            retrieved_task = self.clip_lang_retrieve(current_task)

        return retrieved_task
    
    def clip_lang_retrieve(self, current_task):
        task_embeddings = []
        for task in self.task_list:
            task_embeddings.append(self.clip_model.get_text_feature(task))

        current_task_embedding = self.clip_model.get_text_feature(current_task)

        similarity = self.clip_model.compute_similarity(current_task_embedding, torch.cat(task_embeddings, dim=0))

        max_index = similarity.argmax()

        return self.task_list[max_index]

    def segment_objects(self, retrieved_data_imgs, obj_name, trajs=None):
        print(f"obj name:{obj_name}\ntrajs:{trajs}")
        masked_frames, masks = segment_images(retrieved_data_imgs, trajs, obj_name, self.grounded_dino_model, self.sam_predictor)
        if trajs is not None:
            print(f"trajs are not none:{trajs}")
            test_ori_img= masked_frames[0]
            test_ori_mask= masks[0]
            test_traj= trajs[0]
            #visualize_mask_and_trajectory(test_ori_img,test_ori_mask,test_traj, save_path=os.path.join(self.save_root, "second_retrieved_img_traj.png"))
        self.crop = False
        if self.crop:
            masked_frames, masks, trajs = crop_images(masked_frames, masks, trajs)

        if len(masked_frames) == 1 and len(masks) == 1: # observation
            return masked_frames[0], masks[0]
        return masked_frames, masks, trajs
    
    '''def clip_filtering(self, retrieved_data_dict, obj_prompt):
        query_frame = retrieved_data_dict['masked_query']
        retrieved_frames = retrieved_data_dict['masked_img']
        #text_feature = self.clip_model.get_text_feature(retrieved_data_dict["caption"][0])
        query_object = Image.fromarray(query_frame).convert('RGB')
        query_object_feature = self.clip_model.get_vision_feature(query_object)

        object_features = []
        for retrieved_frame in retrieved_frames:
            object_feature = self.clip_model.get_vision_feature(
                Image.fromarray(retrieved_frame).convert('RGB')
            )
            object_features.append(object_feature)

        visual_similarity = self.clip_model.compute_similarity(query_object_feature, torch.cat(object_features, dim=0))
        #text_similarity = self.clip_model.compute_similarity(text_feature, torch.cat(object_features, dim=0))
        joint_similarity = visual_similarity #* text_similarity

        sorted_index = joint_similarity.argsort()[0, ::-1]
        sorted_retrieved_data_dict = {
            "query_img": retrieved_data_dict["query_img"],
            "query_mask": retrieved_data_dict["query_mask"],
            "masked_query": retrieved_data_dict["masked_query"],
            "img": [],
            "traj": [],
            "masked_img": [],
            "mask": [],
            "caption":retrieved_data_dict.get("caption","Failed getting."),
            "direction":[]
        }
        print(f"A Sorted index:{len(sorted_index)}")
        for idx in range(len(sorted_index)):
            curr = sorted_index[idx]
            prev = sorted_index[idx-1] if idx > 0 else 0
            if (retrieved_data_dict["mask"][curr].all()): # mask is all 1, meaning no mask
                print(f"Continue")
                continue
            adaj_sim=self.clip_model.compute_similarity(object_features[curr], object_features[prev])
            print(f"Adajacent similarity:{adaj_sim}")#<0.8
            print(f"Joint similarity:{joint_similarity}")#>0.1
            # if ( adaj_sim and joint_similarity[0, curr] > 0.1) or idx == 0:
            if True:
                print("Adding a sample.")
                sorted_retrieved_data_dict["img"].append(retrieved_data_dict["img"][sorted_index[idx]])
                sorted_retrieved_data_dict["traj"].append(retrieved_data_dict["traj"][sorted_index[idx]])
                sorted_retrieved_data_dict["masked_img"].append(retrieved_data_dict['masked_img'][sorted_index[idx]])
                sorted_retrieved_data_dict["mask"].append(retrieved_data_dict['mask'][sorted_index[idx]])
                sorted_retrieved_data_dict["direction"].append(retrieved_data_dict['direction'][curr])
                
                if len(sorted_retrieved_data_dict["img"]) >= MAX_IMD_RANKING_NUM:
                    break

        return sorted_retrieved_data_dict'''
    def clip_filtering(self, retrieved_data_dict, obj_prompt):
        query_frame = retrieved_data_dict['masked_query']
        retrieved_frames = retrieved_data_dict['masked_img']
        query_object = Image.fromarray(query_frame).convert('RGB')
        query_object_feature = self.clip_model.get_vision_feature(query_object)

        object_features = []
        for retrieved_frame in retrieved_frames:
            object_feature = self.clip_model.get_vision_feature(
                Image.fromarray(retrieved_frame).convert('RGB')
            )
            object_features.append(object_feature)

        visual_similarity = self.clip_model.compute_similarity(
            query_object_feature, torch.cat(object_features, dim=0)
        )
        joint_similarity = visual_similarity  # 如果需要也可以加入文本相似度

        sorted_index = joint_similarity.argsort()[0, ::-1]

        sorted_retrieved_data_dict = {
            "query_img": retrieved_data_dict["query_img"],
            "query_mask": retrieved_data_dict["query_mask"],
            "masked_query": retrieved_data_dict["masked_query"],
            "img": [],
            "traj": [],
            "masked_img": [],
            "mask": [],
            "caption": retrieved_data_dict.get("caption", "Failed getting."),
            "direction": [],
            "masked_img_feat": []  # 保留对应的缓存特征
        }

        print(f"A Sorted index:{len(sorted_index)}")

        for idx in range(len(sorted_index)):
            curr = sorted_index[idx]
            prev = sorted_index[idx-1] if idx > 0 else 0
            if (retrieved_data_dict["mask"][curr].all()):  # mask 全为1，跳过
                print(f"Continue")
                continue
            adaj_sim = self.clip_model.compute_similarity(object_features[curr], object_features[prev])
            print(f"Adajacent similarity:{adaj_sim}")
            print(f"Joint similarity:{joint_similarity}")

            # if (adaj_sim and joint_similarity[0, curr] > 0.1) or idx == 0:
            if True:
                print("Adding a sample.")
                sorted_retrieved_data_dict["img"].append(retrieved_data_dict["img"][curr])
                sorted_retrieved_data_dict["traj"].append(retrieved_data_dict["traj"][curr])
                sorted_retrieved_data_dict["masked_img"].append(retrieved_data_dict['masked_img'][curr])
                sorted_retrieved_data_dict["mask"].append(retrieved_data_dict['mask'][curr])
                sorted_retrieved_data_dict["direction"].append(retrieved_data_dict['direction'][curr])
                sorted_retrieved_data_dict["masked_img_feat"].append(retrieved_data_dict['masked_img_feat'][curr])

                if len(sorted_retrieved_data_dict["img"]) >= MAX_IMD_RANKING_NUM:
                    break

        return sorted_retrieved_data_dict
    
    def visualize_top5(self, topk_retrieved_data_dict, save_name):
        img_pil_list = []
        img_pil_list.append(Image.fromarray(topk_retrieved_data_dict["masked_query"]).convert('RGB'))
        for img in topk_retrieved_data_dict["masked_img"]:
            img_pil_list.append(Image.fromarray(img).convert('RGB'))

        result_img = concat_images_with_lines(img_pil_list)
        result_img.save(os.path.join(self.save_root, save_name))
    
    '''def imd_ranking(self, sorted_retrieved_data_dict, obj_prompt):
        src_ft = extract_ft(Image.fromarray(sorted_retrieved_data_dict['masked_query']).convert("RGB"), prompt=obj_prompt, ftype='sd') # 1,c,h,w
        src_mask = sorted_retrieved_data_dict["query_mask"]
        imd_distances = []
        for idx in tqdm(range(len(sorted_retrieved_data_dict["img"]))):
            tgt_ft = extract_ft(Image.fromarray(sorted_retrieved_data_dict['masked_img'][idx]).convert("RGB"), prompt=obj_prompt, ftype='sd')
            tgt_mask = sorted_retrieved_data_dict["mask"][idx]
            imd_distances.append(get_distance_imd(src_ft, tgt_ft, src_mask, tgt_mask))
        sorted_index = np.argsort(imd_distances) # from smaller to larger, but the smaller, the better
        topk_retrieved_data_dict = {
            "query_img": sorted_retrieved_data_dict["query_img"],
            "query_mask": sorted_retrieved_data_dict["query_mask"],
            "masked_query": sorted_retrieved_data_dict["masked_query"],
            "img": [],
            "traj": [],
            "masked_img": [],
            "mask": [],
            "caption":sorted_retrieved_data_dict["caption"],
            "direction":[]
        }
        for idx in range(self.topk):
            if idx >= len(sorted_index):
                break
            curr = sorted_index[idx]
            topk_retrieved_data_dict["img"].append(sorted_retrieved_data_dict["img"][curr])
            topk_retrieved_data_dict["traj"].append(sorted_retrieved_data_dict["traj"][curr])
            topk_retrieved_data_dict["masked_img"].append(sorted_retrieved_data_dict['masked_img'][curr])
            topk_retrieved_data_dict["mask"].append(sorted_retrieved_data_dict['mask'][curr])
            topk_retrieved_data_dict["direction"].append(sorted_retrieved_data_dict['direction'][curr])
        return topk_retrieved_data_dict'''
    '''def imd_ranking(self, sorted_retrieved_data_dict, obj_prompt):
        # ✅ 只构造一次 StableDiffusion 特征提取器
        featurizer = self.sd_featurizer
        # 查询图像特征
        src_ft = extract_ft(
            Image.fromarray(sorted_retrieved_data_dict['masked_query']).convert("RGB"),
            prompt=obj_prompt,
            ftype='sd',
            featurizer=featurizer
        )
        src_mask = sorted_retrieved_data_dict["query_mask"]
        imd_distances = []

        # 候选图像特征提取
        for idx in tqdm(range(len(sorted_retrieved_data_dict["img"]))):
            tgt_ft = extract_ft(
                Image.fromarray(sorted_retrieved_data_dict['masked_img'][idx]).convert("RGB"),
                prompt=obj_prompt,
                ftype='sd',
                featurizer=featurizer
            )
            tgt_mask = sorted_retrieved_data_dict["mask"][idx]
            imd_distances.append(get_distance_imd(src_ft, tgt_ft, src_mask, tgt_mask))

        # 距离排序
        sorted_index = np.argsort(imd_distances)  # from smaller to larger

        # 组装 top-k 数据
        topk_retrieved_data_dict = {
            "query_img": sorted_retrieved_data_dict["query_img"],
            "query_mask": sorted_retrieved_data_dict["query_mask"],
            "masked_query": sorted_retrieved_data_dict["masked_query"],
            "img": [],
            "traj": [],
            "masked_img": [],
            "mask": [],
            "caption": sorted_retrieved_data_dict["caption"],
            "direction": []
        }

        for idx in range(self.topk):
            if idx >= len(sorted_index):
                break
            curr = sorted_index[idx]
            topk_retrieved_data_dict["img"].append(sorted_retrieved_data_dict["img"][curr])
            topk_retrieved_data_dict["traj"].append(sorted_retrieved_data_dict["traj"][curr])
            topk_retrieved_data_dict["masked_img"].append(sorted_retrieved_data_dict['masked_img'][curr])
            topk_retrieved_data_dict["mask"].append(sorted_retrieved_data_dict['mask'][curr])
            topk_retrieved_data_dict["direction"].append(sorted_retrieved_data_dict['direction'][curr])

        return topk_retrieved_data_dict'''
    '''def imd_ranking(self, sorted_retrieved_data_dict, obj_prompt):
        featurizer = self.sd_featurizer

        # 查询图像特征
        src_ft = extract_ft(
            Image.fromarray(sorted_retrieved_data_dict['masked_query']).convert("RGB"),
            prompt=obj_prompt,
            ftype='sd',
            featurizer=featurizer
        )
        src_mask = sorted_retrieved_data_dict["query_mask"]

        imd_distances = []

        for idx in tqdm(range(len(sorted_retrieved_data_dict["img"]))):
            # ✅ 使用缓存的特征，转回 cuda
            tgt_ft = sorted_retrieved_data_dict["masked_img_feat"][idx].cuda()
            tgt_mask = sorted_retrieved_data_dict["mask"][idx]
            dist = get_distance_imd(src_ft, tgt_ft, src_mask, tgt_mask)
            imd_distances.append(dist.item())  # ✅ 转为 Python float

        # ✅ 此处 imd_distances 是 list[float]，可以安全使用 np.argsort
        sorted_index = np.argsort(imd_distances)  # 从小到大排序

        # 构造返回的 topk 结果
        topk_retrieved_data_dict = {
            "query_img": sorted_retrieved_data_dict["query_img"],
            "query_mask": sorted_retrieved_data_dict["query_mask"],
            "masked_query": sorted_retrieved_data_dict["masked_query"],
            "img": [],
            "traj": [],
            "masked_img": [],
            "mask": [],
            "caption": sorted_retrieved_data_dict["caption"],
            "direction": [],
            "masked_img_feat": []
        }

        for idx in range(self.topk):
            if idx >= len(sorted_index):
                break
            curr = sorted_index[idx]
            topk_retrieved_data_dict["img"].append(sorted_retrieved_data_dict["img"][curr])
            topk_retrieved_data_dict["traj"].append(sorted_retrieved_data_dict["traj"][curr])
            topk_retrieved_data_dict["masked_img"].append(sorted_retrieved_data_dict['masked_img'][curr])
            topk_retrieved_data_dict["mask"].append(sorted_retrieved_data_dict['mask'][curr])
            topk_retrieved_data_dict["direction"].append(sorted_retrieved_data_dict['direction'][curr])
            topk_retrieved_data_dict["masked_img_feat"].append(sorted_retrieved_data_dict['masked_img_feat'][curr])

        return topk_retrieved_data_dict'''
    def imd_ranking(self, sorted_retrieved_data_dict, obj_prompt):
        featurizer = self.sd_featurizer

        # ✅ 提取查询图像特征，并传入掩膜，确保bbox裁剪+尺度归一化
        src_img = Image.fromarray(sorted_retrieved_data_dict['masked_query']).convert("RGB")
        src_mask = sorted_retrieved_data_dict["query_mask"]
        src_ft,_ = extract_ft(
            src_img,
            prompt=obj_prompt,
            ftype='sd',
            featurizer=featurizer,
            mask=src_mask  # 👈 关键改动
        )

        imd_distances = []

        for idx in tqdm(range(len(sorted_retrieved_data_dict["img"]))):
            # ✅ 使用缓存特征（已在 retrieve 中通过 mask 裁剪过）
            tgt_ft = sorted_retrieved_data_dict["masked_img_feat"][idx].cuda()
            tgt_mask = sorted_retrieved_data_dict["mask"][idx]
            dist = get_distance_imd(src_ft, tgt_ft, src_mask, tgt_mask)
            imd_distances.append(dist.item())

        sorted_index = np.argsort(imd_distances)

        # 构造 top-k 检索结果
        topk_retrieved_data_dict = {
            "query_img": sorted_retrieved_data_dict["query_img"],
            "query_mask": sorted_retrieved_data_dict["query_mask"],
            "masked_query": sorted_retrieved_data_dict["masked_query"],
            "img": [],
            "traj": [],
            "masked_img": [],
            "mask": [],
            "caption": sorted_retrieved_data_dict["caption"],
            "direction": [],
            "masked_img_feat": []
        }

        for i in range(self.topk):
            if i >= len(sorted_index): break
            curr = sorted_index[i]
            for k in ["img", "traj", "masked_img", "mask", "direction", "masked_img_feat"]:
                topk_retrieved_data_dict[k].append(sorted_retrieved_data_dict[k][curr])

        return topk_retrieved_data_dict

    '''def load_retrieved_task_from_pkl(self, retrieved_task):
        task_dir = os.path.join(self.subset_dir, self.data_source, retrieved_task.replace(" ", "_"))
        if os.path.exists(os.path.join(task_dir, retrieved_task.replace(" ", "_") + "_new.pkl")):
            with open(os.path.join(task_dir, retrieved_task.replace(" ", "_") + "_new.pkl"), 'rb') as f:
                retrieved_data_dict = pickle.load(f)
        # print(f"{retrieved_data_dict}")
        return retrieved_data_dict'''
    def load_retrieved_task_from_pkl(self, retrieved_task):
        import pickle

        task_name = retrieved_task.replace(" ", "_")
        task_dir = os.path.join(self.subset_dir, self.data_source, task_name)
        pkl_path = os.path.join(task_dir, task_name + "_new.pkl")

        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"找不到指定的pkl文件: {pkl_path}")

        with open(pkl_path, 'rb') as f:
            retrieved_data_dict = pickle.load(f)

        self.last_retrieved_pkl_path = pkl_path  # ⭐️记录路径，后续覆盖写回用
        return retrieved_data_dict
    def retrieve(self, current_task, current_obs, log=True):
        if log: print("<1> Retrieve the most similar task")
        print(f"Current task: {current_task}")
        retrieved_task = self.language_retrieve(current_task)
        obj_name = "object"
        obj_prompt = f"A photo of {obj_name}"
        if log: print(f"Retrieved task: {retrieved_task}")

        # retrieved_data_dict = load_retrieved_task(subset_dir, retrieved_task) # load from raw data
        if log: print("<2> Load the first frames and trajs of the episodes of the retrieved task")
        retrieved_data_dict = self.load_retrieved_task_from_pkl(retrieved_task) 
        # img, traj, masked_img, mask
        # after modification:img({'hand':IMG,'no_hand':IMG}),caption, traj, image_size, name.
        if log: print("<3> Segment out the object from our observation")
        query_frame, query_mask = self.segment_objects([current_obs], current_task)
        retrieved_data_dict['query_img'] = current_obs
        retrieved_data_dict['query_mask'] = query_mask
        retrieved_data_dict['masked_query'] = query_frame
        
        if "masked_img" not in retrieved_data_dict.keys(): # not preprocessed
            if log: print("<3.5> Retrieved data are not processed, processing...")
            # print(f"{retrieved_data_dict['img']}")
            print(f"{len(retrieved_data_dict['img'])}")
            retrieved_imgs=[item['no_hand'] for item in retrieved_data_dict['img']]
            # retrieved_imgs=retrieved_data_dict['img'][0]['no_hand']
            result = self.segment_objects(retrieved_imgs, obj_name, retrieved_data_dict['traj'])
            if len(result) == 2:  # 如果只返回了2个值
                masked_frame, frame_mask = result
                # 将单个值转换为列表
                masked_frames = [masked_frame]
                frame_masks = [frame_mask]
                trajs = retrieved_data_dict['traj']  # 保持原有轨迹
            else:  # 返回了3个值
                masked_frames, frame_masks, trajs = result
            retrieved_data_dict['masked_img'] = masked_frames
            retrieved_data_dict['mask'] = frame_masks
            retrieved_data_dict['traj'] = trajs
            test_ori_img= retrieved_data_dict["masked_img"][0]
            test_ori_mask= retrieved_data_dict["mask"][0]
            test_traj= retrieved_data_dict["traj"][0]
        # ✅ 仅当不存在缓存的 SD 特征时，才提前提取
        if "masked_img_feat" not in retrieved_data_dict.keys():
            if log: print("[✔] 提前提取 StableDiffusion 特征...")
            retrieved_data_dict['masked_img_feat'] = []
            sd_featurizer = self.sd_featurizer
            for img, mask in zip(retrieved_data_dict['masked_img'], retrieved_data_dict['mask']):
                feat,_ = extract_ft(
                    Image.fromarray(img).convert("RGB"),
                    prompt=obj_prompt,
                    ftype='sd',
                    featurizer=sd_featurizer,
                    mask=mask
                )
                retrieved_data_dict['masked_img_feat'].append(feat.cpu())  # 转 CPU 避免显存占用过多
            # 写回 pickle 缓存，方便下次快速加载
            if hasattr(self, "last_retrieved_pkl_path"):
                import pickle
                with open(self.last_retrieved_pkl_path, "wb") as f:
                    pickle.dump(retrieved_data_dict, f)
                print(f"[✔] 已保存预处理结果（含特征缓存）至: {self.last_retrieved_pkl_path}")
            else:
                print("⚠️ 无法保存回 .pkl 文件，因为未设置 last_retrieved_pkl_path")       
            #visualize_mask_and_trajectory(test_ori_img,test_ori_mask,test_traj, save_path=os.path.join(self.save_root, "ori_retrieved_img_traj.png"))
        '''if log: print("<4> Semantic filtering...")
        sorted_retrieved_data_dict = self.clip_filtering(retrieved_data_dict, obj_prompt)
        test_ori_img= sorted_retrieved_data_dict["masked_img"][0]
        test_ori_mask= sorted_retrieved_data_dict["mask"][0]
        test_traj= sorted_retrieved_data_dict["traj"][0]
        visualize_mask_and_trajectory(test_ori_img,test_ori_mask,test_traj, save_path=os.path.join(self.save_root, "sorted_retrieved_img_traj.png"))
        if not sorted_retrieved_data_dict["img"]:
            print("CLIPFILERTING后为空")'''
        sorted_retrieved_data_dict = retrieved_data_dict
        if log: print("<5> Geometrical retrieval...")
        start_time1 = time.time()
        topk_retrieved_data_dict = self.imd_ranking(sorted_retrieved_data_dict, obj_prompt)
        end_time1 = time.time()
        elapsed_time = end_time1 - start_time1
        test_ori_img= topk_retrieved_data_dict["masked_img"][0]
        test_ori_mask= topk_retrieved_data_dict["mask"][0]
        test_traj= topk_retrieved_data_dict["traj"][0]
        visualize_mask_and_trajectory(test_ori_img,test_ori_mask,test_traj, save_path=os.path.join(self.save_root, "topk_retrieved_img_traj.png"))
        self.visualize_top5(topk_retrieved_data_dict, "imd_top5.png")
        top1_idx = 0
        start_time2 = time.time()
        # 检查结果列表是否为空
        if not topk_retrieved_data_dict["img"]:
            print("警告: 检索结果为空。使用原始查询图像和默认轨迹。")
            # 使用查询图像作为结果
            top1_retrieved_data_dict = {
                "query_img": current_obs,
                "query_mask": query_mask,
                "masked_query": query_frame,
                "img": current_obs,  # 使用查询图像
                "traj": [(current_obs.shape[1]//2, current_obs.shape[0]//2)],  # 图像中心点作为默认轨迹
                "masked_img": query_frame,
                "mask": query_mask,
                "caption": retrieved_data_dict.get("caption", "A generic object"),
                "direction": [(0, 0, 0)]  # 默认方向值
            }
        else:
            top1_retrieved_data_dict = {
                "query_img": current_obs,
                "query_mask": query_mask,
                "masked_query": query_frame,
                "img": topk_retrieved_data_dict["img"][top1_idx],
                "traj": topk_retrieved_data_dict["traj"][top1_idx], # in segmented & cropped space
                "masked_img": topk_retrieved_data_dict["masked_img"][top1_idx],
                "mask": topk_retrieved_data_dict["mask"][top1_idx],
                "caption":topk_retrieved_data_dict["caption"],
                "direction": topk_retrieved_data_dict["direction"][top1_idx]
            }
        end_time2 = time.time()
        elapsed_time += end_time2 - start_time2        # print(f"{top1_retrieved_data_dict}")
        direction_path = os.path.join(self.save_root, "top1_direction.json")
        with open(direction_path, "w") as f:
            json.dump({"direction": top1_retrieved_data_dict["direction"]}, f, indent=4)
        print(f"Top-1 direction saved to: {direction_path}")
        print(f"⏱️ 纯检索耗时: {elapsed_time:.4f} 秒")
        return top1_idx, top1_retrieved_data_dict
