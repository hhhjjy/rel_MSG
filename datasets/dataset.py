# dataloader py
# this file contain the dataset class and corresponding dataloader of the apple arkit dataset, for MSG
# the data directory structure is as follows:
# validation/videoid(each is a video data directory)/ -> sub directory as follows:
# ./videoid_frames/lowres_wide/videoid_frameid.png -> contains the frames of the video

import os
import json
import numpy as np
import torch
# import cv2
import random
from torch.utils.data import Dataset

from torchvision.io import read_image
# from torchvision import tv_tensors
# from torchvision.transforms.v2 import functional as F
# from torchvision.tv_tensors import BoundingBoxes

from torch.nn.utils.rnn import pad_sequence
import pickle
# from ultralytics import SAM
# import cv2
# SAM_model = SAM("sam2_b.pt")

import torch
import matplotlib.pyplot as plt
import torchvision.transforms as T

def matrix_reorder(matrix, dict_a, dict_b):
    """
    支持：
    dict_a 键 = 数字，dict_b 键 = 字符串数字
    或者反过来，都能自动处理
    """
    size = len(matrix)
    
    # 核心：统一键类型，自动匹配 数字 ↔ 字符串数字
    a_idx_map = {}
    for key, idx in dict_a.items():
        # 统一转成字符串，就能完美匹配
        key_str = str(key)
        a_idx_map[key_str] = idx

    # 建立 A序号 → B序号 映射
    a_to_b = [0] * size
    for key, b_idx in dict_b.items():
        key_str = str(key)
        a_idx = a_idx_map[key_str]
        a_to_b[a_idx] = b_idx

    # 重排矩阵
    new_matrix = [[0]*size for _ in range(size)]
    for a_row in range(size):
        for a_col in range(size):
            b_row = a_to_b[a_row]
            b_col = a_to_b[a_col]
            new_matrix[b_row][b_col] = matrix[a_row][a_col]
    
    return torch.tensor(new_matrix)

def draw_and_save_boxes(image_tensor, boxes, save_path='boxed_img.jpg'):
    # 转为 PIL 图像
    img = T.ToPILImage()(image_tensor.cpu().clone())

    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    ax = plt.gca()

    for box in boxes:
        if isinstance(box, torch.Tensor):
            box = box.detach().cpu().numpy()
        
        x1, y1, x2, y2 = box
        # 跳过全0无效框
        if (x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0):
            continue

        w = x2 - x1
        h = y2 - y1
        ax.add_patch(plt.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='green', facecolor='none'))

    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()



def save_sam_result(result, save_dir, frame_name):
    """
    保存 SAM 推理结果
    :param result: Ultralytics SAM 输出的 results[0]
    :param save_dir: 保存文件夹
    :param frame_name: 帧名（例如 000138）
    """
    os.makedirs(save_dir, exist_ok=True)

    # ----------------------
    # 1. 保存带掩码的图片
    # ----------------------
    img_save_path = os.path.join(save_dir, f"seg_{frame_name}.jpg")
    result.save(filename=img_save_path)

    # ----------------------
    # 2. 保存单独的 mask (黑白PNG)
    # ----------------------
    masks = result.masks.data  # [N, H, W]
    if masks.size(0) > 0:
        # 合并所有 mask（可选）
        combined_mask = torch.zeros(masks.shape[1:], dtype=torch.uint8)
        for i, mask in enumerate(masks):
            combined_mask[mask > 0.5] = 255  # 白色

        # 保存
        mask_save_path = os.path.join(save_dir, f"mask_{frame_name}.png")
        cv2.imwrite(mask_save_path, combined_mask.cpu().numpy())

    print(f"✅ 保存完成: {img_save_path}")

class AppleDataHandler:
    '''
    organize videos, provide next video(s) for training process
    '''
    def __init__(self, data_dir, split='train', video_batch_size=1, dataset='3rscan'):
        self.videos = None
        if split == 'train':
            self.videos = [line.strip() for line in open(os.path.join(data_dir, 'train_scans.txt'), 'r')]
        elif split == 'val':
            self.videos = [line.strip() for line in open(os.path.join(data_dir, 'validation_scans.txt'), 'r')]
        elif split == 'test':
            self.videos = [line.strip() for line in open(os.path.join(data_dir, 'test_scans.txt'), 'r')]
        self.data_dir = data_dir
        self.split = split
        if dataset == 'Replica_small':
            self.data_split_dir = os.path.join(self.data_dir, self.split)
        elif dataset == 'Replica_small_split':
            self.data_split_dir = os.path.join(self.data_dir, 'data')
        elif dataset == 'Replica':
            self.data_split_dir = os.path.join(self.data_dir, 'data')
        elif dataset == '3rscan_split':
            self.data_split_dir = os.path.join(self.data_dir, 'data')
        else:
            self.data_split_dir = os.path.join(self.data_dir, self.split)
        if self.videos is None:
            print('use all videos')
            self.videos = sorted([f for f in os.listdir(self.data_split_dir) if os.path.isdir(os.path.join(self.data_split_dir, f))])
        self.num_videos = len(self.videos)
        self.vid_idx = 0 # pointer
        self.video_batch_size = video_batch_size
        

    def __len__(self,):
        return self.num_videos
    
    def __iter__(self,):
        return self
    
    def __next__(self,):
        if self.vid_idx >= self.num_videos:
            self.vid_idx = 0
            raise StopIteration
        else:
            ceil = min(self.vid_idx + self.video_batch_size, self.num_videos)
            current_video_batch = self.videos[self.vid_idx: ceil]
            self.vid_idx += self.video_batch_size
            return current_video_batch
    
    def shuffle(self,):
        # shuffle the videos for training
        random.shuffle(self.videos)

    def reset(self):
        self.vid_idx = 0
    
class VideoDataset_Replica_small_split(Dataset):

    def __init__(self, video_data_dir, video_id, configs, transforms, split="train", scan_dict=None, use_sam = False):

        self.split = split
        self.real_scene = video_id.split('_refine_')[0]
        self.video_data_dir = os.path.join(video_data_dir, self.real_scene)
        self.video_id = video_id
        self.transforms = transforms
        self.ori_image_size = configs['image_size']
        # get annotations
        gt_path = r'/root/autodl-tmp/dataset/Replica_small_split/refine_topo_gt'
        self.gt = json.load(open(os.path.join(gt_path, f'{video_id}.json'))) # 'topo_gt.json'
        if scan_dict is None:
            print('scan_dict is None')
        else:
            # 先使用固定的 rel_matrix_list
            self.rel_matrix = torch.tensor(scan_dict[self.real_scene]['rel_matrix_list'])
            self.obj2id = scan_dict[self.real_scene]['obj2id_dic']
            self.obj2col = self.gt['original_obj2col']
            self.rel_matrix = matrix_reorder(self.rel_matrix, self.obj2id, self.obj2col)

        # self.gt = json.load(open(os.path.join(self.video_data_dir, 'refine_topo_gt.json'))) # 'topo_gt.json'

        # get the frame ids
        self.frame_dir = os.path.join(self.video_data_dir, 'sequence')
        #NOTE: NOT ALL frames are used -> [int(f.split('.')[0].split('_')[1]) for f in os.listdir(os.path.join(self.data_dir, self.video_id, 'videoid_frames', 'lowres_wide'))]
        self.frame_ids = self.gt['sampled_frames'] 
        self.frame_ids.sort()
        self.frame2idx = {frame_id: idx for idx, frame_id in enumerate(self.frame_ids)}
        self.num_frames = len(self.frame_ids)

        self.obj_id_offset = 0
        self.use_sam = use_sam
        self.sam_masks = {}
        self.has_sam_masks = False
        self.cache_file = os.path.join(self.video_data_dir, f'{video_id}_sam_masks.pt')
        if self.use_sam:
            if os.path.exists(self.cache_file):
                self.has_sam_masks = True
                self.sam_masks = torch.load(self.cache_file, map_location='cpu')
            else:
                print('load sam model')
                from ultralytics import SAM
                self.SAM_model = SAM("sam2_b.pt")
                self.SAM_model.to('cuda')  # 放 GPU 加速

        # get grounding dino detections if detector is grounding-dino
        self.use_gdino = False
        if configs["detector"]["model"] == "grounding-dino":
            self.use_gdino = True
            gdino_file = os.path.join(configs["detector"]["result_path"], split, video_id, 'eval_results.json')
            self.gdino_det = json.load(open(gdino_file))


        # other annotations
        self.obj2col = self.gt['obj2col'] # store which column in the gt annotation corresponds to which obj unique id
        self.pp_adj = torch.tensor(self.gt['p-p'])
        self.pp_adj.fill_diagonal_(1)
        # print("gt diag", self.pp_adj.diagonal())
        self.po_adj = torch.tensor(self.gt['p-o'])
        self.uid2obj = dict()
        for object_name in self.gt['uidmap']: # map object id to object name
            for object_id in self.gt['uidmap'][object_name]:
                self.uid2obj[object_id] = object_name
        self.class_map = configs['class_map'] # map object class name to class id, note that background should be set to 0
        self.image_size = configs['image_size']
        self.target_image_size = configs['model_image_size']
        self.new_width = self.target_image_size[1]
        self.new_height = self.target_image_size[0]
        self.orig_width = self.image_size[1]
        self.orig_height = self.image_size[0]

    def __len__(self,):
        return len(self.frame_ids)
    
    def get_det(self, frame_id):
        # read detection annotation, return list of bboxes and list of object ids
        bboxes = []
        obj_ids = []
        obj_labels = []
        frame_id = str(frame_id)
        if frame_id in self.gt['annotations']: #NOTE: else this frame has no gt objct detections.
            det_dict = self.gt['annotations'][frame_id]
            for obj_id, bbox in det_dict.items():
                # print(obj_id, type(obj_id))
                bboxes.append(torch.tensor(bbox))
                obj_ids.append(self.obj2col[obj_id])
                obj_labels.append(self.class_map[self.uid2obj[int(obj_id)]])
                # obj_labels.append(int(self.uid2obj[int(obj_id)]))
            # bboxes = tv_tensors.BoundingBoxes(bboxes, format='XYXY', canvas_size=self.image_size)
            # NOTE: for the older torchvision model BoundingBoxes are not there, gotta work around:
            bboxes = torch.stack(bboxes, dim=0) if bboxes else torch.empty((0, 4), dtype=torch.float32)
            # bboxes = torch.stack(bboxes, dim=0)
        else:
            bboxes = torch.empty((0,4)) # no detection
        obj_ids = torch.as_tensor(obj_ids)
        obj_labels = torch.as_tensor(obj_labels)
        return bboxes, obj_ids, obj_labels
    
    def get_pred_det(self, frame_id):
        bboxes = []
        obj_labels = []
        frame_id = str(frame_id)
        if frame_id in self.gdino_det['detections']:
            det_dict = self.gdino_det['detections'][frame_id]
            for obj_id, det in det_dict.items():
                # print(obj_id, type(obj_id))
                bboxes.append(torch.tensor(det["bbox"], dtype=torch.float))
                obj_labels.append(det["label"])
            if len(bboxes)>0:
                bboxes = torch.stack(bboxes, dim=0)
            else:
                bboxes = torch.empty((0,4), dtype=torch.float)
        else:
            bboxes = torch.empty((0,4), dtype=torch.float) # no detection
        obj_labels = torch.as_tensor(obj_labels)
        return bboxes, obj_labels
        
    
    def get_place_labels(self, frame_idxs):
        """
        index p-p adjencecy, find place labels, use for training place recognition
        Parameters:
         - frame_idxs: the frame idxs in this batch,
        Outputs:
         - place recognition labels, BxB binary matrice
        """
        place_labels = self.pp_adj[frame_idxs][:, frame_idxs]
        return place_labels

    def get_rel_labels(self, obj_idx):
        rel_labels = self.rel_matrix[obj_idx][:, obj_idx]
        return rel_labels
    
    
    def set_objidx_offset(self, offset):
        self.obj_id_offset = offset

    def __getitem__(self, idx):
        # import time
        # aaa = time.time()
        # get the frame id
        frame_id = self.frame_ids[idx]
        bboxes, obj_ids, obj_labels = self.get_det(frame_id)  # (N, 4), (N) two lists
        # bbb = time.time()
        # print('get det time:', bbb-aaa)
        data = dict()
        image_path = os.path.join(self.frame_dir, f'frame-{frame_id[-10:-4]}.color.jpg')

        image = read_image(image_path)
        # ccc = time.time()
        

        # image = image.rot90(3, [1, 2])  # 旋转 270 度
        # print('read image time:', ccc-bbb)
        def get_sam_masks(self, frame_id):
            if self.has_sam_masks:
                return self.sam_masks[frame_id]
            else:
                img = cv2.imread(image_path)
                img_rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

                # ===================== 【核心：SAM 提取 mask】 =====================
                # 确保 self.SAM_model 已经在 __init__ 里加载：self.SAM_model = SAM("sam2_b.pt")
                if bboxes.size(0) == 0:
                    results=[]
                else:
                    with torch.no_grad():  # 禁止计算梯度，提速省显存
                        results = self.SAM_model(
                            img_rotated,
                            bboxes=bboxes.unsqueeze(0),  # 直接用你的 bboxes
                            save=False,      # 这里不需要保存，避免覆盖
                            verbose=False
                        )
                
                # 取出 mask：shape = [N, H, W] （torch tensor）
                if len(results) > 0 and results[0].masks is not None:
                    masks = results[0].masks.data  # [N, H, W]
                else:
                    masks = torch.empty(0, image.shape[1], image.shape[2])
                self.sam_masks[frame_id] = masks
            return masks
        if self.use_sam:
            masks = get_sam_masks(self, frame_id)
        else:
            masks = torch.zeros_like(image)
        # masks
        # ==================================================================
        # ddd = time.time()
        # print('sam time:', ddd-ccc)
        # 原有 transform 逻辑
        if self.transforms is not None:
            image = self.transforms(image)
            # transform boxes accordingly, surrogate when transform API doesn account for bboxes
            bboxes = bboxes.to(torch.float32)
            bboxes[:, 0::2] *= self.new_width / self.orig_width  # (w, h, w ,h)
            bboxes[:, 1::2] *= self.new_height / self.orig_height
            bboxes = bboxes.to(torch.int64)

            # ===================== mask 同步缩放到 transform 后的尺寸 =====================
            H_new, W_new = image.shape[1], image.shape[2]
            if masks.size(0) > 0:
                masks = torch.nn.functional.interpolate(
                    masks.unsqueeze(1).float(),  # [N,1,H,W]
                    size=(H_new, W_new),
                    mode="nearest"
                ).squeeze(1)  # [N, H_new, W_new]
            # ==============================================================================
        # eee = time.time()
        # print('transform time:', eee-ddd)
        # 原有数据
        data['image'] = image
        data['image_idx'] = torch.tensor(self.frame2idx[frame_id])
        data['bbox'] = bboxes
        data['obj_idx'] = obj_ids + self.obj_id_offset
        data['obj_label'] = obj_labels

        # ===================== 【把 mask 存入 data】 =====================
        data['mask'] = masks  # shape: [N, H, W]
        # =================================================================

        if self.use_gdino:
            pred_box, pred_label = self.get_pred_det(frame_id)
            data['pred_bbox'] = pred_box
        # assert 0
        return data


class VideoDataset_Replica_small(Dataset):

    def __init__(self, video_data_dir, video_id, configs, transforms, split="train", scan_dict=None, use_sam = False):

        self.split = split
        self.video_data_dir = os.path.join(video_data_dir, video_id)
        self.video_id = video_id
        self.transforms = transforms
        self.ori_image_size = configs['image_size']
        # get annotations
        gt_path = r'/root/autodl-tmp/dataset/Replica_small/refine_topo_gt'
        self.gt = json.load(open(os.path.join(gt_path, f'{video_id}_refine_topo_gt.json'))) # 'topo_gt.json'
        if scan_dict is None:
            print('scan_dict is None')
        else:
            # 先使用固定的 rel_matrix_list
            self.rel_matrix = torch.tensor(scan_dict[video_id]['rel_matrix_list'])
            self.obj2id = scan_dict[video_id]['obj2id_dic']
            self.obj2col = self.gt['obj2col']
            self.rel_matrix = matrix_reorder(self.rel_matrix, self.obj2id, self.obj2col)

        # self.gt = json.load(open(os.path.join(self.video_data_dir, 'refine_topo_gt.json'))) # 'topo_gt.json'

        # get the frame ids
        self.frame_dir = os.path.join(self.video_data_dir, 'sequence')
        #NOTE: NOT ALL frames are used -> [int(f.split('.')[0].split('_')[1]) for f in os.listdir(os.path.join(self.data_dir, self.video_id, 'videoid_frames', 'lowres_wide'))]
        self.frame_ids = self.gt['sampled_frames'] 
        self.frame_ids.sort()
        self.frame2idx = {frame_id: idx for idx, frame_id in enumerate(self.frame_ids)}
        self.num_frames = len(self.frame_ids)

        self.obj_id_offset = 0
        self.use_sam = use_sam
        self.sam_masks = {}
        self.has_sam_masks = False
        self.cache_file = os.path.join(self.video_data_dir, f'{video_id}_sam_masks.pt')
        if self.use_sam:
            if os.path.exists(self.cache_file):
                self.has_sam_masks = True
                self.sam_masks = torch.load(self.cache_file, map_location='cpu')
            else:
                print('load sam model')
                from ultralytics import SAM
                self.SAM_model = SAM("sam2_b.pt")
                self.SAM_model.to('cuda')  # 放 GPU 加速

        # get grounding dino detections if detector is grounding-dino
        self.use_gdino = False
        if configs["detector"]["model"] == "grounding-dino":
            self.use_gdino = True
            gdino_file = os.path.join(configs["detector"]["result_path"], split, video_id, 'eval_results.json')
            self.gdino_det = json.load(open(gdino_file))


        # other annotations
        self.obj2col = self.gt['obj2col'] # store which column in the gt annotation corresponds to which obj unique id
        self.pp_adj = torch.tensor(self.gt['p-p'])
        self.pp_adj.fill_diagonal_(1)
        # print("gt diag", self.pp_adj.diagonal())
        self.po_adj = torch.tensor(self.gt['p-o'])
        self.uid2obj = dict()
        for object_name in self.gt['uidmap']: # map object id to object name
            for object_id in self.gt['uidmap'][object_name]:
                self.uid2obj[object_id] = object_name
        self.class_map = configs['class_map'] # map object class name to class id, note that background should be set to 0
        self.image_size = configs['image_size']
        self.target_image_size = configs['model_image_size']
        self.new_width = self.target_image_size[1]
        self.new_height = self.target_image_size[0]
        self.orig_width = self.image_size[1]
        self.orig_height = self.image_size[0]

    def __len__(self,):
        return len(self.frame_ids)
    
    def get_det(self, frame_id):
        # read detection annotation, return list of bboxes and list of object ids
        bboxes = []
        obj_ids = []
        obj_labels = []
        frame_id = str(frame_id)
        if frame_id in self.gt['annotations']: #NOTE: else this frame has no gt objct detections.
            det_dict = self.gt['annotations'][frame_id]
            for obj_id, bbox in det_dict.items():
                # print(obj_id, type(obj_id))
                bboxes.append(torch.tensor(bbox))
                obj_ids.append(self.obj2col[obj_id])
                obj_labels.append(self.class_map[self.uid2obj[int(obj_id)]])
                # obj_labels.append(int(self.uid2obj[int(obj_id)]))
            # bboxes = tv_tensors.BoundingBoxes(bboxes, format='XYXY', canvas_size=self.image_size)
            # NOTE: for the older torchvision model BoundingBoxes are not there, gotta work around:
            bboxes = torch.stack(bboxes, dim=0)
        else:
            bboxes = torch.empty((0,4)) # no detection
        obj_ids = torch.as_tensor(obj_ids)
        obj_labels = torch.as_tensor(obj_labels)
        return bboxes, obj_ids, obj_labels
    
    def get_pred_det(self, frame_id):
        bboxes = []
        obj_labels = []
        frame_id = str(frame_id)
        if frame_id in self.gdino_det['detections']:
            det_dict = self.gdino_det['detections'][frame_id]
            for obj_id, det in det_dict.items():
                # print(obj_id, type(obj_id))
                bboxes.append(torch.tensor(det["bbox"], dtype=torch.float))
                obj_labels.append(det["label"])
            if len(bboxes)>0:
                bboxes = torch.stack(bboxes, dim=0)
            else:
                bboxes = torch.empty((0,4), dtype=torch.float)
        else:
            bboxes = torch.empty((0,4), dtype=torch.float) # no detection
        obj_labels = torch.as_tensor(obj_labels)
        return bboxes, obj_labels
        
    
    def get_place_labels(self, frame_idxs):
        """
        index p-p adjencecy, find place labels, use for training place recognition
        Parameters:
         - frame_idxs: the frame idxs in this batch,
        Outputs:
         - place recognition labels, BxB binary matrice
        """
        place_labels = self.pp_adj[frame_idxs][:, frame_idxs]
        return place_labels

    def get_rel_labels(self, obj_idx):
        rel_labels = self.rel_matrix[obj_idx][:, obj_idx]
        return rel_labels
    
    
    def set_objidx_offset(self, offset):
        self.obj_id_offset = offset

    def __getitem__(self, idx):
        # import time
        # aaa = time.time()
        # get the frame id
        frame_id = self.frame_ids[idx]
        bboxes, obj_ids, obj_labels = self.get_det(frame_id)  # (N, 4), (N) two lists
        # bbb = time.time()
        # print('get det time:', bbb-aaa)
        data = dict()
        image_path = os.path.join(self.frame_dir, f'frame-{frame_id[-10:-4]}.color.jpg')

        image = read_image(image_path)
        # ccc = time.time()
        

        # image = image.rot90(3, [1, 2])  # 旋转 270 度
        # print('read image time:', ccc-bbb)
        def get_sam_masks(self, frame_id):
            if self.has_sam_masks:
                return self.sam_masks[frame_id]
            else:
                img = cv2.imread(image_path)
                img_rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

                # ===================== 【核心：SAM 提取 mask】 =====================
                # 确保 self.SAM_model 已经在 __init__ 里加载：self.SAM_model = SAM("sam2_b.pt")
                if bboxes.size(0) == 0:
                    results=[]
                else:
                    with torch.no_grad():  # 禁止计算梯度，提速省显存
                        results = self.SAM_model(
                            img_rotated,
                            bboxes=bboxes.unsqueeze(0),  # 直接用你的 bboxes
                            save=False,      # 这里不需要保存，避免覆盖
                            verbose=False
                        )
                
                # 取出 mask：shape = [N, H, W] （torch tensor）
                if len(results) > 0 and results[0].masks is not None:
                    masks = results[0].masks.data  # [N, H, W]
                else:
                    masks = torch.empty(0, image.shape[1], image.shape[2])
                self.sam_masks[frame_id] = masks
            return masks
        if self.use_sam:
            masks = get_sam_masks(self, frame_id)
        else:
            masks = torch.zeros_like(image)
        # masks
        # ==================================================================
        # ddd = time.time()
        # print('sam time:', ddd-ccc)
        # 原有 transform 逻辑
        if self.transforms is not None:
            image = self.transforms(image)
            # transform boxes accordingly, surrogate when transform API doesn account for bboxes
            bboxes = bboxes.to(torch.float32)
            bboxes[:, 0::2] *= self.new_width / self.orig_width  # (w, h, w ,h)
            bboxes[:, 1::2] *= self.new_height / self.orig_height
            bboxes = bboxes.to(torch.int64)

            # ===================== mask 同步缩放到 transform 后的尺寸 =====================
            H_new, W_new = image.shape[1], image.shape[2]
            if masks.size(0) > 0:
                masks = torch.nn.functional.interpolate(
                    masks.unsqueeze(1).float(),  # [N,1,H,W]
                    size=(H_new, W_new),
                    mode="nearest"
                ).squeeze(1)  # [N, H_new, W_new]
            # ==============================================================================
        # eee = time.time()
        # print('transform time:', eee-ddd)
        # 原有数据
        data['image'] = image
        data['image_idx'] = torch.tensor(self.frame2idx[frame_id])
        data['bbox'] = bboxes
        data['obj_idx'] = obj_ids + self.obj_id_offset
        data['obj_label'] = obj_labels

        # ===================== 【把 mask 存入 data】 =====================
        data['mask'] = masks  # shape: [N, H, W]
        # =================================================================

        if self.use_gdino:
            pred_box, pred_label = self.get_pred_det(frame_id)
            data['pred_bbox'] = pred_box
        # assert 0
        return data

class VideoDataset_Replica(Dataset):

    def __init__(self, video_data_dir, video_id, configs, transforms, split="train", scan_dict=None, use_sam = False):

        self.split = split
        self.video_data_dir = os.path.join(video_data_dir, video_id)
        self.video_id = video_id
        self.transforms = transforms
        self.ori_image_size = configs['image_size']
        # get annotations
        gt_path = r'/root/autodl-tmp/dataset/Replica/refine_topo_gt'
        self.gt = json.load(open(os.path.join(gt_path, f'{video_id}_refine_topo_gt.json'))) # 'topo_gt.json'
        if scan_dict is None:
            print('scan_dict is None')
        else:
            self.rel_matrix = torch.tensor(scan_dict[split][video_id]['rel_matrix_list'])
            self.obj2id = scan_dict[split][video_id]['obj2id_dic']
            self.obj2col = self.gt['obj2col']
            self.rel_matrix = matrix_reorder(self.rel_matrix, self.obj2id, self.obj2col)

        # self.gt = json.load(open(os.path.join(self.video_data_dir, 'refine_topo_gt.json'))) # 'topo_gt.json'

        # get the frame ids
        self.frame_dir = os.path.join(self.video_data_dir, 'sequence')
        #NOTE: NOT ALL frames are used -> [int(f.split('.')[0].split('_')[1]) for f in os.listdir(os.path.join(self.data_dir, self.video_id, 'videoid_frames', 'lowres_wide'))]
        self.frame_ids = self.gt['sampled_frames'] 
        self.frame_ids.sort()
        self.frame2idx = {frame_id: idx for idx, frame_id in enumerate(self.frame_ids)}
        self.num_frames = len(self.frame_ids)

        self.obj_id_offset = 0
        self.use_sam = use_sam
        self.sam_masks = {}
        self.has_sam_masks = False
        self.cache_file = os.path.join(self.video_data_dir, f'{video_id}_sam_masks.pt')
        if self.use_sam:
            if os.path.exists(self.cache_file):
                self.has_sam_masks = True
                self.sam_masks = torch.load(self.cache_file, map_location='cpu')
            else:
                print('load sam model')
                from ultralytics import SAM
                self.SAM_model = SAM("sam2_b.pt")
                self.SAM_model.to('cuda')  # 放 GPU 加速

        # get grounding dino detections if detector is grounding-dino
        self.use_gdino = False
        if configs["detector"]["model"] == "grounding-dino":
            self.use_gdino = True
            gdino_file = os.path.join(configs["detector"]["result_path"], split, video_id, 'eval_results.json')
            self.gdino_det = json.load(open(gdino_file))


        # other annotations
        self.obj2col = self.gt['obj2col'] # store which column in the gt annotation corresponds to which obj unique id
        self.pp_adj = torch.tensor(self.gt['p-p'])
        self.pp_adj.fill_diagonal_(1)
        # print("gt diag", self.pp_adj.diagonal())
        self.po_adj = torch.tensor(self.gt['p-o'])
        self.uid2obj = dict()
        for object_name in self.gt['uidmap']: # map object id to object name
            for object_id in self.gt['uidmap'][object_name]:
                self.uid2obj[object_id] = object_name
        self.class_map = configs['class_map'] # map object class name to class id, note that background should be set to 0
        self.image_size = configs['image_size']
        self.target_image_size = configs['model_image_size']
        self.new_width = self.target_image_size[1]
        self.new_height = self.target_image_size[0]
        self.orig_width = self.image_size[1]
        self.orig_height = self.image_size[0]

    def __len__(self,):
        return len(self.frame_ids)
    
    def get_det(self, frame_id):
        # read detection annotation, return list of bboxes and list of object ids
        bboxes = []
        obj_ids = []
        obj_labels = []
        frame_id = str(frame_id)
        if frame_id in self.gt['annotations']: #NOTE: else this frame has no gt objct detections.
            det_dict = self.gt['annotations'][frame_id]
            for obj_id, bbox in det_dict.items():
                # print(obj_id, type(obj_id))
                bboxes.append(torch.tensor(bbox))
                obj_ids.append(self.obj2col[obj_id])
                obj_labels.append(self.class_map[self.uid2obj[int(obj_id)]])
                # obj_labels.append(int(self.uid2obj[int(obj_id)]))
            # bboxes = tv_tensors.BoundingBoxes(bboxes, format='XYXY', canvas_size=self.image_size)
            # NOTE: for the older torchvision model BoundingBoxes are not there, gotta work around:
            bboxes = torch.stack(bboxes, dim=0)
        else:
            bboxes = torch.empty((0,4)) # no detection
        obj_ids = torch.as_tensor(obj_ids)
        obj_labels = torch.as_tensor(obj_labels)
        return bboxes, obj_ids, obj_labels
    
    def get_pred_det(self, frame_id):
        bboxes = []
        obj_labels = []
        frame_id = str(frame_id)
        if frame_id in self.gdino_det['detections']:
            det_dict = self.gdino_det['detections'][frame_id]
            for obj_id, det in det_dict.items():
                # print(obj_id, type(obj_id))
                bboxes.append(torch.tensor(det["bbox"], dtype=torch.float))
                obj_labels.append(det["label"])
            if len(bboxes)>0:
                bboxes = torch.stack(bboxes, dim=0)
            else:
                bboxes = torch.empty((0,4), dtype=torch.float)
        else:
            bboxes = torch.empty((0,4), dtype=torch.float) # no detection
        obj_labels = torch.as_tensor(obj_labels)
        return bboxes, obj_labels
        
    
    def get_place_labels(self, frame_idxs):
        """
        index p-p adjencecy, find place labels, use for training place recognition
        Parameters:
         - frame_idxs: the frame idxs in this batch,
        Outputs:
         - place recognition labels, BxB binary matrice
        """
        place_labels = self.pp_adj[frame_idxs][:, frame_idxs]
        return place_labels

    def get_rel_labels(self, obj_idx):
        rel_labels = self.rel_matrix[obj_idx][:, obj_idx]
        return rel_labels
    
    
    def set_objidx_offset(self, offset):
        self.obj_id_offset = offset

    def __getitem__(self, idx):
        # import time
        # aaa = time.time()
        # get the frame id
        frame_id = self.frame_ids[idx]
        bboxes, obj_ids, obj_labels = self.get_det(frame_id)  # (N, 4), (N) two lists
        # bbb = time.time()
        # print('get det time:', bbb-aaa)
        data = dict()
        image_path = os.path.join(self.frame_dir, f'frame-{frame_id[-10:-4]}.color.jpg')

        image = read_image(image_path)
        # ccc = time.time()
        

        # image = image.rot90(3, [1, 2])  # 旋转 270 度
        # print('read image time:', ccc-bbb)
        def get_sam_masks(self, frame_id):
            if self.has_sam_masks:
                return self.sam_masks[frame_id]
            else:
                img = cv2.imread(image_path)
                img_rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

                # ===================== 【核心：SAM 提取 mask】 =====================
                # 确保 self.SAM_model 已经在 __init__ 里加载：self.SAM_model = SAM("sam2_b.pt")
                if bboxes.size(0) == 0:
                    results=[]
                else:
                    with torch.no_grad():  # 禁止计算梯度，提速省显存
                        results = self.SAM_model(
                            img_rotated,
                            bboxes=bboxes.unsqueeze(0),  # 直接用你的 bboxes
                            save=False,      # 这里不需要保存，避免覆盖
                            verbose=False
                        )
                
                # 取出 mask：shape = [N, H, W] （torch tensor）
                if len(results) > 0 and results[0].masks is not None:
                    masks = results[0].masks.data  # [N, H, W]
                else:
                    masks = torch.empty(0, image.shape[1], image.shape[2])
                self.sam_masks[frame_id] = masks
            return masks
        if self.use_sam:
            masks = get_sam_masks(self, frame_id)
        else:
            masks = torch.zeros_like(image)
        # masks
        # ==================================================================
        # ddd = time.time()
        # print('sam time:', ddd-ccc)
        # 原有 transform 逻辑
        if self.transforms is not None:
            image = self.transforms(image)
            # transform boxes accordingly, surrogate when transform API doesn account for bboxes
            bboxes = bboxes.to(torch.float32)
            bboxes[:, 0::2] *= self.new_width / self.orig_width  # (w, h, w ,h)
            bboxes[:, 1::2] *= self.new_height / self.orig_height
            bboxes = bboxes.to(torch.int64)

            # ===================== mask 同步缩放到 transform 后的尺寸 =====================
            H_new, W_new = image.shape[1], image.shape[2]
            if masks.size(0) > 0:
                masks = torch.nn.functional.interpolate(
                    masks.unsqueeze(1).float(),  # [N,1,H,W]
                    size=(H_new, W_new),
                    mode="nearest"
                ).squeeze(1)  # [N, H_new, W_new]
            # ==============================================================================
        # eee = time.time()
        # print('transform time:', eee-ddd)
        # 原有数据
        data['image'] = image
        data['image_idx'] = torch.tensor(self.frame2idx[frame_id])
        data['bbox'] = bboxes
        data['obj_idx'] = obj_ids + self.obj_id_offset
        data['obj_label'] = obj_labels

        # ===================== 【把 mask 存入 data】 =====================
        data['mask'] = masks  # shape: [N, H, W]
        # =================================================================

        if self.use_gdino:
            pred_box, pred_label = self.get_pred_det(frame_id)
            data['pred_bbox'] = pred_box
        # assert 0
        return data

class VideoDataset_3RScan_split(Dataset):

    def __init__(self, video_data_dir, video_id, configs, transforms, split="train", scan_dict=None, use_sam = False):

        self.split = split
        self.video_data_dir = os.path.join(video_data_dir, video_id)
        self.video_id = video_id
        self.transforms = transforms
        self.ori_image_size = configs['image_size']
        # get annotations
        gt_path = r'D:\SceneGraph\code\MSG\dataset\simple_3rscan\refine_topo_gt'
        self.gt = json.load(open(os.path.join(gt_path, f'{video_id}_refine_topo_gt.json'))) # 'topo_gt.json'
        if scan_dict is None:
            print('scan_dict is None')
        else:
            self.rel_matrix = torch.tensor(scan_dict[video_id]['rel_matrix_list'])
            self.obj2id = scan_dict[video_id]['obj2id_dic']
            self.obj2col = self.gt['obj2col']
            self.rel_matrix = matrix_reorder(self.rel_matrix, self.obj2id, self.obj2col)

        # self.gt = json.load(open(os.path.join(self.video_data_dir, 'refine_topo_gt.json'))) # 'topo_gt.json'

        # get the frame ids
        self.frame_dir = os.path.join(self.video_data_dir, 'sequence')
        #NOTE: NOT ALL frames are used -> [int(f.split('.')[0].split('_')[1]) for f in os.listdir(os.path.join(self.data_dir, self.video_id, 'videoid_frames', 'lowres_wide'))]
        self.frame_ids = self.gt['sampled_frames'] 
        self.frame_ids.sort()
        self.frame2idx = {frame_id: idx for idx, frame_id in enumerate(self.frame_ids)}
        self.num_frames = len(self.frame_ids)

        self.obj_id_offset = 0
        self.use_sam = use_sam
        self.sam_masks = {}
        self.has_sam_masks = False
        self.cache_file = os.path.join(self.video_data_dir, f'{video_id}_sam_masks.pt')
        if self.use_sam:
            if os.path.exists(self.cache_file):
                self.has_sam_masks = True
                self.sam_masks = torch.load(self.cache_file, map_location='cpu')
            else:
                print('load sam model')
                from ultralytics import SAM
                self.SAM_model = SAM("sam2_b.pt")
                self.SAM_model.to('cuda')  # 放 GPU 加速

        # get grounding dino detections if detector is grounding-dino
        self.use_gdino = False
        if configs["detector"]["model"] == "grounding-dino":
            self.use_gdino = True
            gdino_file = os.path.join(configs["detector"]["result_path"], split, video_id, 'eval_results.json')
            self.gdino_det = json.load(open(gdino_file))


        # other annotations
        self.obj2col = self.gt['obj2col'] # store which column in the gt annotation corresponds to which obj unique id
        self.pp_adj = torch.tensor(self.gt['p-p'])
        self.pp_adj.fill_diagonal_(1)
        # print("gt diag", self.pp_adj.diagonal())
        self.po_adj = torch.tensor(self.gt['p-o'])
        self.uid2obj = dict()
        for object_name in self.gt['uidmap']: # map object id to object name
            for object_id in self.gt['uidmap'][object_name]:
                self.uid2obj[object_id] = object_name
        self.class_map = configs['class_map'] # map object class name to class id, note that background should be set to 0
        self.image_size = configs['image_size']
        self.target_image_size = configs['model_image_size']
        self.new_width = self.target_image_size[1]
        self.new_height = self.target_image_size[0]
        self.orig_width = self.image_size[1]
        self.orig_height = self.image_size[0]

    def __len__(self,):
        return len(self.frame_ids)
    
    def get_det(self, frame_id):
        # read detection annotation, return list of bboxes and list of object ids
        bboxes = []
        obj_ids = []
        obj_labels = []
        frame_id = str(frame_id)
        if frame_id in self.gt['annotations']: #NOTE: else this frame has no gt objct detections.
            det_dict = self.gt['annotations'][frame_id]
            for obj_id, bbox in det_dict.items():
                # print(obj_id, type(obj_id))
                bboxes.append(torch.tensor(bbox))
                obj_ids.append(self.obj2col[obj_id])
                obj_labels.append(self.class_map[self.uid2obj[int(obj_id)]])
                # obj_labels.append(int(self.uid2obj[int(obj_id)]))
            # bboxes = tv_tensors.BoundingBoxes(bboxes, format='XYXY', canvas_size=self.image_size)
            # NOTE: for the older torchvision model BoundingBoxes are not there, gotta work around:
            bboxes = torch.stack(bboxes, dim=0)
        else:
            bboxes = torch.empty((0,4)) # no detection
        obj_ids = torch.as_tensor(obj_ids)
        obj_labels = torch.as_tensor(obj_labels)
        return bboxes, obj_ids, obj_labels
    
    def get_pred_det(self, frame_id):
        bboxes = []
        obj_labels = []
        frame_id = str(frame_id)
        if frame_id in self.gdino_det['detections']:
            det_dict = self.gdino_det['detections'][frame_id]
            for obj_id, det in det_dict.items():
                # print(obj_id, type(obj_id))
                bboxes.append(torch.tensor(det["bbox"], dtype=torch.float))
                obj_labels.append(det["label"])
            if len(bboxes)>0:
                bboxes = torch.stack(bboxes, dim=0)
            else:
                bboxes = torch.empty((0,4), dtype=torch.float)
        else:
            bboxes = torch.empty((0,4), dtype=torch.float) # no detection
        obj_labels = torch.as_tensor(obj_labels)
        return bboxes, obj_labels
        
    
    def get_place_labels(self, frame_idxs):
        """
        index p-p adjencecy, find place labels, use for training place recognition
        Parameters:
         - frame_idxs: the frame idxs in this batch,
        Outputs:
         - place recognition labels, BxB binary matrice
        """
        place_labels = self.pp_adj[frame_idxs][:, frame_idxs]
        return place_labels

    def get_rel_labels(self, obj_idx):
        real_obj = torch.unique(obj_idx[obj_idx != -1]).long()
        rel_labels = self.rel_matrix[real_obj][:, real_obj]
        return rel_labels
    
    
    def set_objidx_offset(self, offset):
        self.obj_id_offset = offset

    def __getitem__(self, idx):
        # import time
        # aaa = time.time()
        # get the frame id
        frame_id = self.frame_ids[idx]
        bboxes, obj_ids, obj_labels = self.get_det(frame_id)  # (N, 4), (N) two lists
        # bbb = time.time()
        # print('get det time:', bbb-aaa)
        data = dict()
        image_path = os.path.join(self.frame_dir, f'frame-{frame_id[37:43]}.color.jpg')

        image = read_image(image_path)
        # ccc = time.time()
        

        image = image.rot90(3, [1, 2])  # 旋转 270 度
        # print('read image time:', ccc-bbb)
        def get_sam_masks(self, frame_id):
            if self.has_sam_masks:
                return self.sam_masks[frame_id]
            else:
                img = cv2.imread(image_path)
                img_rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

                # ===================== 【核心：SAM 提取 mask】 =====================
                # 确保 self.SAM_model 已经在 __init__ 里加载：self.SAM_model = SAM("sam2_b.pt")
                if bboxes.size(0) == 0:
                    results=[]
                else:
                    with torch.no_grad():  # 禁止计算梯度，提速省显存
                        results = self.SAM_model(
                            img_rotated,
                            bboxes=bboxes.unsqueeze(0),  # 直接用你的 bboxes
                            save=False,      # 这里不需要保存，避免覆盖
                            verbose=False
                        )
                
                # 取出 mask：shape = [N, H, W] （torch tensor）
                if len(results) > 0 and results[0].masks is not None:
                    masks = results[0].masks.data  # [N, H, W]
                else:
                    masks = torch.empty(0, image.shape[1], image.shape[2])
                self.sam_masks[frame_id] = masks
            return masks
        if self.use_sam:
            masks = get_sam_masks(self, frame_id)
        else:
            masks = torch.zeros_like(image)
        # masks
        # ==================================================================
        # ddd = time.time()
        # print('sam time:', ddd-ccc)
        # 原有 transform 逻辑
        if self.transforms is not None:
            image = self.transforms(image)
            # transform boxes accordingly, surrogate when transform API doesn account for bboxes
            bboxes = bboxes.to(torch.float32)
            bboxes[:, 0::2] *= self.new_width / self.orig_width  # (w, h, w ,h)
            bboxes[:, 1::2] *= self.new_height / self.orig_height
            bboxes = bboxes.to(torch.int64)

            # ===================== mask 同步缩放到 transform 后的尺寸 =====================
            H_new, W_new = image.shape[1], image.shape[2]
            if masks.size(0) > 0:
                masks = torch.nn.functional.interpolate(
                    masks.unsqueeze(1).float(),  # [N,1,H,W]
                    size=(H_new, W_new),
                    mode="nearest"
                ).squeeze(1)  # [N, H_new, W_new]
            # ==============================================================================
        # eee = time.time()
        # print('transform time:', eee-ddd)
        # 原有数据
        data['image'] = image
        data['image_idx'] = torch.tensor(self.frame2idx[frame_id])
        data['bbox'] = bboxes
        data['obj_idx'] = obj_ids + self.obj_id_offset
        data['obj_label'] = obj_labels

        # ===================== 【把 mask 存入 data】 =====================
        data['mask'] = masks  # shape: [N, H, W]
        # =================================================================

        if self.use_gdino:
            pred_box, pred_label = self.get_pred_det(frame_id)
            data['pred_bbox'] = pred_box
        # assert 0
        return data

    # def __getitem__(self, idx):
    #     # get the frame id
    #     frame_id = self.frame_ids[idx]
    #     bboxes, obj_ids, obj_labels = self.get_det(frame_id) # (N, 4), (N) two lists
        
    #     data = dict()
    #     # image_path = os.path.join(self.frame_dir, f'{self.video_id}_{frame_id}.png')
    #     image_path = os.path.join(self.frame_dir, f'frame-{frame_id[37:43]}.color.jpg')
    #     image = read_image(image_path)
    #     image = image.rot90(3, [1,2])

    # #     if SAM_model is not None:
    # #         results = SAM_model(
    # #         image,
    # #         bboxes=bboxes,
    # #         save=True,        # 开启保存
    # #         save_dir=save_dir # 指定保存地址 ✅
    # #     )

    # #     masks = results[0].masks.data.numpy()
    # #     mask_resized = cv2.resize(
    # #     masks[0].astype(np.float32), 
    # #     dsize=(224, 224), 
    # #     interpolation=cv2.INTER_NEAREST  # 必须这个！
    # # )

    #     # a = torch.tensor(image)
    #     # b = torch.tensor(bboxes)
    #     if self.transforms is not None:
    #         image = self.transforms(image)
    #         # transform boxes accordingly, surrogate when transform API doesn account for bboxes
    #         bboxes = bboxes.to(torch.float32)

    #         bboxes[:, 0::2] *= self.new_width / self.orig_width #(w, h, w ,h)
    #         bboxes[:, 1::2] *= self.new_height / self.orig_height
    #         bboxes = bboxes.to(torch.int64)
    #     # print("after transform", image1.size())
    #     # print("bbox after transform", bboxes1)
            
    #     data['image'] = image
    #     data['image_idx'] = torch.tensor(self.frame2idx[frame_id])

    #     data['bbox'] = bboxes
    #     data['obj_idx'] = obj_ids + self.obj_id_offset
    #     # data['place_label'] = place_label
    #     data['obj_label'] = obj_labels # this is class label
        
    #     if self.use_gdino:
    #         pred_box, pred_label = self.get_pred_det(frame_id)
    #         data['pred_bbox'] = pred_box

    #     # images: 2 x H x W x 3
    #     # detections: 2 x N x 4, N can be not the same
    #     # objects: 2 x N, N can be not the same
    #     # place_label: 1, if the two frames are the same place or not
    #     return data


class VideoDataset_3RScan(Dataset):

    def __init__(self, video_data_dir, video_id, configs, transforms, split="train", scan_dict=None, use_sam = False):

        self.split = split
        self.video_data_dir = os.path.join(video_data_dir, video_id)
        self.video_id = video_id
        self.transforms = transforms
        self.ori_image_size = configs['image_size']
        # get annotations
        gt_path = r'/root/autodl-tmp/dataset/3rscan_msg/refine_topo_gt'
        self.gt = json.load(open(os.path.join(gt_path, f'{video_id}_refine_topo_gt.json'))) # 'topo_gt.json'
        if scan_dict is None:
            print('scan_dict is None')
        else:
            self.rel_matrix = torch.tensor(scan_dict[split][video_id]['rel_matrix_list'])
            self.obj2id = scan_dict[split][video_id]['obj2id_dic']
            self.obj2col = self.gt['obj2col']
            self.rel_matrix = matrix_reorder(self.rel_matrix, self.obj2id, self.obj2col)

        # self.gt = json.load(open(os.path.join(self.video_data_dir, 'refine_topo_gt.json'))) # 'topo_gt.json'

        # get the frame ids
        self.frame_dir = os.path.join(self.video_data_dir, 'sequence')
        #NOTE: NOT ALL frames are used -> [int(f.split('.')[0].split('_')[1]) for f in os.listdir(os.path.join(self.data_dir, self.video_id, 'videoid_frames', 'lowres_wide'))]
        self.frame_ids = self.gt['sampled_frames'] 
        self.frame_ids.sort()
        self.frame2idx = {frame_id: idx for idx, frame_id in enumerate(self.frame_ids)}
        self.num_frames = len(self.frame_ids)

        self.obj_id_offset = 0
        self.use_sam = use_sam
        self.sam_masks = {}
        self.has_sam_masks = False
        self.cache_file = os.path.join(self.video_data_dir, f'{video_id}_sam_masks.pt')
        if self.use_sam:
            if os.path.exists(self.cache_file):
                self.has_sam_masks = True
                self.sam_masks = torch.load(self.cache_file, map_location='cpu')
            else:
                print('load sam model')
                from ultralytics import SAM
                self.SAM_model = SAM("sam2_b.pt")
                self.SAM_model.to('cuda')  # 放 GPU 加速

        # get grounding dino detections if detector is grounding-dino
        self.use_gdino = False
        if configs["detector"]["model"] == "grounding-dino":
            self.use_gdino = True
            gdino_file = os.path.join(configs["detector"]["result_path"], split, video_id, 'eval_results.json')
            self.gdino_det = json.load(open(gdino_file))


        # other annotations
        self.obj2col = self.gt['obj2col'] # store which column in the gt annotation corresponds to which obj unique id
        self.pp_adj = torch.tensor(self.gt['p-p'])
        self.pp_adj.fill_diagonal_(1)
        # print("gt diag", self.pp_adj.diagonal())
        self.po_adj = torch.tensor(self.gt['p-o'])
        self.uid2obj = dict()
        for object_name in self.gt['uidmap']: # map object id to object name
            for object_id in self.gt['uidmap'][object_name]:
                self.uid2obj[object_id] = object_name
        self.class_map = configs['class_map'] # map object class name to class id, note that background should be set to 0
        self.image_size = configs['image_size']
        self.target_image_size = configs['model_image_size']
        self.new_width = self.target_image_size[1]
        self.new_height = self.target_image_size[0]
        self.orig_width = self.image_size[1]
        self.orig_height = self.image_size[0]

    def __len__(self,):
        return len(self.frame_ids)
    
    def get_det(self, frame_id):
        # read detection annotation, return list of bboxes and list of object ids
        bboxes = []
        obj_ids = []
        obj_labels = []
        frame_id = str(frame_id)
        if frame_id in self.gt['annotations']: #NOTE: else this frame has no gt objct detections.
            det_dict = self.gt['annotations'][frame_id]
            for obj_id, bbox in det_dict.items():
                # print(obj_id, type(obj_id))
                bboxes.append(torch.tensor(bbox))
                obj_ids.append(self.obj2col[obj_id])
                obj_labels.append(self.class_map[self.uid2obj[int(obj_id)]])
                # obj_labels.append(int(self.uid2obj[int(obj_id)]))
            # bboxes = tv_tensors.BoundingBoxes(bboxes, format='XYXY', canvas_size=self.image_size)
            # NOTE: for the older torchvision model BoundingBoxes are not there, gotta work around:
            bboxes = torch.stack(bboxes, dim=0)
        else:
            bboxes = torch.empty((0,4)) # no detection
        obj_ids = torch.as_tensor(obj_ids)
        obj_labels = torch.as_tensor(obj_labels)
        return bboxes, obj_ids, obj_labels
    
    def get_pred_det(self, frame_id):
        bboxes = []
        obj_labels = []
        frame_id = str(frame_id)
        if frame_id in self.gdino_det['detections']:
            det_dict = self.gdino_det['detections'][frame_id]
            for obj_id, det in det_dict.items():
                # print(obj_id, type(obj_id))
                bboxes.append(torch.tensor(det["bbox"], dtype=torch.float))
                obj_labels.append(det["label"])
            if len(bboxes)>0:
                bboxes = torch.stack(bboxes, dim=0)
            else:
                bboxes = torch.empty((0,4), dtype=torch.float)
        else:
            bboxes = torch.empty((0,4), dtype=torch.float) # no detection
        obj_labels = torch.as_tensor(obj_labels)
        return bboxes, obj_labels
        
    
    def get_place_labels(self, frame_idxs):
        """
        index p-p adjencecy, find place labels, use for training place recognition
        Parameters:
         - frame_idxs: the frame idxs in this batch,
        Outputs:
         - place recognition labels, BxB binary matrice
        """
        place_labels = self.pp_adj[frame_idxs][:, frame_idxs]
        return place_labels

    def get_rel_labels(self, obj_idx):
        rel_labels = self.rel_matrix[obj_idx][:, obj_idx]
        return rel_labels
    
    
    def set_objidx_offset(self, offset):
        self.obj_id_offset = offset

    def __getitem__(self, idx):
        # import time
        # aaa = time.time()
        # get the frame id
        frame_id = self.frame_ids[idx]
        bboxes, obj_ids, obj_labels = self.get_det(frame_id)  # (N, 4), (N) two lists
        # bbb = time.time()
        # print('get det time:', bbb-aaa)
        data = dict()
        image_path = os.path.join(self.frame_dir, f'frame-{frame_id[37:43]}.color.jpg')

        image = read_image(image_path)
        # ccc = time.time()
        

        image = image.rot90(3, [1, 2])  # 旋转 270 度
        # print('read image time:', ccc-bbb)
        def get_sam_masks(self, frame_id):
            if self.has_sam_masks:
                return self.sam_masks[frame_id]
            else:
                img = cv2.imread(image_path)
                img_rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

                # ===================== 【核心：SAM 提取 mask】 =====================
                # 确保 self.SAM_model 已经在 __init__ 里加载：self.SAM_model = SAM("sam2_b.pt")
                if bboxes.size(0) == 0:
                    results=[]
                else:
                    with torch.no_grad():  # 禁止计算梯度，提速省显存
                        results = self.SAM_model(
                            img_rotated,
                            bboxes=bboxes.unsqueeze(0),  # 直接用你的 bboxes
                            save=False,      # 这里不需要保存，避免覆盖
                            verbose=False
                        )
                
                # 取出 mask：shape = [N, H, W] （torch tensor）
                if len(results) > 0 and results[0].masks is not None:
                    masks = results[0].masks.data  # [N, H, W]
                else:
                    masks = torch.empty(0, image.shape[1], image.shape[2])
                self.sam_masks[frame_id] = masks
            return masks
        if self.use_sam:
            masks = get_sam_masks(self, frame_id)
        else:
            masks = torch.zeros_like(image)
        # masks
        # ==================================================================
        # ddd = time.time()
        # print('sam time:', ddd-ccc)
        # 原有 transform 逻辑
        if self.transforms is not None:
            image = self.transforms(image)
            # transform boxes accordingly, surrogate when transform API doesn account for bboxes
            bboxes = bboxes.to(torch.float32)
            bboxes[:, 0::2] *= self.new_width / self.orig_width  # (w, h, w ,h)
            bboxes[:, 1::2] *= self.new_height / self.orig_height
            bboxes = bboxes.to(torch.int64)

            # ===================== mask 同步缩放到 transform 后的尺寸 =====================
            H_new, W_new = image.shape[1], image.shape[2]
            if masks.size(0) > 0:
                masks = torch.nn.functional.interpolate(
                    masks.unsqueeze(1).float(),  # [N,1,H,W]
                    size=(H_new, W_new),
                    mode="nearest"
                ).squeeze(1)  # [N, H_new, W_new]
            # ==============================================================================
        # eee = time.time()
        # print('transform time:', eee-ddd)
        # 原有数据
        data['image'] = image
        data['image_idx'] = torch.tensor(self.frame2idx[frame_id])
        data['bbox'] = bboxes
        data['obj_idx'] = obj_ids + self.obj_id_offset
        data['obj_label'] = obj_labels

        # ===================== 【把 mask 存入 data】 =====================
        data['mask'] = masks  # shape: [N, H, W]
        # =================================================================

        if self.use_gdino:
            pred_box, pred_label = self.get_pred_det(frame_id)
            data['pred_bbox'] = pred_box
        # assert 0
        return data

    # def __getitem__(self, idx):
    #     # get the frame id
    #     frame_id = self.frame_ids[idx]
    #     bboxes, obj_ids, obj_labels = self.get_det(frame_id) # (N, 4), (N) two lists
        
    #     data = dict()
    #     # image_path = os.path.join(self.frame_dir, f'{self.video_id}_{frame_id}.png')
    #     image_path = os.path.join(self.frame_dir, f'frame-{frame_id[37:43]}.color.jpg')
    #     image = read_image(image_path)
    #     image = image.rot90(3, [1,2])

    # #     if SAM_model is not None:
    # #         results = SAM_model(
    # #         image,
    # #         bboxes=bboxes,
    # #         save=True,        # 开启保存
    # #         save_dir=save_dir # 指定保存地址 ✅
    # #     )

    # #     masks = results[0].masks.data.numpy()
    # #     mask_resized = cv2.resize(
    # #     masks[0].astype(np.float32), 
    # #     dsize=(224, 224), 
    # #     interpolation=cv2.INTER_NEAREST  # 必须这个！
    # # )

    #     # a = torch.tensor(image)
    #     # b = torch.tensor(bboxes)
    #     if self.transforms is not None:
    #         image = self.transforms(image)
    #         # transform boxes accordingly, surrogate when transform API doesn account for bboxes
    #         bboxes = bboxes.to(torch.float32)

    #         bboxes[:, 0::2] *= self.new_width / self.orig_width #(w, h, w ,h)
    #         bboxes[:, 1::2] *= self.new_height / self.orig_height
    #         bboxes = bboxes.to(torch.int64)
    #     # print("after transform", image1.size())
    #     # print("bbox after transform", bboxes1)
            
    #     data['image'] = image
    #     data['image_idx'] = torch.tensor(self.frame2idx[frame_id])

    #     data['bbox'] = bboxes
    #     data['obj_idx'] = obj_ids + self.obj_id_offset
    #     # data['place_label'] = place_label
    #     data['obj_label'] = obj_labels # this is class label
        
    #     if self.use_gdino:
    #         pred_box, pred_label = self.get_pred_det(frame_id)
    #         data['pred_bbox'] = pred_box

    #     # images: 2 x H x W x 3
    #     # detections: 2 x N x 4, N can be not the same
    #     # objects: 2 x N, N can be not the same
    #     # place_label: 1, if the two frames are the same place or not
    #     return data

class VideoDataset(Dataset):

    def __init__(self, video_data_dir, video_id, configs, transforms, split="train"):
        self.split = split
        self.video_data_dir = os.path.join(video_data_dir, video_id)
        self.video_id = video_id
        self.transforms = transforms
        self.ori_image_size = configs['image_size']
        # get annotations
        self.gt = json.load(open(os.path.join(self.video_data_dir, 'refine_topo_gt.json'))) # 'topo_gt.json'
        # get the frame ids
        self.frame_dir = os.path.join(self.video_data_dir, self.video_id+'_frames', 'lowres_wide')
        #NOTE: NOT ALL frames are used -> [int(f.split('.')[0].split('_')[1]) for f in os.listdir(os.path.join(self.data_dir, self.video_id, 'videoid_frames', 'lowres_wide'))]
        self.frame_ids = self.gt['sampled_frames'] 
        self.frame_ids.sort()
        self.frame2idx = {frame_id: idx for idx, frame_id in enumerate(self.frame_ids)}
        self.num_frames = len(self.frame_ids)

        self.obj_id_offset = 0
        
        # get grounding dino detections if detector is grounding-dino
        self.use_gdino = False
        if configs["detector"]["model"] == "grounding-dino":
            self.use_gdino = True
            gdino_file = os.path.join(configs["detector"]["result_path"], split, video_id, 'eval_results.json')
            self.gdino_det = json.load(open(gdino_file))


        # other annotations
        self.obj2col = self.gt['obj2col'] # store which column in the gt annotation corresponds to which obj unique id
        self.pp_adj = torch.tensor(self.gt['p-p'])
        self.pp_adj.fill_diagonal_(1)
        # print("gt diag", self.pp_adj.diagonal())
        self.po_adj = torch.tensor(self.gt['p-o'])
        self.uid2obj = dict()
        for object_name in self.gt['uidmap']: # map object id to object name
            for object_id in self.gt['uidmap'][object_name]:
                self.uid2obj[object_id] = object_name
        self.class_map = configs['class_map'] # map object class name to class id, note that background should be set to 0
        self.image_size = configs['image_size']
        self.target_image_size = configs['model_image_size']
        self.new_width = self.target_image_size[1]
        self.new_height = self.target_image_size[0]
        self.orig_width = self.image_size[1]
        self.orig_height = self.image_size[0]

    def __len__(self,):
        return len(self.frame_ids)
    
    def get_det(self, frame_id):
        # read detection annotation, return list of bboxes and list of object ids
        bboxes = []
        obj_ids = []
        obj_labels = []
        frame_id = str(frame_id)
        if frame_id in self.gt['annotations']: #NOTE: else this frame has no gt objct detections.
            det_dict = self.gt['annotations'][frame_id]
            for obj_id, bbox in det_dict.items():
                # print(obj_id, type(obj_id))
                bboxes.append(torch.tensor(bbox))
                obj_ids.append(self.obj2col[obj_id])
                obj_labels.append(self.class_map[self.uid2obj[obj_id]])
            # bboxes = tv_tensors.BoundingBoxes(bboxes, format='XYXY', canvas_size=self.image_size)
            # NOTE: for the older torchvision model BoundingBoxes are not there, gotta work around:
            bboxes = torch.stack(bboxes, dim=0)
        else:
            bboxes = torch.empty((0,4)) # no detection
        obj_ids = torch.as_tensor(obj_ids)
        obj_labels = torch.as_tensor(obj_labels)
        return bboxes, obj_ids, obj_labels
    
    def get_pred_det(self, frame_id):
        bboxes = []
        obj_labels = []
        frame_id = str(frame_id)
        if frame_id in self.gdino_det['detections']:
            det_dict = self.gdino_det['detections'][frame_id]
            for obj_id, det in det_dict.items():
                # print(obj_id, type(obj_id))
                bboxes.append(torch.tensor(det["bbox"], dtype=torch.float))
                obj_labels.append(det["label"])
            if len(bboxes)>0:
                bboxes = torch.stack(bboxes, dim=0)
            else:
                bboxes = torch.empty((0,4), dtype=torch.float)
        else:
            bboxes = torch.empty((0,4), dtype=torch.float) # no detection
        obj_labels = torch.as_tensor(obj_labels)
        return bboxes, obj_labels
        
    
    def get_place_labels(self, frame_idxs):
        """
        index p-p adjencecy, find place labels, use for training place recognition
        Parameters:
         - frame_idxs: the frame idxs in this batch,
        Outputs:
         - place recognition labels, BxB binary matrice
        """
        place_labels = self.pp_adj[frame_idxs][:, frame_idxs]
        return place_labels
    
    
    def set_objidx_offset(self, offset):
        self.obj_id_offset = offset

    
    def __getitem__(self, idx):
        # get the frame id
        frame_id = self.frame_ids[idx]
        bboxes, obj_ids, obj_labels = self.get_det(frame_id) # (N, 4), (N) two lists
        
        data = dict()
        image_path = os.path.join(self.frame_dir, f'{self.video_id}_{frame_id}.png')
        image = read_image(image_path)
        if self.transforms is not None:
            image = self.transforms(image)
            # transform boxes accordingly, surrogate when transform API doesn account for bboxes
            bboxes = bboxes.to(torch.float32)

            bboxes[:, 0::2] *= self.new_width / self.orig_width #(w, h, w ,h)
            bboxes[:, 1::2] *= self.new_height / self.orig_height
            bboxes = bboxes.to(torch.int64)
        # print("after transform", image1.size())
        # print("bbox after transform", bboxes1)
            
        data['image'] = image
        data['image_idx'] = torch.tensor(self.frame2idx[frame_id])

        data['bbox'] = bboxes
        data['obj_idx'] = obj_ids + self.obj_id_offset
        # data['place_label'] = place_label
        data['obj_label'] = obj_labels # this is class label
        
        if self.use_gdino:
            pred_box, pred_label = self.get_pred_det(frame_id)
            data['pred_bbox'] = pred_box

        # images: 2 x H x W x 3
        # detections: 2 x N x 4, N can be not the same
        # objects: 2 x N, N can be not the same
        # place_label: 1, if the two frames are the same place or not
        return data
    
class MultiVideoDataset(Dataset):
    """
    This class wraps VideoDataset.
    Supports multi video data loading. Only used for training
    """
    def __init__(self, video_data_dir, video_ids, configs, transforms, batch_size, split="train", scan_dict=None):
        self.datasets = []
        self.scan_dict = scan_dict
        objidx_offset_counter = 0
        for vid in video_ids:
            # dt = VideoDataset(video_data_dir, vid, configs, transforms, split=split)
            # import time
            # aaa = time.time()
            # dt = VideoDataset_3RScan(video_data_dir, vid, configs, transforms, split=split, scan_dict=self.scan_dict)
            # dt = VideoDataset_Replica(video_data_dir, vid, configs, transforms, split=split, scan_dict=self.scan_dict)
            # dt = VideoDataset_Replica_small(video_data_dir, vid, configs, transforms, split=split, scan_dict=self.scan_dict)
            dt = VideoDataset_3RScan_split(video_data_dir, vid, configs, transforms, split=split, scan_dict=self.scan_dict)

            # dt = VideoDataset_Replica_small_split(video_data_dir, vid, configs, transforms, split=split, scan_dict=self.scan_dict)

            # bbb = time.time()
            # print('load 3RScan time:', bbb-aaa)
            dt.set_objidx_offset(objidx_offset_counter)
            self.datasets.append(dt)
            # accumulate the number of objects,
            # used to offset the object id so that they don't overlap
            objidx_offset_counter += len(dt.obj2col)

        self.video_ids = video_ids
        self.total_batch_size = batch_size
        self.bs_per_video = batch_size // len(video_ids)

        self.lens_per_video = torch.tensor([len(dt) for dt in self.datasets])
        self.min_len = self.lens_per_video.min().item()
        self.max_len = self.lens_per_video.max().item()

    def __len__(self,):
        return self.max_len
    
    def __getitem__(self, idx):
        batch_data = {}
        # idx = 0
        for dataset, length in zip(self.datasets, self.lens_per_video):
            index = idx % length
            batch_data[dataset.video_id] = dataset[index]

        return batch_data
    
    def get_place_labels(self, frame_idxs, num_per_vid, vid_idx):
        B = frame_idxs.size(0)
        place_labels = torch.zeros(B, B, dtype=torch.int)
        offset = 0
        for didx, dataset in enumerate(self.datasets):
            num_frames = num_per_vid[didx]
            # assert torch.all(vid_idx[offset: offset + num_frames] == int(dataset.video_id))
            block_frame_idx = frame_idxs[offset: offset + num_frames]
            block_place_labels = dataset.get_place_labels(block_frame_idx)
            
            place_labels[offset:offset+num_frames, offset:offset+num_frames] = block_place_labels
            offset += num_frames
        return place_labels

    def get_rel_labels(self, obj_idx):

        rel_matrix_list = []
        for didx, dataset in enumerate(self.datasets):
            rel_matrix_list.append(dataset.rel_matrix)
        rel_matrix = torch.block_diag(*rel_matrix_list)

        real_obj = torch.unique(obj_idx[obj_idx != -1])
        rel_labels = rel_matrix[real_obj][:, real_obj]

        return rel_labels
    

# def multivideo_collate_fn(batch):
#     """
#     1 x batch = bs x subdict 
#     1 subdict = dict{videoId: data point per video}}
#     flatten the batch
#     """
#     scene_ids = []
#     # group by video id
#     groups = {}
#     for subdict in batch:
#         for video_id, item in subdict.items():
#             if video_id not in groups:
#                 groups[video_id] = list()
#                 scene_ids.append(video_id)
#             groups[video_id].append(item)
#     # flatten
#     flat_batch = []
#     num_per_vid = []
#     batch_vid_idx = []
#     for video_id in groups:
#         bs_per_vid = len(groups[video_id])
#         num_per_vid.append(bs_per_vid)
#         batch_vid_idx.extend([scene_ids.index(video_id)]*bs_per_vid)
#         flat_batch.extend(groups[video_id])

#     # collate
#     batch_images = torch.stack([item['image'] for item in flat_batch])
#     batch_bboxes = pad_sequence([item['bbox'] for item in flat_batch], batch_first=True, padding_value=-1)
#     batch_obj_ids = pad_sequence([item['obj_idx'] for item in flat_batch], batch_first=True, padding_value=-1)
#     batch_obj_labels = pad_sequence([item['obj_label'] for item in flat_batch], batch_first=True, padding_value=-1)
#     batch_img_idx = torch.stack([item['image_idx'] for item in flat_batch])

#     # ===================== 【新增：mask 批处理 + 对齐填充】 =====================
#     # 取出所有 mask：每个 item['mask'] 形状 [N, H, W]
#     masks_list = [item['mask'] for item in flat_batch]
    
#     # 获取最大物体数（和 bbox 保持一致）
#     max_obj = batch_bboxes.size(1)
    
#     # 统一 padded masks: [B, max_obj, H, W]
#     batch_masks = []
#     _, H, W = batch_images.shape[-3], batch_images.shape[-2], batch_images.shape[-1]
#     for mask in masks_list:
#         N = mask.size(0)
#         padded = torch.zeros(max_obj, H, W, dtype=mask.dtype, device='cpu')
#         if N > 0:
#             padded[:N] = mask
#         batch_masks.append(padded)
#     batch_masks = torch.stack(batch_masks)  # [B, max_obj, H, W]
#     # ==========================================================================

#     batch_mask = (batch_obj_ids != -1)  # 有效物体掩码
#     batch_vid_idx = torch.tensor(batch_vid_idx)
#     batch_num_per_vid = torch.tensor(num_per_vid)

#     return {
#         'image': batch_images,                # B x 3 x H x W
#         'image_idx': batch_img_idx,            # B
#         'bbox': batch_bboxes,                  # B x padded_N x 4
#         'obj_idx': batch_obj_ids,              # B x padded_N
#         'obj_label': batch_obj_labels,         # B x padded_N
#         'mask': batch_mask,                    # B x padded_N  (有效物体mask)
#         'masks': batch_masks,                  # 【新增】B x padded_N x H x W  (SAM 掩码)
#         'vid_idx': batch_vid_idx,              # (B,)
#         'num_per_vid': batch_num_per_vid,      # (num_videos,)
#     }

def multivideo_collate_fn(batch):
    """
    1 x batch = bs x subdict 
    1 subdict = dict{videoId: data point per video}
    flatten the batch
    """
    # import time
    # aaa = time.time()
    scene_ids = []
    # group by video id
    groups = {}
    for subdict in batch:
        for video_id, item in subdict.items():
            if video_id not in groups:
                groups[video_id] = list()
                scene_ids.append(video_id)
            groups[video_id].append(item)
    # flatten
    flat_batch = []
    num_per_vid = []
    batch_vid_idx = []
    for video_id in groups:
        bs_per_vid = len(groups[video_id])
        num_per_vid.append(bs_per_vid)
        # batch_vid_idx.extend([int(video_id)]*bs_per_vid)
        batch_vid_idx.extend([scene_ids.index(video_id)]*bs_per_vid)
        flat_batch.extend(groups[video_id])

    # collate
    batch_images = torch.stack([item['image'] for item in flat_batch])
    batch_bboxes = pad_sequence([item['bbox'] for item in flat_batch], batch_first=True, padding_value=-1)
    batch_obj_ids = pad_sequence([item['obj_idx'] for item in flat_batch], batch_first=True, padding_value=-1)
    batch_obj_labels = pad_sequence([item['obj_label'] for item in flat_batch], batch_first=True, padding_value=-1)
    batch_mask = (batch_obj_ids != -1)
    batch_img_idx = torch.stack([item['image_idx'] for item in flat_batch])

    batch_num_per_vid = torch.tensor(num_per_vid)
    
    return {
        'image': batch_images, # B x 3 x H x W
        'image_idx': batch_img_idx.type(torch.int64), # B
        'bbox': batch_bboxes, # B x padded N1 x 4
        'obj_idx': batch_obj_ids.type(torch.int64), # B x padded N1
        'obj_label': batch_obj_labels.type(torch.int64), # B x padded N1
        'mask': batch_mask, # B x padded N1 
        'vid_idx': batch_vid_idx, # (B,)
        'num_per_vid': batch_num_per_vid, # (num_videos,)
    }

def generate_mask(sequence, pad_value=0):
    return (sequence != pad_value).any(dim=-1) # if generates mask according to the bounding boxes (last dimension)

def arkit_collate_fn(batch):
    """
    custom collate function for the arkit dataset
    handles padding of the detection bounding boxes and other annotations with various lengths
    generate masks for the padded regions
    """
    # first images, no padding is needed:
    batch_images = torch.stack([item['image'] for item in batch])
    
    # then detections, padding is needed:
    batch_bboxes = pad_sequence([item['bbox'] for item in batch], batch_first=True, padding_value=-1)
    batch_obj_ids = pad_sequence([item['obj_idx'] for item in batch], batch_first=True, padding_value=-1)
    batch_obj_labels = pad_sequence([item['obj_label'] for item in batch], batch_first=True, padding_value=-1)
    
    # mask for padding
    batch_mask = (batch_obj_ids != -1)
    
    batch_img_idx = torch.stack([item['image_idx'] for item in batch])

    ret = {
        'image': batch_images,          # B x 3 x H x W
        'image_idx': batch_img_idx,     # B
        'bbox': batch_bboxes,           # B x padded N1 x 4
        'obj_idx': batch_obj_ids,       # B x padded N1
        'obj_label': batch_obj_labels,  # B x padded N1
        'mask': batch_mask,             # B x padded N1
        # 'masks': batch_masks,           # 👈 新增：SAM 掩码 [B, max_obj, H, W]
    }

    # ===================== 【新增：MASK 批处理 + 填充】 =====================
    # 获取每个样本的 mask
    if "masks" in batch[0]:
        masks_list = [item['masks'] for item in batch]
        max_obj = batch_bboxes.size(1)  # 和 bbox 保持相同的最大物体数
        B = len(batch)
        H, W = batch_images.shape[-2], batch_images.shape[-1]

        batch_masks = []
        for masks in masks_list:
            # 全部保持在 CPU，避免子进程 CUDA 错误
            N = masks.size(0)
            padded_mask = torch.zeros(N, H, W, dtype=torch.float32)  # 按真实数量N创建
            if N > 0:
                padded_mask[:N] = masks.cpu()  # 强制 CPU
            batch_masks.append(padded_mask)

        batch_masks = torch.stack(batch_masks)  # [B, max_obj, H, W]
        ret['masks'] = batch_masks
    # ======================================================================

    if "pred_bbox" in batch[0]:
        batch_pred_bbox = pad_sequence([item['pred_bbox'] for item in batch], batch_first=True, padding_value=-1)
        batch_pred_bbox_mask = (batch_pred_bbox != -1).any(dim=2)
        ret['pred_bbox'] = batch_pred_bbox
        ret['pred_bbox_mask'] = batch_pred_bbox_mask

    return ret

######################################################
# for simple inference, when no ground truth
class SimpleDataset(Dataset):

    def __init__(self, video_data_dir, video_id, configs, transforms, split="train"):
        self.split = split
        self.video_data_dir = os.path.join(video_data_dir, video_id)
        self.video_id = video_id
        self.transforms = transforms
        self.ori_image_size = configs['image_size']
        # get the frame ids
        self.frame_dir = os.path.join(self.video_data_dir, self.video_id+'_frames', 'lowres_wide')
        self.frame_ids = [fid.split(".png")[0].split("_")[-1] for fid in os.listdir(self.frame_dir) if fid.endswith(".png")]
        self.frame_ids.sort()
        self.frame2idx = {frame_id: idx for idx, frame_id in enumerate(self.frame_ids)}
        self.num_frames = len(self.frame_ids)

        self.obj_id_offset = 0
        
        # get grounding dino detections if detector is grounding-dino
        self.use_gdino = False
        if configs["detector"]["model"] == "grounding-dino":
            if configs["detector"]["pre_saved"]:
                self.use_gdino = True
                gdino_file = os.path.join(configs["detector"]["result_path"], split, video_id, 'eval_results.json')
                self.gdino_det = json.load(open(gdino_file))


        self.class_map = configs['class_map'] # map object class name to class id, note that background should be set to 0
        self.image_size = configs['image_size']
        self.target_image_size = configs['model_image_size']
        self.new_width = self.target_image_size[1]
        self.new_height = self.target_image_size[0]
        self.orig_width = self.image_size[1]
        self.orig_height = self.image_size[0]

    def __len__(self,):
        return len(self.frame_ids)
    
    
    def get_pred_det(self, frame_id):
        bboxes = []
        obj_labels = []
        frame_id = str(frame_id)
        if frame_id in self.gdino_det['detections']: 
            det_dict = self.gdino_det['detections'][frame_id]
            if isinstance(det_dict, dict):
                for obj_id, det in det_dict.items():
                    bboxes.append(torch.tensor(det["bbox"], dtype=torch.float))
                    label = det["label"]
                    if isinstance(det["label"], str):
                        label = self.class_map[det["label"]]
                    obj_labels.append(label)
            else:
                for obj_id, det in enumerate(det_dict): # it is actually a list
                    bboxes.append(torch.tensor(det["bbox"], dtype=torch.float))
                    label = det["label"]
                    if isinstance(det["label"], str):
                        label = self.class_map[det["label"]]
                    obj_labels.append(label)
            if len(bboxes)>0:
                bboxes = torch.stack(bboxes, dim=0)
            else:
                bboxes = torch.empty((0,4), dtype=torch.float)
        else:
            bboxes = torch.empty((0,4), dtype=torch.float) # no detection
        obj_labels = torch.as_tensor(obj_labels)
        return bboxes, obj_labels
        
    
    
    def set_objidx_offset(self, offset):
        self.obj_id_offset = offset

    
    def __getitem__(self, idx):
        # get the frame id
        frame_id = self.frame_ids[idx]
        
        data = dict()
        image_path = os.path.join(self.frame_dir, f'{self.video_id}_{frame_id}.png')
        image = read_image(image_path)
        if self.transforms is not None:
            image = self.transforms(image)
            
        data['image'] = image
        data['image_idx'] = torch.tensor(self.frame2idx[frame_id])
        
        if self.use_gdino:
            bboxes, pred_label = self.get_pred_det(frame_id)
            bboxes = bboxes.to(torch.float32)

            bboxes[:, 0::2] *= self.new_width / self.orig_width #(w, h, w ,h)
            bboxes[:, 1::2] *= self.new_height / self.orig_height
            # bboxes = bboxes.to(torch.int64)
            data['pred_bbox'] = bboxes
            data['pred_label'] = pred_label
            

        return data
    
    
def simple_collate_fn(batch):
    batch_images = torch.stack([item['image'] for item in batch])
    
    batch_img_idx = torch.stack([item['image_idx'] for item in batch])
    
    ret = {
        'image': batch_images, # B x 3 x H x W
        'image_idx': batch_img_idx, # B
    }
    
    if "pred_bbox" in batch[0]:
        batch_pred_bbox = pad_sequence([item['pred_bbox'] for item in batch], batch_first=True, padding_value=-1)
        batch_pred_bbox_mask = (batch_pred_bbox != -1).any(dim=2)
        batch_pred_label = pad_sequence([item['pred_label'] for item in batch], batch_first=True, padding_value=-1)
        
        ret['pred_bbox'] = batch_pred_bbox
        ret['pred_bbox_mask'] = batch_pred_bbox_mask
        ret['pred_label'] = batch_pred_label
        
    return ret
        