"""
Query-aware Mapper: 将 query-based 模型输出转换为 evaluator 可接受的 prediction 格式

核心功能：
1. 从 query 输出中提取 object-level 预测（而非 bbox-level）
2. 构建 object feature bank 用于后续评估
3. 支持 place-place 关系构建
4. 兼容旧版 TopoMapperHandler 的输出格式
"""

import os
import json
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment


class QueryMapperHandler(object):
    """
    Query-aware 的 mapper，用于处理 step2/3/4 的 query-based 输出
    
    与旧版 TopoMapperHandler 的区别：
    - 旧版：基于 bbox-level embeddings，通过相似度聚类得到 object
    - 新版：直接基于 query-level embeddings，每个 query 代表一个 object
    """

    def __init__(self, config, video_data_dir, video_id, dataset=None):
        self.config = config
        self.video_id = video_id
        self.video_data_dir = video_data_dir
        
        # 加载 GT 数据
        if dataset is None:
            gt_path = os.path.join(self.video_data_dir, 'refine_topo_gt.json')
            if os.path.exists(gt_path):
                self.gt = json.load(open(gt_path))
                self.frame_ids = self.gt['sampled_frames']
            else:
                self.frame_dir = os.path.join(self.video_data_dir, self.video_id + '_frames', 'lowres_wide')
                self.frame_ids = [fid.split(".png")[0].split("_")[-1] 
                                 for fid in os.listdir(self.frame_dir) if fid.endswith(".png")]
        elif dataset == '3rscan' or dataset == '3rscan_split':
            gt_path = r'/root/autodl-tmp/dataset/3rscan_msg/refine_topo_gt'
            self.gt = json.load(open(os.path.join(gt_path, f'{video_id}_refine_topo_gt.json')))
            self.frame_ids = self.gt['sampled_frames']
        elif dataset == 'Replica':
            gt_path = r'/root/autodl-tmp/dataset/Replica/refine_topo_gt'
            self.gt = json.load(open(os.path.join(gt_path, f'{video_id}_refine_topo_gt.json')))
            self.frame_ids = self.gt['sampled_frames']
        elif dataset == 'Replica_small':
            gt_path = r'/root/autodl-tmp/dataset/Replica_small/refine_topo_gt'
            self.gt = json.load(open(os.path.join(gt_path, f'{video_id}_refine_topo_gt.json')))
            self.frame_ids = self.gt['sampled_frames']
        elif dataset == 'Replica_small_split':
            gt_path = r'/root/autodl-tmp/dataset/Replica_small_split/refine_topo_gt'
            self.gt = json.load(open(os.path.join(gt_path, f'{video_id}.json')))
            self.frame_ids = self.gt['sampled_frames']
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        self.frame_ids.sort()
        self.num_frames = len(self.frame_ids)
        self.frame2idx = {frame_id: idx for idx, frame_id in enumerate(self.frame_ids)}
        
        # 初始化存储
        self.object_bank = {}  # object_id -> {appearance, class, uid, query_idx}
        self.object_feature_bank = None  # M x C
        self.place_feature_bank = dict()  # frame_idx -> embedding
        
        self.pp_adj = torch.zeros((self.num_frames, self.num_frames))
        self.pp_threshold = config.get('pp_threshold', 0.5)
        self.object_threshold = config.get('object_threshold', 0.5)
        self.label2class = self.config.get('inv_class_map', {})
        
        # 用于分析
        self.all_obj_feature = []
        self.all_obj_uid = []
        self.all_obj_label = []
        self.all_query_usage = []  # 记录 query 使用情况

    def map_update(self, batch_data, batch_results):
        """
        更新 mapper，处理 query-based 输出
        
        Args:
            batch_data: 包含 image_idx 等
            batch_results: 模型输出，包含:
                - object_node_feat: [B, Q, C] query features
                - place_node_feat: [B, Q_place, C] place features  
                - object_exist_logits: [B, Q] existence scores
                - object_cls_logits: [B, Q, num_classes] classification logits
                - detections: list of detection dict
        """
        B = batch_data['image_idx'].size(0)
        
        for i in range(B):
            image_idx = batch_data['image_idx'][i].item()
            frame_id = self.frame_ids[image_idx] if image_idx < len(self.frame_ids) else str(image_idx)
            
            # 提取 query features
            obj_queries = batch_results['object_node_feat'][i]  # [Q, C]
            place_queries = batch_results.get('place_node_feat', [None])[i] if 'place_node_feat' in batch_results else None
            
            # 提取存在性分数
            exist_scores = torch.sigmoid(batch_results.get('object_exist_logits', torch.zeros(1))[i]) if 'object_exist_logits' in batch_results else torch.ones(obj_queries.size(0))
            
            # 提取分类预测
            if 'object_cls_logits' in batch_results:
                cls_probs = torch.softmax(batch_results['object_cls_logits'][i], dim=-1)  # [Q, num_classes]
                cls_preds = cls_probs.argmax(dim=-1)  # [Q]
            else:
                cls_preds = torch.zeros(obj_queries.size(0), dtype=torch.long)
            
            # 获取 detections
            detections = batch_results['detections'][i] if isinstance(batch_results['detections'], list) else None
            
            # 更新 place feature bank
            if place_queries is not None:
                # 使用 mean pooling 或第一个 query 作为 place embedding
                place_embedding = place_queries.mean(dim=0) if place_queries.dim() > 0 else place_queries
                self.place_feature_bank[image_idx] = place_embedding
            
            # 更新 object bank（基于 query）
            self._update_object_bank_from_queries(
                image_idx=image_idx,
                frame_id=frame_id,
                obj_queries=obj_queries,
                exist_scores=exist_scores,
                cls_preds=cls_preds,
                detections=detections
            )
            
            # 记录分析数据
            self.all_obj_feature.append(obj_queries)
            if detections is not None:
                self.all_obj_uid.append(detections.get('uids', []))
                self.all_obj_label.append(detections.get('labels', []))
            
            # 记录 query usage
            self.all_query_usage.append({
                'frame_id': frame_id,
                'num_queries': obj_queries.size(0),
                'exist_scores': exist_scores.detach().cpu().numpy(),
                'cls_preds': cls_preds.detach().cpu().numpy(),
            })

    def _update_object_bank_from_queries(self, image_idx, frame_id, obj_queries, exist_scores, cls_preds, detections):
        """
        从 query features 更新 object bank
        
        策略：
        1. 每个 query 代表一个潜在的 object
        2. 使用存在性分数过滤背景 query
        3. 使用匈牙利匹配与已有 object 关联
        """
        Q, C = obj_queries.shape
        
        # 过滤低存在性分数的 query
        valid_mask = exist_scores > self.object_threshold
        valid_queries = obj_queries[valid_mask]  # [K_valid, C]
        valid_cls = cls_preds[valid_mask]
        valid_scores = exist_scores[valid_mask]
        
        if valid_queries.size(0) == 0:
            return
        
        # 初始化 object bank
        if self.object_feature_bank is None:
            self.object_feature_bank = valid_queries.detach().cpu()
            for q_idx in range(valid_queries.size(0)):
                obj_id = len(self.object_bank)
                label = valid_cls[q_idx].item()
                uid = detections['uids'][q_idx].item() if detections is not None and 'uids' in detections and q_idx < len(detections['uids']) else -1
                bbox = detections['boxes'][q_idx] if detections is not None and 'boxes' in detections and q_idx < len(detections['boxes']) else torch.zeros(4)
                
                self.object_bank[obj_id] = {
                    'appearance': [(image_idx, frame_id, bbox, label, valid_scores[q_idx].item())],
                    'class': label,
                    'uid': uid,
                    'query_idx': q_idx,
                    'feature': valid_queries[q_idx].detach().cpu(),
                }
            return
        
        # 与已有 object 进行匹配
        device = valid_queries.device
        bank_features = self.object_feature_bank.to(device)  # [M, C]
        
        # 计算相似度
        sim_matrix = torch.cosine_similarity(
            valid_queries.unsqueeze(1),  # [K_valid, 1, C]
            bank_features.unsqueeze(0),   # [1, M, C]
            dim=-1
        )  # [K_valid, M]
        
        # 匈牙利匹配
        query_ids, existing_ids = linear_sum_assignment(sim_matrix.detach().cpu().numpy(), maximize=True)
        
        matched_queries = set()
        for q_idx, obj_id in zip(query_ids, existing_ids):
            if sim_matrix[q_idx, obj_id] > self.object_threshold:
                matched_queries.add(q_idx)
                # 更新已有 object
                label = valid_cls[q_idx].item()
                uid = detections['uids'][q_idx].item() if detections is not None and 'uids' in detections and q_idx < len(detections['uids']) else -1
                bbox = detections['boxes'][q_idx] if detections is not None and 'boxes' in detections and q_idx < len(detections['boxes']) else torch.zeros(4)
                
                self.object_bank[obj_id]['appearance'].append(
                    (image_idx, frame_id, bbox, label, valid_scores[q_idx].item())
                )
                # 更新特征（加权平均）
                count = len(self.object_bank[obj_id]['appearance'])
                self.object_bank[obj_id]['feature'] = (
                    self.object_bank[obj_id]['feature'] * (count - 1) + valid_queries[q_idx].detach().cpu()
                ) / count
                self.object_feature_bank[obj_id] = self.object_bank[obj_id]['feature']
        
        # 未匹配的 query 作为新 object
        for q_idx in range(valid_queries.size(0)):
            if q_idx not in matched_queries:
                obj_id = len(self.object_bank)
                label = valid_cls[q_idx].item()
                uid = detections['uids'][q_idx].item() if detections is not None and 'uids' in detections and q_idx < len(detections['uids']) else -1
                bbox = detections['boxes'][q_idx] if detections is not None and 'boxes' in detections and q_idx < len(detections['boxes']) else torch.zeros(4)
                
                self.object_bank[obj_id] = {
                    'appearance': [(image_idx, frame_id, bbox, label, valid_scores[q_idx].item())],
                    'class': label,
                    'uid': uid,
                    'query_idx': q_idx,
                    'feature': valid_queries[q_idx].detach().cpu(),
                }
                self.object_feature_bank = torch.cat([
                    self.object_feature_bank, 
                    valid_queries[q_idx].unsqueeze(0).detach().cpu()
                ], dim=0)

    def get_pp(self):
        """计算 place-place 相似度矩阵"""
        if len(self.place_feature_bank) == 0:
            return
        
        place_embeddings = []
        for image_id in sorted(self.place_feature_bank.keys()):
            place_embeddings.append(self.place_feature_bank[image_id])
        place_embeddings = torch.stack(place_embeddings, dim=0)  # [N, C]
        
        # 批量计算相似度
        batch_size = 512
        N = place_embeddings.shape[0]
        self.pp_adj_sim = torch.zeros(N, N, device=place_embeddings.device)
        
        for i in range(0, N, batch_size):
            self.pp_adj_sim[i:i+batch_size] = torch.cosine_similarity(
                place_embeddings[i:i+batch_size].unsqueeze(1),
                place_embeddings.unsqueeze(0),
                dim=-1
            )
        
        self.pp_adj = (self.pp_adj_sim > self.pp_threshold).float()

    def output_mapping(self, save_pp_sim=False, save_emb_dir=None):
        """
        输出 mapping 结果，兼容旧版格式
        
        Returns:
            dict: {
                'video_id': str,
                'p-p': List[List[float]],  # place-place adjacency
                'detections': Dict[frame_id, Dict[obj_id, Dict]],
                'pp-sim': Optional[List[List[float]]],
            }
        """
        self.get_pp()
        
        # 格式化 detections
        detections = {frame_id: dict() for frame_id in self.frame_ids}
        
        if len(self.object_bank) > 0:
            for obj_id, obj_dict in self.object_bank.items():
                for appearance in obj_dict['appearance']:
                    image_idx, frame_id, bbox, label, score = appearance
                    if frame_id not in detections:
                        continue
                    
                    if obj_id not in detections[frame_id]:
                        uniq_label = self.label2class.get(label, f"class_{label}") + f":{int(obj_dict['uid'])}"
                        detections[frame_id][obj_id] = {
                            'bbox': bbox.tolist() if isinstance(bbox, torch.Tensor) else bbox,
                            'label': label,
                            'score': score,
                            'uniq': uniq_label,
                        }
        
        mapping_results = {
            'video_id': self.video_id,
            'p-p': self.pp_adj.tolist(),
            'detections': detections,
        }
        
        if save_pp_sim:
            mapping_results['pp-sim'] = self.pp_adj_sim.tolist()
        
        return mapping_results

    def get_query_usage_stats(self):
        """获取 query 使用统计信息"""
        if len(self.all_query_usage) == 0:
            return {}
        
        total_queries = sum(u['num_queries'] for u in self.all_query_usage)
        total_valid = sum(np.sum(u['exist_scores'] > self.object_threshold) for u in self.all_query_usage)
        
        return {
            'total_frames': len(self.all_query_usage),
            'total_queries': int(total_queries),
            'total_valid_queries': int(total_valid),
            'avg_queries_per_frame': total_queries / len(self.all_query_usage),
            'avg_valid_queries_per_frame': total_valid / len(self.all_query_usage),
            'query_efficiency': total_valid / total_queries if total_queries > 0 else 0,
        }
