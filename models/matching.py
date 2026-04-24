import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


def box_iou(boxes1, boxes2):
    """
    计算两组 bbox 之间的 IoU
    
    Args:
        boxes1: [N, 4] (x1, y1, x2, y2)
        boxes2: [M, 4] (x1, y1, x2, y2)
        
    Returns:
        iou: [N, M] IoU 矩阵
        union: [N, M] Union 面积矩阵
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-8)
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    计算 Generalized IoU (GIoU)
    
    Args:
        boxes1: [N, 4] (x1, y1, x2, y2)
        boxes2: [M, 4] (x1, y1, x2, y2)
        
    Returns:
        giou: [N, M] GIoU 矩阵
    """
    iou, union = box_iou(boxes1, boxes2)
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]
    return iou - (area - union) / (area + 1e-8)


class HungarianMatcher(nn.Module):
    """
    Hungarian Matcher (重构版)
    
    支持两种匹配模式：
    1. Query-to-Object: Query(K) ↔ GT Object(N) (Step 2/3/4)
    2. Query-to-BBox: Query(K) ↔ GT BBox(M) (兼容旧版)
    
    匹配代价：
    - 存在性代价: -sigmoid(exist_logits)
    - 注意力代价: -avg_attention (query 对 object 所有 bbox 的平均注意力)
    - 分类代价: -log_prob(cls_logits, gt_label) (可选)
    """

    def __init__(
        self,
        cost_exist: float = 1.0,
        cost_attn: float = 1.0,
        cost_cls: float = 0.0,
        match_mode: str = 'object',  # 'object' or 'bbox'
    ):
        super().__init__()
        self.cost_exist = cost_exist
        self.cost_attn = cost_attn
        self.cost_cls = cost_cls
        self.match_mode = match_mode
        assert cost_exist != 0 or cost_attn != 0 or cost_cls != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(
        self,
        object_attn,
        object_exist_logits,
        targets,
        valid_key='mask',
        bbox_key='gt_bbox',
        obj_idx_key='obj_idx',
        labels_key='obj_label',
        object_cls_logits=None,
    ):
        """
        执行匈牙利匹配
        
        Args:
            object_attn: [B, Q, V*N] query 对 bbox 的注意力
            object_exist_logits: [B, Q] 存在性 logits
            targets: dict，包含 GT 信息
            valid_key: 有效 mask 的 key
            bbox_key: bbox 的 key
            obj_idx_key: object index 的 key
            labels_key: label 的 key
            object_cls_logits: [B, Q, C] 分类 logits (可选，用于 cost_cls)
            
        Returns:
            indices: list of (pred_idx, tgt_idx) tuples
        """
        B, Q, _ = object_attn.shape
        device = object_attn.device
        
        indices = []
        for bi in range(B):
            tgt_bboxes = targets[bbox_key][bi]
            tgt_valid = targets[valid_key][bi]
            obj_idx = targets.get(obj_idx_key, None)
            
            if obj_idx is not None:
                obj_idx_bi = obj_idx[bi]
            else:
                obj_idx_bi = None
            
            # 确定匹配目标
            if self.match_mode == 'object' and obj_idx_bi is not None:
                # Query-to-Object 匹配
                valid_obj_idx = obj_idx_bi[obj_idx_bi != -1]
                if valid_obj_idx.numel() > 0:
                    unique_obj_ids = torch.unique(valid_obj_idx)
                    num_tgt = len(unique_obj_ids)
                else:
                    unique_obj_ids = None
                    num_tgt = 0
            else:
                # Query-to-BBox 匹配
                unique_obj_ids = None
                num_tgt = tgt_valid.sum().item() if tgt_valid.dim() == 1 else tgt_valid.shape[0]
            
            if num_tgt == 0:
                # 没有有效目标，返回空匹配
                indices.append((
                    torch.tensor([], dtype=torch.int64, device=device),
                    torch.tensor([], dtype=torch.int64, device=device)
                ))
                continue
            
            # 1. 存在性代价: [Q, num_tgt]
            exist_prob = torch.sigmoid(object_exist_logits[bi, :])  # [Q]
            cost_exist = -exist_prob.unsqueeze(1).expand(-1, num_tgt)  # [Q, num_tgt]
            
            # 2. 注意力代价: [Q, num_tgt]
            cost_attn = torch.zeros(Q, num_tgt, device=device)
            
            if unique_obj_ids is not None:
                # Query-to-Object: 计算 query 对每个 object 所有 bbox 的平均注意力
                for tgt_idx in range(num_tgt):
                    obj_id = unique_obj_ids[tgt_idx]
                    obj_mask = (obj_idx_bi == obj_id)  # [V*N] or [M]
                    obj_positions = obj_mask.nonzero(as_tuple=True)[0]
                    
                    if obj_positions.numel() > 0:
                        # 取 query 对这些 bbox 的平均注意力
                        avg_attn = object_attn[bi, :, obj_positions].mean(dim=1)  # [Q]
                        cost_attn[:, tgt_idx] = -avg_attn
            else:
                # Query-to-BBox: 计算 query 对每个有效 bbox 的注意力
                valid_positions = tgt_valid.nonzero(as_tuple=True)[0]
                if valid_positions.numel() > 0 and valid_positions.numel() == num_tgt:
                    cost_attn = -object_attn[bi, :, valid_positions]  # [Q, num_tgt]
            
            # 3. 分类代价: [Q, num_tgt] (可选)
            cost_cls = torch.zeros(Q, num_tgt, device=device)
            if self.cost_cls > 0 and object_cls_logits is not None:
                cls_logits_bi = object_cls_logits[bi]  # [Q, C]
                cls_prob = torch.softmax(cls_logits_bi, dim=-1)  # [Q, C]
                
                labels = targets.get(labels_key, None)
                if labels is not None:
                    labels_bi = labels[bi] if labels.dim() >= 2 else labels
                    # 获取每个 target 的 label
                    for tgt_idx in range(min(num_tgt, labels_bi.shape[0])):
                        label = labels_bi[tgt_idx]
                        if label >= 0:
                            cost_cls[:, tgt_idx] = -torch.log(cls_prob[:, label.long()] + 1e-8)
                        else:
                            cost_cls[:, tgt_idx] = 100.0
            
            # 4. 总代价矩阵
            cost_mat = (
                self.cost_exist * cost_exist +
                self.cost_attn * cost_attn +
                self.cost_cls * cost_cls
            )
            
            # 5. 匈牙利匹配
            pred_indices, tgt_indices = linear_sum_assignment(cost_mat.cpu().numpy())
            
            pred_indices = torch.as_tensor(pred_indices, dtype=torch.int64, device=device)
            tgt_indices = torch.as_tensor(tgt_indices, dtype=torch.int64, device=device)
            
            indices.append((pred_indices, tgt_indices))
        
        return indices


class BoxMatcher(nn.Module):
    """
    BBox-level Matcher: 检测框与 GT 框的匹配
    
    用于评估阶段或传统检测任务
    """

    def __init__(self, cost_bbox: float = 1.0, cost_giou: float = 1.0):
        super().__init__()
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        执行 bbox 匹配
        
        Args:
            outputs: list of dicts，每个包含:
                - "boxes": [num_pred, 4]
                - "scores": [num_pred]
                - "labels": [num_pred]
            targets: list of dicts，每个包含:
                - "boxes": [num_tgt, 4]
                - "labels": [num_tgt]
                
        Returns:
            indices: list of (pred_idx, tgt_idx) tuples
        """
        indices = []
        for bi in range(len(outputs)):
            out_bbox = outputs[bi]["boxes"]
            tgt_bbox = targets[bi]["boxes"]
            tgt_ids = targets[bi]["labels"]

            # L1 代价
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

            # GIoU 代价
            cost_giou = -generalized_box_iou(out_bbox, tgt_bbox)

            cost_mat = self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
            
            pred_indices, tgt_indices = linear_sum_assignment(cost_mat.cpu())
            indices.append((
                torch.as_tensor(pred_indices, dtype=torch.int64),
                torch.as_tensor(tgt_indices, dtype=torch.int64)
            ))
        return indices


def get_match_targets(indices, targets, num_queries, labels_key='obj_label'):
    """
    根据匹配结果获取每个 query 的目标标签
    
    Args:
        indices: list of (pred_idx, tgt_idx) from matcher
        targets: dict or list of target dicts
        num_queries: int, query 数量
        labels_key: label 的 key
        
    Returns:
        exist_targets: [B, Q] bool，表示每个 query 是否匹配到目标
        cls_targets: [B, Q] long，每个 query 的目标类别 (-1 表示无匹配)
    """
    # 处理输入格式
    if isinstance(targets, dict):
        B = targets[labels_key].shape[0] if labels_key in targets else len(indices)
        is_dict_format = True
    else:
        B = len(targets)
        is_dict_format = False
    
    exist_targets = torch.zeros(B, num_queries, dtype=torch.bool, device='cuda' if torch.cuda.is_available() else 'cpu')
    cls_targets = torch.full((B, num_queries), -1, dtype=torch.int64, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    for bi in range(B):
        pred_idx, tgt_idx = indices[bi]
        if len(pred_idx) == 0:
            continue
            
        exist_targets[bi, pred_idx] = True
        
        if is_dict_format and labels_key in targets:
            labels = targets[labels_key][bi]
            
            # 提取有效标签
            if labels.dim() == 1:
                # [N] 格式
                valid_mask = labels != -1
                valid_labels = labels[valid_mask]
                if len(valid_labels) > 0:
                    tgt_idx_clamped = tgt_idx.clamp(0, len(valid_labels) - 1)
                    cls_targets[bi, pred_idx] = valid_labels[tgt_idx_clamped]
            elif labels.dim() == 2:
                # [V, N] 格式，取第一个视角
                labels_flat = labels[0] if labels.shape[0] > 0 else labels.view(-1)
                valid_mask = labels_flat != -1
                valid_labels = labels_flat[valid_mask]
                if len(valid_labels) > 0:
                    tgt_idx_clamped = tgt_idx.clamp(0, len(valid_labels) - 1)
                    cls_targets[bi, pred_idx] = valid_labels[tgt_idx_clamped]
        
    return exist_targets, cls_targets


def match_queries_to_objects(object_attn, object_exist_logits, obj_idx, cost_exist=1.0, cost_attn=1.0):
    """
    便捷的 Query-to-Object 匹配函数
    
    Args:
        object_attn: [B, Q, M] query 对 bbox 的注意力
        object_exist_logits: [B, Q] 存在性 logits
        obj_idx: [B, M] 每个 bbox 属于哪个 object
        cost_exist: 存在性代价权重
        cost_attn: 注意力代价权重
        
    Returns:
        indices: list of (pred_idx, tgt_idx) tuples
    """
    matcher = HungarianMatcher(
        cost_exist=cost_exist,
        cost_attn=cost_attn,
        match_mode='object',
    )
    
    # 构造 targets
    B, M = obj_idx.shape
    targets = {
        'mask': torch.ones(B, M, dtype=torch.bool),
        'gt_bbox': torch.zeros(B, M, 4),
        'obj_idx': obj_idx,
        'obj_label': torch.zeros(B, M, dtype=torch.long),
    }
    
    return matcher(object_attn, object_exist_logits, targets)
