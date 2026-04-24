import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def build_gt_mask(obj_idx, num_obj, num_bbox):
    """
    构造 gt_mask: [B, N, M]
    
    gt_mask[b, j, m] = 1  如果 bbox m 属于 object j，否则 = 0
    
    Args:
        obj_idx: [B, M] 每个 bbox 属于哪个 object（0 ~ N-1），-1 表示无效
        num_obj: int，真实 object 数量 N
        num_bbox: int，bbox 数量 M
        
    Returns:
        gt_mask: [B, N, M]
    """
    B = obj_idx.shape[0]
    device = obj_idx.device
    
    gt_mask = torch.zeros(B, num_obj, num_bbox, device=device)
    
    for b in range(B):
        for m in range(num_bbox):
            obj_id = obj_idx[b, m]
            if obj_id >= 0 and obj_id < num_obj:
                gt_mask[b, obj_id.long(), m] = 1.0
                
    return gt_mask


class ObjectLevelHungarianMatcher(nn.Module):
    """
    Object-level Hungarian Matcher
    
    核心：Query (K) ↔ GT Object (N) 的匹配
    不是 Query ↔ BBox 的匹配
    
    匹配代价：
    - classification cost: CrossEntropy(cls_logits, gt_label)
    - attention cost: BCE(attn, gt_mask)
    """

    def __init__(self, cost_cls: float = 1.0, cost_attn: float = 1.0):
        super().__init__()
        self.cost_cls = cost_cls
        self.cost_attn = cost_attn
        assert cost_cls != 0 or cost_attn != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, object_cls_logits, object_attn, gt_labels, gt_mask):
        """
        Args:
            object_cls_logits: [B, K, C] query 的分类 logits
            object_attn: [B, K, M] query 对 bbox 的 attention
            gt_labels: [B, N] 每个 object 的类别
            gt_mask: [B, N, M] bbox 到 object 的归属 mask
            
        Returns:
            indices: list of (pred_idx, tgt_idx) tuples
        """
        B, K, C = object_cls_logits.shape
        _, N, M = gt_mask.shape
        device = object_cls_logits.device
        
        indices = []
        
        for b in range(B):
            # 1. Classification cost: [K, N]
            cls_logits_b = object_cls_logits[b]  # [K, C]
            gt_labels_b = gt_labels[b]  # [N]
            
            cls_prob = torch.softmax(cls_logits_b, dim=-1)  # [K, C]
            
            cls_cost = torch.zeros(K, N, device=device)
            for n in range(N):
                if gt_labels_b[n] >= 0:
                    cls_cost[:, n] = -torch.log(cls_prob[:, gt_labels_b[n].long()] + 1e-8)
                else:
                    cls_cost[:, n] = 100.0
            
            # 2. Attention cost: [K, N]
            attn_b = object_attn[b]  # [K, M]
            gt_mask_b = gt_mask[b]  # [N, M]
            
            attn_cost = torch.zeros(K, N, device=device)
            for n in range(N):
                if gt_mask_b[n].sum() > 0:
                    target_mask = gt_mask_b[n].unsqueeze(0).expand(K, -1)  # [K, M]
                    bce = F.binary_cross_entropy(
                        attn_b, target_mask, reduction='none'
                    ).sum(dim=-1)  # [K]
                    attn_cost[:, n] = bce
                else:
                    attn_cost[:, n] = 100.0
            
            # 3. Total cost
            cost_mat = self.cost_cls * cls_cost + self.cost_attn * attn_cost  # [K, N]
            
            # 4. Hungarian matching
            pred_indices, tgt_indices = linear_sum_assignment(cost_mat.cpu().numpy())
            
            pred_indices = torch.as_tensor(pred_indices, dtype=torch.int64, device=device)
            tgt_indices = torch.as_tensor(tgt_indices, dtype=torch.int64, device=device)
            
            indices.append((pred_indices, tgt_indices))
        
        return indices


class QueryObjectLoss(nn.Module):
    """
    Query-based Object-level Loss (重构版)
    
    支持两种模式：
    - pure: 仅使用 object-level loss (L_cls + L_attn + L_exist + L_bbox)
    - hybrid: object-level loss + baseline loss (用于渐进式训练)
    
    包含：
    1. L_cls: 分类损失 (matched query)
    2. L_attn: Attention 损失 (matched query)
    3. L_exist: 存在性损失 (all query)
    4. L_bbox: Bbox 回归损失 (optional, attention-weighted)
    """

    def __init__(
        self,
        cost_cls: float = 1.0,
        cost_attn: float = 1.0,
        weight_cls: float = 1.0,
        weight_attn: float = 1.0,
        weight_exist: float = 1.0,
        weight_bbox: float = 0.0,
        loss_mode: str = 'hybrid',  # 'pure' or 'hybrid'
        weight_baseline: float = 1.0,  # hybrid 模式下 baseline loss 的权重
    ):
        super().__init__()
        self.matcher = ObjectLevelHungarianMatcher(cost_cls=cost_cls, cost_attn=cost_attn)
        self.weight_cls = weight_cls
        self.weight_attn = weight_attn
        self.weight_exist = weight_exist
        self.weight_bbox = weight_bbox
        self.loss_mode = loss_mode
        self.weight_baseline = weight_baseline

    def _filter_padding_objects(self, obj_idx, obj_label, bbox_mask):
        """
        严格过滤 padding object，确保 obj_idx / obj_label / bbox_mask 对齐

        规则:
        1. bbox_mask=False 的 bbox 不参与计算
        2. obj_idx=-1 的 bbox 不参与计算
        3. obj_label=-1 的 object 不参与计算
        4. 只保留有至少一个有效 bbox 的 object

        Returns:
            filtered_obj_idx: [B, M] 过滤后的 obj_idx (padding 设为 -1)
            filtered_obj_label: [B, N] 过滤后的 obj_label
            valid_obj_mask: [B, N] 每个 object 是否有效
            num_valid_objs: list 每个 batch 的有效 object 数量
        """
        B, M = obj_idx.shape
        device = obj_idx.device

        # 1. 标记有效 bbox: mask=True 且 obj_idx != -1
        valid_bbox_mask = bbox_mask & (obj_idx != -1)  # [B, M]

        # 2. 收集每个 batch 的有效 object
        filtered_obj_idx = obj_idx.clone()
        filtered_obj_idx[~valid_bbox_mask] = -1

        # 3. 确定有效 object 集合
        num_valid_objs = []
        valid_obj_mask_list = []
        max_n = obj_label.shape[1] if obj_label.dim() >= 2 else 0

        for b in range(B):
            valid_obj_ids = filtered_obj_idx[b][filtered_obj_idx[b] != -1]
            if valid_obj_ids.numel() > 0:
                unique_ids = torch.unique(valid_obj_ids)
                num_valid_objs.append(len(unique_ids))
                # 构建有效 object mask
                mask = torch.zeros(max_n, dtype=torch.bool, device=device)
                for uid in unique_ids:
                    uid_int = int(uid.item())
                    if 0 <= uid_int < max_n:
                        mask[uid_int] = True
                valid_obj_mask_list.append(mask)
            else:
                num_valid_objs.append(0)
                valid_obj_mask_list.append(torch.zeros(max_n, dtype=torch.bool, device=device))

        valid_obj_mask = torch.stack(valid_obj_mask_list, dim=0)  # [B, N]

        # 4. 过滤 obj_label
        if obj_label.dim() == 3:
            obj_label = obj_label[:, 0, :]
        filtered_obj_label = obj_label.clone()
        filtered_obj_label[~valid_obj_mask] = -1

        return filtered_obj_idx, filtered_obj_label, valid_obj_mask, num_valid_objs

    def forward(self, outputs, targets, baseline_loss_fn=None):
        """
        Args:
            outputs: dict containing
                - object_node_feat: [B, K, D]
                - object_attn: [B, K, M] query 对 bbox 的 attention
                - object_cls_logits: [B, K, C]
                - object_exist_logits: [B, K]
            targets: dict containing
                - gt_bbox: [B, M, 4]
                - obj_idx: [B, M] 每个 bbox 属于哪个 object
                - obj_label: [B, N] 每个 object 的类别
                - mask: [B, M] bbox 是否有效
            baseline_loss_fn: callable, hybrid 模式下用于计算 baseline loss

        Returns:
            total_loss: scalar
            loss_dict: dict of individual losses
        """
        object_attn = outputs['object_attn']  # [B, K, M]
        object_cls_logits = outputs['object_cls_logits']  # [B, K, C]
        object_exist_logits = outputs.get('object_exist_logits')  # [B, K]

        B, K, M = object_attn.shape
        device = object_attn.device

        # 从 targets 中提取信息
        obj_idx = targets['obj_idx']  # [B, M]
        obj_label = targets['obj_label']  # [B, N] 或 [B, V, N]
        bbox_mask = targets['mask']  # [B, M]
        gt_bbox = targets.get('gt_bbox')  # [B, M, 4]

        # 严格过滤 padding object，确保对齐
        filtered_obj_idx, filtered_obj_label, valid_obj_mask, num_valid_objs = \
            self._filter_padding_objects(obj_idx, obj_label, bbox_mask)

        max_num_obj = max(num_valid_objs) if num_valid_objs else 0

        if max_num_obj == 0:
            # 没有有效 object，返回 0 loss
            loss_dict = {
                'loss_cls': torch.tensor(0.0, device=device),
                'loss_attn': torch.tensor(0.0, device=device),
                'loss_exist': torch.tensor(0.0, device=device),
            }
            if self.weight_bbox > 0:
                loss_dict['loss_bbox'] = torch.tensor(0.0, device=device)
            return sum(loss_dict.values()), loss_dict

        # 构造 gt_mask: [B, N, M] (使用过滤后的 obj_idx)
        gt_mask = build_gt_mask(filtered_obj_idx, max_num_obj, M)

        # 确保 obj_label 长度与 max_num_obj 一致 (使用过滤后的 label)
        if filtered_obj_label.shape[1] < max_num_obj:
            pad_size = max_num_obj - filtered_obj_label.shape[1]
            filtered_obj_label = torch.cat([
                filtered_obj_label,
                torch.full((B, pad_size), -1, dtype=filtered_obj_label.dtype, device=device)
            ], dim=1)
        else:
            filtered_obj_label = filtered_obj_label[:, :max_num_obj]

        # 1. Hungarian Matching: Query(K) ↔ GT Object(N)
        indices = self.matcher(object_cls_logits, object_attn, filtered_obj_label, gt_mask)

        # 2. 计算各项 loss
        loss_dict = {}

        # --- L_cls: 分类损失 (matched query) ---
        cls_losses = []
        for b in range(B):
            pred_idx, tgt_idx = indices[b]
            if len(pred_idx) == 0:
                continue

            matched_cls_logits = object_cls_logits[b, pred_idx]
            matched_gt_labels = filtered_obj_label[b, tgt_idx].long()

            # 严格过滤: 只计算 label >= 0 的
            valid_mask = matched_gt_labels >= 0
            if valid_mask.any():
                valid_logits = matched_cls_logits[valid_mask]
                valid_labels = matched_gt_labels[valid_mask]
                cls_loss = F.cross_entropy(valid_logits, valid_labels, reduction='mean')
                cls_losses.append(cls_loss)

        if cls_losses:
            loss_dict['loss_cls'] = torch.stack(cls_losses).mean()
        else:
            loss_dict['loss_cls'] = torch.tensor(0.0, device=device, requires_grad=True)

        # --- L_attn: Attention 损失 (matched query) ---
        attn_losses = []
        for b in range(B):
            pred_idx, tgt_idx = indices[b]
            if len(pred_idx) == 0:
                continue

            matched_attn = object_attn[b, pred_idx]
            matched_gt_mask = gt_mask[b, tgt_idx]

            # 严格过滤: 只计算有有效 bbox 的 object 的 attention
            valid_obj_for_attn = matched_gt_mask.sum(dim=-1) > 0  # [num_matched]
            if valid_obj_for_attn.any():
                matched_attn = matched_attn[valid_obj_for_attn]
                matched_gt_mask = matched_gt_mask[valid_obj_for_attn]
                attn_loss = F.binary_cross_entropy(
                    matched_attn, matched_gt_mask, reduction='mean'
                )
                attn_losses.append(attn_loss)

        if attn_losses:
            loss_dict['loss_attn'] = torch.stack(attn_losses).mean()
        else:
            loss_dict['loss_attn'] = torch.tensor(0.0, device=device, requires_grad=True)

        # --- L_exist: 存在性损失 (all query) ---
        if object_exist_logits is not None:
            exist_targets = torch.zeros(B, K, device=device)
            for b in range(B):
                pred_idx, _ = indices[b]
                if len(pred_idx) > 0:
                    exist_targets[b, pred_idx] = 1.0

            loss_dict['loss_exist'] = F.binary_cross_entropy_with_logits(
                object_exist_logits, exist_targets, reduction='mean'
            )
        else:
            loss_dict['loss_exist'] = torch.tensor(0.0, device=device, requires_grad=True)

        # --- L_bbox: Bbox 回归损失 (optional, attention-weighted) ---
        if self.weight_bbox > 0 and gt_bbox is not None:
            bbox_losses = []
            for b in range(B):
                pred_idx, tgt_idx = indices[b]
                if len(pred_idx) == 0:
                    continue

                matched_attn = object_attn[b, pred_idx]
                matched_gt_mask = gt_mask[b, tgt_idx]
                bboxes_b = gt_bbox[b]

                for i in range(len(pred_idx)):
                    attn_weights = matched_attn[i]
                    pred_bbox = (attn_weights.unsqueeze(-1) * bboxes_b).sum(dim=0)

                    gt_mask_i = matched_gt_mask[i]
                    # 严格过滤: 只计算有有效 bbox 的 object
                    if gt_mask_i.sum() > 0:
                        gt_bbox_weighted = (gt_mask_i.unsqueeze(-1) * bboxes_b).sum(dim=0) / (gt_mask_i.sum() + 1e-8)
                        l1_loss = F.l1_loss(pred_bbox, gt_bbox_weighted)
                        bbox_losses.append(l1_loss)

            if bbox_losses:
                loss_dict['loss_bbox'] = torch.stack(bbox_losses).mean()
            else:
                loss_dict['loss_bbox'] = torch.tensor(0.0, device=device, requires_grad=True)

        # 3. 计算 object-level total loss
        object_total_loss = (
            self.weight_cls * loss_dict['loss_cls'] +
            self.weight_attn * loss_dict['loss_attn'] +
            self.weight_exist * loss_dict['loss_exist']
        )
        if self.weight_bbox > 0 and 'loss_bbox' in loss_dict:
            object_total_loss = object_total_loss + self.weight_bbox * loss_dict['loss_bbox']

        loss_dict['loss_object_total'] = object_total_loss
        loss_dict['num_valid_objects'] = sum(num_valid_objs)  # 记录有效 object 数量

        # 4. 根据 loss_mode 返回结果
        if self.loss_mode == 'pure':
            # Pure mode: 仅使用 object-level loss
            return object_total_loss, loss_dict
        elif self.loss_mode == 'hybrid':
            # Hybrid mode: object-level loss + baseline loss
            if baseline_loss_fn is not None:
                try:
                    baseline_total_loss, baseline_logs = baseline_loss_fn()
                    loss_dict.update({f'baseline_{k}': v for k, v in baseline_logs.items()})
                    loss_dict['loss_baseline'] = baseline_total_loss
                    total_loss = object_total_loss + self.weight_baseline * baseline_total_loss
                except Exception as e:
                    # 如果 baseline loss 失败，仅使用 object-level loss
                    print(f"Baseline loss failed in hybrid mode: {e}, using object-level loss only")
                    total_loss = object_total_loss
            else:
                # 没有提供 baseline_loss_fn，仅使用 object-level loss
                total_loss = object_total_loss

            loss_dict['loss_total'] = total_loss
            return total_loss, loss_dict
        else:
            raise ValueError(f"Unknown loss_mode: {self.loss_mode}. Must be 'pure' or 'hybrid'")
