import torch
import torch.nn as nn
import torch.nn.functional as F


def box_iou(boxes1, boxes2):
    """计算 IoU"""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """计算 GIoU"""
    iou, union = box_iou(boxes1, boxes2)
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]
    return iou - (area - union) / area


class InfoNCELoss(nn.Module):
    """InfoNCE Loss 用于物体关联"""
    def __init__(self, temperature=0.1, learnable=False):
        super().__init__()
        if learnable:
            self.temperature = nn.Parameter(torch.ones(1) * temperature)
        else:
            self.register_buffer('temperature', torch.ones(1) * temperature)

    def forward(self, predictions, supervision_matrix, mask):
        """
        Args:
            predictions: [N, N] 相似度矩阵
            supervision_matrix: [N, N] 监督矩阵
            mask: [N, N] 有效掩码
            
        Returns:
            loss: 标量损失
        """
        N, _ = predictions.shape
        
        # 应用掩码
        predictions_masked = predictions * mask
        
        # 温度缩放
        logits = predictions_masked / self.temperature
        
        # 正样本
        positive_mask = supervision_matrix.to(dtype=torch.bool)
        if not positive_mask.any():
            return torch.tensor(0.0, device=predictions.device)
        
        positive_logits = logits[positive_mask]
        
        # 分母 logsumexp
        logsumexp_denominator = torch.logsumexp(logits, dim=-1) + 1e-9
        
        # 计算每个正样本的损失
        losses = positive_logits - logsumexp_denominator[positive_mask.nonzero(as_tuple=True)[0]]
        
        # 平均损失
        valid_entries = mask.sum()
        loss = -torch.sum(losses) / valid_entries if valid_entries > 0 else torch.tensor(0.0, device=predictions.device)
        
        return loss


class MaskBCELoss(nn.Module):
    """带掩码的 BCE Loss"""
    def __init__(self, pos_weight=1.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, predictions, supervision, mask=None):
        """
        Args:
            predictions: [*] 预测 logits
            supervision: [*] 监督
            mask: [*] 掩码
            
        Returns:
            loss: 标量损失
        """
        if mask is None:
            mask = torch.ones_like(predictions, dtype=torch.bool)
        
        valid_predictions = predictions * mask
        loss = F.binary_cross_entropy_with_logits(
            valid_predictions, 
            supervision,
            pos_weight=torch.tensor(self.pos_weight, device=predictions.device) if self.pos_weight != 1.0 else None,
            reduction='none'
        )
        masked_loss = loss * mask
        total_loss = masked_loss.sum() / (mask.sum() + 1e-9)
        
        return total_loss


class FocalLoss(nn.Module):
    """Focal Loss"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predictions, supervision, mask=None):
        """
        Args:
            predictions: [*] 预测 logits
            supervision: [*] 监督
            mask: [*] 掩码
            
        Returns:
            loss: 标量损失
        """
        if mask is None:
            mask = torch.ones_like(predictions, dtype=torch.bool)
        
        bce_loss = F.binary_cross_entropy_with_logits(predictions, supervision, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        masked_loss = focal_loss * mask
        total_loss = masked_loss.sum() / (mask.sum() + 1e-9)
        
        return total_loss


class EdgeLoss(nn.Module):
    """边预测损失"""
    def __init__(self, num_edge_types=50, alpha=0.25, gamma=2.0):
        super().__init__()
        self.num_edge_types = num_edge_types
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)

    def forward(self, logits, rel_matrix, valid_mask=None):
        """
        Args:
            logits: [B, N1, N2, E] 边预测 logits
            rel_matrix: [B, N1, N2, E] 关系矩阵 one-hot
            valid_mask: [B, N1, N2] 有效掩码
            
        Returns:
            loss: 标量损失
        """
        B, N1, N2, E = logits.shape
        
        # 如果 rel_matrix 是索引，转换为 one-hot
        if rel_matrix.dim() == 3:
            rel_matrix = F.one_hot(rel_matrix, num_classes=self.num_edge_types).float()
        
        if valid_mask is None:
            valid_mask = torch.ones(B, N1, N2, dtype=torch.bool, device=logits.device)
        
        # 扩展掩码
        valid_mask = valid_mask.unsqueeze(-1).expand_as(logits)
        
        # 计算损失
        loss = self.focal_loss(logits, rel_matrix, valid_mask)
        
        return loss


class RelationalMSGLoss(nn.Module):
    """
    Relational MSG 整体损失函数
    """

    def __init__(self, num_obj_classes=100, num_edge_types=50, weight_dict=None):
        super().__init__()
        self.num_obj_classes = num_obj_classes
        self.num_edge_types = num_edge_types
        
        if weight_dict is None:
            weight_dict = {
                'loss_obj_exist': 1.0,
                'loss_obj_cls': 1.0,
                'loss_place_exist': 1.0,
                'loss_pp_edge': 1.0,
                'loss_po_edge': 1.0,
            }
        self.weight_dict = weight_dict
        
        # 损失函数
        self.obj_exist_loss = MaskBCELoss(pos_weight=2.0)
        self.obj_cls_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.place_exist_loss = MaskBCELoss(pos_weight=2.0)
        self.edge_loss = EdgeLoss(num_edge_types=num_edge_types)

    def forward(self, outputs, targets, match_indices=None):
        """
        Args:
            outputs: 模型输出字典
            targets: GT targets
            match_indices: 匹配结果
            
        Returns:
            losses: 损失字典
        """
        loss_dict = {}
        device = outputs['object_exist_logits'].device
        
        # Object Exist Loss
        obj_exist_logits = outputs['object_exist_logits']  # [B, Q]
        # 需要从 targets 和 match_indices 构建 exist_targets
        B, Q = obj_exist_logits.shape
        exist_targets = torch.zeros(B, Q, dtype=torch.float32, device=device)
        
        if match_indices is not None:
            for b_idx, (pred_indices, tgt_indices) in enumerate(match_indices):
                exist_targets[b_idx, pred_indices] = 1.0
        
        loss_obj_exist = self.obj_exist_loss(obj_exist_logits, exist_targets)
        loss_dict['loss_obj_exist'] = loss_obj_exist
        
        # Object Class Loss
        obj_cls_logits = outputs['object_cls_logits']  # [B, Q, C]
        # 需要构建 class_targets
        class_targets = torch.full((B, Q), -1, dtype=torch.long, device=device)
        
        if match_indices is not None:
            for b_idx, (pred_indices, tgt_indices) in enumerate(match_indices):
                for pred_idx, tgt_idx in zip(pred_indices, tgt_indices):
                    if len(targets) > b_idx and 'obj_labels' in targets[b_idx]:
                        # 这里需要根据具体的 labels 格式调整
                        # 暂时用 dummy labels
                        tgt_labels = targets[b_idx]['obj_labels']
                        if len(tgt_labels.shape) == 2:
                            # multi-view 格式 [V, N]
                            tgt_label = tgt_labels.view(-1)[tgt_idx]
                        else:
                            tgt_label = tgt_labels[tgt_idx]
                        class_targets[b_idx, pred_idx] = tgt_label
        
        # 只计算匹配样本的损失
        mask = class_targets != -1
        if mask.any():
            loss_obj_cls = self.obj_cls_loss(
                obj_cls_logits[mask], 
                class_targets[mask]
            )
        else:
            loss_obj_cls = torch.tensor(0.0, device=device)
        loss_dict['loss_obj_cls'] = loss_obj_cls
        
        # Place Exist Loss - 简单处理为全存在
        place_exist_logits = outputs['place_exist_logits']  # [B, P]
        place_exist_targets = torch.ones_like(place_exist_logits)
        loss_place_exist = self.place_exist_loss(place_exist_logits, place_exist_targets)
        loss_dict['loss_place_exist'] = loss_place_exist
        
        # Edge Losses - 需要 rel labels
        pp_logits = outputs['pp_logits']  # [B, P, P, E]
        po_logits = outputs['po_logits']  # [B, P, Q, E]
        
        # TODO: 需要完整的 rel labels 实现
        loss_pp_edge = torch.tensor(0.0, device=device)
        loss_po_edge = torch.tensor(0.0, device=device)
        
        loss_dict['loss_pp_edge'] = loss_pp_edge
        loss_dict['loss_po_edge'] = loss_po_edge
        
        # Total Loss
        total_loss = sum([
            loss_dict[k] * self.weight_dict[k] 
            for k in loss_dict if k in self.weight_dict
        ])
        
        loss_dict['total_loss'] = total_loss
        
        return loss_dict
