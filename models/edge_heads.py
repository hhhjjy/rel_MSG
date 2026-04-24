import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeHeads(nn.Module):
    """
    边预测头，显式预测场景图中的边

    包括:
    - place-place edges (支持屏蔽 self-loop)
    - place-object edges

    增强功能:
    - 支持 node mask (忽略无效节点)
    - 支持屏蔽 place-place self-loop
    - 保持接口兼容
    """

    def __init__(self, dim=256, num_edge_types=50, mask_self_loop=True):
        super().__init__()
        self.dim = dim
        self.num_edge_types = num_edge_types
        self.mask_self_loop = mask_self_loop

        self.pp_head = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, num_edge_types)
        )

        self.po_head = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, num_edge_types)
        )

    def forward(self, place_node_feat, object_node_feat, place_mask=None, object_mask=None):
        """
        Args:
            place_node_feat: [B, P, C], P为place数量
            object_node_feat: [B, O, C], O为object数量
            place_mask: [B, P] bool, True表示有效place (可选)
            object_mask: [B, O] bool, True表示有效object (可选)

        Returns:
            pp_logits: [B, P, P, num_edge_types] place-place边预测
            po_logits: [B, P, O, num_edge_types] place-object边预测
            pp_mask: [B, P, P] bool, 有效边掩码 (当提供place_mask时)
            po_mask: [B, P, O] bool, 有效边掩码 (当提供place_mask或object_mask时)
        """
        B, P, C = place_node_feat.shape
        _, O, _ = object_node_feat.shape

        # place-place edges
        p1 = place_node_feat.unsqueeze(2).repeat(1, 1, P, 1)
        p2 = place_node_feat.unsqueeze(1).repeat(1, P, 1, 1)
        pp_pair = torch.cat([p1, p2], dim=-1)
        pp_logits = self.pp_head(pp_pair)

        # place-object edges
        p = place_node_feat.unsqueeze(2).repeat(1, 1, O, 1)
        o = object_node_feat.unsqueeze(1).repeat(1, P, 1, 1)
        po_pair = torch.cat([p, o], dim=-1)
        po_logits = self.po_head(po_pair)

        # 构建边掩码
        pp_mask = None
        po_mask = None

        if place_mask is not None:
            # pp_mask: 两个place都有效时边才有效
            pp_mask = place_mask.unsqueeze(2) & place_mask.unsqueeze(1)  # [B, P, P]

            # po_mask: place和object都有效时边才有效
            po_mask = place_mask.unsqueeze(2) & object_mask.unsqueeze(1) if object_mask is not None else place_mask.unsqueeze(2).expand(B, P, O)

        # 屏蔽 self-loop (place-place 边中 i->i 的边)
        if self.mask_self_loop:
            self_loop_mask = torch.eye(P, dtype=torch.bool, device=place_node_feat.device).unsqueeze(0).expand(B, -1, -1)
            if pp_mask is not None:
                pp_mask = pp_mask & (~self_loop_mask)
            else:
                pp_mask = ~self_loop_mask

            # 将 self-loop 的 logits 设为极小值 (在 softmax 后概率接近0)
            pp_logits = pp_logits.masked_fill(self_loop_mask.unsqueeze(-1), float('-inf'))

        return pp_logits, po_logits, pp_mask, po_mask

    def compute_loss(self, pp_logits, po_logits, gt_pp_rels, gt_po_rels, pp_mask=None, po_mask=None):
        """
        计算边预测损失，支持 mask

        Args:
            pp_logits: [B, P, P, E]
            po_logits: [B, P, O, E]
            gt_pp_rels: [B, P, P] 或 [B, P, P, E]
            gt_po_rels: [B, P, O] 或 [B, P, O, E]
            pp_mask: [B, P, P] bool (可选)
            po_mask: [B, P, O] bool (可选)

        Returns:
            loss_pp: place-place 边损失
            loss_po: place-object 边损失
        """
        # PP loss
        if pp_mask is not None and pp_mask.any():
            # 只计算有效边的损失
            valid_pp = pp_mask.unsqueeze(-1).expand_as(pp_logits)
            pp_logits_flat = pp_logits[valid_pp].view(-1, self.num_edge_types)
            gt_pp_flat = gt_pp_rels[pp_mask].view(-1)
            if pp_logits_flat.numel() > 0 and gt_pp_flat.numel() > 0:
                loss_pp = F.cross_entropy(pp_logits_flat, gt_pp_flat)
            else:
                loss_pp = torch.tensor(0.0, device=pp_logits.device, requires_grad=True)
        else:
            # 全部计算
            loss_pp = F.cross_entropy(
                pp_logits.permute(0, 3, 1, 2),
                gt_pp_rels
            )

        # PO loss
        if po_mask is not None and po_mask.any():
            valid_po = po_mask.unsqueeze(-1).expand_as(po_logits)
            po_logits_flat = po_logits[valid_po].view(-1, self.num_edge_types)
            gt_po_flat = gt_po_rels[po_mask].view(-1)
            if po_logits_flat.numel() > 0 and gt_po_flat.numel() > 0:
                loss_po = F.cross_entropy(po_logits_flat, gt_po_flat)
            else:
                loss_po = torch.tensor(0.0, device=po_logits.device, requires_grad=True)
        else:
            loss_po = F.cross_entropy(
                po_logits.permute(0, 3, 1, 2),
                gt_po_rels
            )

        return loss_pp, loss_po
