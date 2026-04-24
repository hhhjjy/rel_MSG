

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvTiny(nn.Module):
    """
    a tiny cnn, parameters 0.2M
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 7,
        mid_channels: int = 64,
        layers: int = 3,
    ):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels, mid_channels, kernel_size=kernel_size, padding=3
                ),
                nn.ReLU(inplace=True),
            )
        )
        for _ in range(layers - 2):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        mid_channels,
                        mid_channels,
                        kernel_size=kernel_size,
                        padding=3,
                    ),
                    nn.ReLU(inplace=True),
                )
            )
        self.conv_layers.append(
            nn.Sequential(
                nn.Conv2d(
                    mid_channels, out_channels, kernel_size=kernel_size, padding=3
                )
            )
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        for layer in self.conv_layers:
            x = layer(x)
        return x.squeeze(1)


# --------------------------
# 【对齐MMCV】基础Transformer层（支持位置编码 + 完全匹配你的调用参数）
# --------------------------
class BaseTransformerLayer(nn.Module):
    def __init__(
        self,
        embed_dims=256,
        num_heads=8,
        feedforward_channels=2048,
        attn_drop=0.0,
        proj_drop=0.0,
        ffn_drop=0.1,
    ):
        super().__init__()
        # 1. 注意力层（严格对齐参数：batch_first=False, 无dropout）
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dims,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=False,
            bias=True
        )
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dims,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=False,
            bias=True
        )

        # 2. 层归一化（3个，匹配操作顺序）
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)

        # 3. FFN（对齐2层全连接 + ReLU + Dropout）
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, feedforward_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(ffn_drop),
            nn.Linear(feedforward_channels, embed_dims),
            nn.Dropout(proj_drop)
        )

        # 4. 投影Dropout
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        query,
        key,
        value,
        query_pos=None,
        key_pos=None,
        value_pos=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
    ):
        """
        ✅ 完全对齐你的调用参数 + MMCV逻辑
        执行顺序：cross_attn → norm → self_attn → norm → ffn → norm
        """
        # ==================== 1. 交叉注意力（核心：特征 + 位置编码）====================
        # MMCV/DETR标准：query = query + pos, key = key + pos
        q = query + query_pos if query_pos is not None else query
        k = key + key_pos if key_pos is not None else key
        v = value + value_pos if value_pos is not None else value

        # 交叉注意力前向
        attn_output, _ = self.cross_attn(
            query=q,
            key=k,
            value=v,
            key_padding_mask=key_padding_mask,
        )
        # 残差 + 归一化
        query = query + self.proj_drop(attn_output)
        query = self.norm1(query)

        # ==================== 2. 自注意力（查询自身）====================
        q_self = query + query_pos if query_pos is not None else query
        attn_output, _ = self.self_attn(
            query=q_self,
            key=q_self,
            value=q_self,
            key_padding_mask=query_key_padding_mask,
        )
        # 残差 + 归一化
        query = query + self.proj_drop(attn_output)
        query = self.norm2(query)

        # ==================== 3. FFN + 残差 + 归一化 ====================
        ffn_output = self.ffn(query)
        query = query + ffn_output
        query = self.norm3(query)

        return query

# --------------------------
# 【对齐MMCV】Detr解码器（layers可直接遍历调用）
# --------------------------
class DetrTransformerDecoder(nn.Module):
    def __init__(
        self, 
        num_layers=6, 
        return_intermediate=True,
        embed_dims=256,
        num_heads=8,
        dim_feedforward=2048,
    ):
        super().__init__()
        self.return_intermediate = return_intermediate
        self.embed_dims = embed_dims
        self.layers = nn.ModuleList([
            BaseTransformerLayer(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=dim_feedforward,
            ) 
            for _ in range(num_layers)
        ])

    def forward(self, query, memory, **kwargs):
        # 兼容原MMCV整体调用，不影响你逐层调用layers
        intermediate = []
        for layer in self.layers:
            query = layer(query=query, key=memory, value=memory, **kwargs)
            if self.return_intermediate:
                intermediate.append(query)
        return torch.stack(intermediate) if self.return_intermediate else query

class PairNet(nn.Module):
    def __init__(
        self,
        num_classes=133,
        embed_dims=1024,
        num_relations=2,
        num_obj_query=100,
        num_rel_query=100,
        feat_channels=256,
        num_rel_decoder_layers=6,
    ):
        super(PairNet, self).__init__()
        self.embed_dims = embed_dims
        self.num_rel_query = num_rel_query
        self.num_obj_query = num_obj_query
        self.feat_channels = feat_channels
        self.num_relations = num_relations
        self.num_classes = num_classes

        self.query_resize = nn.Linear(self.embed_dims, self.feat_channels)

        self.sub_query_update = nn.Sequential(
            nn.Linear(self.feat_channels, self.feat_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_channels, self.feat_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_channels, self.feat_channels),
        )

        self.obj_query_update = nn.Sequential(
            nn.Linear(self.feat_channels, self.feat_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_channels, self.feat_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_channels, self.feat_channels),
        )

        self.update_importance = ConvTiny()
        self.relation_decoder = DetrTransformerDecoder(
            num_layers=num_rel_decoder_layers, 
            return_intermediate=True,
            embed_dims=feat_channels,
            num_heads=8,
            dim_feedforward=2048,
        )

        self.rel_query_feat = nn.Embedding(self.num_rel_query, self.feat_channels)
        self.rel_query_embed = nn.Embedding(self.num_rel_query, self.feat_channels)
        self.rel_query_embed2 = nn.Embedding(self.num_rel_query * 2, self.feat_channels)
        self.rel_query_embed3 = nn.Embedding(self.num_rel_query * 2, self.feat_channels)

        self.rel_cls_embed = nn.Linear(self.feat_channels, self.num_relations)
        self.obj_cls_embed = nn.Linear(self.feat_channels, self.num_classes + 1)

        self.importance_match_loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.rel_cls_loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.obj_cls_loss_fn = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, query_feats):
        batch_size = query_feats.size(0)
        query_feats = self.query_resize(query_feats)
        query_feats = query_feats.transpose(0, 1)

        num_obj = query_feats.size(0)
        self.num_obj_query = num_obj

        query_feat = query_feats.clone()
        query_feat_list = [query_feat]
        query_feats_stacked = torch.stack(query_feat_list)
        
        sub_embed = self.sub_query_update(query_feats_stacked)
        obj_embed = self.obj_query_update(query_feats_stacked)
        
        sub_embed = F.normalize(sub_embed[-1].transpose(0, 1), p=2, dim=-1, eps=1e-12)
        obj_embed = F.normalize(obj_embed[-1].transpose(0, 1), p=2, dim=-1, eps=1e-12)
        
        importance = torch.matmul(sub_embed, obj_embed.transpose(1, 2))
        importance = self.update_importance(importance)
        return {'importance': importance}
        if self.num_rel_query > num_obj * num_obj:
            k = num_obj * num_obj
        else:
            k = self.num_rel_query
            
        _, updated_importance_idx = torch.topk(
            importance.flatten(-2, -1), k=k
        )
        sub_pos = torch.div(
            updated_importance_idx, num_obj, rounding_mode="trunc"
        )
        obj_pos = torch.remainder(updated_importance_idx, num_obj)

        obj_query_feat = torch.gather(
            query_feat,
            0,
            obj_pos.unsqueeze(-1).repeat(1, 1, self.feat_channels).transpose(0, 1),
        )
        sub_query_feat = torch.gather(
            query_feat,
            0,
            sub_pos.unsqueeze(-1).repeat(1, 1, self.feat_channels).transpose(0, 1),
        )

        rel_query_feat = self.rel_query_feat.weight.unsqueeze(1).repeat(
            (1, batch_size, 1)
        )
        rel_query_embed = self.rel_query_embed.weight.unsqueeze(1).repeat(
            (1, batch_size, 1)
        )
        rel_query_embed2 = self.rel_query_embed2.weight.unsqueeze(1).repeat(
            (1, batch_size, 1)
        )
        rel_query_embed3 = self.rel_query_embed3.weight.unsqueeze(1).repeat(
            (1, batch_size, 1)
        )
        
        pair_feat = torch.cat([sub_query_feat, obj_query_feat], dim=0)
        
        for layer in self.relation_decoder.layers:
            rel_query_feat = layer(
                query=rel_query_feat,
                key=pair_feat,
                value=pair_feat,
                query_pos=rel_query_embed,
                key_pos=None,
                value_pos=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
            )
        
        rel_query = rel_query_feat.transpose(0, 1)
        rel_preds = self.rel_cls_embed(rel_query)
        
        obj_cls_preds = self.obj_cls_embed(query_feat.transpose(0, 1))
        
        sub_cls_preds = torch.gather(
            obj_cls_preds,
            1,
            sub_pos.unsqueeze(-1).expand(-1, -1, obj_cls_preds.shape[-1]),
        )
        obj_cls_preds_gathered = torch.gather(
            obj_cls_preds,
            1,
            obj_pos.unsqueeze(-1).expand(-1, -1, obj_cls_preds.shape[-1]),
        )

        return {
            'importance': importance,
            'rel_preds': rel_preds,
            'sub_pos': sub_pos,
            'obj_pos': obj_pos,
            'sub_cls_preds': sub_cls_preds,
            'obj_cls_preds': obj_cls_preds_gathered,
            'obj_cls_preds_all': obj_cls_preds,
        }

    def compute_loss(self, outputs, gt_rels, gt_importance=None, gt_sub_labels=None, gt_obj_labels=None):
        loss_dict = {}
        
        importance = outputs.get('importance')
        rel_preds = outputs.get('rel_preds')
        sub_cls_preds = outputs.get('sub_cls_preds')
        obj_cls_preds = outputs.get('obj_cls_preds')
        obj_cls_preds_all = outputs.get('obj_cls_preds_all')
        
        if importance is not None and gt_importance is not None:
            pos_count = (gt_importance > 0).sum()
            if pos_count > 0:
                pos_weight = torch.numel(gt_importance) / pos_count
            else:
                pos_weight = torch.tensor(1.0, device=gt_importance.device)
            loss_match = F.binary_cross_entropy_with_logits(
                importance, gt_importance, pos_weight=pos_weight
            )
            loss_dict['loss_match'] = loss_match
        
        if rel_preds is not None and gt_rels is not None and rel_preds.size(1) > 0:
            gt_rels = gt_rels.long()
            if gt_rels.size(1) > rel_preds.size(1):
                gt_rels = gt_rels[:, :rel_preds.size(1)]
            elif gt_rels.size(1) < rel_preds.size(1):
                padding = torch.zeros(
                    gt_rels.size(0),
                    rel_preds.size(1) - gt_rels.size(1),
                    dtype=gt_rels.dtype,
                    device=gt_rels.device
                ).fill_(self.num_relations)
                gt_rels = torch.cat([gt_rels, padding], dim=1)
            
            valid_mask = gt_rels != self.num_relations
            valid_idx = valid_mask.any(dim=1)
            
            if valid_idx.sum() > 0:
                rel_cls_loss = self.rel_cls_loss_fn(
                    rel_preds[valid_idx],
                    gt_rels[valid_idx]
                )
                loss_dict['loss_rel_cls'] = rel_cls_loss
            else:
                loss_dict['loss_rel_cls'] = torch.tensor(
                    0.0, device=rel_preds.device, requires_grad=True
                )
        
        if sub_cls_preds is not None and gt_sub_labels is not None:
            gt_sub_labels = gt_sub_labels.long()
            if gt_sub_labels.size(1) > sub_cls_preds.size(1):
                gt_sub_labels = gt_sub_labels[:, :sub_cls_preds.size(1)]
            elif gt_sub_labels.size(1) < sub_cls_preds.size(1):
                padding = torch.zeros(
                    gt_sub_labels.size(0),
                    sub_cls_preds.size(1) - gt_sub_labels.size(1),
                    dtype=gt_sub_labels.dtype,
                    device=gt_sub_labels.device
                ).fill_(self.num_classes)
                gt_sub_labels = torch.cat([gt_sub_labels, padding], dim=1)
            loss_sub_cls = self.obj_cls_loss_fn(
                sub_cls_preds.view(-1, sub_cls_preds.size(-1)),
                gt_sub_labels.view(-1)
            )
            loss_dict['loss_sub_cls'] = loss_sub_cls
        
        if obj_cls_preds is not None and gt_obj_labels is not None:
            gt_obj_labels = gt_obj_labels.long()
            if gt_obj_labels.size(1) > obj_cls_preds.size(1):
                gt_obj_labels = gt_obj_labels[:, :obj_cls_preds.size(1)]
            elif gt_obj_labels.size(1) < obj_cls_preds.size(1):
                padding = torch.zeros(
                    gt_obj_labels.size(0),
                    obj_cls_preds.size(1) - gt_obj_labels.size(1),
                    dtype=gt_obj_labels.dtype,
                    device=gt_obj_labels.device
                ).fill_(self.num_classes)
                gt_obj_labels = torch.cat([gt_obj_labels, padding], dim=1)
            loss_obj_cls = self.obj_cls_loss_fn(
                obj_cls_preds.view(-1, obj_cls_preds.size(-1)),
                gt_obj_labels.view(-1)
            )
            loss_dict['loss_obj_cls'] = loss_obj_cls
        
        return loss_dict