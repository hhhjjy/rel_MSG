import torch
import torch.nn as nn
import torch.nn.functional as F

from .vggt_layers import Block


class AlternatingAttention(nn.Module):
    """
    Alternating Attention 模块 (VGGT-style)

    对输入的 queries 与 multi-view bbox 特征进行交替注意力：
    - Frame-level attention: 在每个视角内独立做 self-attention + cross-attention
    - Global-level attention: 跨所有视角做 global attention

    输入输出形状保持一致，方便插入到现有 decoder 之前。
    """

    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        aa_block_size: int = 1,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        qk_norm: bool = True,
        init_values: float = 0.01,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.aa_block_size = aa_block_size

        # Frame-level attention blocks
        self.frame_blocks = nn.ModuleList([
            Block(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop=dropout,
                attn_drop=dropout,
                init_values=init_values,
                qk_norm=qk_norm,
                rope=None,
            )
            for _ in range(num_layers)
        ])

        # Global-level attention blocks
        self.global_blocks = nn.ModuleList([
            Block(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop=dropout,
                attn_drop=dropout,
                init_values=init_values,
                qk_norm=qk_norm,
                rope=None,
            )
            for _ in range(num_layers)
        ])

        # 用于把 bbox 特征投影到与 query 相同的维度（如果已经相同则是个 identity 效果）
        self.bbox_proj = nn.Linear(dim, dim, bias=False)

        # 输出 layer norm
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, queries, bbox_feats, bbox_mask=None):
        """
        Args:
            queries: [B, Q, C] 可学习 queries
            bbox_feats: [B, V, N, C] 多视角 bbox 特征
            bbox_mask: [B, V, N] 可选的 mask，True 表示有效 bbox

        Returns:
            refined_queries: [B, Q, C] 经过交替注意力后的 queries
        """
        B, Q, C = queries.shape
        _, V, N, _ = bbox_feats.shape

        # 投影 bbox 特征
        bbox_feats = self.bbox_proj(bbox_feats)

        # 把 queries 扩展为每个视角一份: [B, V, Q, C]
        queries_expanded = queries.unsqueeze(1).expand(B, V, Q, C)

        # 把 bbox 特征展平每个视角内的 N: [B, V, N, C]
        # 拼接 queries 和 bbox: [B, V, Q+N, C]
        tokens = torch.cat([queries_expanded, bbox_feats], dim=2)  # [B, V, Q+N, C]

        # 构建 mask（如果提供）
        if bbox_mask is not None:
            # bbox_mask: [B, V, N] -> 为 queries 部分补 True
            query_mask = torch.ones(B, V, Q, dtype=torch.bool, device=bbox_mask.device)
            combined_mask = torch.cat([query_mask, bbox_mask], dim=2)  # [B, V, Q+N]
            # 转换为 attention mask: False 表示被 mask 掉
            attn_mask = ~combined_mask  # [B, V, Q+N]
        else:
            attn_mask = None

        # Alternating attention
        frame_idx = 0
        global_idx = 0
        num_blocks = self.num_layers // self.aa_block_size

        for _ in range(num_blocks):
            # --- Frame-level attention ---
            # 把 [B, V, Q+N, C] reshape 为 [B*V, Q+N, C]
            tokens_flat = tokens.reshape(B * V, Q + N, C)
            if attn_mask is not None:
                mask_flat = attn_mask.reshape(B * V, Q + N)
                # 转换为 float mask for scaled_dot_product_attention
                mask_flat = mask_flat.unsqueeze(1).unsqueeze(2)  # [B*V, 1, 1, Q+N]
            else:
                mask_flat = None

            for _ in range(self.aa_block_size):
                tokens_flat = self.frame_blocks[frame_idx](tokens_flat)
                frame_idx += 1

            tokens = tokens_flat.reshape(B, V, Q + N, C)

            # --- Global-level attention ---
            # 把 [B, V, Q+N, C] reshape 为 [B, V*(Q+N), C]
            tokens_global = tokens.reshape(B, V * (Q + N), C)
            if attn_mask is not None:
                mask_global = attn_mask.reshape(B, V * (Q + N))
                mask_global = mask_global.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, V*(Q+N)]
            else:
                mask_global = None

            for _ in range(self.aa_block_size):
                tokens_global = self.global_blocks[global_idx](tokens_global)
                global_idx += 1

            tokens = tokens_global.reshape(B, V, Q + N, C)

        # 提取 refined queries: [B, V, Q, C] -> 取平均 -> [B, Q, C]
        refined_queries = tokens[:, :, :Q, :].mean(dim=1)  # [B, Q, C]
        refined_queries = self.out_norm(refined_queries)

        return refined_queries


class ObjectQueryDecoderV2(nn.Module):
    """
    Object Query Decoder V2 (Step 2)

    在原有 ObjectQueryDecoder 基础上，在 decoder 之前增加 AlternatingAttention 模块。
    保持输入输出接口与 V1 完全一致。

    结构:
        queries → AlternatingAttention(queries, bbox_feats) → TransformerDecoder → heads
    """

    def __init__(
        self,
        dim=256,
        num_queries=100,
        num_classes=100,
        num_aa_layers=2,
        num_decoder_layers=2,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
        aa_block_size=1,
    ):
        super().__init__()
        self.dim = dim
        self.num_queries = num_queries
        self.num_classes = num_classes

        # 可学习 queries (与 V1 保持一致)
        self.object_queries = nn.Parameter(torch.randn(num_queries, dim))

        # Step 2 新增: Alternating Attention
        self.alternating_attn = AlternatingAttention(
            dim=dim,
            num_heads=num_heads,
            num_layers=num_aa_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            aa_block_size=aa_block_size,
        )

        # Transformer Decoder (与 V1 保持一致)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # 输出头 (与 V1 保持一致)
        self.exist_head = nn.Linear(dim, 1)
        self.cls_head = nn.Linear(dim, num_classes)
        self.attn_proj = nn.Linear(dim, dim)

    def forward(self, refined_obj_bank, bbox_mask=None):
        """
        Args:
            refined_obj_bank: [B, V, N, C] 跨视角融合后的 object bank
            bbox_mask: [B, V, N] 可选的 mask

        Returns:
            object_node_feat: [B, Q, C]
            object_attn: [B, Q, V*N] attention map
            object_exist_logits: [B, Q]
            object_cls_logits: [B, Q, num_classes]
        """
        B, V, N, C = refined_obj_bank.shape

        memory = refined_obj_bank.reshape(B, V * N, C)

        # 可学习 queries
        queries = self.object_queries.unsqueeze(0).repeat(B, 1, 1)  # [B, Q, C]

        # Step 2 核心: Alternating Attention
        queries = self.alternating_attn(queries, refined_obj_bank, bbox_mask)

        # Transformer Decoder
        object_node_feat = self.decoder(queries, memory)

        # Attention map (用于可视化或后续分析)
        object_attn = torch.bmm(
            self.attn_proj(object_node_feat),
            memory.transpose(1, 2)
        ) / (C ** 0.5)
        object_attn = object_attn.softmax(dim=-1)

        # 预测头
        object_exist_logits = self.exist_head(object_node_feat).squeeze(-1)
        object_cls_logits = self.cls_head(object_node_feat)

        return object_node_feat, object_attn, object_exist_logits, object_cls_logits


class PlaceQueryDecoderV2(nn.Module):
    """
    Place Query Decoder V2 (Step 2)

    在原有 PlaceQueryDecoder 基础上，在 decoder 之前增加 AlternatingAttention 模块。
    保持输入输出接口与 V1 完全一致。
    """

    def __init__(
        self,
        dim=256,
        num_queries=10,
        num_aa_layers=2,
        num_decoder_layers=2,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
        aa_block_size=1,
    ):
        super().__init__()
        self.dim = dim
        self.num_queries = num_queries

        # 可学习 queries
        self.place_queries = nn.Parameter(torch.randn(num_queries, dim))

        # Step 2 新增: Alternating Attention
        # 对于 place，memory 是 image-level 特征 [B, V, C]，需要扩展为 [B, V, 1, C]
        self.alternating_attn = AlternatingAttention(
            dim=dim,
            num_heads=num_heads,
            num_layers=num_aa_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            aa_block_size=aa_block_size,
        )

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # 输出头
        self.exist_head = nn.Linear(dim, 1)
        self.attn_proj = nn.Linear(dim, dim)
        self.obj_fusion = nn.Linear(dim * 2, dim)

    def forward(self, refined_img_bank, object_node_feat, bbox_mask=None):
        """
        Args:
            refined_img_bank: [B, V, C] 跨视角融合后的 image bank
            object_node_feat: [B, Q_obj, C] object node features
            bbox_mask: 不用于 place decoder，保持接口一致

        Returns:
            place_node_feat: [B, Q_place, C]
            place_attn: [B, Q_place, V]
            place_exist_logits: [B, Q_place]
        """
        B, V, C = refined_img_bank.shape

        memory = refined_img_bank  # [B, V, C]

        # 可学习 queries
        queries = self.place_queries.unsqueeze(0).repeat(B, 1, 1)  # [B, Q, C]

        # 融合 object-aware 信息 (与 V1 保持一致)
        obj_agg = object_node_feat.mean(dim=1, keepdim=True)  # [B, 1, C]
        queries = self.obj_fusion(torch.cat([
            queries,
            obj_agg.repeat(1, self.num_queries, 1)
        ], dim=-1))  # [B, Q, C]

        # Step 2 核心: Alternating Attention
        # 把 image bank 从 [B, V, C] 扩展为 [B, V, 1, C]
        img_bank_expanded = refined_img_bank.unsqueeze(2)  # [B, V, 1, C]
        queries = self.alternating_attn(queries, img_bank_expanded, bbox_mask=None)

        # Transformer Decoder
        place_node_feat = self.decoder(queries, memory)

        # Attention map
        place_attn = torch.bmm(
            self.attn_proj(place_node_feat),
            memory.transpose(1, 2)
        ) / (C ** 0.5)
        place_attn = place_attn.softmax(dim=-1)

        # 预测头
        place_exist_logits = self.exist_head(place_node_feat).squeeze(-1)

        return place_node_feat, place_attn, place_exist_logits
