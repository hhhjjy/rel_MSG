import torch
import torch.nn as nn
import torch.nn.functional as F

from .alternating_attention_decoder import AlternatingAttention


class QueryRefiner(nn.Module):
    """
    Query Refiner: 对可学习 queries 做 self-attention 精炼

    让 queries 在进入 cross-attention 之前先互相了解，
    形成更结构化的查询表示。
    """

    def __init__(self, dim=768, num_heads=8, num_layers=2, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.dim = dim

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout,
            batch_first=True
        )
        self.refiner = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, queries):
        """
        Args:
            queries: [B, Q, C]
        Returns:
            refined_queries: [B, Q, C]
        """
        refined = self.refiner(queries)
        refined = self.norm(refined + queries)  # residual
        return refined


class ObjectQueryDecoderV3(nn.Module):
    """
    Object Query Decoder V3 (Step 3)

    核心变化:
    1. Queries 从外部传入（模型级共享参数），而非 decoder 内部创建
    2. 增加 QueryRefiner: queries 先经过 self-attention 精炼
    3. 保留 Step 2 的 AlternatingAttention 与 bbox 特征交互
    4. 最后通过 TransformerDecoder

    结构:
        external_queries → QueryRefiner → AlternatingAttention → TransformerDecoder → heads
    """

    def __init__(
        self,
        dim=256,
        num_queries=100,
        num_classes=100,
        num_refine_layers=1,
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

        # Step 3 新增: Query Refiner
        self.query_refiner = QueryRefiner(
            dim=dim,
            num_heads=num_heads,
            num_layers=num_refine_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        # Step 2 保留: Alternating Attention
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
        self.cls_head = nn.Linear(dim, num_classes)
        self.attn_proj = nn.Linear(dim, dim)

    def forward(self, queries, refined_obj_bank, bbox_mask=None):
        """
        Args:
            queries: [B, Q, C] 外部传入的可学习 queries（模型级共享）
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

        # Step 3 核心 1: Query Refiner (self-attention 精炼)
        queries = self.query_refiner(queries)

        # Step 3 核心 2: Alternating Attention (与 Step 2 保持一致)
        queries = self.alternating_attn(queries, refined_obj_bank, bbox_mask)

        # Transformer Decoder
        object_node_feat = self.decoder(queries, memory)

        # Attention map
        object_attn = torch.bmm(
            self.attn_proj(object_node_feat),
            memory.transpose(1, 2)
        ) / (C ** 0.5)
        object_attn = object_attn.softmax(dim=-1)

        # 预测头
        object_exist_logits = self.exist_head(object_node_feat).squeeze(-1)
        object_cls_logits = self.cls_head(object_node_feat)

        return object_node_feat, object_attn, object_exist_logits, object_cls_logits


class PlaceQueryDecoderV3(nn.Module):
    """
    Place Query Decoder V3 (Step 3)

    与 ObjectQueryDecoderV3 对应，place queries 也改为外部传入 + QueryRefiner
    """

    def __init__(
        self,
        dim=256,
        num_queries=10,
        num_refine_layers=1,
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

        # Step 3 新增: Query Refiner
        self.query_refiner = QueryRefiner(
            dim=dim,
            num_heads=num_heads,
            num_layers=num_refine_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        # Step 2 保留: Alternating Attention
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

    def forward(self, queries, refined_img_bank, object_node_feat, bbox_mask=None):
        """
        Args:
            queries: [B, Q, C] 外部传入的可学习 place queries
            refined_img_bank: [B, V, C] 跨视角融合后的 image bank
            object_node_feat: [B, Q_obj, C] object node features
            bbox_mask: 不用于 place decoder，保持接口一致

        Returns:
            place_node_feat: [B, Q_place, C]
            place_attn: [B, Q_place, V]
            place_exist_logits: [B, Q_place]
        """
        B, V, C = refined_img_bank.shape
        memory = refined_img_bank

        # 融合 object-aware 信息
        obj_agg = object_node_feat.mean(dim=1, keepdim=True)  # [B, 1, C]
        queries = self.obj_fusion(torch.cat([
            queries,
            obj_agg.repeat(1, self.num_queries, 1)
        ], dim=-1))

        # Step 3 核心 1: Query Refiner
        queries = self.query_refiner(queries)

        # Step 3 核心 2: Alternating Attention
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
