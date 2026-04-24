import torch
import torch.nn as nn
import torch.nn.functional as F


class ObjectQueryDecoder(nn.Module):
    """
    Object Query Decoder: 使用可学习的object queries从refined object memory中查询
    
    输出: object node features, attention map, exist logits等
    """

    def __init__(self, dim=256, num_queries=100, num_classes=100):
        super().__init__()
        self.dim = dim
        self.num_queries = num_queries
        self.num_classes = num_classes
        
        self.object_queries = nn.Parameter(torch.randn(num_queries, dim))
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=4,
            dim_feedforward=dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        
        self.exist_head = nn.Linear(dim, 1)
        self.cls_head = nn.Linear(dim, num_classes)
        self.attn_proj = nn.Linear(dim, dim)

    def forward(self, refined_obj_bank):
        """
        Args:
            refined_obj_bank: [B, V, N, C] 跨视角融合后的object bank
            
        Returns:
            object_node_feat: [B, Q, C], Q为query数量
            object_attn: [B, Q, V*N] attention map
            object_exist_logits: [B, Q] existance logits
            object_cls_logits: [B, Q, num_classes] classification logits
        """
        B, V, N, C = refined_obj_bank.shape
        
        memory = refined_obj_bank.reshape(B, V * N, C)
        
        queries = self.object_queries.unsqueeze(0).repeat(B, 1, 1)
        
        object_node_feat = self.decoder(queries, memory)
        
        object_attn = torch.bmm(
            self.attn_proj(object_node_feat),
            memory.transpose(1, 2)
        ) / (C ** 0.5)
        object_attn = object_attn.softmax(dim=-1)
        
        object_exist_logits = self.exist_head(object_node_feat).squeeze(-1)
        object_cls_logits = self.cls_head(object_node_feat)
        
        return object_node_feat, object_attn, object_exist_logits, object_cls_logits


class PlaceQueryDecoder(nn.Module):
    """
    Place Query Decoder: 使用可学习的place queries从refined image memory中查询，
    并融合object-aware信息
    
    输出: place node features等
    """

    def __init__(self, dim=256, num_queries=10):
        super().__init__()
        self.dim = dim
        self.num_queries = num_queries
        
        self.place_queries = nn.Parameter(torch.randn(num_queries, dim))
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=4,
            dim_feedforward=dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        
        self.exist_head = nn.Linear(dim, 1)
        self.attn_proj = nn.Linear(dim, dim)
        
        self.obj_fusion = nn.Linear(dim * 2, dim)

    def forward(self, refined_img_bank, object_node_feat):
        """
        Args:
            refined_img_bank: [B, V, C] 跨视角融合后的image bank
            object_node_feat: [B, Q_obj, C] object node features
            
        Returns:
            place_node_feat: [B, Q_place, C]
            place_attn: [B, Q_place, V]
            place_exist_logits: [B, Q_place]
        """
        B, V, C = refined_img_bank.shape
        
        memory = refined_img_bank
        
        queries = self.place_queries.unsqueeze(0).repeat(B, 1, 1)
        
        obj_agg = object_node_feat.mean(dim=1, keepdim=True)
        queries = self.obj_fusion(torch.cat([
            queries,
            obj_agg.repeat(1, self.num_queries, 1)
        ], dim=-1))
        
        place_node_feat = self.decoder(queries, memory)
        
        place_attn = torch.bmm(
            self.attn_proj(place_node_feat),
            memory.transpose(1, 2)
        ) / (C ** 0.5)
        place_attn = place_attn.softmax(dim=-1)
        
        place_exist_logits = self.exist_head(place_node_feat).squeeze(-1)
        
        return place_node_feat, place_attn, place_exist_logits
