import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationDecoderLayer(nn.Module):
    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        feedforward_channels: int = 2048,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        ffn_drop: float = 0.1,
    ):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dims,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=True,
            bias=True
        )
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dims,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=True,
            bias=True
        )

        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, feedforward_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(ffn_drop),
            nn.Linear(feedforward_channels, embed_dims),
            nn.Dropout(proj_drop)
        )

        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_pos: torch.Tensor = None,
        key_pos: torch.Tensor = None,
        value_pos: torch.Tensor = None,
        query_key_padding_mask: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None,
    ):
        q = query + query_pos if query_pos is not None else query
        k = key + key_pos if key_pos is not None else key
        v = value + value_pos if value_pos is not None else value

        attn_output, _ = self.cross_attn(
            query=q,
            key=k,
            value=v,
            key_padding_mask=key_padding_mask,
        )
        query = query + self.proj_drop(attn_output)
        query = self.norm1(query)

        q_self = query + query_pos if query_pos is not None else query
        attn_output, _ = self.self_attn(
            query=q_self,
            key=q_self,
            value=q_self,
            key_padding_mask=query_key_padding_mask,
        )
        query = query + self.proj_drop(attn_output)
        query = self.norm2(query)

        ffn_output = self.ffn(query)
        query = query + ffn_output
        query = self.norm3(query)

        return query


class RelationDecoder(nn.Module):
    def __init__(
        self,
        num_layers: int = 6,
        return_intermediate: bool = True,
        embed_dims: int = 256,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
    ):
        super().__init__()
        self.return_intermediate = return_intermediate
        self.embed_dims = embed_dims
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            RelationDecoderLayer(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=dim_feedforward,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        rel_query_embed: torch.Tensor,
        pair_feat: torch.Tensor,
        rel_query_pos: torch.Tensor = None,
        pair_pos: torch.Tensor = None,
    ):
        intermediate = []
        query = rel_query_embed
        for layer in self.layers:
            query = layer(
                query=query,
                key=pair_feat,
                value=pair_feat,
                query_pos=rel_query_pos,
                key_pos=pair_pos,
                value_pos=pair_pos,
                query_key_padding_mask=None,
                key_padding_mask=None,
            )
            if self.return_intermediate:
                intermediate.append(query)

        if self.return_intermediate:
            return torch.stack(intermediate)
        return query


class RelationHead(nn.Module):
    def __init__(
        self,
        num_rel_query: int = 100,
        num_relations: int = 56,
        feat_channels: int = 256,
        num_rel_decoder_layers: int = 6,
        num_rel_heads: int = 8,
        dim_rel_feedforward: int = 2048,
    ):
        super().__init__()
        self.num_rel_query = num_rel_query
        self.num_relations = num_relations
        self.feat_channels = feat_channels

        self.rel_query_feat = nn.Embedding(self.num_rel_query, self.feat_channels)
        self.rel_query_embed = nn.Embedding(self.num_rel_query, self.feat_channels)
        self.rel_query_embed2 = nn.Embedding(self.num_rel_query * 2, self.feat_channels)
        self.rel_query_embed3 = nn.Embedding(self.num_rel_query * 2, self.feat_channels)

        self.rel_cls_embed = nn.Linear(self.feat_channels, self.num_relations)

        self.relation_decoder = RelationDecoder(
            num_layers=num_rel_decoder_layers,
            return_intermediate=True,
            embed_dims=feat_channels,
            num_heads=num_rel_heads,
            dim_feedforward=dim_rel_feedforward,
        )

    def forward(
        self,
        sub_query: torch.Tensor,
        obj_query: torch.Tensor,
        sub_cls: torch.Tensor = None,
        obj_cls: torch.Tensor = None,
    ):
        batch_size = sub_query.size(0)
        num_rel = sub_query.size(1)

        pair_feat = torch.cat([sub_query, obj_query], dim=1)

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

        rel_query = self.relation_decoder(
            rel_query_embed=rel_query_feat.transpose(0, 1),  # [batch_size, num_rel_query, C]
            pair_feat=pair_feat,  # [batch_size, 2*k, C] - already batch-first
            rel_query_pos=rel_query_embed.transpose(0, 1),  # [batch_size, num_rel_query, C]
            pair_pos=torch.cat([rel_query_embed2, rel_query_embed3], dim=0).transpose(0, 1),  # [batch_size, 2*num_rel_query, C]
        )

        if self.relation_decoder.return_intermediate:
            rel_query = rel_query[-1]

        rel_query = rel_query.transpose(0, 1)
        rel_pred = self.rel_cls_embed(rel_query)

        outputs = {
            'rel_pred': rel_pred,
            'sub_cls': sub_cls,
            'obj_cls': obj_cls,
        }

        return outputs

    def compute_loss(
        self,
        outputs: dict,
        gt_rels: torch.Tensor = None,
    ):
        loss_dict = {}
        rel_pred = outputs.get('rel_pred')

        if rel_pred is not None and gt_rels is not None and rel_pred.size(1) > 0:
            gt_rels = gt_rels.long()
            if gt_rels.size(1) > rel_pred.size(1):
                gt_rels = gt_rels[:, :rel_pred.size(1)]
            elif gt_rels.size(1) < rel_pred.size(1):
                padding = torch.zeros(
                    gt_rels.size(0),
                    rel_pred.size(1) - gt_rels.size(1),
                    dtype=gt_rels.dtype,
                    device=gt_rels.device
                ).fill_(self.num_relations)
                gt_rels = torch.cat([gt_rels, padding], dim=1)

            valid_mask = gt_rels != self.num_relations
            valid_idx = valid_mask.any(dim=1)

            if valid_idx.sum() > 0:
                rel_cls_loss = F.cross_entropy(
                    rel_pred[valid_idx],
                    gt_rels[valid_idx],
                    reduction='mean',
                )
                loss_dict['loss_rel_cls'] = rel_cls_loss
            else:
                loss_dict['loss_rel_cls'] = torch.tensor(
                    0.0, device=rel_pred.device, requires_grad=True
                )

        return loss_dict
