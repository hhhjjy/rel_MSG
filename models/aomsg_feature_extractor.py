
import torch
import torch.nn as nn
import numpy as np

from .pos_embed import get_2d_sincos_pos_embed
from .aomsg_losses import (
    InfoNCELoss,
    MaskBCELoss,
    FocalLoss,
    MaskMetricLoss,
    MeanSimilarityLoss,
    TotalCodingRate,
    get_match_idx,
    get_association_sv
)


class AOMSGFeatureExtractor(nn.Module):
    """
    AOMSG 特征提取器 - 完整实现
    
    参考实现: reference/MSG/models/msgers/aomsg.py (DecoderAssociator)
    
    该模块实现了 AOMSG 论文中的完整功能，包括:
    - Object/Place 特征投影
    - Box 嵌入
    - 2D Sincos 位置编码
    - Transformer Decoder
    - Scene-level Attention
    - 预测与损失函数
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,
        output_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 1,
        object_dim: int = 768,
        place_dim: int = 768,
        num_img_patches: int = 256,
        model: str = "dinov2-base",
        pr_loss: str = "mse",
        obj_loss: str = "bce",
        pos_weight: float = 10.0,
        **kwargs
    ):
        super().__init__()
        
        self.model_name = model
        self.object_dim = object_dim
        self.place_dim = place_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_img_patches = num_img_patches
        
        # =======================================
        # 1. Transformer Decoder (aomsg.py:97-108)
        # =======================================
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=int(hidden_dim * 4),
            dropout=0.1,
            activation='gelu',
            layer_norm_eps=1e-5,
            batch_first=True,
            norm_first=False,
        )
        decoder_norm = nn.LayerNorm(hidden_dim, eps=1e-5, elementwise_affine=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.num_layers, norm=decoder_norm)
        
        # =======================================
        # 2. Box 嵌入层 (aomsg.py:111-112)
        # =======================================
        self.box_emb = nn.Linear(4, hidden_dim, bias=False)
        self.whole_box = nn.Parameter(
            torch.tensor([0, 0, 224, 224], dtype=torch.float32),
            requires_grad=False
        )
        
        # =======================================
        # 3. Object/Place 特征投影层 (aomsg.py:115-116)
        # =======================================
        self.object_proj = nn.Linear(object_dim, hidden_dim, bias=False)
        self.place_proj = nn.Linear(place_dim, hidden_dim, bias=False)
        
        # =======================================
        # 4. 输出 Heads (aomsg.py:124, 132)
        # =======================================
        self.object_head = nn.Linear(hidden_dim, output_dim)
        self.place_head = nn.Linear(hidden_dim, output_dim)
        
        # =======================================
        # 5. 位置编码 (aomsg.py:134, 196-201)
        # =======================================
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_img_patches + 1, hidden_dim),
            requires_grad=False
        )
        
        # =======================================
        # 6. Scene Decoder (aomsg.py:136-147)
        # =======================================
        scene_decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=int(hidden_dim * 4),
            dropout=0.1,
            activation='gelu',
            layer_norm_eps=1e-5,
            batch_first=True,
            norm_first=False,
        )
        scene_decoder_norm = nn.LayerNorm(hidden_dim, eps=1e-5, elementwise_affine=True)
        self.scene_decoder = nn.TransformerDecoder(scene_decoder_layer, num_layers=1, norm=scene_decoder_norm)
        
        # =======================================
        # 7. 损失函数设置 (aomsg.py:150-178)
        # =======================================
        self.measure_cos_pp = False
        if pr_loss == "bce":
            w = kwargs.get('pp_weight', 1.0)
            self.pr_loss_fn = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([w]))
            self.measure_cos_pp = False
        else:
            self.pr_loss_fn = nn.MSELoss(reduction='none')
            self.measure_cos_pp = True
        
        self.measure_cos_obj = False
        if obj_loss == "bce":
            self.obj_loss_fn = MaskBCELoss(pos_weight=pos_weight)
            self.measure_cos_obj = False
        elif obj_loss == "focal":
            alpha = kwargs.get('alpha', 0.5)
            gamma = kwargs.get('gamma', 2.0)
            self.obj_loss_fn = FocalLoss(alpha=alpha, gamma=gamma)
            self.measure_cos_obj = False
        elif obj_loss == "infonce":
            temperature = kwargs.get('temperature', 0.1)
            self.obj_loss_fn = InfoNCELoss(temperature=temperature, learnable=False)
            self.measure_cos_obj = False
        else:
            self.obj_loss_fn = MaskMetricLoss()
            self.measure_cos_obj = True
        
        self.obj_loss_fn_sim = MeanSimilarityLoss()
        self.obj_tcr = TotalCodingRate(eps=0.2)
        
        # =======================================
        # 8. Pair Net (可选)
        # =======================================
        self.use_pair_net = kwargs.get('use_pair_net', False)
        self.train_pair_net_only = kwargs.get('train_pair_net_only', False)
        if self.use_pair_net or self.train_pair_net_only:
            self.pair_net = None  # 暂不实现 PairNet
        
        # 初始化权重和位置编码
        self.initialize_weights()
    
    def initialize_weights(self):
        """初始化权重 (aomsg.py:191-201)
        """
        self.apply(self._init_weights)
        
        # 初始化位置编码
        grid_size = int(self.num_img_patches ** 0.5)
        pos_embed = get_2d_sincos_pos_embed(
            embed_dim=self.pos_embed.shape[-1],
            grid_size=grid_size,
            cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
    
    def _init_weights(self, m):
        """权重初始化 (aomsg.py:205-213)
        """
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def pad_objects(self, object_emb):
        """填充物体特征 (aomsg.py:223-225)
        """
        if isinstance(object_emb, list):
            padded_obj_emb = torch.nn.utils.rnn.pad_sequence(object_emb, batch_first=True, padding_value=0)
            return padded_obj_emb
        return object_emb
    
    def get_query_mask(self, padded_object_emb):
        """获取 Query mask (aomsg.py:227-237)
        """
        B, L, Ho = padded_object_emb.shape
        img_length = 1
        obj_mask = (padded_object_emb == 0).all(dim=-1).to(padded_object_emb.device)
        place_mask = torch.zeros(B, img_length, dtype=obj_mask.dtype, device=obj_mask.device)
        total_mask = torch.cat([place_mask, obj_mask], dim=1)
        return total_mask
    
    def _process_place_features(self, place_emb):
        """处理场景/图像特征 (aomsg.py:262-266)
        """
        if len(place_emb.size()) == 4:
            B, C, H, W = place_emb.shape
            place_emb = torch.einsum("bchw -> bhwc", place_emb)
            place_emb = place_emb.view(B, -1, C)
        return place_emb
    
    def predict_object(self, padded_obj_feat):
        """物体预测 (aomsg.py:354-368)
        """
        B, K, H = padded_obj_feat.size()
        if self.measure_cos_obj:
            norm = padded_obj_feat.norm(dim=-1, keepdim=True)
            norm = torch.where(norm == 0, torch.ones_like(norm), norm)
            normed_obj_feat = padded_obj_feat / norm
        else:
            normed_obj_feat = padded_obj_feat
        flatten_obj_feat = normed_obj_feat.view(-1, H)
        object_predictions = flatten_obj_feat @ flatten_obj_feat.t()
        return object_predictions
    
    def predict_place(self, place):
        """场景预测 (aomsg.py:370-379)
        """
        if self.measure_cos_pp:
            normed_p = place / place.norm(dim=-1, keepdim=True)
        else:
            normed_p = place
        place_logits = normed_p @ normed_p.t()
        return place_logits
    
    def scene_level_attention(self, object_enc, vid_idx, obj_mask):
        """场景级别注意力 (aomsg.py:306-351)
        """
        B, K, D = object_enc.shape
        device = object_enc.device
        
        vid_idx = vid_idx.to(device)
        
        padded_scene_obj_enc = torch.zeros_like(object_enc)
        
        unique_vids = vid_idx.unique(sorted=True)
        
        for vid in unique_vids:
            vid_mask = (vid_idx == vid)
            vid_obj_enc = object_enc[vid_mask]
            vid_obj_mask = obj_mask[vid_mask]
            
            T, M, D = vid_obj_enc.shape
            
            vid_obj_flat = vid_obj_enc.view(-1, D)
            vid_mask_flat = vid_obj_mask.view(-1)
            
            valid_indices = vid_mask_flat.nonzero(as_tuple=True)[0]
            if len(valid_indices) == 0:
                continue
            
            valid_vid_obj = vid_obj_flat[valid_indices].unsqueeze(0)
            tgt_mask = torch.zeros(1, len(valid_indices), dtype=torch.bool, device=device)
            
            scene_decoded = self.scene_decoder(
                tgt=valid_vid_obj,
                memory=valid_vid_obj,
                tgt_key_padding_mask=tgt_mask,
            )
            
            result_flat = torch.zeros_like(vid_obj_flat)
            result_flat[valid_indices] = scene_decoded.squeeze(0)
            result_3d = result_flat.view(T, M, D)
            
            padded_scene_obj_enc[vid_mask] = result_3d
        
        return padded_scene_obj_enc
    
    def object_similarity_loss(self, embeddings, reorderd_idx):
        """物体相似度损失 (参考 aomsg.py)"""
        sim_loss, mean_dis, id_counts, embeddings_mean = self.obj_loss_fn_sim(embeddings, reorderd_idx)
        tcr = self.obj_tcr(embeddings, reorderd_idx)
        return sim_loss, mean_dis, tcr, id_counts, embeddings_mean
    
    def object_association_loss(self, object_predictions, reorderd_idx):
        """物体关联损失 (参考 aomsg.py)
        """
        supervision_matrix, mask = get_association_sv(reorderd_idx)
        return self.obj_loss_fn(object_predictions, supervision_matrix, mask)
    
    def place_recognition_loss(self, place_predictions, place_labels):
        """场景识别损失 (参考 aomsg.py)
        """
        return self.pr_loss_fn(place_predictions, place_labels)
    
    def get_loss(self, results, additional_info, match_inds, place_labels, rel_labels=None, weights=None):
        """获取损失 (aomsg.py:381-472)
        """
        num_emb = results['embeddings'].size(1)
        reorderd_idx = get_match_idx(match_inds, additional_info, num_emb)
        logs = {}
        
        if self.train_pair_net_only:
            return torch.tensor(0.0), {}
        
        sim_loss, mean_dis, tcr, id_counts, embeddings_mean = self.object_similarity_loss(results['embeddings'], reorderd_idx)
        logs['tcr'] = tcr.item()
        logs['obj_sim_loss'] = sim_loss.item()
        logs['mean_dis'] = mean_dis.item()
        
        object_loss = self.object_association_loss(results['object_predictions'], reorderd_idx)
        logs['running_loss_obj'] = object_loss.item()
        
        place_loss = self.place_recognition_loss(results['place_predictions'], place_labels)
        
        total_loss = object_loss + (weights['pr'] * place_loss if weights else place_loss)
        logs['running_loss_pr'] = place_loss.item()
        
        return total_loss, logs
    
    def forward(self, object_emb, place_emb, detections, vid_idx=None):
        """前向传播 (与 aomsg.py:240-304 保持一致)
        
        input:
            object_emb: list of B elements, each is a tensor of embeddings (K x Ho) of that image but in various lengths K.
            detections: list of B elements, each is a tensor of detections (K x 4) of that image, in various lengths K.
            place_emb: B x L x Hp, or B x H x W x Hp, D is the dimension of the place embeddings
            vid_idx: (B,), LongTensor indicating which scene each frame belongs to
        output:
            object_association_loss, place_recognition_loss
        """
        # ==================== Per-frame processing ====================
        padded_obj_embd = self.pad_objects(object_emb)
        B, K, Ho = padded_obj_embd.shape
        padded_obj_box = self.pad_objects(detections)
        
        whole_box_expanded = self.whole_box.unsqueeze(0).expand(B, 1, -1)
        query = torch.cat([whole_box_expanded, padded_obj_box], dim=1) / 224.0
        query = self.box_emb(query)
        
        query_mask = self.get_query_mask(padded_obj_embd)
        
        # flatten place
        place_emb = self._process_place_features(place_emb)
        
        object_feat = self.object_proj(padded_obj_embd)
        place_feat = self.place_proj(place_emb)
        
        conditioning = torch.cat([place_feat.mean(dim=1, keepdim=True), object_feat], dim=1)
        query = query + conditioning
        
        memory = place_feat + self.pos_embed[:, :place_feat.size(1), :]
        
        # decoding
        decoded_emb = self.decoder(
            tgt=query,
            memory=memory,
            tgt_key_padding_mask=query_mask,
        )
        
        # object and place predictions
        place_enc = self.place_head(decoded_emb[:, 0, :])
        object_enc = self.object_head(decoded_emb[:, 1:, :])
        
        place_logits = self.predict_place(place_enc)
        object_logits = self.predict_object(object_enc)
        
        results = {
            'embeddings': object_enc,
            'place_embeddings': place_enc,
            'place_predictions': place_logits,
            'object_predictions': object_logits,
        }
        
        # ==================== Scene-level Cross-Attention ====================
        if vid_idx is not None:
            scene_object_enc = self.scene_level_attention(
                decoded_emb[:, 1:, :], vid_idx, query_mask[:, 1:]
            )
            results['scene_embeddings'] = scene_object_enc
            results['vid_idx'] = vid_idx
        
        return results
    
    # =======================================
    # 多视角 RelationalMSG 适配函数
    # =======================================
    def forward_multi_view(self, img_feats, bbox_feats, bboxes, bbox_masks=None, vid_idx=None):
        """多视角前向传播，适配 RelationalMSG
        
        这是一个额外的适配函数，用于处理多个视角的输入
        """
        B, V, N, C = bbox_feats.shape
        device = bbox_feats.device
        
        all_results = []
        
        for v in range(V):
            # 获取当前视角的数据
            if bbox_masks is not None:
                valid_mask = bbox_masks[:, v]
                valid_bboxes = bboxes[:, v]
                valid_obj_feat = bbox_feats[:, v]
            else:
                valid_bboxes = bboxes[:, v]
                valid_obj_feat = bbox_feats[:, v]
                valid_mask = torch.ones(B, N, dtype=torch.bool, device=device)
            
            # 为每个视角单独调用原始的 forward
            # 注意：这里我们适配输入格式
            object_emb_list = [valid_obj_feat[b] for b in range(B)]
            detections_list = [valid_bboxes[b] for b in range(B)]
            current_place_feat = img_feats[:, v]
            
            results_v = self.forward(
                object_emb_list,
                current_place_feat,
                detections_list,
                vid_idx
            )
            results_v['decoded_emb'] = None  # 添加 decoded_emb 占位
            all_results.append(results_v)
        
        # =======================================
        # 收集 Refined 特征 (用于 RelationalMSG)
        # =======================================
        refined_obj_list = []
        refined_img_list = []
        
        for v in range(V):
            refined_obj_list.append(all_results[v]['embeddings'])
            refined_img_list.append(all_results[v]['place_embeddings'])
        
        final_refined_obj = torch.stack(refined_obj_list, dim=1)
        final_refined_img = torch.stack(refined_img_list, dim=1)
        
        # Place 特征带位置编码
        first_place_feat = img_feats[:, 0]
        place_feat_flat = self._process_place_features(first_place_feat)
        place_feat_flat = self.place_proj(place_feat_flat)
        place_feat_with_pos = place_feat_flat + self.pos_embed[:, :place_feat_flat.size(1), :]
        
        return final_refined_obj, final_refined_img, place_feat_with_pos, all_results
