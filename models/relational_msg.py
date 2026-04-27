import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import Backbone
from .roi_extractor import RoIExtractor
from .cross_view_encoder import CrossViewEncoder
from .query_decoder import ObjectQueryDecoder, PlaceQueryDecoder
from .alternating_attention_decoder import ObjectQueryDecoderV2, PlaceQueryDecoderV2
from .learnable_query_decoder import ObjectQueryDecoderV3, PlaceQueryDecoderV3
from .edge_heads import EdgeHeads
from .matching import HungarianMatcher, get_match_targets
from .aomsg_feature_extractor import AOMSGFeatureExtractor
from .amosg.associate import Asso_models
from .amosg.matcher import HungarianMatcher as AOMSGMatcher
from .scene_graph import SceneGraphHead, prepare_gt_importance_matrix
from .object_level_loss import QueryObjectLoss
# from .amosg.loss import convert_detections, get_match_idx


class FeatureExtractor(nn.Module):
    """
    特征提取层: 封装 Backbone + RoIExtractor + CrossViewEncoder + AssociationModel
    
    支持三种模式:
    1. aomsg: 直接使用 association_model (Step 1)
    2. vggt: 使用 obj_cross_view + img_cross_view (Step 3/4)
    3. aomsg_refined: 使用 obj_cross_view refine 后再用 association_model (Step 2)
    
    输入: images, bboxes
    输出: 根据模式返回不同格式的特征
    """
    def __init__(self, config, hidden_dim, feature_refine_method):
        super(FeatureExtractor, self).__init__()
        self.hidden_dim = hidden_dim
        self.feature_refine_method = feature_refine_method
        
        self.backbone = Backbone(
            model_type=config['backbone']['model_type'] if config and 'backbone' in config else 'dinov2-base',
            freeze=config['backbone']['freeze'] if config and 'backbone' in config else True,
            weights=config['backbone']['weights'] if config and 'backbone' in config else 'DEFAULT',
        )
        self.roi_extractor = RoIExtractor(
            feat_dim=hidden_dim,
            roi_size=1,
            image_size=config['model_image_size'] if config and 'model_image_size' in config else (224, 224),
        )

        # box embedding
        self.box_emb = nn.Linear(4, hidden_dim, bias=False)
        self.whole_box = nn.Parameter(torch.tensor([0, 0, 224, 224], dtype=torch.float32), requires_grad=False)

        # input adaptor
        self.object_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.place_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # AssociationModel (用于 aomsg 和 aomsg_refined 模式)
        if self.feature_refine_method == 'aomsg':
            loss_keys = ["pr_loss", "obj_loss", "temperature", "alpha", "gamma", "pos_weight", "pp_weight"]
            for k in loss_keys:
                if k in config:
                    config['associator'][k] = config[k]
            self.association_model = Asso_models[config['associator']['model']](**config['associator'])
        elif self.feature_refine_method == 'aomsg_refined':
            loss_keys = ["pr_loss", "obj_loss", "temperature", "alpha", "gamma", "pos_weight", "pp_weight"]
            for k in loss_keys:
                if k in config:
                    config['associator'][k] = config[k]
            self.association_model = Asso_models[config['associator']['model']](**config['associator'])
            self.obj_cross_view = CrossViewEncoder(dim=hidden_dim)
        else:
            self.association_model = None
            self.cross_view = CrossViewEncoder(dim=hidden_dim)
    
    def _compute_conditional_query(self, obj_box, obj_embd, place_emb, obj_mask):
        """
        Args:
            obj_box:  [B, V, K, 4]
            obj_embd: [B, V, K, Ho]
            place_emb: [B, V, N, D]
        Returns:
            query:      [B, V, K+1, C]  🔥 要求的输出形状
            query_mask: [B, V, K+1]    🔥 要求的输出形状
            place_feat: [B, V, N, C]
        """
        # 1. 保存原始 Batch/Video 维度（最后reshape回去用）
        B, V, K, Ho = obj_embd.shape
        N = place_emb.shape[2]  # place特征的序列长度
        C = self.hidden_dim

        # 2. 临时展平 B&V 做计算（兼容原逻辑）
        obj_embd_flat = obj_embd.view(B*V, K, Ho)      # [B*V, K, Ho]
        obj_box_flat = obj_box.view(B*V, K, 4)        # [B*V, K, 4]
        place_emb_flat = place_emb.view(B*V, N, -1)    # [B*V, N, D]
        obj_mask_flat = obj_mask.view(B*V, K)

        place_mask = torch.ones(B*V, 1, dtype = obj_mask.dtype, device = obj_mask.device)
        query_mask_flat = torch.cat([place_mask, obj_mask_flat], dim=1)

        # 3. 原核心计算逻辑（完全不变）
        # BBox编码
        whole_box_expanded = self.whole_box.unsqueeze(0).expand(B*V, 1, -1)
        query_flat = torch.cat([whole_box_expanded, obj_box_flat], dim=1) / 224.0
        query_flat = self.box_emb(query_flat)  # [B*V, K+1, C]

        # 特征投影
        object_feat = self.object_proj(obj_embd_flat)
        place_feat_flat = self.place_proj(place_emb_flat)

        # 条件融合
        conditioning = torch.cat([place_feat_flat.mean(dim=1, keepdim=True), object_feat], dim=1)
        query_flat = query_flat + conditioning

        # 4. 🔥 关键：恢复为 [B, V, ...] 原始形状
        query = query_flat.view(B, V, K+1, C)
        query_mask = query_mask_flat.view(B, V, K+1)

        return query, query_mask

    def forward(self, images_per_scene, bboxes_pos, bboxes_masks, use_cross_view_refine=False):
        """
        Args:
            images_per_scene: [B, V, 3, H, W]
            bboxes_pos: [B, V, N, 4]
            bboxes_masks: [B, V, N]
            use_cross_view_refine: bool, 是否使用 obj_cross_view 对 bbox_feats 进行 refine
                                    (Step 2 使用 True, Step 1 使用 False)
        
        Returns:
            根据 feature_refine_method 返回不同格式:
            - aomsg (use_cross_view_refine=False): (results, img_feats, bbox_feats)
            - aomsg_refined (use_cross_view_refine=True): (results, img_feats, bbox_feats, refined_bbox_feats)
            - vggt: (refined_obj_bank, refined_img_bank, img_feats, bbox_feats)
        """
        B, V = images_per_scene.shape[:2]
        img_feats = self.backbone(images_per_scene)
        bbox_feats = self.roi_extractor(img_feats[:, :, 1:, :], bboxes_pos, bboxes_masks, self.training)
        
        if self.feature_refine_method in ['aomsg', 'aomsg_refined']:
            assert self.association_model is not None, "association_model is None"
            # Step 1: 直接使用 association_model
            if use_cross_view_refine:
                bbox_feats = self.obj_cross_view(bbox_feats, None, bboxes_pos, bboxes_masks)
            results = self.association_model(bbox_feats, img_feats, bboxes_pos)
            return results, img_feats, bbox_feats
        else:
            query, query_mask = self._compute_conditional_query(bboxes_pos, bbox_feats, img_feats, bboxes_masks)

            # vggt 模式 (Step 3/4)
            refined_feat_bank = self.cross_view(query, None, None, query_mask)

            refined_obj_bank = refined_feat_bank[:, :, 1:, :]
            refined_img_bank = refined_feat_bank[:, :, 0, :]
            return refined_obj_bank, refined_img_bank, img_feats, bbox_feats


class QueryDecoderLayer(nn.Module):
    """
    查询解码层: 封装所有版本的 Object/Place Query Decoder
    根据 stage 自动选择对应的 decoder
    """
    def __init__(self, config, hidden_dim, num_obj_queries, num_place_queries, num_obj_classes):
        super(QueryDecoderLayer, self).__init__()
        self.stage = config.get('stage', 'step1')  # step1, step2, step3, step4
        
        # Baseline V1 Decoders
        self.obj_decoder = ObjectQueryDecoder(
            dim=hidden_dim, num_queries=num_obj_queries, num_classes=num_obj_classes,
        )
        # self.place_decoder_v1 = PlaceQueryDecoder(
        #     dim=hidden_dim, num_queries=num_place_queries,
        # )
        
        # # Step 2 Decoders (Alternating Attention)
        # self.obj_decoder_v2 = ObjectQueryDecoderV2(
        #     dim=hidden_dim, num_queries=num_obj_queries, num_classes=num_obj_classes,
        # )
        # self.place_decoder_v2 = PlaceQueryDecoderV2(
        #     dim=hidden_dim, num_queries=num_place_queries,
        # )
        
        # # Step 3/4 Decoders (Learnable Queries + QueryRefiner + AA)
        # self.obj_decoder_v3 = ObjectQueryDecoderV3(
        #     dim=hidden_dim, num_queries=num_obj_queries, num_classes=num_obj_classes,
        # )
        # self.place_decoder_v3 = PlaceQueryDecoderV3(
        #     dim=hidden_dim, num_queries=num_place_queries,
        # )
        
        # # Step 3/4 Learnable Queries
        # self.obj_queries = nn.Parameter(torch.randn(num_obj_queries, hidden_dim))
        # self.place_queries = nn.Parameter(torch.randn(num_place_queries, hidden_dim))
    
    def forward(self, refined_obj_bank, refined_img_bank, bboxes_masks=None, stage=None):
        stage = stage or self.stage
        B = refined_obj_bank.shape[0]
        
        object_node_feat, object_attn, object_exist_logits, object_cls_logits = self.obj_decoder(refined_obj_bank)
        # place_node_feat, place_attn, place_exist_logits = self.place_decoder(refined_img_bank, object_node_feat)

        # if stage in ['step1', 'amosg']:
        #     object_node_feat, object_attn, object_exist_logits, object_cls_logits = self.obj_decoder_v1(refined_obj_bank)
        #     place_node_feat, place_attn, place_exist_logits = self.place_decoder_v1(refined_img_bank, object_node_feat)
        # elif stage == 'step2':
        #     object_node_feat, object_attn, object_exist_logits, object_cls_logits = self.obj_decoder_v2(refined_obj_bank, bbox_mask=bboxes_masks)
        #     place_node_feat, place_attn, place_exist_logits = self.place_decoder_v2(refined_img_bank, object_node_feat, bbox_mask=None)
        # elif stage in ['step3', 'step4']:
        #     obj_queries_batch = self.obj_queries.unsqueeze(0).expand(B, -1, -1)
        #     object_node_feat, object_attn, object_exist_logits, object_cls_logits = self.obj_decoder_v3(
        #         obj_queries_batch, refined_obj_bank, bbox_mask=bboxes_masks
        #     )
        #     place_queries_batch = self.place_queries.unsqueeze(0).expand(B, -1, -1)
        #     place_node_feat, place_attn, place_exist_logits = self.place_decoder_v3(
        #         place_queries_batch, refined_img_bank, object_node_feat, bbox_mask=None
        #     )
        # else:
        #     raise ValueError(f"Unknown stage: {stage}")
        
        return {
            'object_node_feat': object_node_feat,
            'object_attn': object_attn,
            'object_exist_logits': object_exist_logits,
            'object_cls_logits': object_cls_logits,
            'place_node_feat': place_node_feat,
            'place_attn': place_attn,
            'place_exist_logits': place_exist_logits,
        }


class RelationalMSG(nn.Module):
    """
    Query-based Relational Multi-View Scene Graph Generation 主模型 (重构版)
    
    结构解耦:
    1. FeatureExtractor: 特征提取层 (Backbone + RoI + CrossView)
    2. QueryDecoderLayer: 查询解码层 (Object/Place Query Decoder)
    3. EdgeHeads: 边预测层
    4. SceneGraphHead: 场景图预测层 (Step 4)
    
    阶段路由:
    - step1/amosg: baseline AOMSG
    - step2: Alternating Attention
    - step3: Learnable Queries + Object-level Loss
    - step4: Step3 + Scene Graph Head
    """

    def __init__(self, config, device):
        super(RelationalMSG, self).__init__()

        self.hidden_dim = config.get('hidden_model_dim', 768)
        self.num_obj_queries = config.get('num_obj_queries', 100)
        self.num_place_queries = config.get('num_place_queries', 10)
        self.num_obj_classes = config.get('num_obj_classes', 18)
        self.num_edge_types = config.get('num_edge_types', 1)
        self.num_views = config.get('num_views', 4)
        self.num_bboxes_per_view = config.get('num_bboxes_per_view', 20)
        self.stage = config.get('stage', 'step1')
        
        # 根据 stage 设置 feature_refine_method
        if self.stage in ['step1', 'amosg']:
            self.feature_refine_method = 'aomsg'
        elif self.stage == 'step2':
            self.feature_refine_method = 'aomsg_refined'
        else:
            self.feature_refine_method = 'vggt'
        
        # 1. 特征提取层 (所有 stage 都需要)
        self.feature_extractor = FeatureExtractor(config, self.hidden_dim, self.feature_refine_method)

        if self.stage in ['step3']:
            self.obj_decoder = ObjectQueryDecoder(
                dim=self.hidden_dim, num_queries=self.num_obj_queries, num_classes=self.num_obj_classes,
            )
            self.query_view_proj = nn.Sequential(
                nn.Linear(3 * self.hidden_dim, self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            self.vis_pred_head = nn.Sequential(
                nn.Linear(3 * self.hidden_dim, self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_dim, 1),
            )
            self.place_affinity_mlp = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 1)
            )
        
        # # 2. 查询解码层 (仅 Step 3/4 需要)
        # if self.stage in ['step3', 'step4']:
        #     self.query_decoder = QueryDecoderLayer(
        #         config, self.hidden_dim, self.num_obj_queries, 
        #         self.num_place_queries, self.num_obj_classes
        #     )
        # else:
        #     self.query_decoder = None
        
        # # 3. 边预测层 (仅 Step 3/4 需要)
        # if self.stage in ['step3', 'step4']:
        #     self.edge_heads = EdgeHeads(
        #         dim=self.hidden_dim,
        #         num_edge_types=self.num_edge_types,
        #     )
        # else:
        #     self.edge_heads = None
        
        # # 4. 场景图预测层 (仅 Step 4 需要)
        # if self.stage == 'step4':
        #     num_relations = config.get('num_relations', 56)
        #     num_rel_queries = config.get('num_rel_queries', 50)
        #     self.scene_graph_head = SceneGraphHead(
        #         num_classes=self.num_obj_classes,
        #         num_relations=num_relations,
        #         num_obj_query=self.num_obj_queries,
        #         num_rel_query=num_rel_queries,
        #         feat_channels=self.hidden_dim,
        #     )
        #     self.use_scene_graph = True
        # else:
        #     self.scene_graph_head = None
        #     self.use_scene_graph = False
        
        # # 5. Loss 模块
        # # 匹配器 (Step 3/4 需要)
        # if self.stage in ['step3', 'step4']:
        #     self.matcher = HungarianMatcher(cost_exist=1.0, cost_attn=1.0)
        # else:
        #     self.matcher = None
        
        # # AOMSG 匹配器 (Step 1/2 需要)
        # if self.stage in ['step1', 'amosg', 'step2']:
        #     self.aomsg_matcher = AOMSGMatcher()
        # else:
        #     self.aomsg_matcher = None
        
        # # Step 3: Object-level Query Loss (仅 Step 3/4 需要)
        # if self.stage in ['step3', 'step4']:
        #     lq_config = config.get('learnable_queries', {})
        #     self.query_object_loss = QueryObjectLoss(
        #         cost_cls=lq_config.get('cost_cls', 1.0),
        #         cost_attn=lq_config.get('cost_attn', 1.0),
        #         weight_cls=lq_config.get('weight_cls', 1.0),
        #         weight_attn=lq_config.get('weight_attn', 1.0),
        #         weight_exist=lq_config.get('weight_exist', 1.0),
        #         weight_bbox=lq_config.get('weight_bbox', 0.0),
        #     )
        # else:
        #     self.query_object_loss = None

    def _extract_common_features(self, images, bboxes_infos):
        """
        公共特征提取流程，被所有 forward 方法复用
        
        Returns:
            dict: 包含所有中间特征的字典
        """
        # 1. 预处理
        images_per_scene, bboxes_per_scene = self.preprocess(images, bboxes_infos)
        B, V = images_per_scene.shape[:2]
        bboxes_pos = bboxes_per_scene['gt_bbox']
        bboxes_masks = bboxes_per_scene['mask']
        
        # 2. 特征提取
        # Step 2 使用 use_cross_view_refine=True，在 association_model 前插入 obj_cross_view
        use_cross_view_refine = (self.stage == 'step2')
        feat_result = self.feature_extractor(images_per_scene, bboxes_pos, bboxes_masks, use_cross_view_refine=use_cross_view_refine)
        
        if self.stage in ['step1', 'amosg', 'step2']:
            # Step 1: (results, img_feats, bbox_feats)
            results, img_feats, bbox_feats = feat_result
            refined_obj_bank = results.get('embeddings')
            refined_img_bank = results.get('place_embeddings')
            extra_outputs = results
        else:
            # Step 3/4 (vggt): (refined_obj_bank, refined_img_bank, img_feats, bbox_feats)
            refined_obj_bank, refined_img_bank, img_feats, bbox_feats = feat_result
            extra_outputs = {}
        
        # 展平 bbox_feats
        B, V, N, C = bbox_feats.shape
        bbox_feats_flat = bbox_feats.reshape(B, V*N, C)
        
        return {
            'images_per_scene': images_per_scene,
            'bboxes_per_scene': bboxes_per_scene,
            'bboxes_pos': bboxes_pos,
            'bboxes_masks': bboxes_masks,
            'img_feats': img_feats,
            'bbox_feats': bbox_feats,
            'bbox_feats_flat': bbox_feats_flat,
            'refined_obj_bank': refined_obj_bank,
            'refined_img_bank': refined_img_bank,
            'extra_outputs': extra_outputs,
            'B': B,
            'V': V,
        }
    
    def build_legacy_compatible_outputs(
        self,
        object_node_feat,
        place_node_feat,
        bboxes_infos,
        decoder_outputs=None,
        edge_outputs=None,
        scene_graph_outputs=None,
        bbox_feats_flat=None,
    ):
        """
        统一输出适配函数: 将所有 stage 的输出转换为兼容 baseline 的格式
        
        支持:
        - step1/amosg: baseline AOMSG 输出
        - step2: Alternating Attention 输出
        - step3: Learnable Queries 输出
        - step4: Scene Graph Head 输出
        
        Args:
            object_node_feat: [B, Q, C] object query features
            place_node_feat: [B, Q_place, C] place query features
            bboxes_infos: 原始 bbox 信息字典
            decoder_outputs: QueryDecoderLayer 输出字典 (可选)
            edge_outputs: EdgeHeads 输出元组 (pp_logits, po_logits) (可选)
            scene_graph_outputs: SceneGraphHead 输出字典 (可选)
            bbox_feats_flat: [B, V*N, C] 展平后的 bbox 特征 (可选)
            
        Returns:
            dict: 统一格式的输出，包含所有必要的 key
        """
        # 1. 基础 embedding 输出 (所有 stage 共有)
        obj_embeddings = object_node_feat
        place_embeddings = place_node_feat.mean(dim=1)  # [B, C]
        
        # 2. 相似度预测 (baseline 兼容)
        normed_p = place_embeddings / (place_embeddings.norm(dim=-1, keepdim=True) + 1e-8)
        place_predictions = normed_p @ normed_p.t()
        
        B, Q, C = obj_embeddings.shape
        flatten_obj = obj_embeddings.view(-1, C)
        object_predictions = flatten_obj @ flatten_obj.t()
        
        # 3. 构建基础输出字典
        results = {
            'embeddings': obj_embeddings,
            'place_embeddings': place_embeddings,
            'place_predictions': place_predictions,
            'object_predictions': object_predictions,
            'detections': convert_detections(bboxes_infos),
            'object_node_feat': object_node_feat,
            'place_node_feat': place_node_feat,
        }
        
        # 4. 添加 decoder 输出 (step2/3/4)
        if decoder_outputs is not None:
            results.update({
                'object_attn': decoder_outputs.get('object_attn'),
                'object_exist_logits': decoder_outputs.get('object_exist_logits'),
                'object_cls_logits': decoder_outputs.get('object_cls_logits'),
                'place_attn': decoder_outputs.get('place_attn'),
                'place_exist_logits': decoder_outputs.get('place_exist_logits'),
            })
        
        # 5. 添加 edge head 输出 (step2/3/4)
        if edge_outputs is not None:
            pp_logits, po_logits = edge_outputs
            results.update({
                'pp_logits': pp_logits,
                'po_logits': po_logits,
            })
        
        # 6. 添加 bbox 特征 (step2/3/4)
        if bbox_feats_flat is not None:
            results['bbox_feats'] = bbox_feats_flat
        
        # 7. 添加场景图输出 (step4)
        if scene_graph_outputs is not None:
            results.update({
                'sgg_outputs': scene_graph_outputs,
                'sgg_triplets': scene_graph_outputs.get('triplets', []),
                'sgg_rel_pred': scene_graph_outputs.get('rel_pred'),
                'sgg_sub_pos': scene_graph_outputs.get('sub_pos'),
                'sgg_obj_pos': scene_graph_outputs.get('obj_pos'),
                'sgg_importance': scene_graph_outputs.get('importance'),
            })
        
        return results

    def _build_legacy_outputs(self, object_node_feat, place_node_feat, bboxes_infos, extra_outputs=None):
        """
        构建与 baseline 兼容的输出格式 (兼容旧接口)
        
        注意: 新代码应使用 build_legacy_compatible_outputs()
        """
        return self.build_legacy_compatible_outputs(
            object_node_feat=object_node_feat,
            place_node_feat=place_node_feat,
            bboxes_infos=bboxes_infos,
            decoder_outputs=extra_outputs,
        )

    def forward(self, images, bboxes_infos, bbox_masks=None):
        """
        统一前向入口，根据 self.stage 自动路由到对应的方法
        
        Args:
            images: [B, V, 3, H, W] 多视角图像
            bboxes_infos: bbox 信息字典
            bbox_masks: bbox 掩码
            
        Returns:
            outputs: 包含所有输出的字典
        """
        stage_map = {
            'step1': self.forward_amosg,
            'amosg': self.forward_amosg,
            'step2': self.forward_step2,
            'step3': self.forward_step3,
            'step4': self.forward_step4,
        }
        
        if self.stage not in stage_map:
            raise ValueError(f"Unknown stage: {self.stage}. Must be one of {list(stage_map.keys())}")
        
        return stage_map[self.stage](images, bboxes_infos, bbox_masks)

    def forward_step2(self, images, bboxes_infos, bbox_masks=None):
        """
        Step 2: 在 Baseline (Step 1) 前插入 obj_cross_view 进行特征 refine
        
        数据流:
        images, bboxes_infos
            → preprocess
            → backbone + roi_extractor
            → obj_cross_view (新增! 对 bbox_feats 进行 refine)
            → association_model (与 Step 1 相同)
            → 输出 (与 Step 1 格式相同)
        
        与 Step 1 的区别:
        - 在 association_model 前增加了 obj_cross_view 模块
        - 其余流程与 Step 1 完全一致
        """
        # 1. 公共特征提取 (会自动使用 use_cross_view_refine=True)
        features = self._extract_common_features(images, bboxes_infos)
        
        # 2. 直接使用 association_model 的结果 (与 Step 1 相同)
        extra_outputs = features['extra_outputs']
        
        # 3. 构建统一兼容输出 (与 Step 1 相同)
        obj_embeddings = extra_outputs.get('embeddings')
        place_embeddings = extra_outputs.get('place_embeddings')
        
        if obj_embeddings is None or place_embeddings is None:
            raise RuntimeError(
                "Step 2 association_model did not return 'embeddings' or 'place_embeddings'. "
                "Please check the association_model output format."
            )
        
        results = self.build_legacy_compatible_outputs(
            object_node_feat=obj_embeddings,
            place_node_feat=place_embeddings.unsqueeze(1) if place_embeddings.dim() == 2 else place_embeddings,
            bboxes_infos=bboxes_infos,
            bbox_feats_flat=features['bbox_feats_flat'],
        )
        
        # 保留 association_model 原始输出 (兼容旧接口)
        for k, v in extra_outputs.items():
            if k not in results:
                results[k] = v
        
        return results

    def forward_step3(self, images, bboxes_infos, bbox_masks=None):

        # 1. 公共特征提取
        features = self._extract_common_features(images, bboxes_infos)
        
        refined_box_bank = features['refined_obj_bank'].reshape(features['refined_obj_bank'].shape[0], -1, features['refined_obj_bank'].shape[-1])
        refined_img_bank = features['refined_img_bank'].reshape(features['refined_img_bank'].shape[0], -1, features['refined_img_bank'].shape[-1])
        B, N_b, D = refined_box_bank.shape  
        _, N_p, _ = refined_img_bank.shape
        N_q = self.num_obj_queries

        # N_b bbox数量，N_p place数量，N_q query数量
        # 2. Object Query 解码
        object_node_feat, object_attn, object_exist_logits, object_cls_logits = self.obj_decoder(
            refined_box_bank,
            bboxes_masks=features['bboxes_masks'].reshape(features['bboxes_masks'].shape[0], -1),
        )
        bbox_tokens = features['refined_obj_bank']
        # Place之间的位置关系
        img_diff = refined_img_bank.unsqueeze(1) - refined_img_bank.unsqueeze(2)
        place_affinity_logits = self.place_affinity_mlp(img_diff).squeeze(-1) # [B, N_p, N_p]
        PP_matrix = torch.softmax(place_affinity_logits, dim=-1)

        # Query与Place的可见性预测
        obj_expand = object_node_feat.unsqueeze(2).expand(-1, -1, N_p, -1) # [B, N_o, N_p, D]
        img_expand = refined_img_bank.unsqueeze(1).expand(-1, N_q, -1, -1) # [B, N_o, N_p, D]
        vis_input = torch.cat(
        [
            obj_expand,
            img_expand,
            torch.abs(obj_expand - img_expand),
        ],
        dim=-1,
        )
        vis_logits = self.vis_pred_head(vis_input).squeeze(-1) # [B, N_q, N_p]
        PO_matrix = torch.sigmoid(vis_logits) # 连续的可见性概率

        # 每个3D Query根据Place投影为2D Query
        query_view_input = torch.cat(
            [
                obj_expand,
                img_expand,
                torch.abs(obj_expand - img_expand),
            ],
            dim=-1,
        )
        proj_2d_queries = self.query_view_proj(query_view_input) # [B, N_o, D]
        proj_2d_queries = torch.einsum('bpq,boqd->bopd',PP_matrix,proj_2d_queries)
        match_logits = torch.einsum('bopd,bpmd->bopm',proj_2d_queries,bbox_tokens)/ (D ** 0.5)

        bboxes_masks = features['bboxes_masks']
        if bboxes_masks is not None:
            # bboxes_masks: [B, N_p, M], 1 valid / 0 invalid
            invalid_bbox_mask = (bboxes_masks == 0).unsqueeze(1)
            # [B, 1, N_p, M]
            match_logits = match_logits.masked_fill(
                invalid_bbox_mask,
                float('-1000')
            )
        visibility_log_gate = torch.log(PO_matrix.clamp(min=1e-6)).unsqueeze(-1)
        gated_match_logits = match_logits + visibility_log_gate
        prob_bbox_given_obj = torch.softmax(gated_match_logits, dim=-1)
        prob_obj_given_bbox = torch.softmax(gated_match_logits, dim=1)

        results = {
        'object_node_feat': object_node_feat,
        'object_attn': object_attn,
        'object_exist_logits': object_exist_logits,
        'object_cls_logits': object_cls_logits,

        'PP_matrix': PP_matrix,
        'place_affinity_logits': place_affinity_logits,

        'PO_matrix': PO_matrix,
        'vis_logits': vis_logits,

        'proj_2d_queries': proj_2d_queries,

        'match_logits': match_logits,
        'gated_match_logits': gated_match_logits,
        'prob_bbox_given_obj': prob_bbox_given_obj,
        'prob_obj_given_bbox': prob_obj_given_bbox,

        'bbox_tokens': bbox_tokens,
        'bbox_masks': bboxes_masks,
        }
        return results
        # 损失计算
        # 首先将3D Query与3D物体进行匹配得到对应结果

        # 获取3D Query在视角P下投影特征，同时获取与之对应的3D物体在视角P下的二维bbox特征
        # 2D Query与BBOX特征的相似性损失

        # 帧内唯一性损失：每一个3D Query在每一帧内只能对应一个bbox

        # 可视性损失
        # 如果3D Query对于Place不可视，则3D Query在该Place的投影应与bbox特征不相似
        # 如果3D Query对于Place可视，则3D Query在该Place的投影应与bbox特征相似

        # PO矩阵BCE损失
        # PP矩阵BCE损失

        # 同一3D物体在不同视角下的bbox特征应相似


        # # 2. Query 解码 (QueryDecoderLayer 自动使用 V3 + learnable queries)
        # decoder_outputs = self.query_decoder(
        #     features['refined_obj_bank'], 
        #     features['refined_img_bank'],
        #     bboxes_masks=features['bboxes_masks'],
        #     stage='step3'
        # )
        
        # # 3. Edge Head 预测
        # edge_outputs = self.edge_heads(
        #     decoder_outputs['place_node_feat'],
        #     decoder_outputs['object_node_feat']
        # )

        # # 4. 构建统一兼容输出
        # results = self.build_legacy_compatible_outputs(
        #     object_node_feat=decoder_outputs['object_node_feat'],
        #     place_node_feat=decoder_outputs['place_node_feat'],
        #     bboxes_infos=bboxes_infos,
        #     decoder_outputs=decoder_outputs,
        #     edge_outputs=edge_outputs,
        #     bbox_feats_flat=features['bbox_feats_flat'],
        # )

        

    def forward_step4(self, images, bboxes_infos, bbox_masks=None):
        """
        Step 4: Learnable Queries + QueryRefiner + AA + Decoder + 显式场景图预测
        在 Step 3 基础上增加 SceneGraphHead
        """
        # 1. 公共特征提取
        features = self._extract_common_features(images, bboxes_infos)

        # 2. Query 解码
        decoder_outputs = self.query_decoder(
            features['refined_obj_bank'],
            features['refined_img_bank'],
            bboxes_masks=features['bboxes_masks'],
            stage='step4'
        )

        # 3. Edge Head 预测
        edge_outputs = self.edge_heads(
            decoder_outputs['place_node_feat'],
            decoder_outputs['object_node_feat']
        )

        # 4. 显式场景图预测 (Step 4 新增)
        object_cls = torch.softmax(decoder_outputs['object_cls_logits'], dim=-1) if decoder_outputs['object_cls_logits'] is not None else None
        object_scores = torch.sigmoid(decoder_outputs['object_exist_logits']) if decoder_outputs['object_exist_logits'] is not None else None

        sgg_outputs = self.scene_graph_head(
            object_query=decoder_outputs['object_node_feat'],
            object_cls=object_cls,
            object_scores=object_scores,
            mode='sgdet',
        )

        # 5. 构建统一兼容输出
        results = self.build_legacy_compatible_outputs(
            object_node_feat=decoder_outputs['object_node_feat'],
            place_node_feat=decoder_outputs['place_node_feat'],
            bboxes_infos=bboxes_infos,
            decoder_outputs=decoder_outputs,
            edge_outputs=edge_outputs,
            scene_graph_outputs=sgg_outputs,
            bbox_feats_flat=features['bbox_feats_flat'],
        )
        
        return results

    def forward_amosg(self, images, bboxes_infos, bbox_masks=None):
        """
        Baseline: AOMSG 原始流程 (Step 1)
        
        适配当前框架:
        - 使用 _extract_common_features 统一特征提取
        - 使用 build_legacy_compatible_outputs 统一输出格式
        - 保持与 step2/3/4 一致的接口
        
        Args:
            images: [B, V, 3, H, W] 多视角图像   B个场景，每个场景有V个视角
            bboxes_infos: bbox 信息字典
            bbox_masks: [B, V, N] 标记有效的bbox (可选)
            
        Returns:
            outputs: 包含所有输出的字典 (兼容 baseline 格式)
        """
        # 1. 公共特征提取 (统一入口)
        features = self._extract_common_features(images, bboxes_infos)
        
        # 2. AOMSG baseline 直接返回 association_model 结果
        # 注意: _extract_common_features 在 aomsg 模式下返回 extra_outputs 包含 embeddings
        extra_outputs = features['extra_outputs']
        
        # 3. 构建统一兼容输出
        # AOMSG 输出: embeddings [B, Q, C], place_embeddings [B, C]
        obj_embeddings = extra_outputs.get('embeddings')
        place_embeddings = extra_outputs.get('place_embeddings')
        
        if obj_embeddings is None or place_embeddings is None:
            raise RuntimeError(
                "AOMSG association_model did not return 'embeddings' or 'place_embeddings'. "
                "Please check the association_model output format."
            )
        
        # 使用统一的输出适配函数构建结果
        # 对于 step1，object_node_feat = embeddings, place_node_feat = place_embeddings.unsqueeze(1)
        results = self.build_legacy_compatible_outputs(
            object_node_feat=obj_embeddings,
            place_node_feat=place_embeddings.unsqueeze(1) if place_embeddings.dim() == 2 else place_embeddings,
            bboxes_infos=bboxes_infos,
            bbox_feats_flat=features['bbox_feats_flat'],
        )
        
        # 保留 AOMSG 原始输出 (兼容旧接口)
        for k, v in extra_outputs.items():
            if k not in results:
                results[k] = v
        
        return results

    def _compute_loss_core(self, outputs, targets, loss_weights=None):
        """
        核心 loss 计算逻辑，被 compute_loss_amosg / compute_loss_step2 / compute_loss_step3 复用
        """
        if loss_weights is None:
            loss_weights = {
                'pr': 1.0,
                'obj': 1.0,
                'mean': 1.0,
                'tcr': 1.0,
            }
        
        logs = {}
        device = outputs['embeddings'].device

        results = {
            'embeddings': outputs['embeddings'],
            'place_embeddings': outputs['place_embeddings'],
            'place_predictions': outputs.get('place_predictions', None),
            'object_predictions': outputs.get('object_predictions', None),
            'detections': outputs['detections'],
        }
        
        additional_info = {
            'gt_bbox': targets['gt_bbox'],
            'mask': targets['mask'],
            'obj_label': targets['obj_label'],
            'obj_idx': targets['obj_idx'],
            'scene_num': targets.get('scene_num', 1),
        }

        # 使用 AOMSG matcher 进行匹配
        match_inds = self.aomsg_matcher(results['detections'], results['detections'])
        
        num_emb = results['embeddings'].size(1)
        reorderd_idx = get_match_idx(match_inds, additional_info, num_emb)
        place_labels = targets['place_labels']
        
        # 兼容 SepAssociator (返回4个值) 和 DecoderAssociator (返回5个值)
        sim_loss_result = self.object_similarity_loss(results['embeddings'], reorderd_idx)
        sim_loss = sim_loss_result[0]
        mean_dis = sim_loss_result[1]
        tcr = sim_loss_result[2]
        # id_counts = sim_loss_result[3]  # 可选
        # embeddings_mean = sim_loss_result[4] if len(sim_loss_result) > 4 else None
        logs['tcr'] = tcr.item()
        logs['obj_sim_loss'] = sim_loss.item()
        logs['mean_dis'] = mean_dis.item()
        
        object_loss = self.object_association_loss(results['object_predictions'], reorderd_idx)
        logs['running_loss_obj'] = object_loss.item()
        
        place_loss = self.place_recognition_loss(results['place_predictions'], place_labels)
        
        total_loss = object_loss + loss_weights['pr'] * place_loss
        logs['running_loss_pr'] = place_loss.item()
        
        return total_loss, logs

    def compute_loss_amosg(self, outputs, targets, loss_weights=None):
        """
        Baseline (Step 1) 损失计算方法
        """
        device = outputs['embeddings'].device
        if self.feature_refine_method != 'aomsg':
            return torch.tensor(0.0, device=device), {}
        return self._compute_loss_core(outputs, targets, loss_weights)

    def compute_loss_step2(self, outputs, targets, loss_weights=None):
        """
        Step 2 损失计算方法 (与 baseline 相同的 loss，但支持 vggt feature_refine_method)
        """
        return self._compute_loss_core(outputs, targets, loss_weights)

    def compute_loss_step3(self, outputs, targets, loss_weights=None):
        """
        Step 3 损失计算方法: Object-level Query Loss + baseline loss
        
        核心变化:
        1. 使用 QueryObjectLoss 进行 object-level 建模
        2. Matching: Query(K) ↔ GT Object(N)
        3. Loss: L_cls + L_attn + L_exist (+ optional L_bbox)
        """
        # 1. Object-level Query Loss (Step 3 核心)
        if 'object_cls_logits' in outputs and 'object_attn' in outputs:
            # 构造 object-level loss 需要的 targets
            # targets 需要包含: gt_bbox, obj_idx, obj_label, mask
            obj_loss_inputs = {
                'object_node_feat': outputs.get('object_node_feat'),
                'object_attn': outputs['object_attn'],
                'object_cls_logits': outputs['object_cls_logits'],
                'object_exist_logits': outputs.get('object_exist_logits'),
            }
            
            # 从原始 targets 中提取 bbox 信息
            loss_targets = {
                'gt_bbox': targets.get('gt_bbox'),
                'obj_idx': targets.get('obj_idx'),
                'obj_label': targets.get('obj_label'),
                'mask': targets.get('mask'),
            }
            
            # 如果 targets 中没有直接的 bbox 信息，尝试从 detections 获取
            if loss_targets['gt_bbox'] is None and 'detections' in outputs:
                detections = outputs['detections']
                if isinstance(detections, dict):
                    loss_targets['gt_bbox'] = detections.get('gt_bbox')
                    loss_targets['obj_idx'] = detections.get('obj_idx')
                    loss_targets['obj_label'] = detections.get('obj_label')
                    loss_targets['mask'] = detections.get('mask')
            
            # 检查是否有足够的 target 信息
            if loss_targets['obj_idx'] is not None:
                try:
                    query_total_loss, query_loss_dict = self.query_object_loss(obj_loss_inputs, loss_targets)
                except Exception as e:
                    # 如果 object-level loss 失败，回退到 baseline
                    print(f"Object-level loss failed: {e}, falling back to baseline loss")
                    return self._compute_loss_core(outputs, targets, loss_weights)
            else:
                query_total_loss = torch.tensor(0.0, device=outputs['embeddings'].device)
                query_loss_dict = {}
        else:
            query_total_loss = torch.tensor(0.0, device=outputs['embeddings'].device)
            query_loss_dict = {}
        
        # 2. Baseline loss (保持与 Step 1/2 兼容)
        baseline_total_loss, baseline_logs = self._compute_loss_core(outputs, targets, loss_weights)
        
        # 3. 合并 loss
        total_loss = baseline_total_loss + query_total_loss
        
        # 4. 合并 logs
        logs = baseline_logs.copy()
        for k, v in query_loss_dict.items():
            logs[k] = v.item() if isinstance(v, torch.Tensor) else v
        
        return total_loss, logs

    def build_gt_relation_data(self, targets, num_obj, device):
        """
        从 targets 中构建场景图 GT 数据

        Args:
            targets: 包含 rel_labels 的 targets 字典
            num_obj: object query 数量
            device: torch device

        Returns:
            gt_rels_formatted: [B, max_rels, 3] 或 [total_rels, 3] 格式化后的关系
            gt_importance: [B, num_obj, num_obj] importance matrix
            gt_sub_labels: [B, max_rels] subject 标签
            gt_obj_labels: [B, max_rels] object 标签
            valid_mask: [B, max_rels] 有效关系掩码
        """
        gt_rels = targets.get('rel_labels')
        if gt_rels is None:
            return None, None, None, None, None

        B = gt_rels.shape[0] if gt_rels.dim() >= 2 else 1

        # 统一处理 gt_rels 维度
        if gt_rels.dim() == 2:
            # [total_rels, 3] -> 需要按 batch 分组
            # 假设通过 scene_num 或 batch_idx 来区分
            gt_rels = gt_rels.unsqueeze(0).expand(B, -1, -1)

        # gt_rels: [B, max_rels, 3] (sub_idx, obj_idx, rel_cls)
        max_rels = gt_rels.shape[1]

        # 构建 importance matrix 和 valid mask
        gt_importance = torch.zeros((B, num_obj, num_obj), device=device)
        valid_mask = torch.zeros((B, max_rels), dtype=torch.bool, device=device)
        gt_sub_labels = torch.full((B, max_rels), -1, dtype=torch.long, device=device)
        gt_obj_labels = torch.full((B, max_rels), -1, dtype=torch.long, device=device)

        for b in range(B):
            for r in range(max_rels):
                rel = gt_rels[b, r]
                s, o, rel_cls = rel[0].item(), rel[1].item(), rel[2].item()

                # 严格校验: 有效关系需要 rel_cls > 0 且索引在范围内
                if rel_cls > 0 and 0 <= s < num_obj and 0 <= o < num_obj:
                    gt_importance[b, s, o] = 1.0
                    valid_mask[b, r] = True
                    gt_sub_labels[b, r] = s
                    gt_obj_labels[b, r] = o

        # 过滤无效关系，重新格式化 gt_rels
        # 只保留有效关系到新的张量中
        valid_rels_list = []
        for b in range(B):
            batch_valid = gt_rels[b][valid_mask[b]]  # [num_valid, 3]
            valid_rels_list.append(batch_valid)

        # 如果所有 batch 都有相同数量的有效关系，可以堆叠
        # 否则返回列表形式
        if valid_rels_list:
            max_valid = max(vr.shape[0] for vr in valid_rels_list)
            gt_rels_formatted = torch.full((B, max_valid, 3), -1, dtype=torch.long, device=device)
            for b in range(B):
                num_v = valid_rels_list[b].shape[0]
                if num_v > 0:
                    gt_rels_formatted[b, :num_v] = valid_rels_list[b]
        else:
            gt_rels_formatted = torch.full((B, 1, 3), -1, dtype=torch.long, device=device)

        return gt_rels_formatted, gt_importance, gt_sub_labels, gt_obj_labels, valid_mask

    def compute_loss_step4(self, outputs, targets, loss_weights=None):
        """
        Step 4 损失计算方法: baseline loss + 场景图关系 loss

        改进:
        - 使用 build_gt_relation_data() 严格构造 GT
        - 支持 rel_labels 的不同维度格式
        - 严格过滤无效关系 (padding, 越界)
        """
        # 1. 基础 loss (与 Step 3 相同)
        total_loss, logs = self._compute_loss_core(outputs, targets, loss_weights)

        # 2. 场景图关系 loss (Step 4 新增)
        if 'sgg_outputs' in outputs and targets.get('rel_labels') is not None:
            sgg_outputs = outputs['sgg_outputs']

            # 准备维度信息
            B = outputs['embeddings'].shape[0]
            num_obj = outputs['embeddings'].shape[1]
            device = outputs['embeddings'].device

            # 使用统一的 GT 构造函数
            gt_rels_fmt, gt_importance, gt_sub_labels, gt_obj_labels, valid_mask = \
                self.build_gt_relation_data(targets, num_obj, device)

            if gt_rels_fmt is not None and valid_mask.any():
                # 计算场景图 loss
                sgg_loss_dict = self.scene_graph_head.compute_loss(
                    outputs=sgg_outputs,
                    gt_rels=gt_rels_fmt,
                    gt_importance=gt_importance,
                    gt_sub_labels=gt_sub_labels,
                    gt_obj_labels=gt_obj_labels,
                )

                # 合并 loss
                sgg_total_loss = sum(sgg_loss_dict.values())
                total_loss = total_loss + sgg_total_loss

                # 记录场景图 loss
                for k, v in sgg_loss_dict.items():
                    logs[f'sgg_{k}'] = v.item() if isinstance(v, torch.Tensor) else v

                # 记录有效关系数量
                logs['sgg_num_valid_rels'] = valid_mask.sum().item()
            else:
                logs['sgg_num_valid_rels'] = 0

        return total_loss, logs

    def _get_loss_module(self):
        """获取可用的 loss 计算模块"""
        # 优先使用 feature_extractor 中的 association_model (aomsg 模式)
        if hasattr(self, 'feature_extractor') and hasattr(self.feature_extractor, 'association_model'):
            if self.feature_extractor.association_model is not None:
                return self.feature_extractor.association_model
        # 其次使用 step_loss_module (vggt 模式)
        if hasattr(self, 'step_loss_module') and self.step_loss_module is not None:
            return self.step_loss_module
        raise RuntimeError(
            "No loss module available. "
            "For aomsg mode, feature_extractor.association_model should be initialized. "
            "For vggt mode, step_loss_module should be initialized."
        )

    def object_similarity_loss(self, embeddings, matched_idx):
        """计算物体相似度损失"""
        loss_module = self._get_loss_module()
        return loss_module.object_similarity_loss(embeddings, matched_idx)

    def object_association_loss(self, object_predictions, reorderd_idx):
        """计算物体关联损失"""
        loss_module = self._get_loss_module()
        return loss_module.object_association_loss(object_predictions, reorderd_idx)

    def place_recognition_loss(self, place_predictions, place_labels):
        """计算场景识别损失"""
        loss_module = self._get_loss_module()
        return loss_module.place_recognition_loss(place_predictions, place_labels)

    def preprocess(self, images, bboxes_infos):
        # ========== 1. 基础参数提取与合法性校验 ==========
        # 场景总数B
        B = bboxes_infos['scene_num']
        # 总帧数（图像第一维）
        total_frames = images.shape[0]
        # 单场景包含的帧数V
        V = total_frames // B

        # 强校验：确保总帧数可被场景数整除，避免维度错位
        assert B * V == total_frames, \
            f"维度不匹配：总帧数{total_frames} 无法被场景数{B}整除，Images shape {images.shape} 与 B*V={B*V} 不一致"

        # ========== 2. 图像张量 逐场景维度转换 ==========
        # 原shape: [B*V, 3, 224, 224] → 新shape: [B, V, 3, 224, 224]
        # 维度含义：第0维=场景ID，第1维=该场景内的帧序号，后续为图像原有维度
        images_per_scene = images.reshape(B, V, *images.shape[1:])

        # ========== 3. Bbox信息 逐场景维度转换 ==========
        # 初始化逐场景的bbox信息字典，兼容所有字段
        bboxes_per_scene = {}
        for key, value in bboxes_infos.items():
            # 处理标量字段（如scene_num）：直接保留原值
            if isinstance(value, (int, float)):
                bboxes_per_scene[key] = value
                continue

            # 张量字段校验：确保第一维与总帧数对齐
            assert value.shape[0] == total_frames, \
                f"字段{key}维度异常：第一维{value.shape[0]} 与总帧数{total_frames} 不匹配"
            
            # 核心转换：原shape [B*V, *dims] → 新shape [B, V, *dims]
            bboxes_per_scene[key] = value.reshape(B, V, *value.shape[1:])

        return images_per_scene, bboxes_per_scene

    def compute_loss(self, outputs, targets, loss_params=None):
        """
        """
        # 动态权重（如果没有提供则使用默认值，正确更新而不是覆盖）
        default_loss_params = {
            'loss_obj_exist': 1.0,
            'loss_obj_cls': 2.0,
            'loss_place_exist': 1.0,
            'loss_pp_edge': 1.0,
            'loss_po_edge': 1.0,
            'loss_sgg_rel': 1.0,
            'loss_sgg_match': 1.0,
            'loss_sparse': 0.5,       # 动态稀疏损失
            'loss_box_query': 2.0,    # 动态框-Query关联损失
        }
        if loss_params is None:
            loss_params = default_loss_params
        else:
            # 更新默认参数而不是覆盖
            for k, v in default_loss_params.items():
                if k not in loss_params:
                    loss_params[k] = v

        loss_dict = {}
        device = outputs['object_exist_logits'].device

        # 动态获取核心维度（无硬编码）
        B, Q = outputs['object_exist_logits'].shape[:2]  # 批次和Query数量

        # 调整 targets 形状：从 [V*N, ...] 转换为 [B, V*N, ...]
        if 'scene_num' in targets and targets['scene_num'] == B:
            VxN = targets['vid_idx'].shape[0] // B if 'vid_idx' in targets else 1
            targets = self.reshape_targets(targets, B, VxN)

        # 计算每个批次的真实物体数量
        batch_num_gt_obj = [
            len(torch.unique(targets['obj_idx'][bi][targets['obj_idx'][bi] != -1]))
            for bi in range(B)
        ]
        
        num_gt_obj = max(batch_num_gt_obj) if batch_num_gt_obj else 0
        
        # 检查是否有框特征和物体索引
        has_box_feat = 'bbox_feats' in outputs and 'obj_idx' in targets

        # 匈牙利匹配
        match_indices = self.matcher(
            outputs['object_attn'], 
            outputs['object_exist_logits'], 
            targets,
            valid_key='mask',
            bbox_key='gt_bbox',
            obj_idx_key='obj_idx'
        )

        # 获取匹配后的标签
        exist_targets, cls_targets = get_match_targets(
            match_indices, targets, self.num_obj_queries, labels_key='obj_label'
        )
        exist_targets = exist_targets.to(device).float()
        cls_targets = cls_targets.to(device)

        # 物体存在性损失
        obj_exist_logits = outputs['object_exist_logits']
        loss_dict['loss_obj_exist'] = F.binary_cross_entropy_with_logits(
            obj_exist_logits, exist_targets
        )

        # # 物体分类损失（之前缺失！）
        # obj_cls_logits = outputs['object_cls_logits']  # [B, Q, C]
        # valid_mask = exist_targets.bool()  # [B, Q]
        # if valid_mask.sum() > 0:
        #     valid_cls_logits = obj_cls_logits[valid_mask]
        #     valid_cls_targets = cls_targets[valid_mask]
        #     # 确保目标标签有效
        #     valid_cls_mask = valid_cls_targets >= 0
        #     if valid_cls_mask.any():
        #         loss_dict['loss_obj_cls'] = F.cross_entropy(
        #             valid_cls_logits[valid_cls_mask], 
        #             valid_cls_targets[valid_cls_mask]
        #         )
        #     else:
        #         loss_dict['loss_obj_cls'] = torch.tensor(0.0, device=device)
        # else:
        #     loss_dict['loss_obj_cls'] = torch.tensor(0.0, device=device)

        # 场所存在性损失
        place_exist_logits = outputs['place_exist_logits']
        place_exist_targets = torch.ones_like(place_exist_logits).to(device)
        loss_dict['loss_place_exist'] = F.binary_cross_entropy_with_logits(
            place_exist_logits, place_exist_targets
        )

        # 边预测损失
        pp_logits = outputs['pp_logits']
        po_logits = outputs['po_logits']

        gt_pp_rels = targets.get('gt_pp_rels', torch.zeros(pp_logits.shape[:3], dtype=torch.long, device=device))
        loss_dict['loss_pp_edge'] = F.cross_entropy(
            pp_logits.permute(0, 3, 1, 2),
            gt_pp_rels
        )

        gt_po_rels = targets.get('gt_po_rels', torch.zeros(po_logits.shape[:3], dtype=torch.long, device=device))
        loss_dict['loss_po_edge'] = F.cross_entropy(
            po_logits.permute(0, 3, 1, 2),
            gt_po_rels
        )

        # 动态稀疏损失
        loss_dict['loss_sparse'] = self.compute_sparse_loss(batch_num_gt_obj, B, obj_exist_logits, device)

        # 动态框-Query关联损失
        if has_box_feat:
            loss_dict['loss_box_query'] = self.compute_box_query_loss(
                outputs, targets, batch_num_gt_obj, B, device, exist_targets
            )
        else:
            loss_dict['loss_box_query'] = torch.tensor(0.0, device=device)

        # 场景图损失
        if self.use_scene_graph and 'gt_rels' in targets:
            sgg_loss_dict = self.compute_scene_graph_loss(outputs, targets, device)
            for k, v in sgg_loss_dict.items():
                if k in loss_params:
                    loss_dict[k] = v

        # 总损失（动态加权所有损失）
        total_loss = sum(loss_dict[k] * loss_params[k] for k in loss_dict if k in loss_params)
        loss_dict['total_loss'] = total_loss
        
        return total_loss, loss_dict

    def reshape_targets(self, targets, B, VxN):
        for key in ['gt_bbox', 'obj_label', 'obj_idx', 'mask']:
            if targets[key].dim() == 2:
                targets[key] = targets[key].reshape(B, VxN, -1)
            elif targets[key].dim() == 3:
                targets[key] = targets[key].reshape(B, VxN, *targets[key].shape[1:])
        return targets

    def compute_sparse_loss(self, batch_num_gt_obj, B, obj_exist_logits, device):
        exist_prob = torch.sigmoid(obj_exist_logits)
        batch_sparse_losses = [
            1.0 - torch.topk(exist_prob[bi:bi+1], k=batch_num_gt_obj[bi], dim=1)[0].mean()
            if batch_num_gt_obj[bi] > 0 else torch.tensor(0.0, device=device)
            for bi in range(B)
        ]
        return torch.stack(batch_sparse_losses).mean() if batch_sparse_losses else torch.tensor(0.0, device=device)

    def compute_box_query_loss(self, outputs, targets, batch_num_gt_obj, B, device, exist_targets):
        bbox_feats = outputs['bbox_feats']
        query_feats = outputs['object_node_feat']
        
        gt_box_obj = targets.get('gt_box_obj_id', targets['obj_idx'])

        if gt_box_obj.dim() == 3:
            gt_box_obj = gt_box_obj.reshape(B, -1)

        if gt_box_obj.shape[1] != bbox_feats.shape[1]:
            return torch.tensor(0.0, device=device)
        
        batch_box_query_losses = []
        for bi in range(B):
            if batch_num_gt_obj[bi] == 0:
                continue
                
            batch_bbox_feats = bbox_feats[bi:bi+1]
            batch_gt_box_obj = gt_box_obj[bi:bi+1]
            batch_query_feats = query_feats[bi:bi+1]
            
            # 使用 exist_targets 作为有效的查询掩码
            batch_valid_mask = exist_targets[bi:bi+1].bool()
            batch_valid_query_feats = batch_query_feats[batch_valid_mask]
            
            if batch_valid_query_feats.numel() > 0 and batch_num_gt_obj[bi] > 0:
                # 取前 batch_num_gt_obj[bi] 个有效的查询特征
                num_to_use = min(len(batch_valid_query_feats), batch_num_gt_obj[bi])
                batch_valid_query_feats = batch_valid_query_feats[:num_to_use]
                batch_valid_query_feats = batch_valid_query_feats.reshape(1, num_to_use, -1)
                
                # 计算相似度
                sim = F.cosine_similarity(batch_bbox_feats.unsqueeze(2), batch_valid_query_feats.unsqueeze(1), dim=-1)
                
                # 处理 gt_box_obj - 只使用有效的索引
                valid_box_mask = (batch_gt_box_obj >= 0) & (batch_gt_box_obj < num_to_use)
                if valid_box_mask.any():
                    # 创建 gt_assign
                    gt_assign = torch.zeros_like(sim)
                    valid_box_idx = valid_box_mask.nonzero(as_tuple=True)
                    gt_assign[valid_box_idx[0], valid_box_idx[1], batch_gt_box_obj[valid_box_idx]] = 1.0
                    
                    batch_box_query_losses.append(F.binary_cross_entropy_with_logits(sim, gt_assign))
        
        return torch.stack(batch_box_query_losses).mean() if batch_box_query_losses else torch.tensor(0.0, device=device)

    def compute_scene_graph_loss(self, outputs, targets, device):
        sgg_outputs = {
            'importance': outputs.get('sgg_importance'),
            'sub_pos': outputs.get('sgg_sub_pos'),
            'obj_pos': outputs.get('sgg_obj_pos'),
            'rel_pred': outputs.get('sgg_rel_pred'),
        }

        gt_rels = targets['gt_rels']
        gt_importance = None
        if 'obj_labels' in targets and 'bboxes' in targets:
            num_obj = outputs['object_node_feat'].size(1)
            gt_importance = prepare_gt_importance_matrix(
                gt_rels.reshape(-1, 3), num_obj, device
            ).unsqueeze(0).expand(gt_rels.size(0), -1, -1)

        gt_sub_labels = gt_rels[:, :, 0] if gt_rels.size(1) > 0 else None
        gt_obj_labels = gt_rels[:, :, 1] if gt_rels.size(1) > 0 else None

        return self.scene_graph_head.compute_loss(
            outputs=sgg_outputs,
            gt_rels=gt_rels,
            gt_importance=gt_importance,
            gt_sub_labels=gt_sub_labels,
            gt_obj_labels=gt_obj_labels,
        )

    def loss_step3(self, outputs, bboxes_infos, loss_weights=None):
        # if loss_weights is None:
        loss_weights = {
            'pp': 1.0,
            'po': 1.0,
            'assign': 2.0,
            'exist': 1.0,
            'unique': 0.2,
        }

        

        PP_matrix = outputs['PP_matrix']
        place_affinity_logits = outputs['place_affinity_logits']

        PO_matrix = outputs['PO_matrix']
        vis_logits = outputs['vis_logits']

        gated_match_logits = outputs['gated_match_logits']
        prob_bbox_given_obj = outputs['prob_bbox_given_obj']

        object_exist_logits = outputs['object_exist_logits']

        place_labels = bboxes_infos.get('place_labels', None)

        B, N_o, N_p, M = gated_match_logits.shape
        device = gated_match_logits.device


        obj_idx = bboxes_infos['obj_idx'].reshape(B, N_p, -1)
        mask = bboxes_infos['mask'].reshape(B, N_p, -1)

        gt_vis, gt_assign, all_obj_ids = build_instance_targets(obj_idx, mask)
        GT_PO_matrix = gt_vis
        GT_PP_matrix = bboxes_infos['place_labels']
        GT_PP_matrix = torch.stack([GT_PP_matrix[i*N_p:(i+1)*N_p, i*N_p:(i+1)*N_p] for i in range(GT_PP_matrix.shape[0]//N_p)])
        
        total_po_loss = 0.0
        total_assign_loss = 0.0
        total_exist_loss = 0.0
        valid_scene_count = 0

        matched_indices = []
        po_iou_list = []
        pp_iou_list = []
        po_iou_loss_list = []
        pp_iou_loss_list = []
        for b in range(B):
            G = len(all_obj_ids[b])
            batch_gt_PO_matrix = GT_PO_matrix[b, :G]
            batch_gt_PP_matrix = GT_PP_matrix[b]

            batch_pred_PP_matrix = PP_matrix[b]
            batch_pred_PO_matrix = PO_matrix[b]
            if G == 0:
                exist_target = torch.zeros(N_o, device=device)
                total_exist_loss = total_exist_loss + F.binary_cross_entropy_with_logits(
                    object_exist_logits[b],
                    exist_target,
                    reduction='mean'
                )
                matched_indices.append(([], []))
                continue

            cost = compute_query_gt_cost(
                vis_logits[b], #query在视角下可见的probability
                gated_match_logits[b], #query在视角下与bbox匹配的probability
                gt_vis[b, :G],
                gt_assign[b, :G],
            )

            row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())

            row_ind = torch.as_tensor(row_ind, dtype=torch.long, device=device)
            col_ind = torch.as_tensor(col_ind, dtype=torch.long, device=device)

            matched_indices.append((row_ind, col_ind))

            # 损失计算
            pred_po = batch_pred_PO_matrix[row_ind]
            pred_pp = batch_pred_PP_matrix
            gt_po = batch_gt_PO_matrix[col_ind]
            gt_pp = batch_gt_PP_matrix
            # 1. 计算 PO 指标
            po_iou = calculate_binary_iou(pred_po, gt_po)
            po_iou_list.append(po_iou)

            # 2. 计算 PP 指标
            pp_iou = calculate_pp_iou(pred_pp, gt_pp)
            pp_iou_list.append(pp_iou)

            # 3. 计算损失
            po_iou_loss = soft_iou_loss(pred_po, gt_po)
            pp_iou_loss = soft_iou_loss(pred_pp, gt_pp)

            po_iou_loss_list.append(po_iou_loss)
            pp_iou_loss_list.append(pp_iou_loss)


            exist_target = torch.zeros(N_o, device=device)
            exist_target[row_ind] = 1.0

            total_exist_loss = total_exist_loss + F.binary_cross_entropy_with_logits(
                object_exist_logits[b],
                exist_target,
                reduction='mean'
            )

            total_po_loss = total_po_loss + F.binary_cross_entropy_with_logits(
                vis_logits[b, row_ind],
                gt_vis[b, col_ind],
                reduction='mean'
            )

            assign_loss = 0.0
            assign_cnt = 0

            for q, g in zip(row_ind, col_ind):
                for p in range(N_p):
                    target_box = gt_assign[b, g, p]

                    if target_box >= 0:
                        assign_loss = assign_loss + F.cross_entropy(
                            gated_match_logits[b, q, p].unsqueeze(0),
                            target_box.unsqueeze(0),
                            reduction='mean'
                        )
                        assign_cnt += 1

            if assign_cnt > 0:
                assign_loss = assign_loss / assign_cnt
                total_assign_loss = total_assign_loss + assign_loss

            valid_scene_count += 1

        denom = max(valid_scene_count, 1)

        loss_po = total_po_loss / denom
        loss_assign = total_assign_loss / denom
        loss_exist = total_exist_loss / B

        if place_labels is not None:
            loss_pp = F.binary_cross_entropy_with_logits(
                torch.block_diag(*place_affinity_logits),
                place_labels.float(),
                reduction='mean'
            )
        else:
            loss_pp = torch.tensor(0.0, device=device)

        valid_bbox_mask = mask.unsqueeze(1).float()
        selected_sum = (prob_bbox_given_obj * valid_bbox_mask).sum(dim=1)
        loss_unique = F.relu(selected_sum - 1.0).pow(2).mean()

        loss_pp_iou = torch.stack(pp_iou_loss_list).mean()
        loss_po_iou = torch.stack(po_iou_loss_list).mean()

        total_loss = loss_pp_iou + loss_po_iou

        # total_loss = (
        #     loss_weights['pp'] * loss_pp
        #     + loss_weights['po'] * loss_po
        #     + loss_weights['assign'] * loss_assign
        #     + loss_weights['exist'] * loss_exist
        #     + loss_weights['unique'] * loss_unique
        # )

        loss_dict = {
            'loss_total': total_loss,
            'loss_pp': loss_pp,
            'loss_po': loss_po,
            'loss_assign': loss_assign,
            'loss_exist': loss_exist,
            'loss_unique': loss_unique,
            'loss_pp_iou': loss_pp_iou,
            'loss_po_iou': loss_po_iou,
            'pp_iou': torch.stack(pp_iou_list).mean(),
            'po_iou': torch.stack(po_iou_list).mean(),
            # 'matched_indices': matched_indices,
        }

        return total_loss, loss_dict

def calculate_binary_iou(pred, gt, threshold=0.5):
    """
    计算二值化 IoU（指标专用，会做 >0.5 二值化）
    :param pred: 模型预测概率 [0~1]
    :param gt: 真值矩阵
    :return: iou 值
    """
    pred_bin = pred > threshold
    intersection = torch.sum(torch.logical_and(gt, pred_bin))
    union = torch.sum(torch.logical_or(gt, pred_bin))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou

def calculate_pp_iou(pred_pp, gt_pp, threshold=0.5):
    """
    专门计算 PP 矩阵的 IoU（处理对角线）
    """
    # 二值化
    pred_pp_bin = pred_pp > threshold
    num_diag = gt_pp.shape[0]

    # 填充对角线为 1
    gt_pp_diag = gt_pp.clone()
    gt_pp_diag.fill_diagonal_(1)

    pred_pp_diag = pred_pp_bin.clone()
    pred_pp_diag.fill_diagonal_(1)

    # 计算并减去对角线
    intersection = torch.sum(torch.logical_and(gt_pp_diag, pred_pp_diag)) - num_diag
    union = torch.sum(torch.logical_or(gt_pp_diag, pred_pp_diag)) - num_diag
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou

def soft_iou_loss(pred, gt):
    """
    软 IoU 损失（训练专用，使用原始概率，可导）
    """
    intersection = torch.sum(gt * pred + 1e-6)
    union = torch.sum(gt + pred - gt * pred + 1e-6)
    loss = 1 - intersection / union
    return loss


import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def build_instance_targets(obj_idx, mask, invalid_ids=(-1,)):
    B, N_p, M = obj_idx.shape
    device = obj_idx.device

    all_vis_targets = []
    all_assign_targets = []
    all_obj_ids = []

    max_gt = 0

    for b in range(B):
        valid_obj_idx = obj_idx[b][mask[b] > 0]

        for invalid_id in invalid_ids:
            valid_obj_idx = valid_obj_idx[valid_obj_idx != invalid_id]

        obj_ids = torch.unique(valid_obj_idx)
        obj_ids = obj_ids.sort()[0]

        max_gt = max(max_gt, obj_ids.numel())
        all_obj_ids.append(obj_ids)

    vis_targets = torch.zeros(B, max_gt, N_p, device=device)
    assign_targets = torch.full((B, max_gt, N_p), -1, dtype=torch.long, device=device)

    for b in range(B):
        obj_ids = all_obj_ids[b]

        for g, oid in enumerate(obj_ids):
            for p in range(N_p):
                hit = ((obj_idx[b, p] == oid) & (mask[b, p] > 0)).nonzero(as_tuple=False)

                if hit.numel() > 0:
                    box_id = hit[0, 0]
                    vis_targets[b, g, p] = 1.0   #在场景b中，物体g在视角p下可见
                    assign_targets[b, g, p] = box_id  #在场景b中，物体g在视角p下可见的框ID

    return vis_targets, assign_targets, all_obj_ids


def compute_query_gt_cost(vis_logits, match_logits, vis_targets, assign_targets):
    N_o = vis_logits.shape[0]
    G = vis_targets.shape[0]
    N_p = vis_logits.shape[1]

    cost = torch.zeros(N_o, G, device=vis_logits.device)

    for q in range(N_o):
        for g in range(G):
            vis_cost = F.binary_cross_entropy_with_logits(
                vis_logits[q],
                vis_targets[g],
                reduction='mean'
            )

            assign_cost = 0.0
            cnt = 0

            for p in range(N_p):
                target_box = assign_targets[g, p]

                if target_box >= 0:
                    assign_cost = assign_cost + F.cross_entropy(
                        match_logits[q, p].unsqueeze(0),
                        target_box.unsqueeze(0),
                        reduction='mean'
                    )
                    cnt += 1

            if cnt > 0:
                assign_cost = assign_cost / cnt

            cost[q, g] = vis_cost + assign_cost

    return cost




    # def compute_loss(self, outputs, targets, loss_params=None):
    #     """
    #     计算损失（完整补充版 | 动态适配：任意Query/真实物体/候选框数量）
        
    #     Args:
    #         outputs: 模型输出
    #             - bbox_feats: [B, num_boxes, C]  候选框特征（动态数量）
    #             - object_node_feat: [B, Q, C]     Query特征（动态数量）
    #         targets: GT targets，包含:
    #             - gt_obj_labels: [B, num_gt_obj]  物体真实类别（动态真实物体数）
    #             - gt_box_obj_id: [B, num_boxes]  每个候选框对应的真实物体ID
    #             - gt_pp_rels: [B, num_place, num_place] 场所-场所真实关系
    #             - gt_po_rels: [B, num_place, num_obj_queries] 场所-物体真实关系
    #             - gt_rels: [B, N, 3] 场景图三元组真实标签
    #         loss_params: 损失权重等参数
                
    #     Returns:
    #         total_loss: 总损失
    #         loss_dict: 损失字典
    #     """
    #     # ===================== 动态权重（新增稀疏/框关联损失） =====================
    #     if loss_params is None:
    #         loss_params = {
    #             'loss_obj_exist': 1.0,
    #             'loss_obj_cls': 2.0,
    #             'loss_place_exist': 1.0,
    #             'loss_pp_edge': 1.0,
    #             'loss_po_edge': 1.0,
    #             'loss_sgg_rel': 1.0,
    #             'loss_sgg_match': 1.0,
    #             'loss_sparse': 0.5,       # 动态稀疏损失
    #             'loss_box_query': 2.0,    # 动态框-Query关联损失
    #         }

    #     loss_dict = {}
    #     device = outputs['object_exist_logits'].device

    #     # ===================== 动态获取核心维度（无硬编码） =====================
    #     B = outputs['object_exist_logits'].shape[0]                # 批次
    #     Q = outputs['object_exist_logits'].shape[1]                # Query数量（动态）
        
    #     # 调整 targets 形状：从 [V*N, ...] 转换为 [B, V*N, ...]
    #     if 'scene_num' in targets and targets['scene_num'] == B:
    #         if 'vid_idx' in targets:
    #             VxN = targets['vid_idx'].shape[0] // B

    #         # 重新组织 targets
    #         for key in list(targets.keys()):
    #             if key in ['gt_bbox', 'obj_label', 'obj_idx', 'mask']:
    #                 if targets[key].dim() == 2:
    #                     # [V*N, M] -> [B, V*N, M]
    #                     targets[key] = targets[key].reshape(B, VxN, -1)
    #                 elif targets[key].dim() == 3:
    #                     # [V*N, M, 4] -> [B, V*N, M, 4]
    #                     targets[key] = targets[key].reshape(B, VxN, *targets[key].shape[1:])
    #             elif key == 'vid_idx':
    #                 # [V*N] -> [B, V*N]
    #                 targets[key] = targets[key].reshape(B, VxN)
        
    #     # 计算 num_gt_obj：每个 batch 中场景物体的数量（从 obj_idx 提取不同的 id）
    #     batch_num_gt_obj = []
    #     obj_idx = targets['obj_idx']
    #     for bi in range(B):
    #         # 取出当前 batch 的 obj_idx
    #         batch_obj_idx = obj_idx[bi]
    #         # 展平并过滤 -1
    #         valid_ids = batch_obj_idx[batch_obj_idx != -1]
    #         # 统计不同 id 的数量
    #         unique_count = len(torch.unique(valid_ids)) if valid_ids.numel() > 0 else 0
    #         batch_num_gt_obj.append(unique_count)

        
    #     # 计算全局最大物体数量，用于损失计算
    #     num_gt_obj = max(batch_num_gt_obj) if batch_num_gt_obj else 0
        
    #     # 检查是否有框特征和物体索引
    #     has_box_feat = 'bbox_feats' in outputs and ('obj_idx' in targets)

    #     # ===================== 原有匈牙利匹配 + 基础损失（完全不变） =====================
    #     # 匈牙利匹配：预测查询 ↔ 真实目标
    #     match_indices = self.matcher(
    #         outputs['object_attn'], 
    #         outputs['object_exist_logits'], 
    #         targets,
    #         valid_key='mask',
    #         bbox_key='gt_bbox',
    #         obj_idx_key='obj_idx'
    #     )
        
    #     # 获取匹配后的标签
    #     exist_targets, cls_targets = get_match_targets(
    #         match_indices, targets, self.num_obj_queries, labels_key='obj_label'
    #     )
    #     exist_targets = exist_targets.to(device).float()
    #     cls_targets = cls_targets.to(device)  # [B, Q]
        
    #     # 1. 物体存在性损失
    #     obj_exist_logits = outputs['object_exist_logits']  # [B, Q]
    #     loss_obj_exist = F.binary_cross_entropy_with_logits(
    #         obj_exist_logits, exist_targets
    #     )
    #     loss_dict['loss_obj_exist'] = loss_obj_exist
        
    #     # # 2. 物体分类损失
    #     # obj_cls_logits = outputs['object_cls_logits']  # [B, Q, C]
    #     # valid_mask = exist_targets.bool()  # [B, Q]
    #     # if valid_mask.sum() > 0:
    #     #     valid_cls_logits = obj_cls_logits[valid_mask]
    #     #     valid_cls_targets = cls_targets[valid_mask]
    #     #     loss_obj_cls = F.cross_entropy(valid_cls_logits, valid_cls_targets)
    #     # else:
    #     #     loss_obj_cls = torch.tensor(0.0, device=device)
    #     # loss_dict['loss_obj_cls'] = loss_obj_cls
        
    #     # 3. 场所存在性损失
    #     place_exist_logits = outputs['place_exist_logits']  # [B, P]
    #     place_exist_targets = torch.ones_like(place_exist_logits).to(device)
    #     loss_place_exist = F.binary_cross_entropy_with_logits(
    #         place_exist_logits, place_exist_targets
    #     )
    #     loss_dict['loss_place_exist'] = loss_place_exist
        
    #     # 4. 边预测损失
    #     pp_logits = outputs['pp_logits']  # [B, P, P, E]
    #     po_logits = outputs['po_logits']  # [B, P, Q, E]
        
    #     gt_pp_rels = targets.get('gt_pp_rels', torch.zeros(pp_logits.shape[:3], dtype=torch.long, device=device))
    #     loss_pp_edge = F.cross_entropy(
    #         pp_logits.permute(0, 3, 1, 2),
    #         gt_pp_rels
    #     )
    #     loss_dict['loss_pp_edge'] = loss_pp_edge
        
    #     gt_po_rels = targets.get('gt_po_rels', torch.zeros(po_logits.shape[:3], dtype=torch.long, device=device))
    #     loss_po_edge = F.cross_entropy(
    #         po_logits.permute(0, 3, 1, 2),
    #         gt_po_rels
    #     )
    #     loss_dict['loss_po_edge'] = loss_po_edge

    #     # ===================== 【新增1】动态稀疏损失（自动适配真实物体数量） =====================
    #     exist_prob = torch.sigmoid(obj_exist_logits)
    #     # 自动取【真实物体个数】个最高分数Query，抑制其余背景Query
    #     batch_sparse_losses = []
    #     for bi in range(B):
    #         batch_obj_count = batch_num_gt_obj[bi]
    #         if batch_obj_count > 0 and batch_obj_count <= Q:
    #             # 对每个 batch 单独计算 topk
    #             batch_exist_prob = exist_prob[bi:bi+1]
    #             topk_vals, _ = torch.topk(batch_exist_prob, k=batch_obj_count, dim=1)
    #             batch_loss = 1.0 - topk_vals.mean()
    #             batch_sparse_losses.append(batch_loss)
    #         else:
    #             batch_sparse_losses.append(torch.tensor(0.0, device=device))
        
    #     if batch_sparse_losses:
    #         loss_sparse = torch.stack(batch_sparse_losses).mean()
    #     else:
    #         loss_sparse = torch.tensor(0.0, device=device)
    #     loss_dict['loss_sparse'] = loss_sparse

    #     # ===================== 【新增2】动态框-Query关联损失（自动适配任意候选框） =====================
    #     if has_box_feat:
    #         bbox_feats = outputs['bbox_feats']               # [B, num_boxes, C]
    #         query_feats = outputs['object_node_feat']        # [B, Q, C]
            
    #         # 使用 obj_idx 作为 gt_box_obj_id
    #         if 'gt_box_obj_id' in targets:
    #             gt_box_obj = targets['gt_box_obj_id']
    #         else:
    #             gt_box_obj = targets['obj_idx']
            
    #         # 处理形状
    #         if gt_box_obj.dim() == 3:
    #             # [B, V, N] 或 [B, V*N, M] -> [B, V*N] 或 [B, V*N*M]
    #             gt_box_obj = gt_box_obj.reshape(B, -1)
    #         elif gt_box_obj.dim() == 2:
    #             # 已经是 [B, V*N]，不需要展平
    #             pass
            
    #         # 确保 gt_box_obj 与 bbox_feats 形状匹配
    #         if gt_box_obj.shape[1] != bbox_feats.shape[1]:
    #             # 如果不匹配，尝试调整
    #             if gt_box_obj.numel() == bbox_feats.shape[1]:
    #                 gt_box_obj = gt_box_obj.reshape(bbox_feats.shape[0], bbox_feats.shape[1])
    #             else:
    #                 # 形状不匹配，跳过此损失
    #                 loss_dict['loss_box_query'] = torch.tensor(0.0, device=device)
    #         else:
    #             # 对每个 batch 单独计算损失
    #             batch_box_query_losses = []
    #             for bi in range(B):
    #                 batch_obj_count = batch_num_gt_obj[bi]
    #                 if batch_obj_count > 0:
    #                     # 取出当前 batch 的数据
    #                     batch_bbox_feats = bbox_feats[bi:bi+1]  # [1, num_boxes, C]
    #                     batch_gt_box_obj = gt_box_obj[bi:bi+1]  # [1, num_boxes]
    #                     batch_valid_mask = valid_mask[bi:bi+1]  # [1, Q]
    #                     batch_query_feats = query_feats[bi:bi+1]  # [1, Q, C]
                        
    #                     # 筛选有效Query特征
    #                     batch_valid_query_feats = batch_query_feats[batch_valid_mask]
    #                     if batch_valid_query_feats.numel() > 0:
    #                         # 确保可以正确 reshape
    #                         if batch_valid_query_feats.shape[0] == batch_obj_count:
    #                             batch_valid_query_feats = batch_valid_query_feats.reshape(1, batch_obj_count, -1)
    #                             # 计算框与有效Query的相似度
    #                             sim = F.cosine_similarity(batch_bbox_feats.unsqueeze(2), batch_valid_query_feats.unsqueeze(1), dim=-1)
    #                             # 生成真实分配标签
    #                             gt_assign = F.one_hot(batch_gt_box_obj.clamp(0, batch_obj_count-1), num_classes=batch_obj_count).float()
    #                             # 关联损失
    #                             batch_loss = F.binary_cross_entropy_with_logits(sim, gt_assign)
    #                             batch_box_query_losses.append(batch_loss)
                    
    #             if batch_box_query_losses:
    #                 loss_box_query = torch.stack(batch_box_query_losses).mean()
    #                 loss_dict['loss_box_query'] = loss_box_query
    #             else:
    #                 loss_dict['loss_box_query'] = torch.tensor(0.0, device=device)
    #     else:
    #         loss_dict['loss_box_query'] = torch.tensor(0.0, device=device)

    #     # ===================== 原有场景图损失（完全不变） =====================
    #     if self.use_scene_graph and 'gt_rels' in targets:
    #         sgg_outputs = {
    #             'importance': outputs.get('sgg_importance'),
    #             'sub_pos': outputs.get('sgg_sub_pos'),
    #             'obj_pos': outputs.get('sgg_obj_pos'),
    #             'rel_pred': outputs.get('sgg_rel_pred'),
    #         }

    #         gt_rels = targets['gt_rels']
    #         gt_importance = None
    #         if 'obj_labels' in targets and 'bboxes' in targets:
    #             num_obj = outputs['object_node_feat'].size(1)
    #             gt_importance = prepare_gt_importance_matrix(
    #                 gt_rels.reshape(-1, 3), num_obj, device
    #             ).unsqueeze(0).expand(gt_rels.size(0), -1, -1)

    #         gt_sub_labels = gt_rels[:, :, 0] if gt_rels.size(1) > 0 else None
    #         gt_obj_labels = gt_rels[:, :, 1] if gt_rels.size(1) > 0 else None

    #         sgg_loss_dict = self.scene_graph_head.compute_loss(
    #             outputs=sgg_outputs,
    #             gt_rels=gt_rels,
    #             gt_importance=gt_importance,
    #             gt_sub_labels=gt_sub_labels,
    #             gt_obj_labels=gt_obj_labels,
    #         )

    #         for k, v in sgg_loss_dict.items():
    #             if k in loss_params:
    #                 loss_dict[k] = v

    #     # ===================== 总损失（动态加权所有损失） =====================
    #     total_loss = sum(
    #         loss_dict[k] * loss_params[k]
    #         for k in loss_dict.keys() if k in loss_params
    #     )
    #     loss_dict['total_loss'] = total_loss
        
    #     return total_loss, loss_dict


def convert_detections(info):
    batch_bboxes = info['gt_bbox']
    batch_masks = info['mask']
    batch_labels = info['obj_label']
    batch_uids = info['obj_idx']
    
    # 安全获取 masks，不存在时赋值为 None
    batch_img_masks = info.get('masks', None)

    # batch_masks: B x K, binary mask indicating whether the bbox is valid
    # format as if it is the output of the detector
    detections = []
    for bboxes, masks, labels, uids in zip(batch_bboxes, batch_masks, batch_labels, batch_uids):
        # 筛选有效目标
        valid_bboxes = bboxes[masks]
        valid_labels = labels[masks]
        valid_uids = uids[masks]
        
        # 构建基础检测结果
        det = {
            'boxes': valid_bboxes,
            'labels': valid_labels,
            'uids': valid_uids,
            'scores': torch.ones(valid_bboxes.shape[0], device=valid_bboxes.device),
        }
        
        # 只有当 masks 存在时，才添加并索引
        if batch_img_masks is not None:
            # 从 batch_img_masks 中取出当前样本的 mask
            img_mask = batch_img_masks[detections.__len__()]
            det['masks'] = img_mask[masks]
        
        detections.append(det)
        
    return detections

class BBoxDataFlow(object):
    """
    统一 bbox 数据流处理类

    明确区分:
    - gt_bbox: [B, V, N, 4] 或 [B, V*N, 4] Ground Truth bbox 坐标
    - gt_mask: [B, V, N] 或 [B, V*N] bbox 有效性掩码
    - obj_idx: [B, V, N] 或 [B, V*N] 每个 bbox 对应的 object ID
    - obj_label: [B, V, N] 或 [B, V*N] 每个 bbox 对应的 object 类别
    - detections: list of dict, 检测器输出格式 (用于 eval/mapper)

    提供统一接口:
    - flatten_bboxes(): 将 [B, V, N, ...] 展平为 [B, V*N, ...]
    - get_valid_bboxes(): 获取有效 bbox
    - build_detections(): 构建 detections 列表
    """

    def __init__(self, bboxes_per_scene, bbox_source='gt'):
        """
        Args:
            bboxes_per_scene: dict with keys 'gt_bbox', 'mask', 'obj_idx', 'obj_label'
            bbox_source: 'gt' | 'pred' | 'proposal'，明确 bbox 来源
        """
        self.bbox_source = bbox_source
        self.gt_bbox = bboxes_per_scene.get('gt_bbox')
        self.mask = bboxes_per_scene.get('mask')
        self.obj_idx = bboxes_per_scene.get('obj_idx')
        self.obj_label = bboxes_per_scene.get('obj_label')

        if self.gt_bbox is None:
            raise ValueError("bboxes_per_scene must contain 'gt_bbox'")

        self.B = self.gt_bbox.shape[0]
        self.device = self.gt_bbox.device

    def flatten(self):
        """将 [B, V, N, ...] 展平为 [B, V*N, ...]"""
        if self.gt_bbox.dim() == 4:
            V = self.gt_bbox.shape[1]
            N = self.gt_bbox.shape[2]
            flat_bbox = self.gt_bbox.reshape(self.B, V * N, 4)
            flat_mask = self.mask.reshape(self.B, V * N) if self.mask is not None else None
            flat_obj_idx = self.obj_idx.reshape(self.B, V * N) if self.obj_idx is not None else None
            flat_obj_label = self.obj_label.reshape(self.B, V * N) if self.obj_label is not None else None
            return {
                'gt_bbox': flat_bbox,
                'mask': flat_mask,
                'obj_idx': flat_obj_idx,
                'obj_label': flat_obj_label,
            }
        else:
            return {
                'gt_bbox': self.gt_bbox,
                'mask': self.mask,
                'obj_idx': self.obj_idx,
                'obj_label': self.obj_label,
            }

    def get_valid_bboxes(self, batch_idx=None):
        """获取有效 bbox (mask=True 的)"""
        flat = self.flatten()
        if batch_idx is not None:
            mask = flat['mask'][batch_idx] if flat['mask'] is not None else torch.ones(flat['gt_bbox'].shape[1], dtype=torch.bool, device=self.device)
            return {
                'gt_bbox': flat['gt_bbox'][batch_idx][mask],
                'obj_idx': flat['obj_idx'][batch_idx][mask] if flat['obj_idx'] is not None else None,
                'obj_label': flat['obj_label'][batch_idx][mask] if flat['obj_label'] is not None else None,
            }
        else:
            # 返回所有 batch
            result = {'gt_bbox': [], 'obj_idx': [], 'obj_label': []}
            for bi in range(self.B):
                valid = self.get_valid_bboxes(batch_idx=bi)
                result['gt_bbox'].append(valid['gt_bbox'])
                result['obj_idx'].append(valid['obj_idx'])
                result['obj_label'].append(valid['obj_label'])
            return result

    def build_detections(self):
        """构建 detections 列表 (兼容旧接口)"""
        flat = self.flatten()
        detections = []
        for bi in range(self.B):
            mask = flat['mask'][bi] if flat['mask'] is not None else torch.ones(flat['gt_bbox'].shape[1], dtype=torch.bool, device=self.device)
            det = {
                'boxes': flat['gt_bbox'][bi][mask],
                'labels': flat['obj_label'][bi][mask] if flat['obj_label'] is not None else torch.zeros(mask.sum(), dtype=torch.long, device=self.device),
                'uids': flat['obj_idx'][bi][mask] if flat['obj_idx'] is not None else torch.arange(mask.sum(), device=self.device),
                'scores': torch.ones(mask.sum(), device=self.device),
                'bbox_source': self.bbox_source,
            }
            detections.append(det)
        return detections

    def get_num_objects(self):
        """获取每个 batch 的真实 object 数量"""
        flat = self.flatten()
        num_objs = []
        for bi in range(self.B):
            if flat['obj_idx'] is not None:
                valid_ids = flat['obj_idx'][bi][flat['obj_idx'][bi] != -1]
                num_objs.append(len(torch.unique(valid_ids)) if valid_ids.numel() > 0 else 0)
            else:
                mask = flat['mask'][bi] if flat['mask'] is not None else torch.ones(flat['gt_bbox'].shape[1], dtype=torch.bool, device=self.device)
                num_objs.append(mask.sum().item())
        return num_objs


def get_match_idx(match_indices,
                  info,
                    N):
    B = len(match_indices)
    # N = info['gt_bbox'].shape[1]
    # total_N = B * N
    total_reorderd_indx = []
    for bi in range(B):
        pred_indices = match_indices[bi][0]
        gt_indices = match_indices[bi][1]
        # reorder the object ids at gt_indices to pred_indices
        ori_obj_idx = info['obj_idx'][bi]
        # reordered_obj_ids = torch.full_like(ori_obj_idx, -1) # -1
        reordered_obj_ids = torch.full((N,), -1, dtype = ori_obj_idx.dtype, device = ori_obj_idx.device) # -1
        reordered_obj_ids[pred_indices] = ori_obj_idx[gt_indices]

        # print("-------- print each data --------", bi, "--------------")
        # print("pred index", pred_indices)
        # print("gt index", gt_indices)
        # print("original", ori_obj_idx)
        # print("mask", info['mask'][bi])
        # print("reordered idx", reordered_obj_ids)

        total_reorderd_indx.append(reordered_obj_ids)

    total_reorderd_indx = torch.cat(total_reorderd_indx, dim=0) # total_N = B*N = BN

    return total_reorderd_indx