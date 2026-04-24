# decoder style AoMSG
# set the module as a separate file for cleaness
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np

from .loss import get_match_idx, get_association_sv
from .loss import InfoNCELoss, MaskBCELoss, FocalLoss, MaskMetricLoss, MeanSimilarityLoss, TotalCodingRate
from .pair_net import PairNet

def pad_and_stack(tensor_list, dim=0):
    """
    自动补0对齐所有张量形状，然后stack堆叠
    输入：形状不同的tensor列表，如 [1x128, 3x128, 2x128]
    输出：stack后的张量 → [N, max_len, 128]
    """
    # 1. 找到最大长度（第0维的最大值）
    max_len = max(t.shape[0] for t in tensor_list)
    
    padded_list = []
    for t in tensor_list:
        # 2. 计算需要补多少行
        pad_len = max_len - t.shape[0]
        if pad_len > 0:
            # 向下补0 (上、下、左、右)
            padding = (0, 0, 0, pad_len)  # 只补行
            t_padded = torch.nn.functional.pad(t, padding, mode='constant', value=0)
        else:
            t_padded = t
        
        padded_list.append(t_padded)
    
    # 3. 现在形状都一样了，可以stack
    return torch.stack(padded_list, dim=dim)

def pad_square_matrices(matrix_list):
    """
    把一堆大小不同的方阵，全部填充为【最大尺寸的方阵】，补0
    输入：list of 2D tensor（每个都是方阵，大小不同）
    输出：list of 2D tensor（所有矩阵形状完全相同：max_size × max_size）
    """
    # 1. 找到所有方阵里最大的尺寸
    max_size = max(mat.shape[0] for mat in matrix_list)

    padded_matrices = []
    for mat in matrix_list:
        current_size = mat.shape[0]
        pad = max_size - current_size

        if pad > 0:
            # 向下、向右 补 0，把小方阵变成 max_size × max_size
            padded = torch.nn.functional.pad(mat, (0, pad, 0, pad), value=0)
        else:
            padded = mat

        padded_matrices.append(padded)

    return torch.stack(padded_matrices)


def split_diag_blocks(matrix, block_sizes):
    """
    将一个大对角方阵，按 block_sizes 切分成若干个对角子阵
    
    参数：
        matrix: 大的方阵 (total_size x total_size)
        block_sizes: list，每个元素是每个对角块的大小，例如 [2,3,2]
    
    返回：
        list of tensors，每个元素是一个子方阵
    """
    blocks = []
    start = 0
    
    for size in block_sizes:
        # 切片：取出当前对角块
        block = matrix[start:start+size, start:start+size]
        blocks.append(block)
        start += size
    
    return blocks

class DecoderAssociator(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_heads, num_layers, object_dim, place_dim, 
                 num_img_patches, model, pr_loss, obj_loss, **kwargs):
        super(DecoderAssociator, self).__init__()
        self.model_name = model
        self.object_dim = object_dim
        self.place_dim = place_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_img_patches = num_img_patches # 256 # 224//14 ** 2 [CLS]

        # self.sep_token = nn.Parameter(torch.empty(1, hidden_dim))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model = hidden_dim,
            nhead = num_heads,
            dim_feedforward = int(hidden_dim * 4),
            dropout = 0.1,
            activation = 'gelu',
            layer_norm_eps = 1e-5,
            batch_first=True, 
            norm_first=False,
        )
        decoder_norm = nn.LayerNorm(hidden_dim, eps=1e-5, elementwise_affine=True) # 1e-5 or 1e-6?
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.num_layers, norm=decoder_norm)

        # box embedding
        self.box_emb = nn.Linear(4, hidden_dim, bias=False)
        self.whole_box = nn.Parameter(torch.tensor([0, 0, 224, 224], dtype=torch.float32), requires_grad=False)

        # input adaptor
        self.object_proj = nn.Linear(object_dim, hidden_dim, bias=False)
        self.place_proj = nn.Linear(place_dim, hidden_dim, bias=False)
        # output head
        # self.object_head = nn.Sequential(
        #     nn.Linear(hidden_dim, output_dim),
        #     nn.GELU(approximate='tanh'),
        #     nn.LayerNorm(output_dim, elementwise_affine=False, eps=1e-5),
        #     nn.Linear(output_dim, output_dim),
        # )
        self.object_head = nn.Linear(hidden_dim, output_dim)
        
        # self.place_head = nn.Sequential(
        #     nn.Linear(hidden_dim, output_dim),
        #     nn.GELU(approximate='tanh'),
        #     nn.LayerNorm(output_dim, elementwise_affine=False, eps=1e-5),
        #     nn.Linear(output_dim, output_dim),
        # )
        self.place_head = nn.Linear(hidden_dim, output_dim)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_img_patches + 1, hidden_dim), requires_grad=False)

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

        self.measure_cos_pp = False
        if pr_loss == "bce":
            w = kwargs["pp_weight"] if "pp_weight" in kwargs else 1.0
            self.pr_loss_fn = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([w]))
            self.measure_cos_pp = False
        else:
            self.pr_loss_fn = nn.MSELoss(reduction='none')
            self.measure_cos_pp = True

        self.measure_cos_obj = False
        if obj_loss == "bce":
            assert "pos_weight" in kwargs
            pos_weight = kwargs["pos_weight"]
            self.obj_loss_fn = MaskBCELoss(pos_weight=pos_weight)
            self.measure_cos_obj = False
        elif obj_loss == "focal":
            assert "alpha" in kwargs
            assert "gamma" in kwargs
            self.obj_loss_fn = FocalLoss(alpha=kwargs["alpha"], gamma=kwargs["gamma"])
            self.measure_cos_obj = False
        elif obj_loss == "infonce":
            assert "temperature" in kwargs
            self.obj_loss_fn = InfoNCELoss(temperature=kwargs["temperature"], learnable=False)
            self.measure_cos_obj = False
        else:
            self.obj_loss_fn = MaskMetricLoss()
            self.measure_cos_obj = True
        self.obj_loss_fn_sim = MeanSimilarityLoss()
        self.obj_tcr = TotalCodingRate(eps=0.2)
        
        self.use_pair_net = kwargs.get('use_pair_net', False)
        self.train_pair_net_only = kwargs.get('train_pair_net_only', False)
        
        if self.use_pair_net or self.train_pair_net_only:
            self.use_pair_net = True
            self.pair_net = PairNet()
            if self.train_pair_net_only:
                self._freeze_except_pair_net()
        
        self.initialize_weights()


    def initialize_weights(self,):
        
        self.apply(self._init_weights)

        grid_size = int(self.num_img_patches**.5)
        pos_embed = get_2d_sincos_pos_embed(
            embed_dim=self.pos_embed.shape[-1], 
            grid_size=grid_size, 
            cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    # reference: from MAE's code base 
    # https://github.com/facebookresearch/mae/blob/main/models_mae.py#L68
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _freeze_except_pair_net(self):
        for name, param in self.named_parameters():
            if 'pair_net' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True



    def pad_objects(self, object_emb):
        padded_object_emb = pad_sequence(object_emb, batch_first=True, padding_value=0)
        return padded_object_emb
    
    def get_query_mask(self, padded_object_emb):
        """
        Obtain masking for attention, since object embedding are padded.
        padded_object_emb: the real 0/1 masking
        """
        B, L, Ho = padded_object_emb.size()
        img_length = 1 # just 1 token for the whole-box query
        obj_mask = (padded_object_emb == 0).all(dim=-1).to(padded_object_emb.device)
        place_mask = torch.zeros(B, img_length, dtype = obj_mask.dtype, device = obj_mask.device)
        total_mask = torch.cat((place_mask, obj_mask), dim=1)
        return total_mask


    def forward(self, object_emb, place_emb, bbox_pos, vid_idx=None):

        padded_obj_embd = object_emb
        padded_obj_box = bbox_pos

        B, V, K, Ho = padded_obj_embd.shape
        padded_obj_embd = padded_obj_embd.view(B*V, K, Ho)
        padded_obj_box = padded_obj_box.view(B*V, K, 4)
        B = B*V

        whole_box_expanded = self.whole_box.unsqueeze(0).expand(B, 1, -1)
        query = torch.cat([whole_box_expanded, padded_obj_box], dim=1) / 224.0
        query = self.box_emb(query)

        query_mask = self.get_query_mask(padded_obj_embd)

        place_emb = place_emb.reshape(-1, *place_emb.shape[2:])

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

    def scene_level_attention(self, object_enc, vid_idx, obj_mask):
        """
        Perform scene-level cross-attention where all objects within a scene interact.
        object_enc: (B, K, D) - per-frame object features
        vid_idx: (B,) - scene id for each frame
        obj_mask: (B, K) - True for valid objects, False for padding
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


    def predict_object(self, padded_obj_feat):
        # object_embeddings: B x K x Ho, padded from object embedding
        B, K, H = padded_obj_feat.size()
        # # object_predictions: BK x BK
        if self.measure_cos_obj:
            norm = padded_obj_feat.norm(dim=-1, keepdim=True)
            norm = torch.where(norm == 0, torch.ones_like(norm), norm)
            normed_obj_feat = padded_obj_feat / norm # B x K x Ho
        # just dot product:
        else:
            normed_obj_feat = padded_obj_feat
        # --- #
        flatten_obj_feat = normed_obj_feat.view(-1, H) # flatten the first two dimension
        object_predictions = flatten_obj_feat @ flatten_obj_feat.t() # BK x BK
        return object_predictions  
    
    def predict_place(self, place):
        # simple cosine similarity

        # place_predictions: B x B
        if self.measure_cos_pp:
            normed_p = place / place.norm(dim=-1, keepdim=True)
        else:
            normed_p = place
        place_logits = normed_p @ normed_p.t()
        return place_logits
    
    def get_loss(self, results, additional_info, match_inds, place_labels, rel_labels=None, weights=None):

        # prepare
        num_emb = results['embeddings'].size(1)
        reorderd_idx = get_match_idx(match_inds, additional_info, num_emb)
        logs = {}

        if self.train_pair_net_only:
            num_scene = additional_info['scene_num']
            gt_importance = rel_labels
            obj_idx_per_scene = additional_info['obj_idx'].view(num_scene,-1)
            obj_emb_per_scene = results['embeddings'].view(num_scene,-1,1024)
            obj_num_list = []
            embeddings_mean_list = []
            for i in range(num_scene):
                # 1. 取出当前场景的 ID 和 特征，它们是一一对应的
                # current_ids: (N,)
                # current_emb: (N, 1024)
                current_ids = obj_idx_per_scene[i]
                current_emb = obj_emb_per_scene[i]

                # 2. 过滤掉 ID 为 -1 的 padding
                mask = current_ids != -1
                valid_ids = current_ids[mask]
                valid_emb = current_emb[mask]

                # 3. 对当前场景内的 ID 去重，并获取反向索引
                # unique_ids_scene: 当前场景有哪些物体
                # inverse_idx: valid_emb 中的每一行对应 unique_ids_scene 中的第几个
                unique_ids_scene, inverse_idx = valid_ids.unique(return_inverse=True)

                # 4. 初始化累加器 (在这个场景内局部计算)
                num_unique_in_scene = len(unique_ids_scene)
                feat_dim = current_emb.size(-1)
                
                sum_feats = torch.zeros((num_unique_in_scene, feat_dim), 
                                        device=current_emb.device, 
                                        dtype=current_emb.dtype)
                counts = torch.zeros(num_unique_in_scene, 
                                    device=current_emb.device, 
                                    dtype=torch.float)

                # 5. 使用 index_add_ 在当前场景内聚合
                sum_feats.index_add_(0, inverse_idx, valid_emb)
                counts.index_add_(0, inverse_idx, torch.ones_like(inverse_idx, dtype=torch.float))

                # 6. 计算均值
                mean_feats_scene = sum_feats / counts.unsqueeze(1)

                # 7. 存入列表 (保持你原来的接口不变)
                obj_num_list.append(num_unique_in_scene)
                embeddings_mean_list.append(mean_feats_scene)

            gt_importance_list = split_diag_blocks(gt_importance, obj_num_list)
            gt_importance_per_scene = pad_square_matrices(gt_importance_list)
            gt_importance_per_scene = (gt_importance_per_scene != 0).float()

            embeddings_mean_per_scene = pad_and_stack(embeddings_mean_list)
            pair_net_output = self.pair_net(embeddings_mean_per_scene)
            
            loss_dict = self.pair_net.compute_loss(
                pair_net_output, 
                gt_rels=None,
                gt_importance=gt_importance_per_scene
            )
            total_loss = loss_dict.get('loss_match', torch.tensor(0.0, device=gt_importance_per_scene.device))
            logs['rel_matrix_loss'] = total_loss.item()
            return total_loss, logs

        # get loss
        # object similarity loss with TCR regularizer
        sim_loss, mean_dis, tcr, id_counts, embeddings_mean = self.object_similarity_loss(results['embeddings'], reorderd_idx)
        logs['tcr'] = tcr.item()
        logs['obj_sim_loss'] = sim_loss.item()
        # logs['num_obj'] = id_counts.shape[0]
        logs['mean_dis'] = mean_dis.item()
        # logs['avg_num_instances'] = id_counts.sum().item() / (id_counts.shape[0] + 1e-5)

        # object_loss = weights['obj'] * sim_loss + weights['mean'] * mean_dis + weights['tcr'] * tcr
        # # object association loss
        object_loss = self.object_association_loss(results['object_predictions'], reorderd_idx)

        logs['running_loss_obj'] = object_loss.item()
        
        # place recognition loss
        place_loss = self.place_recognition_loss(results['place_predictions'], place_labels)

        total_loss = object_loss + weights['pr'] * place_loss #  weights['tcr'] * tcr
        logs['running_loss_pr'] = place_loss.item()
        # print(sim_loss, mean_dis, tcr, object_loss, place_loss)

        return total_loss, logs

    def place_recognition_loss(self, place_predictions, place_labels): # TODO: check implementation
        # place_predictions: B x 1
        # place_labels: B x 1
        # loss: scalar
        # place_predictions = (place_predictions + 1.0) / 2.0
        
        loss = self.pr_loss_fn(place_predictions, place_labels).mean()
        return loss
    
    def object_association_loss(self, object_predictions, reorderd_idx):
        """
        input:
            object_predictions: BN x BN, cosine similarity matrix
            reorder_idx: BN, padded, reordered gt_indices to match the pred_indices
        intermediate:
            supervision_matrix: BN x BN, binary matrix indicating the object association
            mask: BN x BN, binary matrix indicating the valid entries in the supervision_matrix
        output:
            loss: scalar
        """
        # supervision is already masked by the mask
        supervision_matrix, mask = get_association_sv(reorderd_idx)

        # using wrapped loss
        total_loss = self.obj_loss_fn(object_predictions, supervision_matrix, mask)
        return total_loss
    
    def object_similarity_loss(self, embeddings, matched_idx):
        """
        compute the similarity loss and the regularization
        """
        B, N, h = embeddings.size()
        flatten_embeddings = embeddings.view(-1, h)
        sim_loss, mean_dis_loss, id_counts, embeddings_mean = self.obj_loss_fn_sim(flatten_embeddings, matched_idx)
        tcr = self.obj_tcr(flatten_embeddings, matched_idx)
        return sim_loss, mean_dis_loss, tcr, id_counts, embeddings_mean


# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# MAE: https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
