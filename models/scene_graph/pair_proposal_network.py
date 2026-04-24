import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvTiny(nn.Module):
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


class MatrixLearner(nn.Module):
    def __init__(self, mapper: str = "conv_tiny"):
        super().__init__()
        self.mapper = mapper
        if mapper == "conv_tiny":
            self.network = ConvTiny()
        else:
            raise ValueError(f"Unknown mapper: {mapper}")

    def forward(self, importance_matrix):
        return self.network(importance_matrix)


class PairProposalNetwork(nn.Module):
    def __init__(
        self,
        num_obj_query: int = 100,
        num_rel_query: int = 100,
        feat_channels: int = 256,
        num_classes: int = 133,
        use_matrix_learner: bool = True,
    ):
        super().__init__()
        self.num_obj_query = num_obj_query
        self.num_rel_query = num_rel_query
        self.feat_channels = feat_channels
        self.num_classes = num_classes
        self.use_matrix_learner = use_matrix_learner

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

        if self.use_matrix_learner:
            self.matrix_learner = MatrixLearner(mapper="conv_tiny")
        else:
            self.matrix_learner = None

        self.obj_cls_embed = nn.Linear(self.feat_channels, self.num_classes + 1)

    def forward(
        self,
        object_query: torch.Tensor,
        object_cls: torch.Tensor = None,
        return_importance_only: bool = False,
    ):
        batch_size = object_query.size(0)
        num_obj = object_query.size(1)

        query_feat = object_query

        sub_embed = self.sub_query_update(query_feat)
        obj_embed = self.obj_query_update(query_feat)

        sub_embed = F.normalize(sub_embed, p=2, dim=-1, eps=1e-12)
        obj_embed = F.normalize(obj_embed, p=2, dim=-1, eps=1e-12)

        importance = torch.matmul(sub_embed, obj_embed.transpose(1, 2))

        if self.matrix_learner is not None:
            importance = self.matrix_learner(importance)

        if return_importance_only:
            return {
                'importance': importance,
                'sub_embed': sub_embed,
                'obj_embed': obj_embed,
            }

        k = min(self.num_rel_query, num_obj * num_obj)
        _, topk_idx = torch.topk(
            importance.flatten(-2, -1), k=k
        )
        sub_pos = torch.div(
            topk_idx, num_obj, rounding_mode="trunc"
        )
        obj_pos = torch.remainder(topk_idx, num_obj)

        sub_query = torch.gather(
            query_feat,
            1,
            sub_pos.unsqueeze(-1).expand(-1, -1, self.feat_channels),
        )
        obj_query = torch.gather(
            query_feat,
            1,
            obj_pos.unsqueeze(-1).expand(-1, -1, self.feat_channels),
        )

        if object_cls is not None:
            sub_cls = torch.gather(
                object_cls,
                1,
                sub_pos.unsqueeze(-1).expand(-1, -1, object_cls.size(-1)),
            )
            obj_cls = torch.gather(
                object_cls,
                1,
                obj_pos.unsqueeze(-1).expand(-1, -1, object_cls.size(-1)),
            )
        else:
            sub_cls = None
            obj_cls = None

        return {
            'importance': importance,
            'sub_pos': sub_pos,
            'obj_pos': obj_pos,
            'sub_query': sub_query,
            'obj_query': obj_query,
            'sub_cls': sub_cls,
            'obj_cls': obj_cls,
            'pair_query': torch.cat([sub_query, obj_query], dim=1),
            'num_selected_pairs': k,
        }

    def compute_loss(
        self,
        outputs: dict,
        gt_rels: torch.Tensor = None,
        gt_importance: torch.Tensor = None,
        gt_sub_labels: torch.Tensor = None,
        gt_obj_labels: torch.Tensor = None,
    ):
        loss_dict = {}

        importance = outputs.get('importance')
        sub_cls = outputs.get('sub_cls')
        obj_cls = outputs.get('obj_cls')
        sub_pos = outputs.get('sub_pos')
        obj_pos = outputs.get('obj_pos')

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

        if gt_sub_labels is not None and sub_cls is not None and gt_sub_labels.size(1) > 0:
            max_len = sub_cls.size(1)
            gt_sub_labels_trimmed = gt_sub_labels[:, :max_len]
            gt_obj_labels_trimmed = gt_obj_labels[:, :max_len] if gt_obj_labels is not None else None

            valid_mask_sub = gt_sub_labels_trimmed != self.num_classes
            if valid_mask_sub.any():
                loss_sub_cls = F.cross_entropy(
                    sub_cls[valid_mask_sub],
                    gt_sub_labels_trimmed[valid_mask_sub],
                    reduction='mean',
                )
                loss_dict['loss_sub_cls'] = loss_sub_cls

            if gt_obj_labels_trimmed is not None and obj_cls is not None:
                valid_mask_obj = gt_obj_labels_trimmed != self.num_classes
                if valid_mask_obj.any():
                    loss_obj_cls = F.cross_entropy(
                        obj_cls[valid_mask_obj],
                        gt_obj_labels_trimmed[valid_mask_obj],
                        reduction='mean',
                    )
                    loss_dict['loss_obj_cls'] = loss_obj_cls

        return loss_dict
