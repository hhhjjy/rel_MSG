import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class TripletGenerator:
    def __init__(
        self,
        num_relations: int = 56,
        num_classes: int = 133,
        use_mask: bool = False,
    ):
        self.num_relations = num_relations
        self.num_classes = num_classes
        self.use_mask = use_mask

    def __call__(
        self,
        sub_pos: torch.Tensor,
        obj_pos: torch.Tensor,
        rel_pred: torch.Tensor,
        sub_cls: torch.Tensor = None,
        obj_cls: torch.Tensor = None,
        sub_cls_all: torch.Tensor = None,
        obj_cls_all: torch.Tensor = None,
        obj_scores: torch.Tensor = None,
        mode: str = 'sgdet',
    ):
        return self.generate_triplets(
            sub_pos=sub_pos,
            obj_pos=obj_pos,
            rel_pred=rel_pred,
            sub_cls=sub_cls,
            obj_cls=obj_cls,
            sub_cls_all=sub_cls_all,
            obj_cls_all=obj_cls_all,
            obj_scores=obj_scores,
            mode=mode,
        )

    def generate_triplets(
        self,
        sub_pos: torch.Tensor,
        obj_pos: torch.Tensor,
        rel_pred: torch.Tensor,
        sub_cls: torch.Tensor = None,
        obj_cls: torch.Tensor = None,
        sub_cls_all: torch.Tensor = None,
        obj_cls_all: torch.Tensor = None,
        obj_scores: torch.Tensor = None,
        mode: str = 'sgdet',
    ):
        batch_size = sub_pos.size(0)
        triplets_list = []

        for b in range(batch_size):
            sub_idx = sub_pos[b]
            obj_idx = obj_pos[b]
            rel_logit = rel_pred[b]

            rel_cls = rel_logit.argmax(dim=-1)
            rel_score = rel_logit.softmax(dim=-1)

            if sub_cls is not None and obj_cls is not None:
                sub_label = sub_cls[b].argmax(dim=-1)
                obj_label = obj_cls[b].argmax(dim=-1)
                sub_score = sub_cls[b].softmax(dim=-1)
                obj_score = obj_cls[b].softmax(dim=-1)
            elif sub_cls_all is not None and obj_cls_all is not None:
                sub_label = sub_cls_all[b][sub_idx].argmax(dim=-1)
                obj_label = obj_cls_all[b][obj_idx].argmax(dim=-1)
                sub_score = sub_cls_all[b][sub_idx].softmax(dim=-1)
                obj_score = obj_cls_all[b][obj_idx].softmax(dim=-1)
            else:
                sub_label = None
                obj_label = None
                sub_score = None
                obj_score = None

            if obj_scores is not None:
                obj_score_pred = obj_scores[b]
            else:
                obj_score_pred = None

            num_rel = sub_idx.size(0)
            for i in range(num_rel):
                triplet = {
                    'sub_idx': sub_idx[i].item() if sub_idx.is_cuda else sub_idx[i].item(),
                    'obj_idx': obj_idx[i].item() if obj_idx.is_cuda else obj_idx[i].item(),
                    'rel_cls': rel_cls[i].item() if rel_cls.is_cuda else rel_cls[i].item(),
                    'rel_score': rel_score[i].max().item() if rel_score.is_cuda else rel_score[i].max().item(),
                    'sub_label': sub_label[i].item() if sub_label is not None and hasattr(sub_label[i], 'item') else sub_label[i],
                    'obj_label': obj_label[i].item() if obj_label is not None and hasattr(obj_label[i], 'item') else obj_label[i],
                    'sub_score': sub_score[i].max().item() if sub_score is not None and sub_score.is_cuda else (sub_score[i].max().item() if sub_score is not None else None),
                    'obj_score': obj_score[i].max().item() if obj_score is not None and obj_score.is_cuda else (obj_score[i].max().item() if obj_score is not None else None),
                }

                if mode == 'sgdet' and obj_score_pred is not None:
                    triplet['obj_score'] = obj_score_pred[obj_idx[i]].item() if obj_score_pred.is_cuda else obj_score_pred[obj_idx[i]].item()

                triplets_list.append(triplet)

        return triplets_list

    @staticmethod
    def triplets_to_predictions(
        triplets: List[Dict],
        sub_cls_all: torch.Tensor = None,
        obj_cls_all: torch.Tensor = None,
        obj_scores: torch.Tensor = None,
        mode: str = 'sgdet',
    ):
        predictions = []
        for triplet in triplets:
            pred = {
                'sub_idx': triplet['sub_idx'],
                'obj_idx': triplet['obj_idx'],
                'rel_cls': triplet['rel_cls'],
                'rel_score': triplet['rel_score'],
            }

            if 'sub_label' in triplet and triplet['sub_label'] is not None:
                pred['sub_label'] = triplet['sub_label']
            if 'obj_label' in triplet and triplet['obj_label'] is not None:
                pred['obj_label'] = triplet['obj_label']
            if 'sub_score' in triplet and triplet['sub_score'] is not None:
                pred['sub_score'] = triplet['sub_score']
            if 'obj_score' in triplet and triplet['obj_score'] is not None:
                pred['obj_score'] = triplet['obj_score']

            predictions.append(pred)

        return predictions


def convert_triplets_to_matrix(
    triplets: List[Dict],
    num_obj: int,
    num_rel: int,
    device: torch.device = torch.device('cpu'),
):
    relation_matrix = torch.zeros((num_obj, num_obj), dtype=torch.long, device=device)
    score_matrix = torch.zeros((num_obj, num_obj), dtype=torch.float32, device=device)

    for triplet in triplets:
        s = triplet['sub_idx']
        o = triplet['obj_idx']
        r = triplet['rel_cls']
        score = triplet.get('rel_score', 1.0)

        if 0 <= s < num_obj and 0 <= o < num_obj:
            relation_matrix[s, o] = r
            score_matrix[s, o] = score

    return relation_matrix, score_matrix


def convert_groundtruth_to_triplets(
    gt_rels: torch.Tensor,
    gt_labels: torch.Tensor = None,
):
    num_rel = gt_rels.size(0)
    triplets = []

    for i in range(num_rel):
        triplet = {
            'sub_idx': gt_rels[i, 0].item() if isinstance(gt_rels[i, 0], torch.Tensor) else gt_rels[i, 0],
            'obj_idx': gt_rels[i, 1].item() if isinstance(gt_rels[i, 1], torch.Tensor) else gt_rels[i, 1],
            'rel_cls': gt_rels[i, 2].item() if isinstance(gt_rels[i, 2], torch.Tensor) else gt_rels[i, 2],
        }

        if gt_labels is not None:
            triplet['sub_label'] = gt_labels[i, 0].item() if isinstance(gt_labels[i, 0], torch.Tensor) else gt_labels[i, 0]
            triplet['obj_label'] = gt_labels[i, 1].item() if isinstance(gt_labels[i, 1], torch.Tensor) else gt_labels[i, 1]

        triplets.append(triplet)

    return triplets


class PairFeatureExtractor(nn.Module):
    def __init__(
        self,
        feat_channels: int = 256,
        output_dim: int = 512,
    ):
        super().__init__()
        self.feat_channels = feat_channels
        self.output_dim = output_dim

        self.pair_mlp = nn.Sequential(
            nn.Linear(feat_channels * 2, feat_channels * 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels * 2, output_dim),
        )

    def forward(self, sub_feat: torch.Tensor, obj_feat: torch.Tensor):
        pair_feat = torch.cat([sub_feat, obj_feat], dim=-1)
        return self.pair_mlp(pair_feat)


from torch import nn
