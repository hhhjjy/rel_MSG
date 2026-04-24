import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from .pair_proposal_network import PairProposalNetwork, MatrixLearner
from .relation_decoder import RelationDecoder, RelationDecoderLayer, RelationHead
from .triplet_generator import TripletGenerator, convert_triplets_to_matrix, convert_groundtruth_to_triplets
from .sgg_evaluator import SceneGraphEvaluator, evaluate_scene_graph


class SceneGraphHead(nn.Module):
    def __init__(
        self,
        num_classes: int = 133,
        num_relations: int = 56,
        num_obj_query: int = 100,
        num_rel_query: int = 100,
        feat_channels: int = 256,
        use_matrix_learner: bool = True,
        num_rel_decoder_layers: int = 6,
        num_rel_heads: int = 8,
        dim_rel_feedforward: int = 2048,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_relations = num_relations
        self.num_obj_query = num_obj_query
        self.num_rel_query = num_rel_query
        self.feat_channels = feat_channels

        self.pair_proposal_network = PairProposalNetwork(
            num_obj_query=num_obj_query,
            num_rel_query=num_rel_query,
            feat_channels=feat_channels,
            num_classes=num_classes,
            use_matrix_learner=use_matrix_learner,
        )

        self.relation_head = RelationHead(
            num_rel_query=num_rel_query,
            num_relations=num_relations,
            feat_channels=feat_channels,
            num_rel_decoder_layers=num_rel_decoder_layers,
            num_rel_heads=num_rel_heads,
            dim_rel_feedforward=dim_rel_feedforward,
        )

        self.triplet_generator = TripletGenerator(
            num_relations=num_relations,
            num_classes=num_classes,
        )

    def forward(
        self,
        object_query: torch.Tensor,
        object_cls: torch.Tensor = None,
        object_scores: torch.Tensor = None,
        mode: str = 'sgdet',
    ):
        ppn_outputs = self.pair_proposal_network(
            object_query=object_query,
            object_cls=object_cls,
        )

        rel_outputs = self.relation_head(
            sub_query=ppn_outputs['sub_query'],
            obj_query=ppn_outputs['obj_query'],
            sub_cls=ppn_outputs['sub_cls'],
            obj_cls=ppn_outputs['obj_cls'],
        )

        triplets = self.triplet_generator(
            sub_pos=ppn_outputs['sub_pos'],
            obj_pos=ppn_outputs['obj_pos'],
            rel_pred=rel_outputs['rel_pred'],
            sub_cls=ppn_outputs['sub_cls'],
            obj_cls=ppn_outputs['obj_cls'],
            obj_scores=object_scores,
            mode=mode,
        )

        outputs = {
            'importance': ppn_outputs['importance'],
            'sub_pos': ppn_outputs['sub_pos'],
            'obj_pos': ppn_outputs['obj_pos'],
            'rel_pred': rel_outputs['rel_pred'],
            'sub_cls': ppn_outputs['sub_cls'],
            'obj_cls': ppn_outputs['obj_cls'],
            'triplets': triplets,
            'num_selected_pairs': ppn_outputs['num_selected_pairs'],
        }

        return outputs

    def compute_loss(
        self,
        outputs: dict,
        gt_rels: torch.Tensor = None,
        gt_importance: torch.Tensor = None,
        gt_sub_labels: torch.Tensor = None,
        gt_obj_labels: torch.Tensor = None,
    ):
        loss_dict = {}

        ppn_loss = self.pair_proposal_network.compute_loss(
            outputs=outputs,
            gt_rels=gt_rels,
            gt_importance=gt_importance,
            gt_sub_labels=gt_sub_labels,
            gt_obj_labels=gt_obj_labels,
        )
        loss_dict.update(ppn_loss)

        rel_loss = self.relation_head.compute_loss(
            outputs=outputs,
            gt_rels=gt_rels,
        )
        loss_dict.update(rel_loss)

        return loss_dict


class SceneGraphModule(nn.Module):
    def __init__(
        self,
        num_classes: int = 133,
        num_relations: int = 56,
        num_obj_query: int = 100,
        num_rel_query: int = 100,
        feat_channels: int = 256,
        embed_dims: int = 1024,
        use_matrix_learner: bool = True,
        num_rel_decoder_layers: int = 6,
        num_rel_heads: int = 8,
        dim_rel_feedforward: int = 2048,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_relations = num_relations
        self.num_obj_query = num_obj_query
        self.num_rel_query = num_rel_query
        self.feat_channels = feat_channels
        self.embed_dims = embed_dims

        self.query_resize = nn.Linear(self.embed_dims, self.feat_channels)

        self.scene_graph_head = SceneGraphHead(
            num_classes=num_classes,
            num_relations=num_relations,
            num_obj_query=num_obj_query,
            num_rel_query=num_rel_query,
            feat_channels=feat_channels,
            use_matrix_learner=use_matrix_learner,
            num_rel_decoder_layers=num_rel_decoder_layers,
            num_rel_heads=num_rel_heads,
            dim_rel_feedforward=dim_rel_feedforward,
        )

    def forward(
        self,
        object_query: torch.Tensor,
        object_cls: torch.Tensor = None,
        object_scores: torch.Tensor = None,
        mode: str = 'sgdet',
    ):
        if object_query.size(-1) != self.feat_channels:
            object_query = self.query_resize(object_query)

        if object_query.dim() == 2:
            object_query = object_query.unsqueeze(0)

        outputs = self.scene_graph_head(
            object_query=object_query,
            object_cls=object_cls,
            object_scores=object_scores,
            mode=mode,
        )

        return outputs

    def compute_loss(
        self,
        outputs: dict,
        gt_rels: torch.Tensor = None,
        gt_importance: torch.Tensor = None,
        gt_sub_labels: torch.Tensor = None,
        gt_obj_labels: torch.Tensor = None,
    ):
        return self.scene_graph_head.compute_loss(
            outputs=outputs,
            gt_rels=gt_rels,
            gt_importance=gt_importance,
            gt_sub_labels=gt_sub_labels,
            gt_obj_labels=gt_obj_labels,
        )


def prepare_gt_importance_matrix(
    gt_rels: torch.Tensor,
    num_obj: int,
    device: torch.device = torch.device('cpu'),
):
    importance = torch.zeros((num_obj, num_obj), dtype=torch.float32, device=device)
    if gt_rels is not None and len(gt_rels) > 0:
        for rel in gt_rels:
            s, o = rel[0].item(), rel[1].item()
            if s < num_obj and o < num_obj:
                importance[s, o] = 1.0
    return importance


def format_predictions_for_evaluation(
    triplets: List[Dict],
    sub_cls_all: torch.Tensor = None,
    obj_cls_all: torch.Tensor = None,
    obj_scores_all: torch.Tensor = None,
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

        if sub_cls_all is not None and obj_cls_all is not None:
            b = 0
            pred['sub_label'] = triplet.get('sub_label', sub_cls_all[b][triplet['sub_idx']].argmax().item())
            pred['obj_label'] = triplet.get('obj_label', obj_cls_all[b][triplet['obj_idx']].argmax().item())

        if obj_scores_all is not None:
            b = 0
            pred['obj_score'] = obj_scores_all[b][triplet['obj_idx']].item()

        predictions.append(pred)

    return predictions


def format_groundtruth_for_evaluation(
    gt_rels: torch.Tensor,
    gt_labels: torch.Tensor = None,
):
    groundtruths = []
    if gt_labels is not None:
        for i in range(len(gt_rels)):
            groundtruths.append({
                'sub_label': gt_labels[i, 0].item(),
                'obj_label': gt_labels[i, 1].item(),
                'rel_cls': gt_rels[i, 2].item(),
            })
    else:
        for i in range(len(gt_rels)):
            groundtruths.append({
                'sub_idx': gt_rels[i, 0].item(),
                'obj_idx': gt_rels[i, 1].item(),
                'rel_cls': gt_rels[i, 2].item(),
            })

    return groundtruths
