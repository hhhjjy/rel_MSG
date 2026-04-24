import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from functools import reduce


def intersect_2d(x1, x2):
    if x1.shape[1] != x2.shape[1]:
        raise ValueError("Input arrays must have same #columns")
    res = (x1[..., None] == x2.T[None, ...]).all(1)
    return res


def argsort_desc(scores):
    return np.column_stack(np.unravel_index(np.argsort(-scores.ravel()), scores.shape))


def compute_iou(boxes_a, boxes_b):
    if isinstance(boxes_a, torch.Tensor):
        boxes_a = boxes_a.cpu().numpy()
    if isinstance(boxes_b, torch.Tensor):
        boxes_b = boxes_b.cpu().numpy()

    x1 = np.maximum(boxes_a[:, 0, None], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, 1, None], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, 2, None], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, 3, None], boxes_b[None, :, 3])

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    union = area_a[:, None] + area_b[None, :] - intersection
    iou = intersection / (union + 1e-10)
    return iou


def _triplet_bbox(rels, classes, boxes, rel_scores=None, obj_scores=None):
    if isinstance(rels, torch.Tensor):
        rels = rels.cpu().numpy()
    if isinstance(classes, torch.Tensor):
        classes = classes.cpu().numpy()
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()

    if rel_scores is not None:
        if isinstance(rel_scores, torch.Tensor):
            rel_scores = rel_scores.cpu().numpy()
        pred_labels = 1 + rel_scores[:, 1:].argmax(1)
        pred_scores = rel_scores[:, 1:].max(1)
    else:
        pred_labels = rels[:, 2]
        pred_scores = np.ones(len(rels))

    if obj_scores is not None:
        if isinstance(obj_scores, torch.Tensor):
            obj_scores = obj_scores.cpu().numpy()
        obj_scores_per_rel = obj_scores[rels[:, :2]].prod(1)
        pred_scores = pred_scores * obj_scores_per_rel

    triplets = np.column_stack([
        classes[rels[:, 0]],
        classes[rels[:, 1]],
        pred_labels
    ])

    triplet_boxes = np.column_stack([
        boxes[rels[:, 0]],
        boxes[rels[:, 1]]
    ])

    return triplets, triplet_boxes, pred_scores


def _triplet_panseg(rels, classes, masks, rel_scores=None, obj_scores=None):
    if isinstance(rels, torch.Tensor):
        rels = rels.cpu().numpy()
    if isinstance(classes, torch.Tensor):
        classes = classes.cpu().numpy()

    if rel_scores is not None:
        if isinstance(rel_scores, torch.Tensor):
            rel_scores = rel_scores.cpu().numpy()
        pred_labels = 1 + rel_scores[:, 1:].argmax(1)
        pred_scores = rel_scores[:, 1:].max(1)
    else:
        pred_labels = rels[:, 2]
        pred_scores = np.ones(len(rels))

    if obj_scores is not None:
        if isinstance(obj_scores, torch.Tensor):
            obj_scores = obj_scores.cpu().numpy()
        obj_scores_per_rel = obj_scores[rels[:, :2]].prod(1)
        pred_scores = pred_scores * obj_scores_per_rel

    triplets = np.column_stack([
        classes[rels[:, 0]],
        classes[rels[:, 1]],
        pred_labels
    ])

    return triplets, None, pred_scores


def _compute_pred_matches_bbox(
    gt_triplets,
    pred_triplets,
    gt_boxes,
    pred_boxes,
    iou_thresh,
    phrdet=False,
    ignore_rel=False,
):
    pred_triplets = pred_triplets.astype(int)
    gt_triplets = gt_triplets.astype(int)

    if len(pred_triplets) == 0 or len(gt_triplets) == 0:
        return [np.array([]) for _ in range(100)]

    iou = compute_iou(pred_boxes, gt_boxes)

    gt_match = np.zeros(len(gt_triplets), dtype=int)
    pred_match = np.zeros(len(pred_triplets), dtype=int)

    for gt_idx, gt_triplet in enumerate(gt_triplets):
        sub_i, obj_i, rel_i = gt_triplet

        for pred_idx, pred_triplet in enumerate(pred_triplets):
            if pred_match[pred_idx] != 0:
                continue
            if ignore_rel and rel_i != pred_triplet[2]:
                continue

            if sub_i == pred_triplet[0] and obj_i == pred_triplet[1]:
                if rel_i == pred_triplet[2]:
                    if iou[pred_idx, gt_idx] >= iou_thresh:
                        gt_match[gt_idx] = pred_idx
                        pred_match[pred_idx] = gt_idx
                        break

    match_matrix = pred_match >= 0
    pred_to_gt = []
    for pred_idx in range(len(pred_triplets)):
        if match_matrix[pred_idx]:
            pred_to_gt.append(np.array([np.where(pred_match == pred_idx)[0][0]]))
        else:
            pred_to_gt.append(np.array([]))

    while len(pred_to_gt) < 100:
        pred_to_gt.append(np.array([]))

    return pred_to_gt


def _compute_pred_matches_panseg(
    gt_triplets,
    pred_triplets,
    gt_masks,
    pred_masks,
    iou_thresh,
    phrdet=False,
    ignore_rel=False,
):
    pred_triplets = pred_triplets.astype(int)
    gt_triplets = gt_triplets.astype(int)

    if len(pred_triplets) == 0 or len(gt_triplets) == 0:
        return [np.array([]) for _ in range(100)]

    gt_match = np.zeros(len(gt_triplets), dtype=int)
    pred_match = np.zeros(len(pred_triplets), dtype=int)

    for gt_idx, gt_triplet in enumerate(gt_triplets):
        sub_i, obj_i, rel_i = gt_triplet

        for pred_idx, pred_triplet in enumerate(pred_triplets):
            if pred_match[pred_idx] != 0:
                continue
            if ignore_rel and rel_i != pred_triplet[2]:
                continue

            if sub_i == pred_triplet[0] and obj_i == pred_triplet[1]:
                if rel_i == pred_triplet[2]:
                    gt_match[gt_idx] = pred_idx
                    pred_match[pred_idx] = gt_idx
                    break

    match_matrix = pred_match >= 0
    pred_to_gt = []
    for pred_idx in range(len(pred_triplets)):
        if match_matrix[pred_idx]:
            pred_to_gt.append(np.array([np.where(pred_match == pred_idx)[0][0]]))
        else:
            pred_to_gt.append(np.array([]))

    while len(pred_to_gt) < 100:
        pred_to_gt.append(np.array([]))

    return pred_to_gt


class SceneGraphEvaluation:
    def __init__(
        self,
        mode: str = 'sgdet',
        num_predicates: int = 56,
        iou_thresh: float = 0.5,
        detection_method: str = 'bbox',
    ):
        self.mode = mode
        self.num_predicates = num_predicates
        self.iou_thresh = iou_thresh
        self.detection_method = detection_method

        self.result_dict = {
            mode + "_recall": {20: [], 50: [], 100: []},
            mode + "_mean_recall": {20: 0.0, 50: 0.0, 100: 0.0},
        }

        self.gt_rels_list = []
        self.pred_rels_list = []

        if detection_method == 'bbox':
            self.generate_triplet = _triplet_bbox
            self.compute_pred_matches = _compute_pred_matches_bbox
        else:
            self.generate_triplet = _triplet_panseg
            self.compute_pred_matches = _compute_pred_matches_panseg

    def register_container(self, mode):
        self.result_dict[mode + "_recall"] = {20: [], 50: [], 100: []}
        self.result_dict[mode + "_mean_recall"] = {20: 0.0, 50: 0.0, 100: 0.0}

    def calculate_recall(
        self,
        gt_rels,
        pred_rels,
        gt_classes,
        pred_classes,
        gt_boxes=None,
        pred_boxes=None,
        gt_masks=None,
        pred_masks=None,
        rel_scores=None,
        obj_scores=None,
    ):
        if self.detection_method == 'bbox':
            gt_det_results = gt_boxes
            pred_det_results = pred_boxes
        else:
            gt_det_results = gt_masks
            pred_det_results = pred_masks

        if isinstance(gt_rels, torch.Tensor):
            gt_rels = gt_rels.cpu().numpy()
        if isinstance(gt_classes, torch.Tensor):
            gt_classes = gt_classes.cpu().numpy()
        if isinstance(pred_classes, torch.Tensor):
            pred_classes = pred_classes.cpu().numpy()
        if isinstance(pred_rels, torch.Tensor):
            pred_rels = pred_rels.cpu().numpy()

        gt_triplets, gt_triplet_det_results, _ = self.generate_triplet(
            gt_rels, gt_classes, gt_det_results
        )

        if rel_scores is not None:
            if isinstance(rel_scores, torch.Tensor):
                rel_scores = rel_scores.cpu().numpy()
            pred_rels_formatted = np.column_stack((
                pred_rels[:, 0],
                pred_rels[:, 1],
                1 + rel_scores[:, 1:].argmax(1)
            ))
        else:
            pred_rels_formatted = pred_rels

        pred_triplets, pred_triplet_det_results, pred_scores = self.generate_triplet(
            pred_rels_formatted, pred_classes, pred_det_results, rel_scores, obj_scores
        )

        if self.detection_method == 'bbox':
            pred_to_gt = self.compute_pred_matches(
                gt_triplets,
                pred_triplets,
                gt_triplet_det_results,
                pred_triplet_det_results,
                self.iou_thresh,
            )
        else:
            pred_to_gt = self.compute_pred_matches(
                gt_triplets,
                pred_triplets,
                gt_triplet_det_results,
                pred_triplet_det_results,
                self.iou_thresh,
            )

        for k in [20, 50, 100]:
            match = reduce(np.union1d, pred_to_gt[:k])
            rec_i = float(len(match)) / float(gt_rels.shape[0]) if gt_rels.shape[0] > 0 else 0.0
            self.result_dict[self.mode + "_recall"][k].append(rec_i)

        return pred_to_gt

    def calculate_mean_recall(
        self,
        gt_rels,
        pred_rels,
        gt_classes,
        pred_classes,
        gt_boxes=None,
        pred_boxes=None,
        rel_scores=None,
        obj_scores=None,
    ):
        if isinstance(gt_rels, torch.Tensor):
            gt_rels = gt_rels.cpu().numpy()
        if isinstance(gt_classes, torch.Tensor):
            gt_classes = gt_classes.cpu().numpy()
        if isinstance(pred_classes, torch.Tensor):
            pred_classes = pred_classes.cpu().numpy()
        if isinstance(pred_rels, torch.Tensor):
            pred_rels = pred_rels.cpu().numpy()
        if isinstance(rel_scores, torch.Tensor):
            rel_scores = rel_scores.cpu().numpy()

        pred_rel_labels = 1 + rel_scores[:, 1:].argmax(1) if rel_scores is not None else pred_rels[:, 2]
        pred_scores_per_rel = rel_scores[:, 1:].max(1) if rel_scores is not None else np.ones(len(pred_rels))

        if obj_scores is not None:
            if isinstance(obj_scores, torch.Tensor):
                obj_scores = obj_scores.cpu().numpy()
            obj_scores_per_rel = obj_scores[pred_rels[:, :2]].prod(1)
            pred_scores_per_rel = pred_scores_per_rel * obj_scores_per_rel

        pred_triplets = np.column_stack([
            pred_classes[pred_rels[:, 0]],
            pred_classes[pred_rels[:, 1]],
            pred_rel_labels
        ])

        gt_triplets = np.column_stack([
            gt_classes[gt_rels[:, 0]],
            gt_classes[gt_rels[:, 1]],
            gt_rels[:, 2]
        ])

        recall_sum = {20: 0.0, 50: 0.0, 100: 0.0}
        num_valid = 0

        for k in [20, 50, 100]:
            top_k_idx = np.argsort(-pred_scores_per_rel)[:k]
            top_k_pred_triplets = pred_triplets[top_k_idx]

            matched_count = 0
            for pred_triplet in top_k_pred_triplets:
                for gt_idx, gt_triplet in enumerate(gt_triplets):
                    if (pred_triplet[0] == gt_triplet[0] and
                        pred_triplet[1] == gt_triplet[1] and
                        pred_triplet[2] == gt_triplet[2]):
                        matched_count += 1
                        break

            if len(gt_triplets) > 0:
                recall_sum[k] += matched_count / len(gt_triplets)
                num_valid += 1

        if num_valid > 0:
            for k in [20, 50, 100]:
                mean_recall = recall_sum[k] / num_valid
                old = self.result_dict[self.mode + "_mean_recall"][k]
                self.result_dict[self.mode + "_mean_recall"][k] = (old + mean_recall) / 2

        return self.result_dict[self.mode + "_mean_recall"]

    def get_result(self):
        result = {}
        for key, values in self.result_dict.items():
            if isinstance(values, dict):
                for k, v in values.items():
                    result[f"{key}_{k}"] = np.mean(v) if len(v) > 0 else 0.0
            else:
                result[key] = values
        return result


class SGRecall:
    def __init__(self, result_dict, nogc_result_dict, nogc_thres_num, detection_method="bbox"):
        self.result_dict = result_dict
        self.nogc_result_dict = nogc_result_dict
        self.nogc_thres_num = nogc_thres_num
        self.detection_method = detection_method

        if detection_method == "bbox":
            self.generate_triplet = _triplet_bbox
            self.compute_pred_matches = _compute_pred_matches_bbox
        else:
            self.generate_triplet = _triplet_panseg
            self.compute_pred_matches = _compute_pred_matches_panseg

    def register_container(self, mode):
        self.result_dict[mode + "_recall"] = {20: [], 50: [], 100: []}

    def calculate_recall(
        self,
        global_container,
        local_container,
        mode,
    ):
        pred_rel_inds = local_container["pred_rel_inds"]
        rel_scores = local_container["rel_scores"]
        gt_rels = local_container["gt_rels"]
        gt_classes = local_container["gt_classes"]
        pred_classes = local_container["pred_classes"]
        obj_scores = local_container.get("obj_scores")

        if self.detection_method == "bbox":
            gt_det_results = local_container.get("gt_boxes")
            pred_det_results = local_container.get("pred_boxes")
        else:
            gt_det_results = local_container.get("gt_masks")
            pred_det_results = local_container.get("pred_masks")

        gt_triplets, gt_triplet_det_results, _ = self.generate_triplet(
            gt_rels, gt_classes, gt_det_results
        )

        pred_rels = np.column_stack((
            pred_rel_inds,
            1 + rel_scores[:, 1:].argmax(1)
        ))
        pred_scores = rel_scores[:, 1:].max(1)

        pred_triplets, pred_triplet_det_results, pred_triplet_scores = self.generate_triplet(
            pred_rels, pred_classes, pred_det_results, pred_scores, obj_scores
        )

        pred_to_gt = self.compute_pred_matches(
            gt_triplets,
            pred_triplets,
            gt_triplet_det_results,
            pred_triplet_det_results,
            global_container["iou_thrs"],
        )

        local_container["pred_to_gt"] = pred_to_gt

        for k in [20, 50, 100]:
            match = reduce(np.union1d, pred_to_gt[:k])
            rec_i = float(len(match)) / float(gt_rels.shape[0]) if gt_rels.shape[0] > 0 else 0.0
            self.result_dict[mode + "_recall"][k].append(rec_i)

        return local_container

    def generate_print_string(self, mode):
        result_str = "SGG eval: "
        for k, v in self.result_dict[mode + "_recall"].items():
            result_str += f" R @{k}: {np.mean(v):.4f}; "
        result_str += f" for mode={mode}"
        return result_str


class SGMeanRecall:
    def __init__(
        self,
        result_dict,
        nogc_result_dict,
        nogc_thres_num,
        num_predicates,
        ind_to_predicates,
        detection_method="bbox",
        print_detail=False,
    ):
        self.result_dict = result_dict
        self.nogc_result_dict = nogc_result_dict
        self.nogc_thres_num = nogc_thres_num
        self.num_predicates = num_predicates
        self.ind_to_predicates = ind_to_predicates
        self.detection_method = detection_method
        self.print_detail = print_detail

    def register_container(self, mode):
        self.result_dict[mode + "_mean_recall"] = {20: 0.0, 50: 0.0, 100: 0.0}
        self.nogc_result_dict[mode + "_mean_recall"] = {
            ngc: {20: 0.0, 50: 0.0, 100: 0.0} for ngc in self.nogc_thres_num
        }

    def calculate_mean_recall(self, mode):
        return self.result_dict[mode + "_mean_recall"]

    def collect_mean_recall_items(self, global_container, local_container, mode):
        return local_container

    def generate_print_string(self, mode, predicate_freq=None):
        result_str = "SGG eval Mean Recall: "
        for k, v in self.result_dict[mode + "_mean_recall"].items():
            result_str += f" mR @{k}: {v:.4f}; "
        result_str += f" for mode={mode}"
        return result_str


class SceneGraphEvaluator:
    def __init__(
        self,
        mode: str = 'sgdet',
        num_predicates: int = 56,
        iou_thresh: float = 0.5,
        detection_method: str = 'bbox',
        recall_k_values: List[int] = [20, 50, 100],
    ):
        self.mode = mode
        self.num_predicates = num_predicates
        self.iou_thresh = iou_thresh
        self.detection_method = detection_method
        self.recall_k_values = recall_k_values

        self.groundtruths = []
        self.predictions = []

    def update(
        self,
        gt_rels,
        pred_rels,
        gt_classes,
        pred_classes,
        gt_boxes=None,
        pred_boxes=None,
        gt_masks=None,
        pred_masks=None,
        rel_scores=None,
        obj_scores=None,
    ):
        self.groundtruths.append({
            'rels': gt_rels,
            'classes': gt_classes,
            'boxes': gt_boxes,
            'masks': gt_masks,
        })

        self.predictions.append({
            'rel_pair_idxes': pred_rels,
            'rel_dists': rel_scores,
            'labels': pred_classes,
            'refine_bboxes': pred_boxes,
            'masks': pred_masks,
        })

    def compute(self):
        evaluator = SceneGraphEvaluation(
            mode=self.mode,
            num_predicates=self.num_predicates,
            iou_thresh=self.iou_thresh,
            detection_method=self.detection_method,
        )
        evaluator.register_container(self.mode)

        all_results = {}

        for gt, pred in zip(self.groundtruths, self.predictions):
            gt_rels = gt['rels']
            pred_rels = pred['rel_pair_idxes']
            gt_classes = gt['classes']
            pred_classes = pred['labels']
            gt_boxes = gt.get('boxes')
            pred_boxes = pred.get('refine_bboxes')
            rel_scores = pred.get('rel_dists')
            obj_scores = pred.get('obj_scores') if 'obj_scores' in pred else None

            if pred_boxes is not None and len(pred_boxes) > 0:
                obj_scores = pred_boxes[:, -1] if pred_boxes.shape[1] > 4 else None

            evaluator.calculate_recall(
                gt_rels=gt_rels,
                pred_rels=pred_rels,
                gt_classes=gt_classes,
                pred_classes=pred_classes,
                gt_boxes=gt_boxes,
                pred_boxes=pred_boxes,
                rel_scores=rel_scores,
                obj_scores=obj_scores,
            )

            if rel_scores is not None:
                evaluator.calculate_mean_recall(
                    gt_rels=gt_rels,
                    pred_rels=pred_rels,
                    gt_classes=gt_classes,
                    pred_classes=pred_classes,
                    gt_boxes=gt_boxes,
                    pred_boxes=pred_boxes,
                    rel_scores=rel_scores,
                    obj_scores=obj_scores,
                )

        all_results.update(evaluator.get_result())

        return all_results

    def reset(self):
        self.groundtruths = []
        self.predictions = []


def evaluate_scene_graph(
    groundtruths: List[Dict],
    predictions: List[Dict],
    mode: str = 'sgdet',
    num_predicates: int = 56,
    iou_thresh: float = 0.5,
    detection_method: str = 'bbox',
):
    evaluator = SceneGraphEvaluator(
        mode=mode,
        num_predicates=num_predicates,
        iou_thresh=iou_thresh,
        detection_method=detection_method,
    )

    for gt, pred in zip(groundtruths, predictions):
        evaluator.update(
            gt_rels=gt.get('rels', gt.get('gt_rels')),
            pred_rels=pred.get('rel_pair_idxes', pred.get('pred_rels')),
            gt_classes=gt.get('classes', gt.get('labels')),
            pred_classes=pred.get('labels', pred.get('pred_classes')),
            gt_boxes=gt.get('boxes', gt.get('gt_boxes')),
            pred_boxes=pred.get('refine_bboxes', pred.get('pred_boxes')),
            rel_scores=pred.get('rel_dists', pred.get('rel_scores')),
            obj_scores=pred.get('obj_scores'),
        )

    return evaluator.compute()
