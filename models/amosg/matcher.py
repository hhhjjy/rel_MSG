# detection bounding box matcher class from DeTR at: 
# https://github.com/facebookresearch/detr/blob/main/models/matcher.py
# with modification:
# 1) outputs is not a dict but but a list of detections. len(detections) == batch size, each is detection is
# a dictionary of {'boxes', 'labels', 'scores'}, which is the same format as the groundtruth
# cls labels are not used for matching

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torchvision.ops.boxes import box_area
# from detr's code, for computing geIoU
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a list of outputs (len(outputs) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_predict_boxes] (where num_predict_boxes is the number of ground-truth
                           objects in the output) containing the class labels
                 "boxes": Tensor of dim [num_predict_boxes, 4] containing the predict box coordinates
                 "scores": Tensor of dim [num_predict_boxes] containing the prediction logits

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        # # We flatten to compute the cost matrices in a batch
        # out_bbox = torch.cat([v["boxes"] for v in outputs])
        # out_sizes = [len(v["boxes"]) for v in outputs]

        # # Also concat the target labels and boxes
        # tgt_ids = torch.cat([v["labels"] for v in targets])
        # tgt_bbox = torch.cat([v["boxes"] for v in targets])
        # tgt_sizes = [len(v["boxes"]) for v in targets]
        
        # handle each data separately
        indices = []
        for bi in range(len(outputs)):
            out_bbox = outputs[bi]["boxes"]
            out_probs = outputs[bi]["scores"]

            tgt_bbox = targets[bi]["boxes"]
            tgt_ids = targets[bi]["labels"]

            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

            cost_giou = -generalized_box_iou(out_bbox, tgt_bbox)

            cost_mat = self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
            
            pred_indices, tgt_indices = linear_sum_assignment(cost_mat.cpu())
            indices.append(
                (torch.as_tensor(pred_indices, dtype=torch.int64), torch.as_tensor(tgt_indices, dtype=torch.int64))
            )
        return indices



# def build_matcher(args):
#     return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)

# if __name__ == "__main__":
#     # do sanity check
#     matcher = HungarianMatcher()
#     det1 = {'boxes': torch.as_tensor([[48, 110, 236, 190], [100, 24, 184, 112]], dtype=torch.float64), 'labels': torch.as_tensor([1,2]), 'scores': torch.as_tensor([1.0, 1.0])}
#     det2 = {'boxes': torch.as_tensor([[64, 32, 226, 190]],dtype=torch.float64), 'labels': torch.as_tensor([1,]), 'scores': torch.as_tensor([1.0,])}
#     detections = [det1, det2]
#     matches = matcher(detections, detections)
#     print(matches)