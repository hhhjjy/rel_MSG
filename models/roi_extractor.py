import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align


def enlarge_boxes(boxes, image_size=(224, 224), scale=1.1):
    """
    Enlarge bounding boxes, and keep the enlarged box withint the image frame
    boxes are: tensor (N, 4), (x1, y1, x2, y2)
    image size: a tuple, here we use 224 x 224 since it has been preprocessed
    """
    box_centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    box_sizes = boxes[:, 2:] - boxes[:, :2]
    new_box_sizes = box_sizes * scale

    new_boxes = torch.zeros_like(boxes)
    new_boxes[:, :2] = torch.max(box_centers - new_box_sizes / 2, torch.tensor([0, 0], dtype=boxes.dtype, device=boxes.device))
    new_boxes[:, 2:] = torch.min(box_centers + new_box_sizes / 2, torch.tensor(image_size, dtype=boxes.dtype, device=boxes.device))
    new_boxes[:, 2:] = torch.max(new_boxes[:, 2:], new_boxes[:, :2] + 1e-6)

    return new_boxes


def random_shift_boxes(bboxes, image_size=(224,224), shift_ratio=0.2):
    """
    random shift `shift_ratio` of a bound box
    boxes are: tensor (N, 4), (x1, y1, x2, y2)
    image size: a tuple, here we use 224 x 224 since it has been preprocessed
    """
    N = bboxes.size(0)
    if N == 0:
        return bboxes
    bbox_widths = bboxes[:, 2] - bboxes[:, 0]
    bbox_heights = bboxes[:, 3] - bboxes[:, 1]

    max_shifts_x = (bbox_widths * shift_ratio).int()
    max_shifts_y = (bbox_heights * shift_ratio).int()

    devc = bboxes.device

    shifts_x = torch.stack([torch.randint(-max_shift, max_shift + 1, (1,), device=devc) for max_shift in max_shifts_x])
    shifts_y = torch.stack([torch.randint(-max_shift, max_shift + 1, (1,), device=devc) for max_shift in max_shifts_y])

    bboxes[:, [0, 2]] += shifts_x
    bboxes[:, [1, 3]] += shifts_y

    width, height = image_size
    bboxes[:, [0, 2]] = torch.clamp(bboxes[:, [0, 2]], min=0, max=width)
    bboxes[:, [1, 3]] = torch.clamp(bboxes[:, [1, 3]], min=0, max=height)

    return bboxes


class RoIExtractor(nn.Module):
    """
    ROI特征提取器，从图像特征中提取bbox对应的region-level特征
    
    输入: image-level features + bbox proposals
    输出: bbox features
    """

    def __init__(self, feat_dim=256, roi_size=1, image_size=(224, 224)):
        super().__init__()
        self.feat_dim = feat_dim
        self.roi_size = roi_size
        self.image_size = image_size

    def forward(self, img_feats, bboxes, bbox_masks=None, training=False):
        """
        Args:
            img_feats: [B, V, L, C] 图像空间特征
            bboxes: [B, V, N, 4], N为每个视角的bbox位置, (x1, y1, x2, y2)
            bbox_masks: [B, V, N], 标记有效的bbox
            training: bool, 是否训练模式

        Returns:
            bbox_feats: [B, V, N, C] 提取的bbox特征
        """
        B, V, L, C = img_feats.shape
        H = W = int(L ** 0.5)
        assert H * W == L, f"L={L} 不能还原成方形特征图"

        # [B, V, L, C] -> [B, V, H, W, C] -> [B, V, C, H, W] -> [B*V, C, H, W]
        x = img_feats.reshape(B, V, H, W, C)
        x = x.permute(0, 1, 4, 2, 3).contiguous()
        x = x.reshape(B * V, C, H, W)

        feature_dim = C
        output_size = (1, 1)

        # 和参考代码一致：特征图相对原图的缩放比例
        spatial_scale = H * 1.0 / 224

        num_boxes_per_view = []
        list_boxes = []

        for b in range(B):
            for v in range(V):
                boxes = bboxes[b, v]   # [N, 4]

                if bbox_masks is not None:
                    valid_mask = bbox_masks[b, v].bool()
                    boxes = boxes[valid_mask]

                if boxes.numel() > 0:
                    boxes = enlarge_boxes(boxes, scale=1.1)
                    if training:
                        boxes = random_shift_boxes(boxes, shift_ratio=0.2)

                    # 可选：裁剪到图像范围内
                    boxes = boxes.clone()
                    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, 224)
                    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, 224)

                num_boxes_per_view.append(boxes.size(0))
                list_boxes.append(boxes)

        total_boxes = sum(num_boxes_per_view)

        if total_boxes > 0:
            # roi_align 输入:
            # x: [B*V, C, H, W]
            # list_boxes: len=B*V, each [num_boxes_i, 4]
            all_bbox_feats = roi_align(
                x,
                list_boxes,
                output_size=output_size,
                spatial_scale=spatial_scale
            )  # [total_boxes, C, 1, 1]
            all_bbox_feats = all_bbox_feats.squeeze(-1).squeeze(-1)  # [total_boxes, C]
        else:
            all_bbox_feats = x.new_zeros((0, feature_dim))

        # 还原回 [B, V, N, C]
        N = bboxes.shape[2]
        bbox_feats = x.new_zeros((B, V, N, feature_dim))

        start = 0
        for b in range(B):
            for v in range(V):
                idx = b * V + v
                count = num_boxes_per_view[idx]

                if count == 0:
                    continue

                cur_feats = all_bbox_feats[start:start + count]  # [count, C]

                if bbox_masks is not None:
                    valid_mask = bbox_masks[b, v].bool()
                    bbox_feats[b, v, valid_mask] = cur_feats
                else:
                    bbox_feats[b, v, :count] = cur_feats

                start += count

        return bbox_feats