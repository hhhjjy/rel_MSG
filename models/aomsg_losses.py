
# AOMSG 损失函数模块
# 参考: reference/MSG/models/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------- 监督信号生成 ----------------------------- #
def get_match_idx(match_indices, info, N):
    B = len(match_indices)
    total_reorderd_indx = []
    for bi in range(B):
        pred_indices = match_indices[bi][0]
        gt_indices = match_indices[bi][1]
        ori_obj_idx = info['obj_idx'][bi]
        reordered_obj_ids = torch.full((N,), -1, dtype=ori_obj_idx.dtype, device=ori_obj_idx.device)
        reordered_obj_ids[pred_indices] = ori_obj_idx[gt_indices]
        total_reorderd_indx.append(reordered_obj_ids)
    total_reorderd_indx = torch.cat(total_reorderd_indx, dim=0)
    return total_reorderd_indx


def get_association_sv(total_reorderd_indx):
    expanded = total_reorderd_indx.unsqueeze(1)
    batch_equality = (expanded == expanded.transpose(0, 1)).int()
    mask = (total_reorderd_indx == -1).int() * -1
    expanded_mask = mask.unsqueeze(1)
    batch_mask = (expanded_mask + expanded_mask.transpose(0, 1) == 0).int()
    batch_supervision_matrix = (batch_equality * batch_mask).float()
    return batch_supervision_matrix, batch_mask


# ---------------------------- 损失函数模块 ----------------------- #
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1, learnable=False):
        super(InfoNCELoss, self).__init__()
        self.temp = nn.Parameter(torch.ones(1) * temperature)
        if learnable:
            self.temp.requires_grad = True
        else:
            self.temp.requires_grad = False

    def forward(self, predictions, supervision_matrix_masked, mask):
        BN, _ = predictions.shape
        predictions_masked = predictions * mask
        logits = predictions_masked / self.temp
        positive_mask = supervision_matrix_masked.to(dtype=torch.bool)
        positive_logits = logits[positive_mask]
        logsumexp_denominator = torch.logsumexp(logits, dim=-1) + 1e-9
        losses = positive_logits - logsumexp_denominator[positive_mask.nonzero(as_tuple=True)[0]]
        valid_entries = mask.sum()
        loss = -torch.sum(losses) / valid_entries if valid_entries > 0 else torch.tensor(0.0).to(predictions.device)
        return loss


class MaskBCELoss(nn.Module):
    def __init__(self, pos_weight=5.0):
        super(MaskBCELoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([pos_weight]))

    def forward(self, predictions, supervision, mask):
        valid_object_predictions = predictions
        loss = self.bce_loss(valid_object_predictions, supervision)
        masked_loss = loss * mask
        total_loss = masked_loss.sum() / (mask.sum() + 1e-9)
        return total_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, predictions, supervision, mask):
        bceloss = self.bce_loss(predictions, supervision)
        pt = torch.exp(-bceloss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bceloss
        masked_loss = focal_loss * mask
        total_loss = masked_loss.sum() / (mask.sum() + 1e-9)
        return total_loss


class MaskMetricLoss(nn.Module):
    def __init__(self):
        super(MaskMetricLoss, self).__init__()
        self.loss_fn = nn.MSELoss(reduction='none')

    def forward(self, predictions, supervision, mask):
        valid_object_predictions = (predictions + 1.0) / 2 * mask
        loss = self.loss_fn(valid_object_predictions, supervision)
        masked_loss = loss * mask
        total_loss = masked_loss.sum() / (mask.sum() + 1e-9)
        return total_loss


class MeanSimilarityLoss(nn.Module):
    def __init__(self):
        super(MeanSimilarityLoss, self).__init__()

    def mean_reg(self, means):
        means_norm = torch.nn.functional.normalize(means, p=2, dim=1)
        cos_sim = torch.mm(means_norm, means_norm.t())
        mean_cos_sim = off_diagonal(cos_sim).sum() / (means.shape[0] ** 2 - means.shape[0] + 1e-9)
        return mean_cos_sim

    def forward(self, embeddings, flatten_idx):
        h = embeddings.size(1)
        valid_entry = (flatten_idx != -1).float()

        unique_ids = flatten_idx.unique()
        unique_ids = unique_ids[unique_ids != -1]

        embeddings_sum = torch.zeros((len(unique_ids), h), dtype=torch.float).to(embeddings.device)
        counts = torch.zeros(len(unique_ids), dtype=torch.float).to(embeddings.device)
        for i, unique_id in enumerate(unique_ids):
            id_mask = (flatten_idx == unique_id)
            embeddings_sum[i] = embeddings[id_mask].sum(dim=0)
            counts[i] = id_mask.sum()

        embeddings_mean = embeddings_sum / counts.unsqueeze(1)
        embeddings_mean_expanded = torch.zeros_like(embeddings)
        for i, unique_id in enumerate(unique_ids):
            embeddings_mean_expanded[flatten_idx == unique_id] = embeddings_mean[i]

        cos_distance = 1 - F.cosine_similarity(embeddings, embeddings_mean_expanded)
        avg_dis = (cos_distance * valid_entry).sum() / (valid_entry.sum() + 1e-9)
        mean_dis = self.mean_reg(embeddings_mean)

        return avg_dis, mean_dis, counts, embeddings_mean


class TotalCodingRate(nn.Module):
    def __init__(self, eps=0.01):
        super(TotalCodingRate, self).__init__()
        self.eps = eps

    def compute_discrimn_loss(self, W):
        p, m = W.shape
        I = torch.eye(p, device=W.device)
        scalar = p * m / ((m + 1e-5) * (m + 1e-5) * self.eps)
        logdet = torch.logdet(I + scalar * W.matmul(W.T))
        return logdet / 2.

    def forward(self, embeddings, idx):
        validity_mask = idx != -1
        X = embeddings[validity_mask]
        nX = F.normalize(X)
        return - self.compute_discrimn_loss(nX.T)


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

