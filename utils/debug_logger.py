"""
日志与调试工具

记录:
- attention 分布
- query usage 统计
- entropy (分类/存在性)
- 梯度统计
- 损失分解
"""

import torch
import torch.nn.functional as F
import json
from collections import defaultdict


class DebugLogger(object):
    """
    调试日志记录器

    用于记录模型训练/评估过程中的可解释性指标
    """

    def __init__(self, log_interval=100):
        self.log_interval = log_interval
        self.step_count = 0
        self.history = defaultdict(list)

    def log_attention(self, object_attn, place_attn=None, prefix=''):
        """
        记录 attention 统计信息

        Args:
            object_attn: [B, Q, M] object attention
            place_attn: [B, P, P] or [B, P, O] place attention (可选)
            prefix: 日志前缀
        """
        stats = {}

        # Object attention stats
        if object_attn is not None:
            attn_flat = object_attn.view(-1, object_attn.shape[-1])
            stats[f'{prefix}obj_attn_mean'] = attn_flat.mean().item()
            stats[f'{prefix}obj_attn_std'] = attn_flat.std().item()
            stats[f'{prefix}obj_attn_max'] = attn_flat.max().item()
            stats[f'{prefix}obj_attn_min'] = attn_flat.min().item()

            # 每个 query 的 attention entropy (衡量 query 是否专注于特定 bbox)
            attn_probs = F.softmax(attn_flat, dim=-1)
            entropy = -(attn_probs * torch.log(attn_probs + 1e-8)).sum(dim=-1)
            stats[f'{prefix}obj_attn_entropy_mean'] = entropy.mean().item()
            stats[f'{prefix}obj_attn_entropy_std'] = entropy.std().item()

        # Place attention stats
        if place_attn is not None:
            p_flat = place_attn.view(-1, place_attn.shape[-1])
            stats[f'{prefix}place_attn_mean'] = p_flat.mean().item()
            stats[f'{prefix}place_attn_std'] = p_flat.std().item()

        self._add_to_history(stats)
        return stats

    def log_query_usage(self, object_exist_logits, object_cls_logits=None, prefix=''):
        """
        记录 query usage 统计

        Args:
            object_exist_logits: [B, Q] object 存在性 logits
            object_cls_logits: [B, Q, C] object 分类 logits (可选)
            prefix: 日志前缀
        """
        stats = {}

        if object_exist_logits is not None:
            exist_probs = torch.sigmoid(object_exist_logits)

            # 存在性统计
            stats[f'{prefix}exist_prob_mean'] = exist_probs.mean().item()
            stats[f'{prefix}exist_prob_std'] = exist_probs.std().item()

            # 高置信度 query 数量 (exist_prob > 0.5)
            high_conf = (exist_probs > 0.5).float().sum(dim=-1)  # [B]
            stats[f'{prefix}num_active_queries_mean'] = high_conf.mean().item()
            stats[f'{prefix}num_active_queries_max'] = high_conf.max().item()

            # 存在性 entropy
            p = exist_probs
            entropy = -(p * torch.log(p + 1e-8) + (1-p) * torch.log(1-p + 1e-8))
            stats[f'{prefix}exist_entropy_mean'] = entropy.mean().item()

        # 分类 entropy (衡量分类置信度)
        if object_cls_logits is not None:
            cls_probs = F.softmax(object_cls_logits, dim=-1)
            cls_entropy = -(cls_probs * torch.log(cls_probs + 1e-8)).sum(dim=-1)
            stats[f'{prefix}cls_entropy_mean'] = cls_entropy.mean().item()
            stats[f'{prefix}cls_entropy_std'] = cls_entropy.std().item()

            # Top-1 分类置信度
            top1_conf = cls_probs.max(dim=-1)[0]
            stats[f'{prefix}cls_top1_conf_mean'] = top1_conf.mean().item()

        self._add_to_history(stats)
        return stats

    def log_gradients(self, model, prefix=''):
        """
        记录梯度统计

        Args:
            model: nn.Module
            prefix: 日志前缀
        """
        stats = {}

        total_norm = 0.0
        param_norms = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                param_norms[name] = param_norm
                total_norm += param_norm ** 2

        total_norm = total_norm ** 0.5
        stats[f'{prefix}grad_total_norm'] = total_norm

        # 记录关键模块的梯度
        for module_name in ['query_decoder', 'edge_heads', 'scene_graph_head']:
            module_norm = sum(
                v ** 2 for k, v in param_norms.items()
                if module_name in k
            ) ** 0.5
            if module_norm > 0:
                stats[f'{prefix}grad_{module_name}_norm'] = module_norm

        self._add_to_history(stats)
        return stats

    def log_loss_decomposition(self, loss_dict, prefix=''):
        """
        记录损失分解

        Args:
            loss_dict: 损失字典
            prefix: 日志前缀
        """
        stats = {}
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            stats[f'{prefix}loss_{k}'] = v

        self._add_to_history(stats)
        return stats

    def log_matching(self, match_indices, num_queries, prefix=''):
        """
        记录匹配统计

        Args:
            match_indices: list of (pred_idx, tgt_idx) tuples
            num_queries: int, query 数量
            prefix: 日志前缀
        """
        stats = {}

        total_matched = sum(len(m[0]) for m in match_indices)
        num_batches = len(match_indices)
        stats[f'{prefix}matched_queries_mean'] = total_matched / num_batches if num_batches > 0 else 0
        stats[f'{prefix}matched_queries_ratio'] = total_matched / (num_batches * num_queries) if num_batches > 0 else 0

        self._add_to_history(stats)
        return stats

    def _add_to_history(self, stats):
        """添加到历史记录"""
        for k, v in stats.items():
            self.history[k].append(v)

    def get_summary(self, window=100):
        """
        获取最近 window 步的统计摘要

        Returns:
            summary: dict of mean/std for each metric
        """
        summary = {}
        for k, values in self.history.items():
            if len(values) == 0:
                continue
            recent = values[-window:]
            summary[f'{k}_mean'] = sum(recent) / len(recent)
            if len(recent) > 1:
                mean = summary[f'{k}_mean']
                variance = sum((x - mean) ** 2 for x in recent) / len(recent)
                summary[f'{k}_std'] = variance ** 0.5
        return summary

    def print_summary(self, window=100):
        """打印统计摘要"""
        summary = self.get_summary(window)
        print("\n" + "=" * 50)
        print(f"Debug Summary (last {window} steps)")
        print("=" * 50)
        for k in sorted(summary.keys()):
            print(f"  {k}: {summary[k]:.6f}")
        print("=" * 50)

    def save_summary(self, path, window=100):
        """保存统计摘要到 JSON"""
        summary = self.get_summary(window)
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2)

    def reset(self):
        """重置历史记录"""
        self.history.clear()
        self.step_count = 0


class QueryUsageTracker(object):
    """
    Query 使用追踪器

    追踪每个 query 在训练过程中的使用情况
    """

    def __init__(self, num_queries):
        self.num_queries = num_queries
        self.query_usage_count = torch.zeros(num_queries, dtype=torch.long)
        self.query_match_count = torch.zeros(num_queries, dtype=torch.long)

    def update(self, match_indices):
        """
        更新 query 使用统计

        Args:
            match_indices: list of (pred_idx, tgt_idx) tuples
        """
        for pred_idx, _ in match_indices:
            self.query_usage_count[pred_idx] += 1

    def get_stats(self):
        """获取 query 使用统计"""
        total_usage = self.query_usage_count.sum().item()
        if total_usage == 0:
            return {
                'total_usage': 0,
                'active_queries': 0,
                'usage_entropy': 0.0,
            }

        usage_probs = self.query_usage_count.float() / total_usage
        active = (self.query_usage_count > 0).sum().item()

        # 使用 entropy 衡量 query 使用均匀度
        entropy = -(usage_probs * torch.log(usage_probs + 1e-8)).sum().item()

        return {
            'total_usage': total_usage,
            'active_queries': active,
            'inactive_queries': self.num_queries - active,
            'usage_entropy': entropy,
            'most_used_query': self.query_usage_count.argmax().item(),
            'least_used_query': self.query_usage_count.argmin().item(),
        }

    def reset(self):
        """重置统计"""
        self.query_usage_count.zero_()
        self.query_match_count.zero_()


def log_model_structure(model, logger=None):
    """
    记录模型结构信息

    Args:
        model: nn.Module
        logger: 日志记录器 (可选)
    """
    info = []
    info.append("=" * 50)
    info.append("Model Structure")
    info.append("=" * 50)

    total_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params

    info.append(f"Total parameters: {total_params:,}")
    info.append(f"Trainable parameters: {trainable_params:,}")
    info.append(f"Non-trainable parameters: {total_params - trainable_params:,}")

    # 按模块统计
    module_params = {}
    for name, module in model.named_children():
        module_params[name] = sum(p.numel() for p in module.parameters())

    info.append("\nParameters by module:")
    for name, num in sorted(module_params.items(), key=lambda x: -x[1]):
        info.append(f"  {name}: {num:,}")

    info.append("=" * 50)

    output = "\n".join(info)
    if logger is not None:
        logger.info(output)
    else:
        print(output)

    return output
