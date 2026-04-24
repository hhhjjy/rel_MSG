"""
统一训练/评估调度器

提供 get_stage_runner() 函数，根据 config['experiment_stage'] 自动选择:
- 前向传播函数
- 损失计算函数
- 评估函数
"""

import torch


class StageRunner(object):
    """
    阶段运行器: 封装特定阶段的训练/评估逻辑
    """

    def __init__(self, model, config, device='cuda'):
        self.model = model
        self.config = config
        self.device = device
        self.stage = config.get('experiment_stage', config.get('stage', 'step1'))

        # 验证 stage
        valid_stages = ['step1', 'step2', 'step3', 'step4', 'amosg']
        if self.stage not in valid_stages:
            raise ValueError(f"Unknown stage: {self.stage}. Must be one of {valid_stages}")

    def forward(self, images, bboxes_infos):
        """统一前向入口"""
        return self.model(images, bboxes_infos)

    def compute_loss(self, outputs, targets, loss_weights=None):
        """统一损失计算入口"""
        return self.model.compute_loss(outputs, targets, loss_weights)

    def get_stage_name(self):
        """获取阶段名称"""
        stage_names = {
            'step1': 'AoMSG Baseline',
            'amosg': 'AoMSG Baseline',
            'step2': 'Alternating Attention',
            'step3': 'Learnable Queries',
            'step4': 'Scene Graph Head',
        }
        return stage_names.get(self.stage, 'Unknown')

    def get_model_components(self):
        """获取当前阶段使用的模型组件"""
        components = {
            'feature_extractor': True,
            'query_decoder': self.stage in ['step2', 'step3', 'step4'],
            'edge_heads': self.stage in ['step2', 'step3', 'step4'],
            'scene_graph_head': self.stage == 'step4',
            'learnable_queries': self.stage in ['step3', 'step4'],
            'alternating_attention': self.stage == 'step2',
        }
        return components

    def log_stage_info(self, logger=None):
        """记录阶段信息"""
        info = f"Stage: {self.stage} ({self.get_stage_name()})"
        components = self.get_model_components()
        info += f"\n  Active components: {', '.join([k for k, v in components.items() if v])}"

        if logger is not None:
            logger.info(info)
        else:
            print(info)

        return info


def get_stage_runner(model, config, device='cuda'):
    """
    获取阶段运行器

    Args:
        model: RelationalMSG 模型实例
        config: 配置字典
        device: 计算设备

    Returns:
        StageRunner: 阶段运行器实例
    """
    return StageRunner(model, config, device)


def dispatch_train_step(runner, batch, optimizer, scaler=None):
    """
    统一的训练步骤调度

    Args:
        runner: StageRunner 实例
        batch: 数据批次
        optimizer: 优化器
        scaler: GradScaler (混合精度训练)

    Returns:
        loss: 标量损失值
        logs: 日志字典
    """
    model = runner.model
    model.train()

    images = batch['image'].to(runner.device)

    # 构建 additional_info
    additional_info = {
        'gt_bbox': batch['bbox'].type(torch.FloatTensor).to(runner.device),
        'obj_label': batch['obj_label'].to(runner.device),
        'obj_idx': batch['obj_idx'].to(runner.device),
        'mask': batch['mask'].to(runner.device),
        'scene_num': batch.get('scene_num', 1),
    }

    # 前向传播
    if scaler is not None:
        with torch.cuda.amp.autocast():
            outputs = runner.forward(images, additional_info)
            loss, logs = runner.compute_loss(outputs, additional_info)
    else:
        outputs = runner.forward(images, additional_info)
        loss, logs = runner.compute_loss(outputs, additional_info)

    # 反向传播
    optimizer.zero_grad()
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    return loss.item(), logs


def dispatch_eval_step(runner, batch):
    """
    统一的评估步骤调度

    Args:
        runner: StageRunner 实例
        batch: 数据批次

    Returns:
        outputs: 模型输出
        loss: 损失值 (如果有)
        logs: 日志字典
    """
    model = runner.model
    model.eval()

    images = batch['image'].to(runner.device)

    additional_info = {
        'gt_bbox': batch['bbox'].type(torch.FloatTensor).to(runner.device),
        'obj_label': batch['obj_label'].to(runner.device),
        'obj_idx': batch['obj_idx'].to(runner.device),
        'mask': batch['mask'].to(runner.device),
        'scene_num': batch.get('scene_num', 1),
    }

    with torch.no_grad():
        outputs = runner.forward(images, additional_info)
        loss, logs = runner.compute_loss(outputs, additional_info)

    return outputs, loss.item(), logs
