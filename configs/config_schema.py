"""
统一配置结构定义与验证

统一字段:
- experiment_stage: 'step1' | 'step2' | 'step3' | 'step4'
- feature_encoder_type: 'aomsg' | 'vggt'
- query_type: 'v1' | 'v2' | 'v3'
- graph_head_type: 'none' | 'scene_graph'
"""

import copy


# 默认配置模板
DEFAULT_CONFIG = {
    # 实验阶段 (核心字段)
    'experiment_stage': 'step1',  # 'step1' | 'step2' | 'step3' | 'step4'

    # 特征编码器类型
    'feature_encoder_type': 'aomsg',  # 'aomsg' | 'vggt'

    # Query 类型
    'query_type': 'v1',  # 'v1' (baseline) | 'v2' (alternating attention) | 'v3' (learnable)

    # 图头类型
    'graph_head_type': 'none',  # 'none' | 'scene_graph'

    # 模型维度
    'hidden_model_dim': 768,
    'num_obj_queries': 100,
    'num_place_queries': 10,
    'num_obj_classes': 18,
    'num_edge_types': 1,
    'num_views': 4,
    'num_bboxes_per_view': 20,

    # Backbone 配置
    'backbone': {
        'model_type': 'dinov2-base',
        'freeze': True,
        'weights': 'DEFAULT',
    },

    # 图像尺寸
    'model_image_size': (224, 224),

    # 场景图配置 (Step 4)
    'scene_graph': {
        'num_relations': 56,
        'num_rel_queries': 50,
        'use_matrix_learner': True,
        'num_rel_decoder_layers': 6,
        'num_rel_heads': 8,
        'dim_rel_feedforward': 2048,
    },

    # Learnable Queries 配置 (Step 3/4)
    'learnable_queries': {
        'cost_cls': 1.0,
        'cost_attn': 1.0,
        'weight_cls': 1.0,
        'weight_attn': 1.0,
        'weight_exist': 1.0,
        'weight_bbox': 0.0,
        'loss_mode': 'hybrid',  # 'pure' | 'hybrid'
        'weight_baseline': 1.0,
    },

    # Loss 权重
    'loss_weights': {
        'pr': 1.0,
        'obj': 1.0,
        'mean': 1.0,
        'tcr': 1.0,
        'loss_obj_exist': 1.0,
        'loss_obj_cls': 2.0,
        'loss_place_exist': 1.0,
        'loss_pp_edge': 1.0,
        'loss_po_edge': 1.0,
        'loss_sgg_rel': 1.0,
        'loss_sgg_match': 1.0,
        'loss_sparse': 0.5,
        'loss_box_query': 2.0,
    },

    # 训练配置
    'training': {
        'batch_size': 4,
        'num_epochs': 100,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'warmup_epochs': 5,
    },

    # 评估配置
    'evaluation': {
        'eval_interval': 5,
        'save_best': True,
        'metrics': ['recall@1', 'recall@5', 'recall@10'],
    },
}


# 阶段到配置字段的映射
STAGE_CONFIG_MAP = {
    'step1': {
        'experiment_stage': 'step1',
        'feature_encoder_type': 'aomsg',
        'query_type': 'v1',
        'graph_head_type': 'none',
    },
    'amosg': {
        'experiment_stage': 'step1',
        'feature_encoder_type': 'aomsg',
        'query_type': 'v1',
        'graph_head_type': 'none',
    },
    'step2': {
        'experiment_stage': 'step2',
        'feature_encoder_type': 'vggt',
        'query_type': 'v2',
        'graph_head_type': 'none',
    },
    'step3': {
        'experiment_stage': 'step3',
        'feature_encoder_type': 'vggt',
        'query_type': 'v3',
        'graph_head_type': 'none',
    },
    'step4': {
        'experiment_stage': 'step4',
        'feature_encoder_type': 'vggt',
        'query_type': 'v3',
        'graph_head_type': 'scene_graph',
    },
}


def build_config(user_config=None, stage=None):
    """
    构建完整配置，自动填充默认值并验证

    Args:
        user_config: 用户提供的配置字典
        stage: 实验阶段，如果提供则自动设置相关字段

    Returns:
        config: 完整的配置字典
    """
    config = copy.deepcopy(DEFAULT_CONFIG)

    if user_config is not None:
        # 递归更新配置
        _deep_update(config, user_config)

    # 如果指定了 stage，自动设置相关字段
    if stage is not None:
        stage = stage.lower()
        if stage in STAGE_CONFIG_MAP:
            stage_defaults = STAGE_CONFIG_MAP[stage]
            for k, v in stage_defaults.items():
                config[k] = v
        else:
            raise ValueError(f"Unknown stage: {stage}. Must be one of {list(STAGE_CONFIG_MAP.keys())}")

    # 兼容旧字段
    _backward_compatible(config)

    # 验证配置
    validate_config(config)

    return config


def validate_config(config):
    """验证配置合法性"""
    # 检查必要字段
    required_fields = ['experiment_stage', 'hidden_model_dim', 'num_obj_queries']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required config field: {field}")

    # 检查 stage 合法性
    valid_stages = ['step1', 'step2', 'step3', 'step4', 'amosg']
    if config['experiment_stage'] not in valid_stages:
        raise ValueError(f"Invalid experiment_stage: {config['experiment_stage']}. Must be one of {valid_stages}")

    # 检查 feature_encoder_type
    valid_encoders = ['aomsg', 'vggt']
    if config.get('feature_encoder_type') not in valid_encoders:
        raise ValueError(f"Invalid feature_encoder_type: {config.get('feature_encoder_type')}. Must be one of {valid_encoders}")

    # 检查 query_type
    valid_queries = ['v1', 'v2', 'v3']
    if config.get('query_type') not in valid_queries:
        raise ValueError(f"Invalid query_type: {config.get('query_type')}. Must be one of {valid_queries}")

    # 检查 graph_head_type
    valid_graphs = ['none', 'scene_graph']
    if config.get('graph_head_type') not in valid_graphs:
        raise ValueError(f"Invalid graph_head_type: {config.get('graph_head_type')}. Must be one of {valid_graphs}")

    # 检查维度一致性
    if config['num_obj_queries'] <= 0:
        raise ValueError("num_obj_queries must be positive")
    if config['num_place_queries'] <= 0:
        raise ValueError("num_place_queries must be positive")

    return True


def _deep_update(base_dict, update_dict):
    """递归更新字典"""
    for key, value in update_dict.items():
        if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
            _deep_update(base_dict[key], value)
        else:
            base_dict[key] = value


def _backward_compatible(config):
    """处理旧配置字段的兼容性"""
    # 兼容旧字段 'stage'
    if 'stage' in config and 'experiment_stage' not in config:
        config['experiment_stage'] = config['stage']

    # 兼容旧字段 'forward_version'
    if 'forward_version' in config and 'experiment_stage' not in config:
        config['experiment_stage'] = config['forward_version']

    # 兼容旧字段 'feature_refine_method'
    if 'feature_refine_method' in config and 'feature_encoder_type' not in config:
        method = config['feature_refine_method']
        if method == 'aomsg':
            config['feature_encoder_type'] = 'aomsg'
        else:
            config['feature_encoder_type'] = 'vggt'

    # 兼容旧字段 'use_scene_graph'
    if 'use_scene_graph' in config and 'graph_head_type' not in config:
        config['graph_head_type'] = 'scene_graph' if config['use_scene_graph'] else 'none'

    # 确保 stage 存在 (用于模型内部)
    config['stage'] = config.get('experiment_stage', 'step1')

    # 确保 feature_refine_method 存在 (用于 FeatureExtractor)
    if 'feature_refine_method' not in config:
        config['feature_refine_method'] = 'aomsg' if config.get('feature_encoder_type') == 'aomsg' else 'vggt'

    # 确保 use_scene_graph 存在 (用于 SceneGraphHead)
    if 'use_scene_graph' not in config:
        config['use_scene_graph'] = config.get('graph_head_type') == 'scene_graph'


def get_stage_config(stage):
    """获取指定阶段的默认配置"""
    return build_config(stage=stage)


def print_config(config, indent=0):
    """打印配置 (用于调试)"""
    for key, value in config.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key}:")
            print_config(value, indent + 1)
        else:
            print("  " * indent + f"{key}: {value}")
