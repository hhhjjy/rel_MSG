"""
Query-aware Evaluation Script

用于评估 step2/3/4 的 query-based 模型输出。
与旧版 eval.py 的区别：
- 使用 QueryMapperHandler 替代 TopoMapperHandler
- 支持 query-level 存在性过滤
- 保留 Evaluator（指标层）不变
"""

import yaml
import json
import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import logging

from util.config_utils import get_configs
from util.transforms import get_transform
from util.box_utils import BBoxReScaler
from util.monitor import TrainingMonitor
from torch.utils.data import DataLoader
from datasets.dataset import (
    AppleDataHandler, VideoDataset, arkit_collate_fn,
    VideoDataset_3RScan, VideoDataset_Replica, VideoDataset_3RScan_split
)

from run.query_mapper import QueryMapperHandler
from run.evaluator import Evaluator
from models.relational_msg import RelationalMSG
from util.checkpointing import load_checkpoint


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_logger(output_dir, output_file):
    """Create logger for evaluation records"""
    logfile = output_file.split('.')[0] + "_query_eval.log"
    print('create logger:', logfile)
    logpath = os.path.join(output_dir, logfile)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(logpath, mode='w')]
    )
    logger = logging.getLogger(__name__)
    return logger


def get_forward_method(model, stage):
    """
    根据 stage 获取对应的 forward 方法
    
    Args:
        model: RelationalMSG 模型
        stage: str, one of ['step1', 'step2', 'step3', 'step4', 'amosg']
    
    Returns:
        callable: forward 方法
    """
    stage_map = {
        'step1': model.forward_amosg,
        'amosg': model.forward_amosg,
        'step2': model.forward_step2,
        'step3': model.forward_step3,
        'step4': model.forward_step4,
    }
    
    if stage not in stage_map:
        raise ValueError(f"Unknown stage: {stage}. Must be one of {list(stage_map.keys())}")
    
    return stage_map[stage]


def get_loss_method(model, stage):
    """
    根据 stage 获取对应的 loss 计算方法
    
    Args:
        model: RelationalMSG 模型
        stage: str, one of ['step1', 'step2', 'step3', 'step4', 'amosg']
    
    Returns:
        callable: loss 计算方法
    """
    stage_map = {
        'step1': model.compute_loss_amosg,
        'amosg': model.compute_loss_amosg,
        'step2': model.compute_loss_step2,
        'step3': model.compute_loss_step3,
        'step4': model.compute_loss_step4,
    }
    
    if stage not in stage_map:
        raise ValueError(f"Unknown stage: {stage}. Must be one of {list(stage_map.keys())}")
    
    return stage_map[stage]


def eval_per_video_query(dataset, dataloader, config, mapper, model, device, backproc, logger, stage='step3'):
    """
    Query-aware 的视频级评估
    
    Args:
        dataset: VideoDataset
        dataloader: DataLoader
        config: config dict
        mapper: QueryMapperHandler
        model: RelationalMSG
        device: torch.device
        backproc: BBoxReScaler
        logger: logger
        stage: str, evaluation stage
    
    Returns:
        dict: evaluation metrics
    """
    model.eval()
    
    # 获取 forward 和 loss 方法
    forward_fn = get_forward_method(model, stage)
    loss_fn = get_loss_method(model, stage)
    
    local_monitor = TrainingMonitor()
    local_monitor.add('running_loss_total')
    local_monitor.add('steps')
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Eval {dataset.video_id}"):
            images = batch['image'].to(device)
            
            # 构建 additional_info
            additional_info = {
                'gt_bbox': batch['bbox'].type(torch.FloatTensor).to(device),
                'obj_label': batch['obj_label'].to(device),
                'obj_idx': batch['obj_idx'].to(device),
                'mask': batch['mask'].to(device),
                'scene_num': 1,
            }
            
            if 'pred_bbox' in batch:
                additional_info['pred_bbox'] = batch['pred_bbox'].to(device)
                additional_info['pred_bbox_mask'] = batch['pred_bbox_mask'].to(device)
            
            if 'masks' in batch:
                additional_info['masks'] = batch['masks'].to(device)
            
            # 前向传播
            results = forward_fn(images, additional_info)
            
            # 计算 loss（如果适用）
            if stage in ['step1', 'amosg']:
                additional_info['place_labels'] = dataset.get_place_labels(batch['image_idx']).type(torch.FloatTensor).to(device)
                additional_info['rel_labels'] = dataset.get_rel_labels(batch['obj_idx']).type(torch.FloatTensor).to(device)
                total_loss, logs = loss_fn(results, additional_info)
                local_monitor.add('running_loss_total', total_loss.item())
                local_monitor.update(logs)
            
            local_monitor.add('steps', 1)
            
            # 后处理 bbox
            if 'detections' in results:
                results['detections'] = backproc.post_rescale_bbox(results['detections'])
            
            # 更新 mapper
            mapper.map_update(batch, results)
    
    # 日志输出
    logger.info(json.dumps(local_monitor.get_avg(), indent=4))
    logger.info(local_monitor.export_logging())
    
    # 获取 query usage 统计
    query_stats = mapper.get_query_usage_stats()
    if query_stats:
        logger.info("Query Usage Stats: %s", json.dumps(query_stats, indent=4))
    
    # 输出 mapping
    output_path = os.path.join(
        config['eval_output_dir'], 
        config['eval_split'], 
        dataset.video_id, 
        "query_eval_results.json"
    )
    
    map_results = mapper.output_mapping(save_pp_sim=True)
    
    # 评估
    evaluator = Evaluator(
        video_data_dir=dataset.video_data_dir,
        video_id=dataset.video_id,
        gt=dataset.gt,
        pred=map_results,
        out_path=os.path.dirname(output_path),
        inv_class_map=config['inv_class_map'],
        dataset='3rscan',
    )
    
    if config.get('vis_det', False):
        detector_type = config['detector']['model']
        evaluator.visualize_det(det_type=detector_type)
    
    eval_results = evaluator.get_metrics()
    map_results['eval_metrics'] = eval_results
    map_results['query_stats'] = query_stats
    
    # 保存结果
    if config.get("save_every", True):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(map_results, f)
    
    return eval_results


def eval_map_query(config, logger, stage='step3'):
    """
    Query-aware 的地图级评估
    
    Args:
        config: config dict
        logger: logger
        stage: str, evaluation stage
    
    Returns:
        dict: 每个 video 的评估结果
    """
    arkit_data = AppleDataHandler(
        config['dataset_path'], 
        split=config['eval_split'], 
        dataset='3rscan_split'
    )
    
    logger.info(f"Number of videos in the validation set: {len(arkit_data)}")
    
    # 构建模型
    backproc = BBoxReScaler(
        orig_size=config['image_size'], 
        new_size=config['model_image_size'], 
        device='cpu'
    )
    
    device_no = config['device']
    device = torch.device(f"cuda:{device_no}" if torch.cuda.is_available() else "cpu")
    model = RelationalMSG(config, device)
    
    # 加载 checkpoint
    if config.get("eval_chkpt") is not None:
        chkpt_path = os.path.join(config["eval_output_dir"], "checkpoints", config["eval_chkpt"])
        load_checkpoint(model=model, checkpoint_path=chkpt_path, logger=logger)
        logger.info(f"Loading model from checkpoint: {chkpt_path}")
    
    model = model.to(device)
    
    eval_results = dict()
    
    # 加载 scan dict
    import pickle
    scan_dict_path = r'/root/autodl-tmp/dataset/3rscan_msg_split/scan_dict.pkl'
    with open(scan_dict_path, "rb") as f:
        scan_dict = pickle.load(f)
    
    for i, next_video_id in enumerate(arkit_data.videos):
        logger.info(f"Processing video {next_video_id}, progress {i+1}/{len(arkit_data)}")
        next_video_path = os.path.join(arkit_data.data_split_dir, next_video_id)
        
        # 构建 dataset
        dataset = VideoDataset_3RScan_split(
            arkit_data.data_split_dir, 
            next_video_id, 
            config, 
            get_transform(config['model_image_size']), 
            split=config['eval_split'], 
            scan_dict=scan_dict
        )
        
        # 使用 QueryMapperHandler
        mapper = QueryMapperHandler(
            config, 
            next_video_path, 
            next_video_id, 
            dataset='3rscan_split'
        )
        
        dataloader = DataLoader(
            dataset, 
            batch_size=config.get("eval_bs", 1), 
            shuffle=True, 
            num_workers=config.get("num_workers", 0), 
            collate_fn=arkit_collate_fn
        )
        
        eval_result_per_video = eval_per_video_query(
            dataset, 
            dataloader, 
            config, 
            mapper, 
            model, 
            device, 
            backproc,
            logger,
            stage=stage,
        )
        
        logger.info("Result per video: %s", json.dumps(eval_result_per_video, indent=4))
        eval_results[next_video_id] = eval_result_per_video
    
    return eval_results


if __name__ == '__main__':
    DEFAULT_EXPERIMENT = 'aomsg_3rscan_split'
    
    parser = argparse.ArgumentParser(description="Query-aware Evaluation")
    parser.add_argument('--experiment', default=DEFAULT_EXPERIMENT, type=str, help='Name of the experiment config')
    parser.add_argument('--split', type=str, help='Name of the split to evaluate')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--output_file', type=str, help='Output file name')
    parser.add_argument('--device', type=int, help="Device number")
    parser.add_argument('--eval_split', type=str, help="Evaluation split")
    parser.add_argument('--eval_chkpt', type=str, help="Checkpoint file")
    parser.add_argument('--vis_det', type=bool, help="Output visualization")
    parser.add_argument('--object_threshold', type=float, help="Object threshold")
    parser.add_argument('--pp_threshold', type=float, help="Place threshold")
    parser.add_argument('--stage', type=str, default='step3', 
                       choices=['step1', 'step2', 'step3', 'step4', 'amosg'],
                       help="Evaluation stage")
    
    args = parser.parse_args()
    
    base_config_dir = './configs/defaults'
    config = get_configs(base_config_dir, args, creat_subdir=False)
    
    eval_file = config.get("eval_split", "eval")
    if config.get("eval_chkpt") is not None:
        eval_file = config["eval_chkpt"].split(".")[0]
    
    logger = create_logger(config["eval_output_dir"], eval_file)
    logger.info("Query-aware Evaluation config: %s\n", json.dumps(config, indent=4))
    
    set_seed(config.get('seed', 42))
    
    # 执行评估
    eval_results = eval_map_query(config, logger, stage=args.stage)
    
    # 计算平均指标
    avg_pp = 0.
    avg_po = 0.
    avg_graph = 0.
    
    for vid, res in eval_results.items():
        avg_pp += res.get('pp_iou', 0)
        avg_po += res.get('po_iou', 0)
        avg_graph += res.get('graph_iou', 0)
    
    avg_pp /= len(eval_results) if eval_results else 1
    avg_po /= len(eval_results) if eval_results else 1
    avg_graph /= len(eval_results) if eval_results else 1
    
    logger.info(f"Final Average avg pp: {avg_pp:.4f}, avg_po: {avg_po:.4f}, avg_graph_iou: {avg_graph:.4f}")
    logger.info(f"Query-aware Evaluation done!")
