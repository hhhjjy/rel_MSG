# AOMSG 特征提取集成指南

## 概述

本文档详细说明了如何将 `reference/MSG/models/msgers/aomsg.py` 中的关键特征提取方法以模块化方式集成到 RelationalMSG 框架中。

## 设计原则

✅ **保持兼容性**: 原有 `RelationalMSG` 类结构完全不变  
✅ **模块化设计**: AOMSG 功能封装在独立的 `AOMSGFeatureExtractor` 类中  
✅ **可选启用**: 通过配置参数 `use_aomsg_features` 控制是否启用  
✅ **向后兼容**: 不启用时完全退回到原有行为  

---

## 文件结构

```
models/
├── relational_msg.py          # 主模型类 (保持不变，新增集成)
├── aomsg_feature_extractor.py # 新增: AOMSG 特征提取器 (独立模块)
├── pos_embed.py               # 新增: 2D Sincos 位置编码工具
├── backbone.py
├── roi_extractor.py
├── cross_view_encoder.py
├── query_decoder.py
├── edge_heads.py
└── matching.py

docs/
└── AOMSG_INTEGRATION_GUIDE.md # 本文件
```

---

## 核心组件

### 1. `AOMSGFeatureExtractor` 类

**位置**: `models/aomsg_feature_extractor.py`

**功能**: 独立的 AOMSG 特征提取模块，可单独使用或集成到主模型中。

**关键属性**:
```python
class AOMSGFeatureExtractor(nn.Module):
    self.object_proj      # Object 特征投影层 (Linear)
    self.place_proj       # Place 特征投影层 (Linear)
    self.box_emb          # Box 嵌入层 (Linear)
    self.whole_box        # Whole Box 参数 [0,0,224,224]
    self.pos_embed        # 2D Sincos 位置编码
```

**关键方法**:
```python
def __init__(hidden_dim=768, num_img_patches=256)
def _initialize_pos_embed()      # 初始化位置编码
def _process_place_features()    # 处理场景特征
def forward(img_feats, global_feats, bbox_feats, bboxes, bbox_masks)
```

**输出**:
- `refined_obj`: 增强后的物体特征 [B, V, N, C]
- `refined_img`: 增强后的场景特征 [B, V, C]
- `place_feat_with_pos`: 带位置编码的场景特征 [B, V, L, C]

---

### 2. 集成到 `RelationalMSG`

**修改位置**: `models/relational_msg.py`

**修改内容** (保持原有结构):

1. **新增导入**:
```python
from .aomsg_feature_extractor import AOMSGFeatureExtractor
```

2. **新增配置参数**:
```python
self.use_aomsg_features = config.get('use_aomsg_features', False)
if self.use_aomsg_features:
    self.aomsg_extractor = AOMSGFeatureExtractor(...)
```

3. **更新 forward**:
```python
# 可选: AOMSG 特征增强
if self.use_aomsg_features:
    aomsg_refined_obj, aomsg_refined_img, ... = self.aomsg_extractor(...)
    input_obj_feats = aomsg_refined_obj
    input_img_feats = aomsg_refined_img.unsqueeze(2)
else:
    input_obj_feats = bbox_feats
    input_img_feats = global_feats.unsqueeze(2)
```

---

## 使用方法

### 方法 1: 独立使用 AOMSGFeatureExtractor

```python
from models.aomsg_feature_extractor import AOMSGFeatureExtractor

# 初始化
extractor = AOMSGFeatureExtractor(
    hidden_dim=768,
    num_img_patches=256
)

# 前向传播
refined_obj, refined_img, place_with_pos = extractor(
    img_feats,      # [B, V, C, H, W] 或 [B, V, L, C]
    global_feats,   # [B, V, C]
    bbox_feats,     # [B, V, N, C]
    bboxes,         # [B, V, N, 4]
    bbox_masks      # [B, V, N] (可选)
)
```

---

### 方法 2: 集成到 RelationalMSG (推荐)

#### 基础模式 (不使用 AOMSG，保持原有行为)
```python
from models.relational_msg import RelationalMSG

model = RelationalMSG(
    hidden_dim=768,
    num_obj_queries=100,
    num_place_queries=10,
    num_obj_classes=18,
    num_edge_types=50,
    config=None  # 或不包含 use_aomsg_features
)
```

#### AOMSG 增强模式
```python
config = {
    # 启用 AOMSG 特征提取
    'use_aomsg_features': True,
    'num_img_patches': 256,
    
    # 原有配置保持不变
    'backbone': {...},
    'matcher': {...}
}

model = RelationalMSG(
    hidden_dim=768,
    num_obj_queries=100,
    num_place_queries=10,
    num_obj_classes=18,
    num_edge_types=50,
    config=config
)

# forward 调用不变
outputs = model(images, bboxes, bbox_masks)

# 获取 AOMSG 增强输出
if 'aomsg_refined_obj' in outputs:
    aomsg_obj = outputs['aomsg_refined_obj']
    aomsg_img = outputs['aomsg_refined_img']
```

---

## 完整配置示例

```python
# 配置字典
config = {
    # AOMSG 特征提取配置
    'use_aomsg_features': True,
    'num_img_patches': 256,
    
    # Backbone 配置 (保持原有)
    'backbone': {
        'model_type': 'dinov2-base',
        'freeze': True,
        'weights': 'DEFAULT'
    },
    
    # Matcher 配置 (保持原有)
    'matcher': {
        'cost_bbox': 1.0,
        'cost_giou': 1.0,
        'cost_exist': 1.0
    }
}

# 初始化模型
model = RelationalMSG(
    hidden_dim=768,
    num_obj_queries=100,
    num_place_queries=10,
    num_obj_classes=18,
    num_edge_types=50,
    config=config
)
```

---

## 输出说明

### 基础模式输出 (不变)
```python
outputs = {
    'bbox_feats',
    'refined_obj_bank',
    'refined_img_bank',
    'object_node_feat',
    'object_attn',
    'object_exist_logits',
    'object_cls_logits',
    'place_node_feat',
    'place_attn',
    'place_exist_logits',
    'pp_logits',
    'po_logits'
}
```

### AOMSG 模式新增输出
```python
outputs = {
    # 原有输出保持不变
    ...,
    # 新增 AOMSG 输出
    'aomsg_refined_obj',      # 增强后的物体特征 [B, V, N, C]
    'aomsg_refined_img',      # 增强后的场景特征 [B, V, C]
    'place_feat_with_pos'     # 带位置编码的场景特征 [B, V, L, C]
}
```

---

## 架构对比

### 原有架构
```
images → Backbone → ROI Extractor → Cross View → Queries → Edges
```

### AOMSG 增强架构
```
images → Backbone → ROI Extractor
                    ↓
        (可选) AOMSGFeatureExtractor
                    ↓
            Cross View → Queries → Edges
```

---

## 测试

运行测试脚本验证集成:
```bash
python test_aomsg_integration_v2.py
```

测试内容:
- ✅ `AOMSGFeatureExtractor` 独立组件
- ✅ `RelationalMSG` 基础模式
- ✅ `RelationalMSG` AOMSG 模式

---

## 迁移指南

### 从原有代码迁移

1. **无需代码修改**: 原有代码无需任何改动即可正常运行
2. **可选启用**: 如需使用 AOMSG 特征，只需在配置中添加 `use_aomsg_features: True`

### 配置迁移
```python
# 原有配置
config_old = {
    'backbone': {...},
    'matcher': {...}
}

# 新配置 (可选启用 AOMSG)
config_new = {
    **config_old,
    'use_aomsg_features': True,  # 新增
    'num_img_patches': 256       # 新增 (可选)
}
```

---

## 关键特性

| 特性 | 说明 |
|-----|------|
| **模块化设计** | AOMSG 功能完全独立封装 |
| **可选启用** | 通过配置开关控制 |
| **向后兼容** | 不启用时保持原有行为 |
| **完整注释** | 代码包含详细说明和参考来源 |
| **PyTorch 规范** | 遵循 PyTorch Module 设计 |

---

## 参考来源

- AOMSG 原实现: `reference/MSG/models/msgers/aomsg.py`
- 位置编码: MAE (https://github.com/facebookresearch/mae)
- Box Utils: `reference/MSG/util/box_utils.py`

---

## 下一步

- [ ] 在实际训练数据上验证效果
- [ ] 对比 AOMSG 启用/禁用的性能差异
- [ ] 进一步集成 AOMSG 的损失函数 (如需要)
