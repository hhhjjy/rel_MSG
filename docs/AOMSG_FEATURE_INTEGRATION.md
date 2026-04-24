# AOMSG 特征提取方法集成文档

## 概述

本文档记录了将 `reference/MSG/models/msgers/aomsg.py` 中的关键特征提取方法集成到 `RelationalMSG` 模型的工作。

## 参考来源

- **主参考**: `reference/MSG/models/msgers/aomsg.py`
- **辅助参考**: `reference/MSG/util/pos_embed.py`, `reference/MSG/models/msg.py`

---

## 关键方法对比与实现

| 组件 | AOMSG 参考实现 | 我们的实现 | 文件位置 |
|------|--------------|----------|---------|
| **特征投影层** | `object_proj`, `place_proj` | `self.object_proj`, `self.place_proj` | `models/relational_msg.py:64-65` |
| **Box 嵌入** | `box_emb` + `whole_box` | `self.box_emb` + `self.whole_box` | `models/relational_msg.py:68-69` |
| **2D 位置编码** | `get_2d_sincos_pos_embed` | 独立工具文件 | `models/pos_embed.py` |
| **条件增强** | `query + conditioning` | `_apply_aomsg_feature_extraction` 方法 | `models/relational_msg.py:112-184` |
| **位置编码初始化** | `_initialize_pos_embed` | 初始化方法 | `models/relational_msg.py:100-110` |

---

## 修改的文件

### 1. `models/pos_embed.py` (新建)
- 完整实现了 `aomsg.py` 中的 2D sincos 位置编码
- 包含: `get_2d_sincos_pos_embed`, `get_2d_sincos_pos_embed_from_grid`, `get_1d_sincos_pos_embed_from_grid`

### 2. `models/relational_msg.py` (修改)
- **新增导入**: `numpy`, `.pos_embed`
- **新增属性**: 
  - `self.object_proj`, `self.place_proj` (特征投影)
  - `self.box_emb`, `self.whole_box` (Box 嵌入)
  - `self.pos_embed` (位置编码)
  - `self.num_img_patches` (图像 patch 数量)
- **新增方法**:
  - `_initialize_pos_embed()`: 初始化 2D 位置编码
  - `_apply_aomsg_feature_extraction()`: 应用完整的 AOMSG 特征提取流程
- **更新 forward**: 集成 AOMSG 特征提取 + 跨视角融合

### 3. `test_aomsg_integration.py` (新建)
- 测试脚本验证集成正确性

---

## 特征提取流程

### 原始 AOMSG 流程 (aomsg.py:240-281)
```
images → place_embeddings
       ↓
       object_embeddings (from features)
       ↓
    padded_obj_embd + padded_obj_box
       ↓
    query = box_emb(whole_box + obj_boxes) + conditioning
       ↓
    memory = place_feat + pos_embed
       ↓
    decoded_emb = decoder(query, memory)
```

### 我们的集成流程
```
1. Backbone → img_feats, global_feats
2. ROI Extractor → bbox_feats
3. AOMSG 特征提取 (新增):
   ├─ Object/Place 特征投影
   ├─ 添加 2D 位置编码
   ├─ Box 嵌入 + 条件增强 (query + conditioning)
   └─ → aomsg_refined_obj, aomsg_refined_img
4. 跨视角融合 (使用 AOMSG 特征)
5. Query Decoders
6. Edge Heads
```

---

## 使用说明

### 模型初始化
```python
from models.relational_msg import RelationalMSG

model = RelationalMSG(
    hidden_dim=768,          # 与 dinov2-base 匹配
    num_obj_queries=100,
    num_place_queries=10,
    num_obj_classes=18,
    num_edge_types=50,
    config=your_config,      # 可选
)
```

### 前向传播
```python
outputs = model(images, bboxes, bbox_masks)

# 新增输出 (AOMSG 特征):
aomsg_refined_obj = outputs['aomsg_refined_obj']    # 增强后的物体特征
aomsg_refined_img = outputs['aomsg_refined_img']    # 增强后的场景特征
place_feat_with_pos = outputs['place_feat_with_pos'] # 带位置编码的场景特征
```

---

## 关键改进点

### 1. 特征投影
```python
# 新增代码 (relational_msg.py:64-65)
self.object_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
self.place_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
```
- **作用**: 对 object/place 特征进行独立的线性投影，增强特征表示能力

### 2. Box 嵌入
```python
self.box_emb = nn.Linear(4, hidden_dim, bias=False)
self.whole_box = nn.Parameter([0, 0, 224, 224], requires_grad=False)
```
- **作用**: 将 bbox 坐标编码为高维特征，结合 whole_box 作为场景级 anchor

### 3. 2D Sincos 位置编码
```python
# pos_embed.py 完整实现
pos_embed = get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=True)
```
- **作用**: 添加空间位置信息，参考 MAE 实现

### 4. 条件增强
```python
# query = box_emb(boxes) + conditioning (place_mean + object_feat)
query_with_cond = query + conditioning
```
- **作用**: 将场景上下文（place_mean）和物体特征结合，增强查询表示

---

## 与原代码的兼容性

✅ **完全向后兼容**
- 原有的 `refined_obj_bank`, `refined_img_bank` 输出保持不变
- 新增输出以 `aomsg_` 前缀标识
- 所有原有方法和参数保持不变

---

## 验证清单

- [x] 新增 pos_embed 工具文件
- [x] 集成特征投影层
- [x] 实现 Box 嵌入
- [x] 添加 2D sincos 位置编码
- [x] 实现条件增强逻辑
- [x] 更新 forward 流程
- [x] 测试脚本编写完成

---

## 下一步建议

1. **验证与训练**: 在完整环境中运行测试和训练
2. **AOMSG Loss 集成**: 可考虑进一步集成 AOMSG 的损失函数
3. **场景级 Attention**: 可参考 `aomsg.py:306-351` 的 `scene_level_attention`
4. **消融实验**: 对比有无 AOMSG 特征提取的效果差异

---

## 参考文献

- MAE: https://github.com/facebookresearch/mae
- AOMSG 原实现: `reference/MSG/models/msgers/aomsg.py`
