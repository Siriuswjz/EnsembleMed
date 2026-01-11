# Design Document

## Overview

本设计文档描述了一个从零开始实现的 Vision Transformer (ViT) 二分类系统。系统采用 PyTorch 框架实现，包含完整的数据处理流程、模型架构、训练循环和推理模块。系统设计遵循模块化原则，便于维护和扩展。

## Architecture

系统采用分层架构，主要包含以下模块：

```
vit_classifier/
├── model.py           # ViT 模型架构实现
├── dataset.py         # 数据集和数据加载器
├── train.py           # 训练脚本
├── inference.py       # 推理脚本
└── config.py          # 配置参数
```

### 数据流

1. **训练阶段**: 图像 → 数据增强 → Patch Embedding → Transformer Encoder → 分类头 → 损失计算 → 反向传播
2. **推理阶段**: 图像 → 预处理 → Patch Embedding → Transformer Encoder → 分类头 → 预测结果

## Components and Interfaces

### 1. Configuration Module (config.py)

配置类集中管理所有超参数：

```python
class Config:
    # 数据参数
    data_dir: str = "data/TrainSet"
    test_dir: str = "data/TestSet"
    img_size: int = 224
    batch_size: int = 16
    val_split: float = 0.2
    num_workers: int = 4
    
    # 模型参数
    patch_size: int = 16
    embed_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    
    # 训练参数
    num_epochs: int = 100
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 5
    
    # 其他
    checkpoint_dir: str = "checkpoints"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
```

### 2. Dataset Module (dataset.py)

#### DiseaseDataset Class

自定义 PyTorch Dataset 类，负责加载和预处理图像数据。

**接口**:
- `__init__(root_dir, transform, split='train')`: 初始化数据集
- `__len__()`: 返回数据集大小
- `__getitem__(idx)`: 返回单个样本 (image, label)

**数据增强策略**:
- 训练集: RandomResizedCrop, RandomHorizontalFlip, RandomRotation, ColorJitter, Normalize
- 验证/测试集: Resize, CenterCrop, Normalize

#### DataLoader Factory

创建训练、验证和测试数据加载器的工厂函数。

```python
def create_dataloaders(config):
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
    return train_loader, val_loader
```

### 3. Model Module (model.py)

#### PatchEmbedding Class

将输入图像分割为固定大小的 patches 并投影到嵌入空间。

**参数**:
- `img_size`: 输入图像尺寸
- `patch_size`: Patch 尺寸
- `in_channels`: 输入通道数 (RGB=3)
- `embed_dim`: 嵌入维度

**输出**: `(batch_size, num_patches, embed_dim)`

**实现细节**:
- 使用 Conv2d 层实现 patch 提取和线性投影
- `num_patches = (img_size // patch_size) ** 2`

#### MultiHeadAttention Class

实现多头自注意力机制。

**参数**:
- `embed_dim`: 嵌入维度
- `num_heads`: 注意力头数量
- `dropout`: Dropout 比率

**操作流程**:
1. 线性投影生成 Q, K, V
2. 分割为多个注意力头
3. 计算缩放点积注意力: `Attention(Q,K,V) = softmax(QK^T/√d_k)V`
4. 拼接多头输出并线性投影

#### TransformerBlock Class

Transformer 编码器块，包含多头注意力和前馈网络。

**结构**:
```
Input
  ↓
LayerNorm → MultiHeadAttention → Residual Connection
  ↓
LayerNorm → MLP (Linear → GELU → Dropout → Linear) → Residual Connection
  ↓
Output
```

#### VisionTransformer Class

完整的 ViT 模型架构。

**组件**:
1. **Patch Embedding**: 将图像转换为 patch 序列
2. **Class Token**: 可学习的分类 token，添加到序列开头
3. **Positional Encoding**: 可学习的位置编码，添加到所有 tokens
4. **Transformer Encoder**: 堆叠的 TransformerBlock 层
5. **Classification Head**: MLP 分类头 (LayerNorm → Linear)

**前向传播**:
```python
def forward(x):
    # x: (B, 3, H, W)
    x = patch_embed(x)  # (B, N, D)
    cls_token = repeat(cls_token, '1 1 d -> b 1 d', b=B)
    x = concat([cls_token, x], dim=1)  # (B, N+1, D)
    x = x + pos_embed  # (B, N+1, D)
    x = transformer_encoder(x)  # (B, N+1, D)
    cls_output = x[:, 0]  # (B, D)
    logits = classification_head(cls_output)  # (B, 2)
    return logits
```

### 4. Training Module (train.py)

#### Trainer Class

封装训练逻辑的类。

**方法**:
- `train_epoch()`: 执行一个训练 epoch
- `validate()`: 在验证集上评估模型
- `save_checkpoint()`: 保存模型检查点
- `load_checkpoint()`: 加载模型检查点

**训练循环**:
```python
for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch()
    val_loss, val_acc = validate()
    
    if val_acc > best_val_acc:
        save_checkpoint()
        best_val_acc = val_acc
    
    log_metrics(epoch, train_loss, train_acc, val_loss, val_acc)
```

**优化器配置**:
- Optimizer: AdamW
- Learning Rate Schedule: Cosine Annealing with Warmup
- Loss Function: Binary Cross Entropy with Logits

### 5. Inference Module (inference.py)

#### Predictor Class

负责模型推理和结果生成。

**方法**:
- `load_model()`: 加载训练好的模型
- `predict_image()`: 对单张图像进行预测
- `predict_batch()`: 批量预测
- `generate_submission()`: 生成提交文件

**推理流程**:
```python
def predict():
    model.eval()
    with torch.no_grad():
        for images, image_ids in test_loader:
            logits = model(images)
            predictions = torch.argmax(logits, dim=1)
            results.append((image_ids, predictions))
    
    save_to_csv(results, "submission.csv")
```

## Data Models

### 输入数据格式

**训练数据结构**:
```
data/TrainSet/
├── disease/
│   ├── disease_000005.jpg
│   ├── disease_000006.jpg
│   └── ...
└── normal/
    ├── normal_000001.jpg
    ├── normal_000002.jpg
    └── ...
```

**测试数据结构**:
```
data/TestSet/
├── 1.jpg
├── 2.jpg
└── ...
```

### 标签编码

- `0`: normal (正常)
- `1`: disease (疾病)

### 输出格式

**submission.csv**:
```csv
image_id,label
1,0
2,1
3,0
...
```

## Error Handling

### 数据加载错误

- **图像读取失败**: 记录错误并跳过该样本
- **图像格式不支持**: 尝试转换为 RGB 格式
- **空目录**: 抛出 ValueError 并提示用户

### 训练错误

- **OOM (Out of Memory)**: 捕获异常，建议减小 batch_size
- **NaN Loss**: 检测到 NaN 时停止训练，记录日志
- **梯度爆炸**: 使用梯度裁剪 (clip_grad_norm)

### 推理错误

- **模型文件不存在**: 抛出 FileNotFoundError
- **图像尺寸不匹配**: 自动调整到目标尺寸
- **设备不匹配**: 自动将模型和数据移动到正确设备

## Testing Strategy

### 单元测试

1. **PatchEmbedding 测试**
   - 验证输出形状正确性
   - 验证 patch 数量计算

2. **MultiHeadAttention 测试**
   - 验证注意力权重形状
   - 验证输出维度

3. **VisionTransformer 测试**
   - 验证端到端前向传播
   - 验证输出 logits 形状

### 集成测试

1. **数据加载测试**
   - 验证数据集正确加载
   - 验证数据增强应用正确

2. **训练流程测试**
   - 小数据集快速训练测试
   - 验证损失下降
   - 验证检查点保存

3. **推理流程测试**
   - 验证模型加载
   - 验证预测输出格式
   - 验证 CSV 文件生成

### 性能测试

1. **训练速度**: 记录每个 epoch 的训练时间
2. **内存使用**: 监控 GPU 内存占用
3. **推理速度**: 测量单张图像推理时间

## Design Decisions and Rationales

### 1. 为什么从零开始训练而不使用预训练模型？

**决策**: 完全从随机初始化开始训练 ViT 模型

**理由**:
- 用户明确要求不使用预训练权重
- 可以完全控制模型架构和训练过程
- 适合特定领域的数据集（医学图像）

**权衡**: 需要更多训练数据和更长训练时间，但获得更好的可解释性

### 2. 为什么选择 ViT 而不是 CNN？

**决策**: 使用 Vision Transformer 架构

**理由**:
- ViT 在图像分类任务上表现优异
- 自注意力机制可以捕获全局依赖关系
- 符合用户的技术选型要求

### 3. Patch Size 选择

**决策**: 默认使用 16x16 patch size

**理由**:
- 平衡计算效率和特征粒度
- 对于 224x224 图像，产生 196 个 patches，序列长度适中
- 可配置以适应不同需求

### 4. 位置编码方式

**决策**: 使用可学习的位置编码而非固定的正弦编码

**理由**:
- 更灵活，可以适应数据特性
- 在 ViT 原始论文中表现更好
- 实现简单

### 5. 优化器选择

**决策**: 使用 AdamW 优化器

**理由**:
- AdamW 在 Transformer 训练中表现稳定
- 内置权重衰减，有助于正则化
- 对学习率不太敏感

### 6. 学习率调度

**决策**: 使用 Warmup + Cosine Annealing

**理由**:
- Warmup 阶段避免训练初期的不稳定
- Cosine Annealing 平滑降低学习率
- 在 ViT 训练中被广泛验证有效

### 7. 数据增强策略

**决策**: 使用中等强度的数据增强

**理由**:
- 医学图像需要保持关键特征
- 避免过度增强导致信息丢失
- 包含基本的几何和颜色变换

### 8. 模型规模

**决策**: 默认使用 ViT-Base 配置 (12 layers, 768 dim, 12 heads)

**理由**:
- 在性能和计算成本之间取得平衡
- 适合中等规模数据集
- 可配置以调整模型容量
