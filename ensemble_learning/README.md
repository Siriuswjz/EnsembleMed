# 集成学习分类器

使用度量学习 (Triplet Loss + KNN) 进行医学图像分类，达到 **0.93** 准确率。

## 最佳方案 ⭐

**双模型投票**: 0.90 + 0.92 → **0.93**

```bash
python vote_ensemble.py
```

## 方法说明

### 度量学习 (metric_ensemble.py)

**核心思路**: 不直接分类，而是学习有区分性的特征表示

```
图像 → ConvNeXt + ViT → 特征拼接 → Adapter → 128维embedding
                                              ↓
                                        L2归一化
                                              ↓
                                    Triplet Loss训练
                                              ↓
                                        KNN分类
```

**训练**:
```bash
python metric_ensemble.py
```

**输出**: 
- `metric_checkpoints/best_metric_ensemble.pth` - 模型权重
- `0.92.csv` - 预测结果

### 双模型投票 (vote_ensemble.py)

结合两个高准确率模型的预测:
- Model 1: best_0.90 (ConvNeXt + ViT @ 448)
- Model 2: metric_ensemble (0.92)

**投票策略**:
- 软投票: 按准确率加权平均概率
- 硬投票: 多数投票
- 保守投票: 两个都预测disease才是disease

**运行**:
```bash
python vote_ensemble.py
```

**输出**:
- `submission_vote_soft.csv` - 软投票 (推荐)
- `submission_vote_hard.csv` - 硬投票
- `submission_vote_conservative.csv` - 保守投票

### 预测分析 (analyze_predictions.py)

分析多个模型的预测差异和互补性:

```bash
python analyze_predictions.py
```

**功能**:
- 计算模型间一致率
- 分析互补性
- 模拟不同投票策略
- 给出集成建议

## 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| IMG_SIZE | 448 | 高分辨率对医学图像很重要 |
| BATCH_SIZE | 16-20 | Triplet Loss需要较大batch |
| LR_HEAD | 1e-3 | Adapter学习率 |
| LR_BACKBONE | 5e-6 | Backbone微调学习率 |
| MARGIN | 0.5 | Triplet Loss margin |
| EPOCHS | 20-25 | 训练轮数 |

## 为什么度量学习效果好？

1. **小样本友好**: 医学图像数据量通常较小
2. **特征质量高**: Triplet Loss直接优化类间距离
3. **利用全部数据**: KNN可以用全部训练数据
4. **灵活性**: 可以动态调整K值

## 实验结果

| 模型 | 准确率 | 备注 |
|------|--------|------|
| best_0.90 | 0.90 | ConvNeXt + ViT |
| metric_ensemble | 0.92 | 改进训练策略 |
| **双模型投票** | **0.93** | ✅ 最佳 |
| metric_v3 | 0.82 | 不同策略，效果差 |

## 失败的尝试

❌ **Swin + EfficientNetV2**: 0.74-0.79，效果差
❌ **ConvNeXtV2 + BEiT**: 0.71，效果差
❌ **三模型投票**: 低准确率模型拖后腿

## 依赖

```bash
pip install torch timm scikit-learn pandas numpy opencv-python
```
