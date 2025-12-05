# 医学图像分类 - ViT实现

基于Vision Transformer (ViT-Large)的眼底图像二分类项目,用于识别患病(Disease)和正常(Normal)眼底图像。

## 项目结构

```
ML_Person/
├── data/                      # 数据目录
│   ├── TrainSet/             # 训练集
│   │   ├── disease/          # 患病图像
│   │   └── normal/           # 正常图像
│   └── TestSet/              # 测试集
├── config.py                  # 配置文件
├── dataset.py                 # 数据加载模块
├── model.py                   # 模型定义
├── train.py                   # 训练脚本
├── predict.py                 # 预测脚本
├── requirements.txt           # 依赖包
└── README.md                  # 说明文档
```

## 环境配置

### 安装依赖

```bash
pip install -r requirements.txt
```

### 主要依赖
- PyTorch >= 2.0.0
- timm >= 0.9.0 (用于加载预训练ViT模型)
- torchvision
- pandas
- scikit-learn
- tqdm

## 使用说明

### 1. 训练模型

```bash
python train.py
```

训练过程会:
- 自动下载预训练的ViT-Large模型
- 将训练集分为90%训练/10%验证
- 使用数据增强提高模型泛化能力
- 每个epoch后在验证集上评估
- 保存最佳模型到 `checkpoints/best_model.pth`
- 生成训练历史图 `training_history.png`

### 2. 预测测试集

```bash
python predict.py
```

预测过程会:
- 加载训练好的最佳模型
- 对测试集的250张图像进行预测
- 生成 `submission.csv` 文件,格式为:
  ```
  id,label
  1.jpg,0
  2.jpg,1
  ...
  ```

### 3. 提交结果

将生成的 `submission.csv` 上传到Kaggle竞赛页面。

## 模型配置

### 模型架构
- **模型**: ViT-Large-Patch16-224
- **参数量**: ~307M
- **输入尺寸**: 224×224
- **预训练**: ImageNet-21k

### 训练超参数
- **批次大小**: 16
- **学习率**: 1e-4
- **优化器**: AdamW
- **学习率调度**: 余弦退火
- **训练轮数**: 30 (带早停机制)
- **数据增强**: 旋转、翻转、颜色抖动

### 数据增强策略
- 随机旋转: ±15度
- 水平翻转: 50%概率
- 垂直翻转: 30%概率
- 颜色抖动: 亮度、对比度、饱和度、色调变化

## 注意事项

1. **GPU要求**: 建议使用至少8GB显存的GPU。如果显存不足,可以在 `config.py` 中减小 `BATCH_SIZE`。

2. **训练时间**: 使用GPU大约需要1-2小时完成训练(取决于硬件配置)。

3. **随机种子**: 已设置随机种子(42)以确保结果可复现。

4. **混合精度训练**: 默认启用混合精度训练以加速训练并减少显存占用。

5. **早停机制**: 如果验证准确率连续7个epoch未提升,训练会自动停止。

## 评估指标

训练过程会输出以下指标:
- **准确率 (Accuracy)**: 整体分类准确率
- **F1分数**: 二分类的F1分数
- **混淆矩阵**: 显示具体的分类情况

## 文件说明

- `config.py`: 所有配置参数的集中管理
- `dataset.py`: 数据集加载、预处理和数据增强
- `model.py`: ViT模型定义和检查点管理
- `train.py`: 完整的训练流程,包括验证和早停
- `predict.py`: 测试集预测和CSV生成

## 常见问题

### Q: 训练时显存不足怎么办?
A: 在 `config.py` 中将 `BATCH_SIZE` 改小,例如改为8或4。

### Q: 没有GPU可以训练吗?
A: 可以,但训练会非常慢。代码会自动检测并使用CPU。

### Q: 如何修改训练参数?
A: 所有参数都在 `config.py` 中,修改后重新运行 `train.py` 即可。

### Q: 预测结果的格式是什么?
A: CSV文件包含两列:
- `id`: 测试图像文件名(如 "1.jpg")
- `label`: 预测标签 (0=Normal, 1=Disease)

## 许可证

本项目仅供学习和研究使用。
