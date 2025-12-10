# 医学图像二分类项目

基于 Kaggle "Medical Image Detection" 比赛的数据，使用 `timm` 的强力 Transformer 模型（默认 `eva02_large_patch14_448.mim_m38m`）对眼底图像进行二分类（0=Normal，1=Disease）。在原有 baseline 0.75 左右的基础上，本项目集成了高分方案常用的增强、Mixup/CutMix、EMA、Warmup+Cosine 学习率、TTA 等策略，实测可以显著提升公榜成绩。

## 环境准备

```bash
pip install -r requirements.txt
```

推荐使用 CUDA 设备，若只在 CPU 运行请在 `config.py` 中将 `DEVICE` 改为 `"cpu"` 并适当调低 `IMG_SIZE`、`BATCH_SIZE`。

## 数据结构

```
data/
  TrainSet/
    disease/*.jpg
    normal/*.jpg
  TestSet/*.jpg
```

将 Kaggle 训练/测试图像解压到上述结构即可。

## 训练模型

```bash
python train.py
```

关键特性：

- Stratified K-Fold/随机分割：在 `config.py` 中通过 `NUM_FOLDS`、`CURRENT_FOLD` 控制，也可使用 `TRAIN_SPLIT` 进行单折划分。
- 448px RandomResizedCrop + TrivialAugmentWide + Blur 等强增广，配合 Mixup/CutMix 与 Label Smoothing 防止过拟合。
- EMA 模型（`USE_EMA`）自动保存，验证和推理均使用 EMA 权重。
- Warmup + Cosine 学习率（`WARMUP_EPOCHS` 与 `MIN_LR`），并支持梯度裁剪、AMP 训练。
- 训练曲线会保存到 `training_history.png`，最佳模型保存为 `checkpoints/best_model.pth`。

如需多折训练，可循环设置 `CURRENT_FOLD=0...NUM_FOLDS-1`，最后在推理时平均多个模型输出。

## 推理与提交

```bash
python predict.py
```

推理阶段默认执行多种 TTA（水平/竖直翻转、转置等），对每张图像的概率取平均再决定标签，并自动生成 `submission.csv`：

```
id,label
1.jpg,0
2.jpg,1
...
```

上传该 CSV 至 Kaggle 即可。若需要关闭/调整 TTA，可在 `config.py` 的 `TTA_TRANSFORMS` 中修改。

## 调参与进阶建议

- **模型与分辨率**：换成 `convnextv2_large`、`coatnet` 等更大 backbone，或提升 `IMG_SIZE` 到 384/448，可进一步提升精度。
- **批大小与优化器**：根据显存调整 `BATCH_SIZE`，LR 可按线性比例缩放；如需 LAMB、SGD 等可在 `train.py` 中修改。
- **半监督与伪标签**：当公榜分数 >0.80 后，可将高置信度测试预测加入训练集合，以获得额外收益。
- **集成策略**：多折模型 + 不同 backbone 进行 logit 平均往往能提升 1-2 个百分点。

祝你在比赛中取得更高的成绩！
