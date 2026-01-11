# Requirements Document

## Introduction

本文档定义了一个基于 Vision Transformer (ViT) 的二分类系统的需求，用于从零开始训练模型以区分正常和疾病图像。系统将使用 TrainSet 中的标注数据进行训练，并对 TestSet 中的图像进行预测。

## Glossary

- **ViT_System**: Vision Transformer 二分类训练和推理系统
- **Training_Module**: 负责模型训练的模块
- **Inference_Module**: 负责模型推理和预测的模块
- **Data_Loader**: 数据加载和预处理组件
- **ViT_Model**: Vision Transformer 模型架构
- **Checkpoint**: 训练过程中保存的模型状态
- **Prediction_Output**: 包含图像ID和预测类别的CSV文件

## Requirements

### Requirement 1

**User Story:** 作为开发者，我希望从零开始构建 ViT 模型架构，以便完全控制模型结构而不依赖预训练权重

#### Acceptance Criteria

1. THE ViT_System SHALL implement patch embedding layer that converts input images into fixed-size patches
2. THE ViT_System SHALL implement positional encoding mechanism for patch sequences
3. THE ViT_System SHALL implement multi-head self-attention mechanism with configurable number of heads
4. THE ViT_System SHALL implement transformer encoder blocks with layer normalization and feed-forward networks
5. THE ViT_System SHALL implement classification head that outputs binary predictions

### Requirement 2

**User Story:** 作为开发者，我希望系统能够加载和预处理训练数据，以便模型能够有效学习

#### Acceptance Criteria

1. THE Data_Loader SHALL read images from data/TrainSet/disease and data/TrainSet/normal directories
2. WHEN loading images, THE Data_Loader SHALL apply image resizing to consistent dimensions
3. THE Data_Loader SHALL apply data normalization to pixel values
4. THE Data_Loader SHALL split training data into training and validation subsets with configurable ratio
5. THE Data_Loader SHALL support batch processing with configurable batch size

### Requirement 3

**User Story:** 作为开发者，我希望实现数据增强策略，以便提高模型的泛化能力

#### Acceptance Criteria

1. WHEN processing training images, THE Data_Loader SHALL apply random horizontal flipping
2. WHEN processing training images, THE Data_Loader SHALL apply random rotation within specified angle range
3. WHEN processing training images, THE Data_Loader SHALL apply random color jittering
4. WHEN processing validation images, THE Data_Loader SHALL apply only normalization without augmentation

### Requirement 4

**User Story:** 作为开发者，我希望训练模型并监控训练过程，以便获得最佳性能

#### Acceptance Criteria

1. THE Training_Module SHALL initialize ViT_Model with random weights
2. THE Training_Module SHALL use binary cross-entropy loss function for optimization
3. THE Training_Module SHALL use Adam optimizer with configurable learning rate
4. WHEN training, THE Training_Module SHALL compute training loss and accuracy for each epoch
5. WHEN training, THE Training_Module SHALL compute validation loss and accuracy for each epoch
6. THE Training_Module SHALL save Checkpoint when validation accuracy improves
7. THE Training_Module SHALL log training metrics including loss and accuracy to console

### Requirement 5

**User Story:** 作为开发者，我希望系统能够对测试集进行推理，以便生成预测结果

#### Acceptance Criteria

1. THE Inference_Module SHALL load trained model from saved Checkpoint
2. THE Inference_Module SHALL read all images from data/TestSet directory
3. WHEN processing test images, THE Inference_Module SHALL apply same preprocessing as validation data
4. THE Inference_Module SHALL generate predictions for each test image
5. THE Inference_Module SHALL create Prediction_Output in CSV format with columns for image ID and predicted class

### Requirement 6

**User Story:** 作为开发者，我希望配置模型超参数，以便灵活调整模型性能

#### Acceptance Criteria

1. THE ViT_System SHALL support configurable image size parameter
2. THE ViT_System SHALL support configurable patch size parameter
3. THE ViT_System SHALL support configurable embedding dimension parameter
4. THE ViT_System SHALL support configurable number of transformer layers parameter
5. THE ViT_System SHALL support configurable number of attention heads parameter
6. THE ViT_System SHALL support configurable MLP hidden dimension ratio parameter
7. THE Training_Module SHALL support configurable number of training epochs parameter
8. THE Training_Module SHALL support configurable batch size parameter
9. THE Training_Module SHALL support configurable learning rate parameter

### Requirement 7

**User Story:** 作为开发者，我希望系统能够在 GPU 上训练（如果可用），以便加速训练过程

#### Acceptance Criteria

1. WHEN GPU is available, THE ViT_System SHALL automatically use CUDA device for computation
2. WHEN GPU is not available, THE ViT_System SHALL use CPU device for computation
3. THE ViT_System SHALL move model and data tensors to selected device during training and inference
