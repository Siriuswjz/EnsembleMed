"""
度量学习集成分类器
采用 Triplet Loss + KNN 的方式，与 best_0.90 方法一致
多个不同backbone提取特征后融合
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


# ==========================================
# 配置
# ==========================================
class Config:
    ROOT_DIR = './data/TrainSet'
    TEST_DIR = './data/TestSet'
    
    # 高分辨率很重要
    IMG_SIZE = 448
    BATCH_SIZE = 16  # Triplet Loss需要较大batch
    NUM_WORKERS = 4
    EPOCHS = 20
    
    # 差分学习率 - 这是成功的关键
    LR_HEAD = 1e-3
    LR_BACKBONE = 5e-6
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    seed = 42
    
    SAVE_DIR = './ensemble_learning/metric_checkpoints'


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(Config.seed)


# ==========================================
# 数据集
# ==========================================
class MedicalDataset(Dataset):
    def __init__(self, df, root_dir, transform=None, is_test=False):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_rel_path = self.df.iloc[idx]['image_id']
        label = self.df.iloc[idx]['label'] if not self.is_test else -1
        
        img_path = os.path.join(self.root_dir, img_rel_path)
        try:
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            image = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)

        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms(is_train=False):
    if is_train:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((Config.IMG_SIZE + 32, Config.IMG_SIZE + 32)),
            transforms.RandomCrop((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def load_data():
    """加载训练数据"""
    class_map = {'normal': 0, 'disease': 1}
    paths, labels = [], []
    
    for class_name, label in class_map.items():
        class_dir = os.path.join(Config.ROOT_DIR, class_name)
        if os.path.exists(class_dir):
            for file in os.listdir(class_dir):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    paths.append(os.path.join(class_name, file))
                    labels.append(label)
    
    df = pd.DataFrame({'image_id': paths, 'label': labels})
    print(f"总样本数: {len(df)}, 正常: {(df['label']==0).sum()}, 疾病: {(df['label']==1).sum()}")
    return df


# ==========================================
# Triplet Loss
# ==========================================
class TripletLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, features, labels):
        n = features.size(0)
        
        # 计算距离矩阵
        dist = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(features, features.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()

        # 正样本对: 同类最远距离
        mask_pos = labels.expand(n, n).eq(labels.expand(n, n).t())
        dist_ap, _ = torch.max(dist * mask_pos.float(), dim=1)

        # 负样本对: 异类最近距离
        mask_neg = labels.expand(n, n).ne(labels.expand(n, n).t())
        dist_an, _ = torch.min(dist + 1e5 * (~mask_neg).float(), dim=1)

        loss = F.relu(dist_ap - dist_an + self.margin)
        return loss.mean()


# ==========================================
# 集成度量学习模型
# ==========================================
class MetricEnsembleModel(nn.Module):
    """
    多backbone集成的度量学习模型
    融合多个预训练模型的特征，输出归一化的embedding
    """
    def __init__(self, model_configs: Dict[str, str] = None):
        super().__init__()
        
        if model_configs is None:
            # 默认使用3个强力backbone
            model_configs = {
                'convnext': 'convnext_large.fb_in22k_ft_in1k',
                'vit': 'vit_large_patch16_224.augreg_in21k_ft_in1k',
                'swin': 'swin_large_patch4_window12_384.ms_in22k_ft_in1k',
            }
        
        self.backbones = nn.ModuleDict()
        self.feature_dims = {}
        total_dim = 0
        
        print("=== 初始化集成度量学习模型 ===")
        
        for name, model_name in model_configs.items():
            print(f"  加载 {name}: {model_name}")
            
            # 根据模型类型设置参数
            kwargs = {'pretrained': True, 'num_classes': 0}
            if 'vit' in model_name.lower() or 'swin' in model_name.lower():
                kwargs['img_size'] = Config.IMG_SIZE
            
            backbone = timm.create_model(model_name, **kwargs)
            
            # 冻结所有参数
            for param in backbone.parameters():
                param.requires_grad = False
            
            # 解冻最后几层
            self._unfreeze_last_layers(backbone, name)
            
            self.backbones[name] = backbone
            self.feature_dims[name] = backbone.num_features
            total_dim += backbone.num_features
            
            print(f"    特征维度: {backbone.num_features}")
        
        print(f"  总特征维度: {total_dim}")
        
        # 特征融合adapter
        self.adapter = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128)
        )
        
    def _unfreeze_last_layers(self, backbone, name):
        """解冻backbone最后几层"""
        if hasattr(backbone, 'stages'):  # ConvNeXt
            print(f"    解冻 {name} Stage 3")
            for param in backbone.stages[-1].parameters():
                param.requires_grad = True
        
        if hasattr(backbone, 'blocks'):  # ViT, Swin
            total_blocks = len(backbone.blocks)
            print(f"    解冻 {name} Last 2 Blocks")
            for i in range(max(0, total_blocks - 2), total_blocks):
                for param in backbone.blocks[i].parameters():
                    param.requires_grad = True
        
        if hasattr(backbone, 'layers'):  # Swin
            print(f"    解冻 {name} Last Layer")
            for param in backbone.layers[-1].parameters():
                param.requires_grad = True
        
        # 解冻norm层
        if hasattr(backbone, 'norm'):
            for param in backbone.norm.parameters():
                param.requires_grad = True
    
    def forward(self, x):
        features = []
        for name, backbone in self.backbones.items():
            f = backbone(x)
            features.append(f)
        
        # 拼接所有特征
        combined = torch.cat(features, dim=1)
        
        # 通过adapter
        embedding = self.adapter(combined)
        
        # L2归一化
        return F.normalize(embedding, p=2, dim=1)


# ==========================================
# 特征提取与TTA
# ==========================================
def extract_features(loader, model, device):
    """提取特征"""
    model.eval()
    features = []
    labels = []
    
    with torch.no_grad():
        for imgs, batch_labels in loader:
            imgs = imgs.to(device)
            feats = model(imgs)
            features.append(feats.cpu().numpy())
            labels.append(batch_labels.numpy())
    
    return np.concatenate(features), np.concatenate(labels)


def extract_features_with_tta(loader, model, device):
    """使用TTA提取特征 (4视角)"""
    model.eval()
    features = []
    labels = []
    
    print("  -> 启用 4-View TTA (原图 + H翻转 + V翻转 + 旋转90)")
    
    with torch.no_grad():
        for imgs, batch_labels in loader:
            imgs = imgs.to(device)
            
            # 4视角TTA
            f1 = model(imgs)                                    # 原图
            f2 = model(torch.flip(imgs, dims=[3]))              # 水平翻转
            f3 = model(torch.flip(imgs, dims=[2]))              # 垂直翻转
            f4 = model(torch.rot90(imgs, k=1, dims=[2, 3]))     # 旋转90度
            
            # 平均并归一化
            f_avg = (f1 + f2 + f3 + f4) / 4.0
            f_avg = F.normalize(f_avg, p=2, dim=1)
            
            features.append(f_avg.cpu().numpy())
            labels.append(batch_labels.numpy())
    
    return np.concatenate(features), np.concatenate(labels)


def evaluate_auto_k(model, train_loader, val_loader, device):
    """自动搜索最佳K值"""
    X_train, y_train = extract_features(train_loader, model, device)
    X_val, y_val = extract_features(val_loader, model, device)
    
    best_acc = 0
    best_k = 15
    
    candidates = [5, 9, 15, 20, 25, 30, 40, 50]
    
    for k in candidates:
        knn = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='euclidean')
        knn.fit(X_train, y_train)
        preds = knn.predict(X_val)
        acc = accuracy_score(y_val, preds)
        if acc > best_acc:
            best_acc = acc
            best_k = k
    
    return best_acc, best_k


# ==========================================
# 训练
# ==========================================
def train_model(model, train_loader, train_loader_eval, val_loader, device):
    """训练度量学习模型"""
    criterion = TripletLoss(margin=0.5)
    
    # 分离参数: adapter用大学习率，backbone用小学习率
    backbone_params = []
    adapter_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'adapter' in name:
            adapter_params.append(param)
        else:
            backbone_params.append(param)
    
    print(f"可训练参数: adapter={len(adapter_params)}, backbone={len(backbone_params)}")
    
    optimizer = optim.AdamW([
        {'params': adapter_params, 'lr': Config.LR_HEAD},
        {'params': backbone_params, 'lr': Config.LR_BACKBONE}
    ], weight_decay=1e-3)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    
    print(f"\n>>> 开始训练 (Epochs: {Config.EPOCHS})...")
    
    best_acc = 0
    best_k = 15
    best_state = None
    
    for epoch in range(Config.EPOCHS):
        model.train()
        total_loss = 0
        
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            features = model(imgs)
            loss = criterion(features, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        
        # 验证
        val_acc, optimal_k = evaluate_auto_k(model, train_loader_eval, val_loader, device)
        
        print(f"Epoch [{epoch+1}/{Config.EPOCHS}] Loss: {total_loss/len(train_loader):.4f} | "
              f"Val Acc: {val_acc:.4f} (Best K={optimal_k})")
        
        if val_acc >= best_acc:
            best_acc = val_acc
            best_k = optimal_k
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  [Save] 最佳模型 (Acc: {best_acc:.4f})")
    
    # 保存最佳模型
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    torch.save(best_state, os.path.join(Config.SAVE_DIR, 'best_metric_ensemble.pth'))
    
    # 加载最佳权重
    model.load_state_dict(best_state)
    
    return model, best_k, best_acc


# ==========================================
# 预测
# ==========================================
def generate_submission(model, best_k, device):
    """生成提交文件"""
    print(f"\n>>> 生成提交文件 (Best K={best_k})...")
    
    # 全量训练数据建库
    df = load_data()
    full_train_ds = MedicalDataset(df, Config.ROOT_DIR, transform=get_transforms(False))
    full_train_loader = DataLoader(full_train_ds, batch_size=Config.BATCH_SIZE, 
                                   shuffle=False, num_workers=Config.NUM_WORKERS)
    
    # 测试数据
    test_files = [f for f in os.listdir(Config.TEST_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    try:
        test_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    except ValueError:
        test_files.sort()
    
    test_df = pd.DataFrame({'image_id': test_files, 'label': 0})
    test_ds = MedicalDataset(test_df, Config.TEST_DIR, transform=get_transforms(False), is_test=True)
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, 
                             shuffle=False, num_workers=Config.NUM_WORKERS)
    
    print("构建特征索引...")
    X_train, y_train = extract_features(full_train_loader, model, device)
    
    print("提取测试集特征 (4-View TTA)...")
    X_test, _ = extract_features_with_tta(test_loader, model, device)
    
    print(f"使用 K={best_k} 进行 KNN 预测...")
    knn = KNeighborsClassifier(n_neighbors=best_k, weights='distance', metric='euclidean')
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)
    
    # 保存结果
    submission = pd.DataFrame({'id': test_files, 'label': preds})
    submission.to_csv('./ensemble_learning/submission_metric_ensemble.csv', index=False)
    
    print(f"✅ 提交文件已保存: ./ensemble_learning/submission_metric_ensemble.csv")
    print(f"预测分布: Normal={sum(preds==0)}, Disease={sum(preds==1)}")
    
    return preds


# ==========================================
# 主程序
# ==========================================
def main():
    print("="*60)
    print("度量学习集成分类器")
    print("="*60)
    print(f"设备: {Config.device}")
    print(f"图像大小: {Config.IMG_SIZE}")
    print(f"批次大小: {Config.BATCH_SIZE}")
    print(f"训练轮数: {Config.EPOCHS}")
    print("="*60)
    
    # 加载数据
    df = load_data()
    train_df, val_df = train_test_split(df, test_size=0.15, stratify=df['label'], random_state=Config.seed)
    
    # 创建数据加载器
    train_ds = MedicalDataset(train_df, Config.ROOT_DIR, transform=get_transforms(True))
    val_ds = MedicalDataset(val_df, Config.ROOT_DIR, transform=get_transforms(False))
    train_ds_eval = MedicalDataset(train_df, Config.ROOT_DIR, transform=get_transforms(False))
    
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, 
                              num_workers=Config.NUM_WORKERS, drop_last=True)
    train_loader_eval = DataLoader(train_ds_eval, batch_size=Config.BATCH_SIZE, shuffle=False, 
                                   num_workers=Config.NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, 
                            num_workers=Config.NUM_WORKERS)
    
    # 创建模型 - 使用多个强力backbone
    model_configs = {
        'convnext': 'convnext_large.fb_in22k_ft_in1k',
        'vit': 'vit_large_patch16_224.augreg_in21k_ft_in1k',
    }
    
    model = MetricEnsembleModel(model_configs).to(Config.device)
    
    # 训练
    model, best_k, best_acc = train_model(model, train_loader, train_loader_eval, val_loader, Config.device)
    
    print(f"\n训练完成! 最佳验证准确率: {best_acc:.4f}, 最佳K值: {best_k}")
    
    # 生成提交
    generate_submission(model, best_k, Config.device)


if __name__ == "__main__":
    main()
