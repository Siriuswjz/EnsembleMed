"""
双模型联合投票
结合 best_0.90 和 metric_ensemble (0.92) 两个模型的预测
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')


# ==========================================
# 配置
# ==========================================
class Config:
    ROOT_DIR = './data/TrainSet'
    TEST_DIR = './data/TestSet'
    
    IMG_SIZE_MODEL1 = 448  # best_0.90 模型
    IMG_SIZE_MODEL2 = 448  # metric_ensemble 模型
    
    BATCH_SIZE = 16
    NUM_WORKERS = 4
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    seed = 42
    
    # 模型权重路径
    MODEL1_PATH = './best_0.90/robust_pro_model.pth'
    MODEL2_PATH = './ensemble_learning/metric_checkpoints/best_metric_ensemble.pth'


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
            image = np.zeros((448, 448, 3), dtype=np.uint8)

        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms(img_size):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
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
    
    return pd.DataFrame({'image_id': paths, 'label': labels})


# ==========================================
# 模型1: best_0.90 (ConvNeXt + ViT)
# ==========================================
class Model1(nn.Module):
    """best_0.90 模型结构"""
    def __init__(self):
        super().__init__()
        self.backbone1 = timm.create_model('convnext_large.fb_in22k_ft_in1k', pretrained=False, num_classes=0)
        self.backbone2 = timm.create_model('vit_large_patch16_224.augreg_in21k_ft_in1k', pretrained=False, num_classes=0, img_size=448)
        
        input_dim = 1536 + 1024
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128)
        )
        
    def forward(self, x):
        f1 = self.backbone1(x)
        f2 = self.backbone2(x)
        features = torch.cat([f1, f2], dim=1)
        embedding = self.adapter(features)
        return F.normalize(embedding, p=2, dim=1)


# ==========================================
# 模型2: metric_ensemble (ConvNeXt + ViT with ModuleDict)
# ==========================================
class Model2(nn.Module):
    """metric_ensemble 模型结构"""
    def __init__(self):
        super().__init__()
        
        self.backbones = nn.ModuleDict()
        
        self.backbones['convnext'] = timm.create_model('convnext_large.fb_in22k_ft_in1k', pretrained=False, num_classes=0)
        self.backbones['vit'] = timm.create_model('vit_large_patch16_224.augreg_in21k_ft_in1k', pretrained=False, num_classes=0, img_size=448)
        
        total_dim = 1536 + 1024
        
        self.adapter = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128)
        )
    
    def forward(self, x):
        features = []
        for name, backbone in self.backbones.items():
            f = backbone(x)
            features.append(f)
        
        combined = torch.cat(features, dim=1)
        embedding = self.adapter(combined)
        return F.normalize(embedding, p=2, dim=1)


# ==========================================
# 特征提取
# ==========================================
def extract_features(loader, model, device):
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
    """4视角TTA"""
    model.eval()
    features = []
    labels = []
    
    with torch.no_grad():
        for imgs, batch_labels in loader:
            imgs = imgs.to(device)
            
            f1 = model(imgs)
            f2 = model(torch.flip(imgs, dims=[3]))
            f3 = model(torch.flip(imgs, dims=[2]))
            f4 = model(torch.rot90(imgs, k=1, dims=[2, 3]))
            
            f_avg = (f1 + f2 + f3 + f4) / 4.0
            f_avg = F.normalize(f_avg, p=2, dim=1)
            
            features.append(f_avg.cpu().numpy())
            labels.append(batch_labels.numpy())
    
    return np.concatenate(features), np.concatenate(labels)


# ==========================================
# 主程序
# ==========================================
def main():
    print("="*60)
    print("双模型联合投票")
    print("="*60)
    
    device = Config.device
    print(f"设备: {device}")
    
    # 加载训练数据
    df = load_data()
    print(f"训练样本: {len(df)}")
    
    # 加载测试数据
    test_files = [f for f in os.listdir(Config.TEST_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    try:
        test_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    except:
        test_files.sort()
    
    test_df = pd.DataFrame({'image_id': test_files, 'label': 0})
    print(f"测试样本: {len(test_df)}")
    
    # ==========================================
    # 模型1预测
    # ==========================================
    print("\n>>> 加载模型1 (best_0.90)...")
    model1 = Model1().to(device)
    model1.load_state_dict(torch.load(Config.MODEL1_PATH, map_location=device))
    model1.eval()
    
    transform1 = get_transforms(Config.IMG_SIZE_MODEL1)
    train_ds1 = MedicalDataset(df, Config.ROOT_DIR, transform=transform1)
    test_ds1 = MedicalDataset(test_df, Config.TEST_DIR, transform=transform1, is_test=True)
    
    train_loader1 = DataLoader(train_ds1, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)
    test_loader1 = DataLoader(test_ds1, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)
    
    print("  提取训练集特征...")
    X_train1, y_train1 = extract_features(train_loader1, model1, device)
    print("  提取测试集特征 (TTA)...")
    X_test1, _ = extract_features_with_tta(test_loader1, model1, device)
    
    # KNN预测
    knn1 = KNeighborsClassifier(n_neighbors=25, weights='distance', metric='euclidean')
    knn1.fit(X_train1, y_train1)
    proba1 = knn1.predict_proba(X_test1)  # 获取概率
    preds1 = knn1.predict(X_test1)
    
    del model1
    torch.cuda.empty_cache()
    
    # ==========================================
    # 模型2预测
    # ==========================================
    print("\n>>> 加载模型2 (metric_ensemble 0.92)...")
    model2 = Model2().to(device)
    model2.load_state_dict(torch.load(Config.MODEL2_PATH, map_location=device))
    model2.eval()
    
    transform2 = get_transforms(Config.IMG_SIZE_MODEL2)
    train_ds2 = MedicalDataset(df, Config.ROOT_DIR, transform=transform2)
    test_ds2 = MedicalDataset(test_df, Config.TEST_DIR, transform=transform2, is_test=True)
    
    train_loader2 = DataLoader(train_ds2, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)
    test_loader2 = DataLoader(test_ds2, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)
    
    print("  提取训练集特征...")
    X_train2, y_train2 = extract_features(train_loader2, model2, device)
    print("  提取测试集特征 (TTA)...")
    X_test2, _ = extract_features_with_tta(test_loader2, model2, device)
    
    # KNN预测
    knn2 = KNeighborsClassifier(n_neighbors=25, weights='distance', metric='euclidean')
    knn2.fit(X_train2, y_train2)
    proba2 = knn2.predict_proba(X_test2)
    preds2 = knn2.predict(X_test2)
    
    del model2
    torch.cuda.empty_cache()
    
    # ==========================================
    # 联合投票
    # ==========================================
    print("\n>>> 联合投票...")
    
    # 方法1: 硬投票 (多数投票)
    hard_vote = ((preds1 + preds2) >= 1).astype(int)  # 有一个预测为1就是1
    
    # 方法2: 软投票 (概率加权平均)
    # 给0.92的模型更高权重
    weight1 = 0.90
    weight2 = 0.92
    avg_proba = (proba1 * weight1 + proba2 * weight2) / (weight1 + weight2)
    soft_vote = np.argmax(avg_proba, axis=1)
    
    # 方法3: 保守投票 (两个都预测为1才是1)
    conservative_vote = ((preds1 + preds2) == 2).astype(int)
    
    print(f"\n模型1预测分布: Normal={sum(preds1==0)}, Disease={sum(preds1==1)}")
    print(f"模型2预测分布: Normal={sum(preds2==0)}, Disease={sum(preds2==1)}")
    print(f"硬投票分布: Normal={sum(hard_vote==0)}, Disease={sum(hard_vote==1)}")
    print(f"软投票分布: Normal={sum(soft_vote==0)}, Disease={sum(soft_vote==1)}")
    print(f"保守投票分布: Normal={sum(conservative_vote==0)}, Disease={sum(conservative_vote==1)}")
    
    # 两个模型预测一致的数量
    agree = sum(preds1 == preds2)
    print(f"\n两模型一致: {agree}/{len(preds1)} ({100*agree/len(preds1):.1f}%)")
    
    # 保存所有结果
    pd.DataFrame({'id': test_files, 'label': soft_vote}).to_csv(
        './ensemble_learning/submission_vote_soft.csv', index=False)
    pd.DataFrame({'id': test_files, 'label': hard_vote}).to_csv(
        './ensemble_learning/submission_vote_hard.csv', index=False)
    pd.DataFrame({'id': test_files, 'label': conservative_vote}).to_csv(
        './ensemble_learning/submission_vote_conservative.csv', index=False)
    
    print("\n✅ 已生成三种投票结果:")
    print("  - submission_vote_soft.csv (软投票，推荐)")
    print("  - submission_vote_hard.csv (硬投票)")
    print("  - submission_vote_conservative.csv (保守投票)")


if __name__ == "__main__":
    main()
