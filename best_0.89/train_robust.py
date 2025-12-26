import os
import cv2
import time
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
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 配置 (微调进阶版)
# ==========================================
class Config:
    # --- [关键修改] 开启重新训练 ---
    DO_TRAINING = True 
    
    # 路径检测
    ROOT_DIR = './data/TrainSet'
    TEST_DIR = './data/TestSet'

    
    print(f">> 训练集: {ROOT_DIR}")
    
    IMG_SIZE = 384
    # 显存允许的话，Batch Size 越大 Triplet 效果越好
    BATCH_SIZE = 24 
    NUM_WORKERS = 8
    
    EPOCHS = 15          
    
    # [关键] 差分学习率配置
    LR_HEAD = 1e-3      # 适配层（新层）学习率大一点
    LR_BACKBONE = 5e-6  # 骨干网络（解冻部分）学习率极其微小
    
    MODEL_1 = 'convnext_large.fb_in22k_ft_in1k'
    MODEL_2 = 'vit_large_patch16_224.augreg_in21k_ft_in1k'
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    seed = 42
    
    MODEL_SAVE_PATH = 'finetuned_ensemble_model.pth'

def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

seed_everything(Config.seed)

# ==========================================
# 2. 数据集 (TTA支持)
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
        if 'id' in self.df.columns:
            img_rel_path = self.df.iloc[idx]['id']
        else:
            img_rel_path = self.df.iloc[idx]['image_id']
        
        if not self.is_test:
            label = self.df.iloc[idx]['label']
        else:
            label = -1
            
        img_path = os.path.join(self.root_dir, img_rel_path)
        try:
            image = cv2.imread(img_path)
            if image is None: raise FileNotFoundError
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
            transforms.ColorJitter(brightness=0.1, contrast=0.1), 
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

# ==========================================
# 3. 鲁棒模型 (部分解冻版)
# ==========================================
class RobustEnsembleModel(nn.Module):
    def __init__(self):
        super().__init__()
        print(f"=== 初始化模型: 解冻最后层级 + 差分学习率 ===")
        
        self.backbone1 = timm.create_model(Config.MODEL_1, pretrained=True, num_classes=0)
        self.backbone2 = timm.create_model(Config.MODEL_2, pretrained=True, num_classes=0, img_size=Config.IMG_SIZE)
        
        # --- 策略：先冻结所有，再解冻最后一部分 ---
        
        # 1. 冻结所有参数
        for param in self.backbone1.parameters():
            param.requires_grad = False
        for param in self.backbone2.parameters():
            param.requires_grad = False
            
        # 2. 解冻 ConvNeXt 的最后一个 Stage (通常包含最丰富的高级语义)
        # timm 的 ConvNeXt 实现中，stages 是一个 nn.Sequential
        # 我们解冻 stages[3] (即第4个stage) 和 norm 层
        print("  -> 解冻 ConvNeXt Stage 3 & Norm")
        if hasattr(self.backbone1, 'stages'):
            for param in self.backbone1.stages[3].parameters():
                param.requires_grad = True
        if hasattr(self.backbone1, 'norm'): # Final norm
             for param in self.backbone1.norm.parameters():
                param.requires_grad = True

        # 3. 解冻 ViT 的最后 2 个 Block
        # timm 的 ViT 实现中，blocks 是 nn.Sequential
        print("  -> 解冻 ViT Last 2 Blocks & Norm")
        if hasattr(self.backbone2, 'blocks'):
            total_blocks = len(self.backbone2.blocks)
            for i in range(total_blocks - 2, total_blocks):
                for param in self.backbone2.blocks[i].parameters():
                    param.requires_grad = True
        if hasattr(self.backbone2, 'norm'):
             for param in self.backbone2.norm.parameters():
                param.requires_grad = True

        # 输入维度
        input_dim = 1536 + 1024 
        
        # 适配层
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(512, 128) 
        )
        
    def forward(self, x):
        # 此时 backbone 的部分层会计算梯度
        f1 = self.backbone1(x)
        f2 = self.backbone2(x)
        features = torch.cat([f1, f2], dim=1)
        
        embedding = self.adapter(features)
        return F.normalize(embedding, p=2, dim=1)

# ==========================================
# 4. 训练逻辑
# ==========================================
class TripletLoss(nn.Module):
    def __init__(self, margin=0.5): 
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, features, labels):
        n = features.size(0)
        dist = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(features, features.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt() 

        mask_pos = labels.expand(n, n).eq(labels.expand(n, n).t())
        dist_ap, _ = torch.max(dist * mask_pos.float(), dim=1)

        mask_neg = labels.expand(n, n).ne(labels.expand(n, n).t())
        dist_an, _ = torch.min(dist + 1e5 * (~mask_neg).float(), dim=1)

        loss = F.relu(dist_ap - dist_an + self.margin)
        return loss.mean()

def train_robust(model, train_loader, val_loader, device):
    criterion = TripletLoss(margin=0.5)
    
    # [关键] 差分学习率优化器设置
    # 分组：Adapter参数用大LR，Backbone解冻参数用小LR
    
    backbone_params = []
    adapter_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'adapter' in name:
            adapter_params.append(param)
        else:
            backbone_params.append(param)
            
    optimizer = optim.AdamW([
        {'params': adapter_params, 'lr': Config.LR_HEAD},
        {'params': backbone_params, 'lr': Config.LR_BACKBONE}
    ], weight_decay=1e-3)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        
    print(f"\n>>> 开始微调 (Backbone LR: {Config.LR_BACKBONE}, Head LR: {Config.LR_HEAD})...")
    
    best_acc = 0
    
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
        
        if (epoch + 1) % 1 == 0: # 每个 Epoch 都验证
            val_acc = evaluate(model, train_loader, val_loader, device)
            print(f"Epoch [{epoch+1}/{Config.EPOCHS}] Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")
            
            if val_acc >= best_acc:
                best_acc = val_acc
                state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                torch.save(state_dict, Config.MODEL_SAVE_PATH)
                print(f"  [Save] 最佳模型保存 (Acc: {best_acc:.4f})")
    
    return model

# ==========================================
# 5. TTA 特征提取 (提分关键)
# ==========================================
def extract_features_with_tta(loader, model, device):
    """使用测试时增强 (TTA) 提取特征"""
    model.eval()
    features = []
    labels = []
    
    print("  -> 启用 TTA (原图 + 水平翻转 + 垂直翻转) ...")
    
    with torch.no_grad():
        for imgs, batch_labels in loader:
            imgs = imgs.to(device)
            
            # 1. 原图
            f1 = model(imgs)
            
            # 2. 水平翻转
            f2 = model(torch.flip(imgs, dims=[3]))
            
            # 3. 垂直翻转
            f3 = model(torch.flip(imgs, dims=[2]))
            
            # 平均并重新归一化
            f_avg = (f1 + f2 + f3) / 3.0
            f_avg = F.normalize(f_avg, p=2, dim=1)
            
            features.append(f_avg.cpu().numpy())
            labels.append(batch_labels.numpy())
            
    return np.concatenate(features), np.concatenate(labels)

# 基础提取 (用于训练集建库，不一定要TTA，为了速度可以不用)
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

def evaluate(model, train_loader, val_loader, device):
    # 验证时用普通提取即可
    X_train, y_train = extract_features(train_loader, model, device)
    X_val, y_val = extract_features(val_loader, model, device)
    
    knn = KNeighborsClassifier(n_neighbors=15, weights='distance', metric='euclidean') 
    knn.fit(X_train, y_train)
    preds = knn.predict(X_val)
    return accuracy_score(y_val, preds)

def generate_submission(model, device):
    if Config.TEST_DIR is None: return
    print(f"\n>>> 生成提交文件 (启用 TTA)...")
    
    class_map = {'normal': 0, 'disease': 1}
    paths, labels = [], []
    for f, l in class_map.items():
        p = os.path.join(Config.ROOT_DIR, f)
        if os.path.exists(p):
            for file in os.listdir(p):
                if file.lower().endswith(('.jpg','.png')):
                    paths.append(os.path.join(f, file))
                    labels.append(l)
    
    full_train_df = pd.DataFrame({'image_id': paths, 'label': labels})
    full_train_ds = MedicalDataset(full_train_df, Config.ROOT_DIR, transform=get_transforms(False))
    full_train_loader = DataLoader(full_train_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)
    
    test_files = [f for f in os.listdir(Config.TEST_DIR) if f.lower().endswith(('.jpg','.png'))]
    
    # 智能排序
    try:
        test_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
        print(">> 检测到数字文件名，已执行数值排序。")
    except ValueError:
        test_files.sort()
        print(">> 执行字母排序。")
        
    test_df = pd.DataFrame({'id': test_files, 'label': 0})
    test_ds = MedicalDataset(test_df, Config.TEST_DIR, transform=get_transforms(False), is_test=True)
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)
    
    print("构建索引 (训练集)...")
    # 训练集建库时也可以考虑 TTA，但为了效率先不用，或者只用 flip
    X_train, y_train = extract_features(full_train_loader, model, device)
    
    print("预测测试集 (TTA)...")
    # 测试集必须用 TTA！
    X_test, _ = extract_features_with_tta(test_loader, model, device)
    
    # 使用加权 KNN
    # 增大 K 值到 25，因为用了 TTA 特征更稳，可以看更远
    knn = KNeighborsClassifier(n_neighbors=25, weights='distance', metric='euclidean')
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)
    
    pd.DataFrame({'id': test_files, 'label': preds}).to_csv('submission_finetuned_tta.csv', index=False)
    print("✅ submission_finetuned_tta.csv 生成完毕")

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    # 初始化模型
    model = RobustEnsembleModel().to(Config.device)
    
    # 1. 尝试训练
    if Config.DO_TRAINING:
        class_map = {'normal': 0, 'disease': 1}
        paths, labels = [], []
        for f, l in class_map.items():
            p = os.path.join(Config.ROOT_DIR, f)
            if os.path.exists(p):
                for file in os.listdir(p):
                    if file.lower().endswith(('.jpg','.png')):
                        paths.append(os.path.join(f, file))
                        labels.append(l)
        df = pd.DataFrame({'image_id': paths, 'label': labels})
        train_df, val_df = train_test_split(df, test_size=0.15, stratify=df['label'], random_state=Config.seed)
        
        train_ds = MedicalDataset(train_df, Config.ROOT_DIR, transform=get_transforms(True))
        val_ds = MedicalDataset(val_df, Config.ROOT_DIR, transform=get_transforms(False))
        
        train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS, drop_last=True)
        val_loader_eval = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)
        
        model = train_robust(model, train_loader, val_loader_eval, Config.device)
    else:
        print(">>> 跳过训练步骤...")

    # 2. 加载权重
    if os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"Loading weights from {Config.MODEL_SAVE_PATH}...")
        state = torch.load(Config.MODEL_SAVE_PATH)
        if isinstance(model, nn.DataParallel): 
            model.module.load_state_dict(state)
        else: 
            model.load_state_dict(state)
    else:
        if not Config.DO_TRAINING:
             print(f"错误: 找不到权重文件 {Config.MODEL_SAVE_PATH}。请先设置 DO_TRAINING=True 运行一次。")
             exit()
        
    # 3. 生成提交
    generate_submission(model, Config.device)