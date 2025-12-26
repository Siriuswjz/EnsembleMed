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
# 1. 配置 (Pro 版)
# ==========================================
class Config:
    DO_TRAINING = True  # 如果已经跑完想直接预测，改为 False
    
    # 路径检测
    ROOT_DIR = './data/TrainSet'
    TEST_DIR = './data/TestSet'
    
    print(f">> 训练集: {ROOT_DIR}")
    
    # [升级 1] 分辨率提升至 448 (医学图像细节关键)
    IMG_SIZE = 448
    
    # 如果 448 爆显存，请手动改回 384 或调小 Batch Size
    # Triplet Loss 依然需要尽量大的 Batch
    BATCH_SIZE = 24 
    NUM_WORKERS = 8
    
    # [升级 2] 增加训练轮数，让微调更充分
    EPOCHS = 25          
    
    # 差分学习率 (保持不变，这是成功的关键)
    LR_HEAD = 1e-3      
    LR_BACKBONE = 5e-6  
    
    MODEL_1 = 'convnext_large.fb_in22k_ft_in1k'
    MODEL_2 = 'vit_large_patch16_224.augreg_in21k_ft_in1k'
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    seed = 42
    
    MODEL_SAVE_PATH = 'robust_pro_model.pth'

def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

seed_everything(Config.seed)

# ==========================================
# 2. 数据集
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
            transforms.ColorJitter(brightness=0.15, contrast=0.15), #稍微加强一点颜色扰动
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
# 3. 鲁棒模型
# ==========================================
class RobustEnsembleModel(nn.Module):
    def __init__(self):
        super().__init__()
        print(f"=== 初始化 Pro 模型: 448x448 分辨率 + 差分微调 ===")
        
        # 显式传入 img_size=448，让 ViT 进行插值
        self.backbone1 = timm.create_model(Config.MODEL_1, pretrained=True, num_classes=0)
        self.backbone2 = timm.create_model(Config.MODEL_2, pretrained=True, num_classes=0, img_size=Config.IMG_SIZE)
        
        # 1. 先冻结所有
        for param in self.backbone1.parameters():
            param.requires_grad = False
        for param in self.backbone2.parameters():
            param.requires_grad = False
            
        # 2. 解冻 ConvNeXt Stage 3 (High-level features)
        print("  -> 解冻 ConvNeXt Stage 3")
        if hasattr(self.backbone1, 'stages'):
            for param in self.backbone1.stages[3].parameters():
                param.requires_grad = True
        if hasattr(self.backbone1, 'norm'):
             for param in self.backbone1.norm.parameters():
                param.requires_grad = True

        # 3. 解冻 ViT Last 2 Blocks
        print("  -> 解冻 ViT Last 2 Blocks")
        if hasattr(self.backbone2, 'blocks'):
            total_blocks = len(self.backbone2.blocks)
            for i in range(total_blocks - 2, total_blocks):
                for param in self.backbone2.blocks[i].parameters():
                    param.requires_grad = True
        if hasattr(self.backbone2, 'norm'):
             for param in self.backbone2.norm.parameters():
                param.requires_grad = True

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
    
    backbone_params = []
    adapter_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if 'adapter' in name: adapter_params.append(param)
        else: backbone_params.append(param)
            
    optimizer = optim.AdamW([
        {'params': adapter_params, 'lr': Config.LR_HEAD},
        {'params': backbone_params, 'lr': Config.LR_BACKBONE}
    ], weight_decay=1e-3)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        
    print(f"\n>>> 开始微调 (Epochs: {Config.EPOCHS})...")
    
    best_acc = 0
    best_k = 15 # 默认值
    
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
        
        # 每轮验证
        val_acc, optimal_k = evaluate_auto_k(model, train_loader, val_loader, device)
        print(f"Epoch [{epoch+1}/{Config.EPOCHS}] Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f} (Best K={optimal_k})")
        
        if val_acc >= best_acc:
            best_acc = val_acc
            best_k = optimal_k
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_dict, Config.MODEL_SAVE_PATH)
            print(f"  [Save] 最佳模型保存 (Acc: {best_acc:.4f})")
    
    return model, best_k

# ==========================================
# 5. TTA 与 Auto-K
# ==========================================
def extract_features_with_tta(loader, model, device):
    """[升级] 4视角 TTA"""
    model.eval()
    features = []
    labels = []
    
    print("  -> 启用增强 TTA (原图 + H翻转 + V翻转 + 旋转90) ...")
    
    with torch.no_grad():
        for imgs, batch_labels in loader:
            imgs = imgs.to(device)
            
            # 1. 原图
            f1 = model(imgs)
            # 2. H翻转
            f2 = model(torch.flip(imgs, dims=[3]))
            # 3. V翻转
            f3 = model(torch.flip(imgs, dims=[2]))
            # 4. 旋转 90度 (新增)
            f4 = model(torch.rot90(imgs, k=1, dims=[2, 3]))
            
            f_avg = (f1 + f2 + f3 + f4) / 4.0
            f_avg = F.normalize(f_avg, p=2, dim=1)
            
            features.append(f_avg.cpu().numpy())
            labels.append(batch_labels.numpy())
            
    return np.concatenate(features), np.concatenate(labels)

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

def evaluate_auto_k(model, train_loader, val_loader, device):
    """[升级] 自动搜索最佳 K 值"""
    X_train, y_train = extract_features(train_loader, model, device)
    X_val, y_val = extract_features(val_loader, model, device)
    
    best_acc = 0
    best_k = 15
    
    # 搜索范围
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

def generate_submission(model, best_k_from_training, device):
    if Config.TEST_DIR is None: return
    print(f"\n>>> 生成提交文件 (Best K={best_k_from_training})...")
    
    # 全量数据建库
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
    try:
        test_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    except ValueError:
        test_files.sort()
        
    test_df = pd.DataFrame({'id': test_files, 'label': 0})
    test_ds = MedicalDataset(test_df, Config.TEST_DIR, transform=get_transforms(False), is_test=True)
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)
    
    print("构建索引...")
    # 也可以在这里用 TTA 建库，但为了效率先用基础特征
    X_train, y_train = extract_features(full_train_loader, model, device)
    
    print("预测测试集 (4-View TTA)...")
    X_test, _ = extract_features_with_tta(test_loader, model, device)
    
    print(f"使用最佳 K={best_k_from_training} 进行 KNN 预测...")
    knn = KNeighborsClassifier(n_neighbors=best_k_from_training, weights='distance', metric='euclidean')
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)
    
    pd.DataFrame({'id': test_files, 'label': preds}).to_csv('submission_pro.csv', index=False)
    print("✅ submission_pro.csv 生成完毕")

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    model = RobustEnsembleModel().to(Config.device)
    
    best_k = 25 # 默认 fallback
    
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
        # 用来做验证的 train loader (无 shuffle)
        train_loader_eval = DataLoader(MedicalDataset(train_df, Config.ROOT_DIR, transform=get_transforms(False)), 
                                       batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)
        val_loader_eval = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)
        
        model, best_k = train_robust(model, train_loader, val_loader_eval, Config.device)
        print(f"训练结束，最佳 K 值: {best_k}")
        
    else:
        print(">>> 跳过训练步骤...")
        # 如果跳过训练，尝试加载之前保存的 K 值（这里简单起见，默认 25，或者你可以手动修改）
        # 实际项目中，最好把 best_k 也存到文件里
        best_k = 25 

    if os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"Loading weights from {Config.MODEL_SAVE_PATH}...")
        state = torch.load(Config.MODEL_SAVE_PATH)
        if isinstance(model, nn.DataParallel): model.module.load_state_dict(state)
        else: model.load_state_dict(state)
    else:
        if not Config.DO_TRAINING:
             print("错误: 找不到权重文件。")
             exit()
        
    generate_submission(model, best_k, Config.device)