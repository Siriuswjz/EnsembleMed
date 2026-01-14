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
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


class Config:
    ROOT_DIR = './data/TrainSet'
    TEST_DIR = './data/TestSet'
    IMG_SIZE = 448
    BATCH_SIZE = 16
    NUM_WORKERS = 4
    EPOCHS = 20
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
    print(f"Total: {len(df)}, Normal: {(df['label']==0).sum()}, Disease: {(df['label']==1).sum()}")
    return df


class TripletLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
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


class MetricEnsembleModel(nn.Module):
    def __init__(self, model_configs=None):
        super().__init__()

        if model_configs is None:
            model_configs = {
                'convnext': 'convnext_large.fb_in22k_ft_in1k',
                'vit': 'vit_large_patch16_224.augreg_in21k_ft_in1k',
            }

        self.backbones = nn.ModuleDict()
        total_dim = 0

        for name, model_name in model_configs.items():
            kwargs = {'pretrained': True, 'num_classes': 0}
            if 'vit' in model_name.lower() or 'swin' in model_name.lower():
                kwargs['img_size'] = Config.IMG_SIZE

            backbone = timm.create_model(model_name, **kwargs)

            for param in backbone.parameters():
                param.requires_grad = False

            self._unfreeze_last_layers(backbone)

            self.backbones[name] = backbone
            total_dim += backbone.num_features

        self.adapter = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128)
        )

    def _unfreeze_last_layers(self, backbone):
        if hasattr(backbone, 'stages'):
            for param in backbone.stages[-1].parameters():
                param.requires_grad = True

        if hasattr(backbone, 'blocks'):
            for block in backbone.blocks[-2:]:
                for param in block.parameters():
                    param.requires_grad = True

        if hasattr(backbone, 'norm'):
            for param in backbone.norm.parameters():
                param.requires_grad = True

    def forward(self, x):
        features = [backbone(x) for backbone in self.backbones.values()]
        combined = torch.cat(features, dim=1)
        embedding = self.adapter(combined)
        return F.normalize(embedding, p=2, dim=1)


def extract_features(loader, model, device):
    model.eval()
    features, labels = [], []

    with torch.no_grad():
        for imgs, batch_labels in loader:
            imgs = imgs.to(device)
            features.append(model(imgs).cpu().numpy())
            labels.append(batch_labels.numpy())

    return np.concatenate(features), np.concatenate(labels)


def extract_features_with_tta(loader, model, device):
    model.eval()
    features, labels = [], []

    with torch.no_grad():
        for imgs, batch_labels in loader:
            imgs = imgs.to(device)

            f1 = model(imgs)
            f2 = model(torch.flip(imgs, dims=[3]))
            f3 = model(torch.flip(imgs, dims=[2]))
            f4 = model(torch.rot90(imgs, k=1, dims=[2, 3]))

            f_avg = F.normalize((f1 + f2 + f3 + f4) / 4.0, p=2, dim=1)

            features.append(f_avg.cpu().numpy())
            labels.append(batch_labels.numpy())

    return np.concatenate(features), np.concatenate(labels)


def evaluate_auto_k(model, train_loader, val_loader, device):
    X_train, y_train = extract_features(train_loader, model, device)
    X_val, y_val = extract_features(val_loader, model, device)

    best_acc, best_k = 0, 15

    for k in [5, 9, 15, 20, 25, 30, 40, 50]:
        knn = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='euclidean')
        knn.fit(X_train, y_train)
        acc = accuracy_score(y_val, knn.predict(X_val))
        if acc > best_acc:
            best_acc, best_k = acc, k

    return best_acc, best_k


def train_model(model, train_loader, train_loader_eval, val_loader, device):
    criterion = TripletLoss(margin=0.5)

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

    best_acc, best_k, best_state = 0, 15, None

    for epoch in range(Config.EPOCHS):
        model.train()
        total_loss = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        val_acc, optimal_k = evaluate_auto_k(model, train_loader_eval, val_loader, device)

        print(f"Epoch {epoch+1}/{Config.EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Acc: {val_acc:.4f} | K: {optimal_k}")

        if val_acc >= best_acc:
            best_acc, best_k = val_acc, optimal_k
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    torch.save(best_state, os.path.join(Config.SAVE_DIR, 'best_metric_ensemble.pth'))
    model.load_state_dict(best_state)

    return model, best_k, best_acc


def generate_submission(model, best_k, device):
    df = load_data()
    full_train_ds = MedicalDataset(df, Config.ROOT_DIR, transform=get_transforms(False))
    full_train_loader = DataLoader(full_train_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)

    test_files = sorted([f for f in os.listdir(Config.TEST_DIR) if f.lower().endswith(('.jpg', '.png'))],
                       key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else x)

    test_df = pd.DataFrame({'image_id': test_files, 'label': 0})
    test_ds = MedicalDataset(test_df, Config.TEST_DIR, transform=get_transforms(False), is_test=True)
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)

    X_train, y_train = extract_features(full_train_loader, model, device)
    X_test, _ = extract_features_with_tta(test_loader, model, device)

    knn = KNeighborsClassifier(n_neighbors=best_k, weights='distance', metric='euclidean')
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)

    pd.DataFrame({'id': test_files, 'label': preds}).to_csv('./ensemble_learning/submission_metric_ensemble.csv', index=False)
    print(f"Saved: submission_metric_ensemble.csv (Normal={sum(preds==0)}, Disease={sum(preds==1)})")

    return preds


def main():
    print(f"Device: {Config.device}, Image Size: {Config.IMG_SIZE}, Batch: {Config.BATCH_SIZE}, Epochs: {Config.EPOCHS}")

    df = load_data()
    train_df, val_df = train_test_split(df, test_size=0.15, stratify=df['label'], random_state=Config.seed)

    train_ds = MedicalDataset(train_df, Config.ROOT_DIR, transform=get_transforms(True))
    val_ds = MedicalDataset(val_df, Config.ROOT_DIR, transform=get_transforms(False))
    train_ds_eval = MedicalDataset(train_df, Config.ROOT_DIR, transform=get_transforms(False))

    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS, drop_last=True)
    train_loader_eval = DataLoader(train_ds_eval, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)

    model = MetricEnsembleModel().to(Config.device)
    model, best_k, best_acc = train_model(model, train_loader, train_loader_eval, val_loader, Config.device)

    print(f"\nBest Val Acc: {best_acc:.4f}, Best K: {best_k}")
    generate_submission(model, best_k, Config.device)


if __name__ == "__main__":
    main()
