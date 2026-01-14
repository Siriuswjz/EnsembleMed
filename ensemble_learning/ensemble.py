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


class Config:
    ROOT_DIR = './data/TrainSet'
    TEST_DIR = './data/TestSet'
    IMG_SIZE = 448
    BATCH_SIZE = 16
    NUM_WORKERS = 4
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    seed = 42
    MODEL1_PATH = './ensemble_learning/metric_checkpoints/robust_pro_model.pth'
    MODEL2_PATH = './ensemble_learning/metric_checkpoints/best_metric_ensemble.pth'


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
            image = np.zeros((448, 448, 3), dtype=np.uint8)

        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms():
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

    return pd.DataFrame({'image_id': paths, 'label': labels})


class Model1(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone1 = timm.create_model('convnext_large.fb_in22k_ft_in1k', pretrained=False, num_classes=0)
        self.backbone2 = timm.create_model('vit_large_patch16_224.augreg_in21k_ft_in1k', pretrained=False, num_classes=0, img_size=448)

        self.adapter = nn.Sequential(
            nn.Linear(1536 + 1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128)
        )

    def forward(self, x):
        f1 = self.backbone1(x)
        f2 = self.backbone2(x)
        return F.normalize(self.adapter(torch.cat([f1, f2], dim=1)), p=2, dim=1)


class Model2(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbones = nn.ModuleDict()
        self.backbones['convnext'] = timm.create_model('convnext_large.fb_in22k_ft_in1k', pretrained=False, num_classes=0)
        self.backbones['vit'] = timm.create_model('vit_large_patch16_224.augreg_in21k_ft_in1k', pretrained=False, num_classes=0, img_size=448)

        self.adapter = nn.Sequential(
            nn.Linear(1536 + 1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128)
        )

    def forward(self, x):
        features = [backbone(x) for backbone in self.backbones.values()]
        return F.normalize(self.adapter(torch.cat(features, dim=1)), p=2, dim=1)


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


def main():
    device = Config.device
    df = load_data()

    test_files = sorted([f for f in os.listdir(Config.TEST_DIR) if f.lower().endswith(('.jpg', '.png'))],
                       key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else x)

    test_df = pd.DataFrame({'image_id': test_files, 'label': 0})

    transform = get_transforms()

    print("Loading Model 1...")
    model1 = Model1().to(device)
    model1.load_state_dict(torch.load(Config.MODEL1_PATH, map_location=device))
    model1.eval()

    train_ds1 = MedicalDataset(df, Config.ROOT_DIR, transform=transform)
    test_ds1 = MedicalDataset(test_df, Config.TEST_DIR, transform=transform, is_test=True)
    train_loader1 = DataLoader(train_ds1, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)
    test_loader1 = DataLoader(test_ds1, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)

    X_train1, y_train1 = extract_features(train_loader1, model1, device)
    X_test1, _ = extract_features_with_tta(test_loader1, model1, device)

    knn1 = KNeighborsClassifier(n_neighbors=25, weights='distance', metric='euclidean')
    knn1.fit(X_train1, y_train1)
    proba1 = knn1.predict_proba(X_test1)

    del model1
    torch.cuda.empty_cache()

    print("Loading Model 2...")
    model2 = Model2().to(device)
    model2.load_state_dict(torch.load(Config.MODEL2_PATH, map_location=device))
    model2.eval()

    train_ds2 = MedicalDataset(df, Config.ROOT_DIR, transform=transform)
    test_ds2 = MedicalDataset(test_df, Config.TEST_DIR, transform=transform, is_test=True)
    train_loader2 = DataLoader(train_ds2, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)
    test_loader2 = DataLoader(test_ds2, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)

    X_train2, y_train2 = extract_features(train_loader2, model2, device)
    X_test2, _ = extract_features_with_tta(test_loader2, model2, device)

    knn2 = KNeighborsClassifier(n_neighbors=25, weights='distance', metric='euclidean')
    knn2.fit(X_train2, y_train2)
    proba2 = knn2.predict_proba(X_test2)

    del model2
    torch.cuda.empty_cache()

    avg_proba = (proba1 * 0.90 + proba2 * 0.92) / 1.82
    soft_vote = np.argmax(avg_proba, axis=1)

    print(f"Prediction dist: Normal={sum(soft_vote==0)}, Disease={sum(soft_vote==1)}")

    pd.DataFrame({'id': test_files, 'label': soft_vote}).to_csv('./ensemble_learning/submission_vote_soft.csv', index=False)
    print("Saved: submission_vote_soft.csv")


if __name__ == "__main__":
    main()
