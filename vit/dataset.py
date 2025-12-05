import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
import numpy as np

class MedicalImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        self.samples = []
        
        if is_train:
            self._load_train_data()
        else:
            self._load_test_data()
    
    def _load_train_data(self):
        disease_dir = os.path.join(self.root_dir, "disease")
        if os.path.exists(disease_dir):
            for img_name in os.listdir(disease_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(disease_dir, img_name), 1))
        
        normal_dir = os.path.join(self.root_dir, "normal")
        if os.path.exists(normal_dir):
            for img_name in os.listdir(normal_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(normal_dir, img_name), 0))
        
        print(f"训练集: {len(self.samples)}张 (Disease: {sum(1 for _, l in self.samples if l == 1)}, Normal: {sum(1 for _, l in self.samples if l == 0)})")
    
    def _load_test_data(self):
        for img_name in os.listdir(self.root_dir):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                self.samples.append((os.path.join(self.root_dir, img_name), -1))
        
        print(f"测试集: {len(self.samples)}张")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        
        return image, label, img_id


def get_transforms(config, is_train=True):
    if is_train:
        transform = transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.RandomRotation(config.AUGMENTATION['rotation']),
            transforms.RandomHorizontalFlip(p=config.AUGMENTATION['horizontal_flip']),
            transforms.RandomVerticalFlip(p=config.AUGMENTATION['vertical_flip']),
            transforms.ColorJitter(
                brightness=config.AUGMENTATION['color_jitter']['brightness'],
                contrast=config.AUGMENTATION['color_jitter']['contrast'],
                saturation=config.AUGMENTATION['color_jitter']['saturation'],
                hue=config.AUGMENTATION['color_jitter']['hue']
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.NORMALIZE_MEAN,
                std=config.NORMALIZE_STD
            )
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.NORMALIZE_MEAN,
                std=config.NORMALIZE_STD
            )
        ])
    
    return transform


def create_data_loaders(config):
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    
    full_dataset = MedicalImageDataset(
        root_dir=config.TRAIN_DIR,
        transform=None,
        is_train=True
    )
    
    total_size = len(full_dataset)
    train_size = int(total_size * config.TRAIN_SPLIT)
    
    indices = list(range(total_size))
    random.shuffle(indices)
    
    train_samples = [full_dataset.samples[i] for i in indices[:train_size]]
    val_samples = [full_dataset.samples[i] for i in indices[train_size:]]
    
    train_dataset = MedicalImageDataset(config.TRAIN_DIR, get_transforms(config, True), True)
    train_dataset.samples = train_samples
    
    val_dataset = MedicalImageDataset(config.TRAIN_DIR, get_transforms(config, False), True)
    val_dataset.samples = val_samples
    
    print(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, 
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, 
                            shuffle=False, num_workers=0, pin_memory=True)
    
    return train_loader, val_loader


def create_test_loader(config):
    test_dataset = MedicalImageDataset(config.TEST_DIR, get_transforms(config, False), False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return test_loader
