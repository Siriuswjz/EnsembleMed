import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode, TrivialAugmentWide
import random
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

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
    normalize = transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
    if is_train:
        aug_cfg = config.AUGMENTATION
        transform_list = []
        if aug_cfg.get("random_resized_crop", True):
            transform_list.append(
                transforms.RandomResizedCrop(
                    config.IMG_SIZE,
                    scale=aug_cfg.get("scale", (0.8, 1.0)),
                    interpolation=InterpolationMode.BICUBIC
                )
            )
        else:
            resize_size = int(config.IMG_SIZE * 1.15)
            transform_list.extend([
                transforms.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(config.IMG_SIZE)
            ])
        transform_list.append(
            transforms.RandomRotation(
                aug_cfg.get("rotation", 0),
                interpolation=InterpolationMode.BILINEAR,
                fill=0
            )
        )
        transform_list.append(transforms.RandomHorizontalFlip(p=aug_cfg.get("horizontal_flip", 0.0)))
        transform_list.append(transforms.RandomVerticalFlip(p=aug_cfg.get("vertical_flip", 0.0)))
        if aug_cfg.get("use_trivial_augment", False):
            transform_list.append(TrivialAugmentWide(num_magnitude_bins=int(aug_cfg.get("trivial_magnitude", 31))))
        color_jitter = transforms.ColorJitter(
            brightness=aug_cfg['color_jitter']['brightness'],
            contrast=aug_cfg['color_jitter']['contrast'],
            saturation=aug_cfg['color_jitter']['saturation'],
            hue=aug_cfg['color_jitter']['hue']
        )
        transform_list.append(transforms.RandomApply([color_jitter], p=0.8))
        blur_prob = aug_cfg.get("gaussian_blur_p", 0.0)
        if blur_prob > 0:
            transform_list.append(
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))],
                    p=blur_prob
                )
            )
        transform_list.extend([
            transforms.ToTensor(),
            normalize
        ])
        transform = transforms.Compose(transform_list)
    else:
        resize_size = int(config.IMG_SIZE * 1.1)
        transform = transforms.Compose([
            transforms.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(config.IMG_SIZE),
            transforms.ToTensor(),
            normalize
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
    indices = np.arange(total_size)
    labels = np.array([label for _, label in full_dataset.samples])
    
    if config.NUM_FOLDS > 1:
        skf = StratifiedKFold(n_splits=config.NUM_FOLDS, shuffle=True, random_state=config.SEED)
        fold_pairs = list(skf.split(indices, labels))
        fold_idx = config.CURRENT_FOLD % config.NUM_FOLDS
        train_idx, val_idx = fold_pairs[fold_idx]
        print(f"使用第 {fold_idx+1}/{config.NUM_FOLDS} 折进行训练")
    else:
        if config.USE_STRATIFIED_SPLIT:
            train_idx, val_idx = train_test_split(
                indices,
                test_size=1 - config.TRAIN_SPLIT,
                random_state=config.SEED,
                stratify=labels
            )
        else:
            random.shuffle(indices)
            split = int(total_size * config.TRAIN_SPLIT)
            train_idx, val_idx = indices[:split], indices[split:]
    
    train_samples = [full_dataset.samples[i] for i in train_idx]
    val_samples = [full_dataset.samples[i] for i in val_idx]
    
    train_dataset = MedicalImageDataset(config.TRAIN_DIR, get_transforms(config, True), True)
    train_dataset.samples = train_samples
    val_dataset = MedicalImageDataset(config.TRAIN_DIR, get_transforms(config, False), True)
    val_dataset.samples = val_samples
    
    train_labels = sum(1 for _, label in train_samples if label == 1)
    val_labels = sum(1 for _, label in val_samples if label == 1)
    print(
        f"训练集: {len(train_dataset)} (Disease: {train_labels}, Normal: {len(train_dataset) - train_labels})"
    )
    print(
        f"验证集: {len(val_dataset)} (Disease: {val_labels}, Normal: {len(val_dataset) - val_labels})"
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    return train_loader, val_loader


def create_test_loader(config):
    test_dataset = MedicalImageDataset(config.TEST_DIR, get_transforms(config, False), False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    return test_loader
