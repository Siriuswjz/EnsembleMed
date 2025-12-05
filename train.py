import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import random
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

from config import Config
from dataset import create_data_loaders
from model import create_model, save_checkpoint

PROXY_ADDRESS = "http://10.61.113.61:7897"
os.environ['http_proxy'] = PROXY_ADDRESS
os.environ['https_proxy'] = PROXY_ADDRESS

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, config, epoch):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [训练]")
    
    for batch_idx, (images, labels, _) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        if config.USE_AMP:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            optimizer.step()
        
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        if (batch_idx + 1) % config.LOG_INTERVAL == 0:
            pbar.set_postfix({'loss': f'{running_loss / (batch_idx + 1):.4f}'})
    
    return running_loss / len(train_loader), accuracy_score(all_labels, all_preds)


def validate(model, val_loader, criterion, device, config, epoch):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [验证]")
    
    with torch.no_grad():
        for images, labels, _ in pbar:
            images, labels = images.to(device), labels.to(device)
            
            if config.USE_AMP:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='binary')
    
    cm = confusion_matrix(all_labels, all_preds)
    print(f"\n混淆矩阵: Normal={cm[0][0]}/{cm[0][1]}, Disease={cm[1][0]}/{cm[1][1]}")
    
    return epoch_loss, epoch_acc, epoch_f1


def plot_training_history(history, save_path='training_history.png'):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].plot(history['train_loss'], label='训练Loss', marker='o')
    axes[0].plot(history['val_loss'], label='验证Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(history['train_acc'], label='训练Acc', marker='o')
    axes[1].plot(history['val_acc'], label='验证Acc', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"训练历史图: {save_path}")
    plt.close()


def train_model():
    config = Config()
    set_seed(config.SEED)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"\n设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    train_loader, val_loader = create_data_loaders(config)
    model = create_model(config, device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS, eta_min=config.MIN_LR)
    scaler = GradScaler() if config.USE_AMP else None
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    best_val_acc = 0.0
    patience_counter = 0
    
    print("开始训练")
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, config, epoch)
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device, config, epoch)
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        print(f"训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, scheduler, epoch, val_acc, config.BEST_MODEL_PATH)
            print(f"最佳 Acc: {val_acc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"未改善 ({patience_counter}/{config.EARLY_STOPPING_PATIENCE})")
        
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"\n早停触发")
            break

    print(f"训练完成! 最佳Acc: {best_val_acc:.4f}")
    
    plot_training_history(history)


if __name__ == "__main__":
    train_model()
