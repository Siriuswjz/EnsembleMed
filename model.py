import torch
import torch.nn as nn
import timm

class ImageClassifier(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True):
        super(ImageClassifier, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"模型: {model_name}, 参数量: {total_params/1e6:.1f}M")
    
    def forward(self, x):
        return self.model(x)


def create_model(config, device):
    model = ImageClassifier(config.MODEL_NAME, config.NUM_CLASSES, config.PRETRAINED)
    return model.to(device)


def save_checkpoint(model, optimizer, scheduler, epoch, val_acc, save_path, ema_state_dict=None):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'val_acc': val_acc
    }
    if ema_state_dict is not None:
        checkpoint['ema_state_dict'] = ema_state_dict
    torch.save(checkpoint, save_path)
    print(f"模型已保存: {save_path}")


def load_checkpoint(model, checkpoint_path, device, optimizer=None, scheduler=None, use_ema_weights=True):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if use_ema_weights and 'ema_state_dict' in checkpoint:
        state_dict = checkpoint['ema_state_dict']
    else:
        state_dict = checkpoint['model_state_dict']
    model.load_state_dict(state_dict)
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    val_acc = checkpoint.get('val_acc', 0.0)
    print(f"已加载模型: epoch={epoch}, val_acc={val_acc:.4f}")
    
    return epoch, val_acc
