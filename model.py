import torch
import torch.nn as nn
import timm

try:
    import open_clip
except ImportError:
    open_clip = None

class TimmImageClassifier(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"模型(timm): {model_name}, 参数量: {total_params/1e6:.1f}M")
    
    def forward(self, x):
        return self.model(x)


class OpenClipClassifier(nn.Module):
    def __init__(self, model_name, pretrained, num_classes):
        super().__init__()
        if open_clip is None:
            raise ImportError("open_clip_torch 未安装，请先 pip install open_clip_torch")
        clip_model, _, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            cache_dir=None
        )
        self.backbone = clip_model.visual
        embed_dim = getattr(self.backbone, 'output_dim', getattr(clip_model, 'embed_dim', 1024))
        self.head = nn.Linear(embed_dim, num_classes)
        self._print_params(model_name)
    
    def _print_params(self, model_name):
        total_params = sum(p.numel() for p in self.parameters())
        print(f"模型(OpenCLIP): {model_name}, 参数量: {total_params/1e6:.1f}M")
    
    def forward(self, x):
        dtype = next(self.backbone.parameters()).dtype
        x = x.to(dtype)
        features = self.backbone(x)
        if isinstance(features, tuple):
            features = features[0]
        if features.dim() > 2:
            features = features.mean(dim=1)
        features = features.to(self.head.weight.dtype)
        logits = self.head(features)
        return logits


def create_model(config, device):
    if config.MODEL_LIBRARY.lower() == "open_clip":
        model = OpenClipClassifier(config.OPENCLIP_MODEL, config.OPENCLIP_PRETRAINED, config.NUM_CLASSES)
    else:
        model = TimmImageClassifier(config.MODEL_NAME, config.NUM_CLASSES, config.PRETRAINED)
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
