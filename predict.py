import os
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm
import pandas as pd

from config import Config
from dataset import create_test_loader
from model import create_model, load_checkpoint


def apply_tta(images, tta_type):
    if tta_type == "none":
        return images
    if tta_type == "hflip":
        transformed = torch.flip(images, dims=[3])
    elif tta_type == "vflip":
        transformed = torch.flip(images, dims=[2])
    elif tta_type == "hvflip":
        transformed = torch.flip(images, dims=[2, 3])
    elif tta_type == "transpose":
        transformed = images.transpose(-1, -2)
    elif tta_type == "rot90":
        transformed = torch.rot90(images, k=1, dims=(2, 3))
    elif tta_type == "rot180":
        transformed = torch.rot90(images, k=2, dims=(2, 3))
    elif tta_type == "rot270":
        transformed = torch.rot90(images, k=3, dims=(2, 3))
    else:
        transformed = images
    return transformed.contiguous()


def predict():
    config = Config()
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    
    if not os.path.exists(config.BEST_MODEL_PATH):
        print(f"错误: 未找到模型 {config.BEST_MODEL_PATH}")
        return
    test_loader = create_test_loader(config)
    model = create_model(config, device)
    load_checkpoint(model, config.BEST_MODEL_PATH, device, use_ema_weights=True)
    model.eval()
    
    predictions, image_ids = [], []
    
    with torch.no_grad():
        for images, _, img_ids in tqdm(test_loader, desc="预测"):
            images = images.to(device)
            prob_sum = None
            for tta_name in config.TTA_TRANSFORMS:
                aug_images = apply_tta(images, tta_name)
                if config.USE_AMP:
                    with autocast():
                        logits = model(aug_images)
                else:
                    logits = model(aug_images)
                probs = torch.softmax(logits, dim=1)
                prob_sum = probs if prob_sum is None else (prob_sum + probs)
            avg_probs = prob_sum / max(1, len(config.TTA_TRANSFORMS))
            preds = torch.argmax(avg_probs, dim=1)
            predictions.extend(preds.cpu().numpy().tolist())
            image_ids.extend(img_ids)
    
    formatted_ids = []
    for img_id in image_ids:
        try:
            num_id = int(''.join(filter(str.isdigit, img_id)))
            formatted_ids.append(f"{num_id}.jpg")
        except:
            formatted_ids.append(img_id if '.jpg' in img_id else f"{img_id}.jpg")
    
    submission_df = pd.DataFrame({'id': formatted_ids, 'label': predictions})
    submission_df['sort_key'] = submission_df['id'].apply(lambda x: int(''.join(filter(str.isdigit, x))))
    submission_df = submission_df.sort_values('sort_key').drop('sort_key', axis=1)
    submission_df.to_csv(config.SUBMISSION_FILE, index=False)
    
    print(f"\n提交文件: {config.SUBMISSION_FILE} ({len(submission_df)}张)")
    


if __name__ == "__main__":
    predict()
