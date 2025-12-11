import os

class Config:
    DATA_DIR = "data"
    TRAIN_DIR = os.path.join(DATA_DIR, "TrainSet")
    TEST_DIR = os.path.join(DATA_DIR, "TestSet")
    
    MODEL_LIBRARY = "open_clip"  # open_clip 或 timm
    MODEL_NAME = "eva02_large_patch14_448.mim_m38m"  # 备用 timm 模型
    OPENCLIP_MODEL = "ViT-bigG-14"
    OPENCLIP_PRETRAINED = "laion2b_s39b_b160k"
    NUM_CLASSES = 2
    PRETRAINED = True
    
    IMG_SIZE = 448
    BATCH_SIZE = 4
    NUM_WORKERS = 8
    NUM_EPOCHS = 35
    LEARNING_RATE = 3e-5
    WEIGHT_DECAY = 5e-4
    
    TRAIN_SPLIT = 0.85
    NUM_FOLDS = 5
    CURRENT_FOLD = 0
    USE_STRATIFIED_SPLIT = True
    
    OPTIMIZER = "adamw"
    SCHEDULER = "cosine"
    MIN_LR = 1e-6
    WARMUP_EPOCHS = 3
    
    EARLY_STOPPING_PATIENCE = 7
    GRAD_CLIP = 1.0
    LABEL_SMOOTHING = 0.1
    USE_MIXUP = True
    MIXUP_ALPHA = 0.3
    CUTMIX_ALPHA = 0.3
    MIXUP_PROB = 1.0
    MIXUP_SWITCH_PROB = 0.0
    MIXUP_MODE = "batch"
    
    USE_EMA = True
    EMA_DECAY = 0.9998
    
    AUGMENTATION = {
        "random_resized_crop": True,
        "scale": (0.85, 1.0),
        "rotation": 15,
        "horizontal_flip": 0.5,
        "vertical_flip": 0.25,
        "use_trivial_augment": True,
        "trivial_magnitude": 15,
        "color_jitter": {
            "brightness": 0.15,
            "contrast": 0.15,
            "saturation": 0.15,
            "hue": 0.05
        },
        "gaussian_blur_p": 0.2
    }
    
    NORMALIZE_MEAN = [0.48145466, 0.4578275, 0.40821073]
    NORMALIZE_STD = [0.26862954, 0.26130258, 0.27577711]
    
    CHECKPOINT_DIR = "checkpoints"
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    SUBMISSION_FILE = "submission.csv"
    
    DEVICE = "cuda"
    SEED = 42
    CLASS_NAMES = {0: "Normal", 1: "Disease"}
    USE_AMP = True
    LOG_INTERVAL = 10
    
    # 多次TTA在Predict阶段平均概率
    TTA_TRANSFORMS = ["none", "hflip", "vflip", "transpose"]
