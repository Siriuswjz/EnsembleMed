import os

class Config:
    DATA_DIR = "data"
    TRAIN_DIR = os.path.join(DATA_DIR, "TrainSet")
    TEST_DIR = os.path.join(DATA_DIR, "TestSet")
    
    MODEL_NAME = "eva02_large_patch14_224.mim_m38m"
    NUM_CLASSES = 2
    PRETRAINED = True
    
    IMG_SIZE = 224
    BATCH_SIZE = 16
    NUM_EPOCHS = 30
    LEARNING_RATE = 5e-5
    WEIGHT_DECAY = 5e-4
    
    TRAIN_SPLIT = 0.8
    OPTIMIZER = "adamw"
    SCHEDULER = "cosine"
    MIN_LR = 1e-6
    WARMUP_EPOCHS = 3
    
    EARLY_STOPPING_PATIENCE = 5
    GRAD_CLIP = 1.0
    
    AUGMENTATION = {
        "rotation": 10,
        "horizontal_flip": 0.5,
        "vertical_flip": 0.2,
        "color_jitter": {
            "brightness": 0.1,
            "contrast": 0.1,
            "saturation": 0.1,
            "hue": 0.05
        }
    }
    
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]
    
    CHECKPOINT_DIR = "checkpoints"
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    SUBMISSION_FILE = "submission.csv"
    
    DEVICE = "cuda"
    SEED = 42
    CLASS_NAMES = {0: "Normal", 1: "Disease"}
    USE_AMP = True
    LOG_INTERVAL = 10
