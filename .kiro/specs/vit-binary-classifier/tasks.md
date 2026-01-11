# Implementation Plan

- [x] 1. Set up project structure and configuration module
  - Create vit_classifier directory with __init__.py
  - Implement config.py with Config class containing all hyperparameters (data paths, model params, training params)
  - Create checkpoints directory for saving model weights
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.1, 7.2_

- [x] 2. Implement dataset module with data loading and augmentation
  - [x] 2.1 Create DiseaseDataset class in dataset.py
    - Implement __init__ to scan disease and normal directories and build file list with labels
    - Implement __len__ to return dataset size
    - Implement __getitem__ to load image, apply transforms, and return (image, label) tuple
    - Add error handling for corrupted images and RGB conversion
    - _Requirements: 2.1, 2.2, 2.3_
  
  - [x] 2.2 Implement data augmentation transforms
    - Create get_train_transforms() function with RandomResizedCrop, RandomHorizontalFlip, RandomRotation, ColorJitter, ToTensor, and Normalize
    - Create get_val_transforms() function with Resize, CenterCrop, ToTensor, and Normalize
    - _Requirements: 2.2, 2.3, 3.1, 3.2, 3.3, 3.4_
  
  - [x] 2.3 Create dataloader factory function
    - Implement create_dataloaders() to split data into train/val sets and return DataLoader instances
    - Configure batch_size, shuffle, num_workers, and pin_memory parameters
    - _Requirements: 2.4, 2.5_

- [x] 3. Implement Vision Transformer model architecture
  - [x] 3.1 Implement PatchEmbedding class in model.py
    - Create Conv2d layer to extract and project patches
    - Calculate num_patches based on img_size and patch_size
    - Implement forward method to output (batch_size, num_patches, embed_dim)
    - _Requirements: 1.1_
  
  - [x] 3.2 Implement MultiHeadAttention class
    - Create linear projections for Q, K, V
    - Implement scaled dot-product attention computation
    - Split into multiple heads and concatenate outputs
    - Add dropout for regularization
    - _Requirements: 1.3_
  
  - [x] 3.3 Implement TransformerBlock class
    - Create LayerNorm + MultiHeadAttention with residual connection
    - Create LayerNorm + MLP (Linear → GELU → Dropout → Linear) with residual connection
    - _Requirements: 1.4_
  
  - [x] 3.4 Implement VisionTransformer class
    - Initialize PatchEmbedding layer
    - Create learnable class token parameter
    - Create learnable positional encoding parameter
    - Stack multiple TransformerBlock layers
    - Implement classification head (LayerNorm → Linear) for binary output
    - Implement forward method combining all components
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 4. Implement training module
  - [x] 4.1 Create Trainer class in train.py
    - Initialize model, optimizer (AdamW), loss function (BCEWithLogitsLoss), and learning rate scheduler
    - Implement device selection logic (CUDA if available, else CPU)
    - _Requirements: 4.1, 4.2, 4.3, 7.1, 7.2, 7.3_
  
  - [x] 4.2 Implement train_epoch method
    - Set model to training mode
    - Iterate through training dataloader
    - Perform forward pass, compute loss, backward pass, and optimizer step
    - Compute and accumulate training loss and accuracy
    - Add gradient clipping to prevent gradient explosion
    - _Requirements: 4.4_
  
  - [x] 4.3 Implement validate method
    - Set model to evaluation mode
    - Iterate through validation dataloader without gradient computation
    - Compute validation loss and accuracy
    - _Requirements: 4.5_
  
  - [x] 4.4 Implement checkpoint management
    - Create save_checkpoint method to save model state, optimizer state, epoch, and best accuracy
    - Create load_checkpoint method to restore training state
    - Save checkpoint only when validation accuracy improves
    - _Requirements: 4.6_
  
  - [x] 4.5 Implement main training loop
    - Create training loop iterating over epochs
    - Call train_epoch and validate for each epoch
    - Log metrics (epoch, train_loss, train_acc, val_loss, val_acc) to console
    - Save best model checkpoint
    - Add NaN detection and early stopping on training failure
    - _Requirements: 4.4, 4.5, 4.6, 4.7_

- [ ] 5. Implement inference module
  - [ ] 5.1 Create TestDataset class in dataset.py
    - Scan TestSet directory and extract image IDs from filenames
    - Implement __getitem__ to return (image, image_id)
    - Apply validation transforms (no augmentation)
    - _Requirements: 5.2, 5.3_
  
  - [ ] 5.2 Create Predictor class in inference.py
    - Implement load_model method to load trained checkpoint
    - Set model to evaluation mode and move to correct device
    - _Requirements: 5.1, 7.3_
  
  - [ ] 5.3 Implement prediction methods
    - Create predict_batch method to process batches of test images
    - Extract predictions from model logits using argmax
    - Collect (image_id, prediction) pairs
    - _Requirements: 5.4_
  
  - [ ] 5.4 Implement submission file generation
    - Create generate_submission method to write predictions to CSV
    - Format CSV with columns: image_id, label
    - Sort by image_id for consistent output
    - _Requirements: 5.5_

- [ ] 6. Create main execution scripts
  - [ ] 6.1 Create main training script
    - Write main() function in train.py to initialize config, create dataloaders, initialize model and trainer
    - Add command-line argument parsing for config overrides
    - Add try-except block for OOM errors with helpful message
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7_
  
  - [ ] 6.2 Create main inference script
    - Write main() function in inference.py to load config, create test dataloader, initialize predictor
    - Run prediction and generate submission.csv
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [ ] 7. Add requirements.txt and documentation
  - Create requirements.txt with torch, torchvision, Pillow, numpy, and tqdm
  - Create README.md with usage instructions for training and inference
  - Document hyperparameter configuration options
  - _Requirements: All_
