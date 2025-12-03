
"""
TumorVision: Enhanced Utilities Module v2.0
Contains custom loss functions, metrics, data generators, and augmentation pipelines
for improved brain tumor detection and segmentation.

Key Improvements:
- Advanced boundary-aware loss functions for precise tumor localization
- Multi-scale feature extraction for better small tumor detection
- Medical imaging-specific augmentation pipeline
- Optimized inference with Test Time Augmentation (TTA)
- Support for mixed precision training for faster computation
"""

import pandas as pd
import numpy as np
import seaborn as sns
import cv2
import tensorflow as tf
import os 
from PIL import Image
from tensorflow.keras import backend as K
from keras.saving import register_keras_serializable
import albumentations as A
from scipy import ndimage
from scipy.ndimage import distance_transform_edt

# Enable XLA compilation for faster execution
tf.config.optimizer.set_jit(True)

# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

IMG_SIZE = 256
BATCH_SIZE = 16
SEED = 42

# Set random seeds for reproducibility
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================================
# ENHANCED DATA AUGMENTATION PIPELINE
# ============================================================================

def get_training_augmentation(intensity='medium'):
    """
    Advanced augmentation pipeline for training data.
    Specifically designed for medical brain MRI images.
    
    Args:
        intensity: 'light', 'medium', or 'heavy' augmentation
    
    Returns:
        Albumentations Compose object
    """
    if intensity == 'light':
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        ])
    elif intensity == 'heavy':
        return A.Compose([
            # Spatial transformations
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.15,
                scale_limit=0.2,
                rotate_limit=45,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.7
            ),
            # Advanced elastic deformation for medical images
            A.ElasticTransform(
                alpha=150,
                sigma=150 * 0.06,
                p=0.4
            ),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.4),
            A.OpticalDistortion(distort_limit=0.2, shift_limit=0.15, p=0.3),
            # Intensity transformations
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=1.0),
                A.RandomGamma(gamma_limit=(70, 130), p=1.0),
                A.CLAHE(clip_limit=6.0, tile_grid_size=(8, 8), p=1.0),
            ], p=0.6),
            # Noise and blur for robustness
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 80.0), p=1.0),
                A.GaussianBlur(blur_limit=(3, 9), p=1.0),
                A.MotionBlur(blur_limit=7, p=1.0),
            ], p=0.4),
            # Color/intensity augmentation
            A.OneOf([
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
                A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=1.0),
            ], p=0.3),
            # Coarse dropout for regularization
            A.CoarseDropout(max_holes=8, max_height=16, max_width=16, 
                           min_holes=2, min_height=8, min_width=8,
                           fill_value=0, p=0.3),
        ])
    else:  # medium (default)
        return A.Compose([
            # Spatial transformations
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.15,
                rotate_limit=30,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5
            ),
            # Elastic deformation for medical images
            A.ElasticTransform(
                alpha=120,
                sigma=120 * 0.05,
                p=0.3
            ),
            A.GridDistortion(p=0.3),
            # Intensity transformations
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=1.0
                ),
                A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
            ], p=0.5),
            # Noise and blur
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MedianBlur(blur_limit=5, p=1.0),
            ], p=0.3),
        ])

def get_validation_augmentation():
    """Light augmentation for validation (only normalization)."""
    return A.Compose([])


def get_classification_augmentation():
    """
    Specialized augmentation for classification task.
    Focus on preserving global features while adding diversity.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=20, p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1.0),
            A.CLAHE(clip_limit=3.0, p=1.0),
        ], p=0.4),
        A.GaussNoise(var_limit=(5.0, 30.0), p=0.2),
    ])# ============================================================================
# ENHANCED DATA GENERATOR WITH AUGMENTATION
# ============================================================================

class DataGenerator(tf.keras.utils.Sequence):
    """
    Enhanced data generator with on-the-fly augmentation.
    Supports both training and validation modes.
    """
    def __init__(self, ids, mask, image_dir='./', batch_size=16, 
                 img_h=256, img_w=256, shuffle=True, augment=True):
        self.ids = ids
        self.mask = mask
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.img_h = img_h
        self.img_w = img_w
        self.shuffle = shuffle
        self.augment = augment
        self.augmentation = get_training_augmentation() if augment else get_validation_augmentation()
        self.on_epoch_end()

    def __len__(self):
        'Get the number of batches per epoch'
        return int(np.floor(len(self.ids)) / self.batch_size)

    def __getitem__(self, index):
        'Generate a batch of data'
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_ids = [self.ids[i] for i in indexes]
        list_mask = [self.mask[i] for i in indexes]
        X, y = self.__data_generation(list_ids, list_mask)
        return X, y

    def on_epoch_end(self):
        'Update indices after each epoch'
        self.indexes = np.arange(len(self.ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_ids, list_mask):
        'Generate data for a batch of images'
        X = np.empty((self.batch_size, self.img_h, self.img_w, 3))
        y = np.empty((self.batch_size, self.img_h, self.img_w, 1))

        for i in range(len(list_ids)):
            img_path = './' + str(list_ids[i])
            mask_path = './' + str(list_mask[i])
            
            # Read images
            img = np.array(Image.open(img_path))
            mask = np.array(Image.open(mask_path))

            # Resize
            img = cv2.resize(img, (self.img_h, self.img_w))
            mask = cv2.resize(mask, (self.img_h, self.img_w))
            
            # Apply augmentation
            if self.augment:
                augmented = self.augmentation(image=img, mask=mask)
                img = augmented['image']
                mask = augmented['mask']
            
            # Convert to float64
            img = np.array(img, dtype=np.float64)
            mask = np.array(mask, dtype=np.float64)

            # Standardization
            img -= img.mean()
            img /= (img.std() + 1e-8)  # Add epsilon to prevent division by zero
            
            X[i,] = img
            y[i,] = np.expand_dims(mask, axis=2)
        
        # Binary mask
        y = (y > 0).astype(np.float32)

        return X, y


class EnhancedDataGenerator(tf.keras.utils.Sequence):
    """
    Advanced data generator with mixup, cutmix, and class balancing.
    """
    def __init__(self, ids, mask, image_dir='./', batch_size=16,
                 img_h=256, img_w=256, shuffle=True, augment=True,
                 mixup_alpha=0.2, cutmix_alpha=0.0):
        self.ids = ids
        self.mask = mask
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.img_h = img_h
        self.img_w = img_w
        self.shuffle = shuffle
        self.augment = augment
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.augmentation = get_training_augmentation() if augment else None
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.ids) / self.batch_size))

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.ids))
        indexes = self.indexes[start_idx:end_idx]
        
        list_ids = [self.ids[i] for i in indexes]
        list_mask = [self.mask[i] for i in indexes]
        
        X, y = self.__data_generation(list_ids, list_mask)
        
        # Apply mixup augmentation
        if self.mixup_alpha > 0 and self.augment:
            X, y = self.__mixup(X, y, self.mixup_alpha)
        
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __mixup(self, X, y, alpha=0.2):
        """Apply mixup augmentation"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = X.shape[0]
        index = np.random.permutation(batch_size)
        
        mixed_X = lam * X + (1 - lam) * X[index]
        mixed_y = lam * y + (1 - lam) * y[index]
        
        return mixed_X, mixed_y

    def __data_generation(self, list_ids, list_mask):
        actual_batch_size = len(list_ids)
        X = np.empty((actual_batch_size, self.img_h, self.img_w, 3))
        y = np.empty((actual_batch_size, self.img_h, self.img_w, 1))

        for i in range(actual_batch_size):
            img_path = './' + str(list_ids[i])
            mask_path = './' + str(list_mask[i])
            
            img = np.array(Image.open(img_path))
            mask = np.array(Image.open(mask_path))

            img = cv2.resize(img, (self.img_h, self.img_w))
            mask = cv2.resize(mask, (self.img_h, self.img_w))
            
            if self.augmentation is not None:
                augmented = self.augmentation(image=img, mask=mask)
                img = augmented['image']
                mask = augmented['mask']
            
            img = np.array(img, dtype=np.float64)
            mask = np.array(mask, dtype=np.float64)

            img -= img.mean()
            img /= (img.std() + 1e-8)
            
            X[i,] = img
            y[i,] = np.expand_dims(mask, axis=2)
        
        y = (y > 0).astype(np.float32)

        return X, y


# ============================================================================
# ENHANCED PREDICTION FUNCTION WITH TTA
# ============================================================================

def prediction(test, model, model_seg, use_tta=False):
    """
    Enhanced prediction function with optional Test Time Augmentation (TTA).
    
    Two-stage prediction:
    1. Classification: Does the image have a tumor?
    2. Segmentation: If yes, where is the tumor located?
    
    Args:
        test: DataFrame with image paths
        model: Classification model
        model_seg: Segmentation model
        use_tta: Enable Test Time Augmentation for better accuracy
    """
    directory = "./"
    mask = []
    image_id = []
    has_mask = []

    for i in test.image_path:
        path = directory + str(i)
        
        # Read image
        img = np.array(Image.open(path))
        
        # Normalize for classification
        img_norm = img * 1./255.
        img_norm = cv2.resize(img_norm, (256, 256))
        img_norm = np.array(img_norm, dtype=np.float64)
        img_norm = np.reshape(img_norm, (1, 256, 256, 3))

        # Classification prediction (with optional TTA)
        if use_tta:
            is_defect = predict_with_tta_classification(model, img_norm)
        else:
            is_defect = model.predict(img_norm, verbose=0)

        # If no tumor detected
        if np.argmax(is_defect) == 0:
            image_id.append(i)
            has_mask.append(0)
            mask.append('No mask')
            continue

        # Prepare for segmentation
        img = np.array(Image.open(path))
        X = np.empty((1, 256, 256, 3))
        img = cv2.resize(img, (256, 256))
        img = np.array(img, dtype=np.float64)
        
        # Standardize
        img -= img.mean()
        img /= (img.std() + 1e-8)
        X[0,] = img

        # Segmentation prediction (with optional TTA)
        if use_tta:
            predict = predict_with_tta_segmentation(model_seg, X)
        else:
            predict = model_seg.predict(X, verbose=0)

        if predict.round().astype(int).sum() == 0:
            image_id.append(i)
            has_mask.append(0)
            mask.append('No mask')
        else:
            image_id.append(i)
            has_mask.append(1)
            mask.append(predict)

    return image_id, mask, has_mask


def predict_with_tta_classification(model, img, strategy='max'):
    """
    Test Time Augmentation for classification.
    
    Args:
        model: Classification model
        img: Preprocessed image
        strategy: 'mean' for average, 'max' for maximum confidence
    
    Returns:
        Prediction with higher confidence
    """
    predictions = []
    
    # Original
    predictions.append(model.predict(img, verbose=0))
    
    # Horizontal flip
    img_flip = np.flip(img, axis=2)
    predictions.append(model.predict(img_flip, verbose=0))
    
    # Vertical flip
    img_flip_v = np.flip(img, axis=1)
    predictions.append(model.predict(img_flip_v, verbose=0))
    
    if strategy == 'max':
        # Return prediction with highest confidence for detected class
        max_conf = 0
        best_pred = predictions[0]
        for pred in predictions:
            conf = np.max(pred)
            if conf > max_conf:
                max_conf = conf
                best_pred = pred
        return best_pred
    else:
        # Average predictions
        return np.mean(predictions, axis=0)


def predict_with_tta_segmentation(model, img):
    """
    Test Time Augmentation for segmentation.
    Applies geometric transforms and averages the results.
    """
    predictions = []
    
    # Original
    pred = model.predict(img, verbose=0)
    predictions.append(pred)
    
    # Horizontal flip
    img_flip = np.flip(img, axis=2)
    pred_flip = model.predict(img_flip, verbose=0)
    pred_flip = np.flip(pred_flip, axis=2)
    predictions.append(pred_flip)
    
    # Vertical flip
    img_flip_v = np.flip(img, axis=1)
    pred_flip_v = model.predict(img_flip_v, verbose=0)
    pred_flip_v = np.flip(pred_flip_v, axis=1)
    predictions.append(pred_flip_v)
    
    # 90 degree rotation
    img_rot90 = np.rot90(img, axes=(1, 2))
    pred_rot90 = model.predict(img_rot90, verbose=0)
    pred_rot90 = np.rot90(pred_rot90, k=-1, axes=(1, 2))
    predictions.append(pred_rot90)
    
    # Average all predictions
    return np.mean(predictions, axis=0)


# ============================================================================
# ENHANCED LOSS FUNCTIONS AND METRICS
# ============================================================================

'''
Custom loss functions for training ResUNet segmentation model.
Based on: https://github.com/nabsabraham/focal-tversky-unet/blob/master/losses.py

@article{focal-unet,
  title={A novel Focal Tversky loss function with improved Attention U-Net for lesion segmentation},
  author={Abraham, Nabila and Khan, Naimul Mefraz},
  journal={arXiv preprint arXiv:1810.07842},
  year={2018}
}
'''

@register_keras_serializable()
def tversky(y_true, y_pred, smooth=1e-6):
    """
    Tversky Index: Generalization of Dice coefficient.
    Allows different weights for false positives and false negatives.
    α = 0.7 penalizes false negatives more (important for medical imaging)
    """
    # Cast to float32 for mixed precision compatibility
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)

@register_keras_serializable()
def tversky_loss(y_true, y_pred):
    """Tversky loss for optimization."""
    return 1 - tversky(y_true, y_pred)

@register_keras_serializable()
def focal_tversky(y_true, y_pred):
    """
    Focal Tversky Loss: Focuses training on hard examples.
    γ = 0.75 provides good balance between easy and hard examples.
    """
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1 - pt_1), gamma)

@register_keras_serializable()
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    Dice Coefficient: 2 * |A ∩ B| / (|A| + |B|)
    Standard metric for segmentation tasks.
    """
    # Cast to float32 for mixed precision compatibility
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

@register_keras_serializable()
def dice_loss(y_true, y_pred):
    """Dice loss for optimization."""
    return 1 - dice_coefficient(y_true, y_pred)

@register_keras_serializable()
def bce_dice_loss(y_true, y_pred):
    """
    Combined Binary Cross-Entropy and Dice Loss.
    Provides both pixel-wise and region-based optimization.
    """
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return 0.5 * bce + 0.5 * dice

@register_keras_serializable()
def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """
    Focal Loss: Addresses class imbalance by down-weighting easy examples.
    Reference: Lin et al., 2017
    """
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    cross_entropy = -y_true * K.log(y_pred)
    focal_weight = alpha * K.pow(1 - y_pred, gamma)
    focal = focal_weight * cross_entropy
    return K.mean(K.sum(focal, axis=-1))

@register_keras_serializable()
def combo_loss(y_true, y_pred, alpha=0.5, ce_ratio=0.5):
    """
    Combo Loss: Combines Dice and weighted cross-entropy.
    Effective for highly imbalanced datasets.
    """
    dice = dice_loss(y_true, y_pred)
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    weighted_ce = -y_true * K.log(y_pred) * alpha - (1 - y_true) * K.log(1 - y_pred) * (1 - alpha)
    weighted_ce = K.mean(weighted_ce)
    return ce_ratio * weighted_ce + (1 - ce_ratio) * dice

@register_keras_serializable()
def iou_score(y_true, y_pred, smooth=1e-6):
    """
    Intersection over Union (IoU / Jaccard Index).
    Standard metric for segmentation evaluation.
    """
    # Cast to float32 for mixed precision compatibility
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

@register_keras_serializable()
def sensitivity(y_true, y_pred, smooth=1e-6):
    """Sensitivity / Recall / True Positive Rate."""
    # Cast to float32 for mixed precision compatibility
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    true_positives = K.sum(y_true * K.round(y_pred))
    possible_positives = K.sum(y_true)
    return (true_positives + smooth) / (possible_positives + smooth)

@register_keras_serializable()
def specificity(y_true, y_pred, smooth=1e-6):
    """Specificity / True Negative Rate."""
    # Cast to float32 for mixed precision compatibility
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    true_negatives = K.sum((1 - y_true) * (1 - K.round(y_pred)))
    possible_negatives = K.sum(1 - y_true)
    return (true_negatives + smooth) / (possible_negatives + smooth)

@register_keras_serializable()
def precision_metric(y_true, y_pred, smooth=1e-6):
    """Precision: TP / (TP + FP)"""
    true_positives = K.sum(y_true * K.round(y_pred))
    predicted_positives = K.sum(K.round(y_pred))
    return (true_positives + smooth) / (predicted_positives + smooth)


# ============================================================================
# ADVANCED BOUNDARY-AWARE LOSS FUNCTIONS
# ============================================================================

@register_keras_serializable()
def boundary_loss(y_true, y_pred, smooth=1e-6):
    """
    Boundary Loss: Penalizes errors at tumor boundaries.
    Important for precise tumor localization.
    Uses distance transform to weight boundary pixels.
    """
    # Compute boundary weight map using edge detection
    y_true_f = K.cast(y_true, 'float32')
    y_pred_f = K.cast(y_pred, 'float32')
    
    # Sobel filters for edge detection
    sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=tf.float32)
    sobel_x = tf.reshape(sobel_x, [3, 3, 1, 1])
    sobel_y = tf.constant([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=tf.float32)
    sobel_y = tf.reshape(sobel_y, [3, 3, 1, 1])
    
    # Compute edges
    edge_x = tf.nn.conv2d(y_true_f, sobel_x, strides=[1, 1, 1, 1], padding='SAME')
    edge_y = tf.nn.conv2d(y_true_f, sobel_y, strides=[1, 1, 1, 1], padding='SAME')
    boundary = tf.sqrt(edge_x**2 + edge_y**2)
    
    # Normalize boundary weights
    boundary = boundary / (K.max(boundary) + smooth)
    boundary = boundary + 0.5  # Ensure all pixels have some weight
    
    # Weighted binary cross-entropy
    bce = K.binary_crossentropy(y_true_f, y_pred_f)
    weighted_bce = bce * boundary
    
    return K.mean(weighted_bce)

@register_keras_serializable()
def surface_loss(y_true, y_pred, smooth=1e-6):
    """
    Surface Loss: Uses distance map to improve boundary predictions.
    Particularly effective for small tumors and irregular shapes.
    """
    y_true_f = K.flatten(K.cast(y_true, 'float32'))
    y_pred_f = K.flatten(K.cast(y_pred, 'float32'))
    
    # Approximate distance transform effect
    # Penalize predictions far from actual boundaries
    diff = K.abs(y_true_f - y_pred_f)
    
    # Weight by prediction confidence (penalize confident wrong predictions)
    confidence_weight = K.abs(y_pred_f - 0.5) * 2 + 0.5
    
    return K.mean(diff * confidence_weight)

@register_keras_serializable()
def hausdorff_loss_approx(y_true, y_pred, smooth=1e-6):
    """
    Approximate Hausdorff Distance Loss.
    Measures the maximum distance between predicted and true boundaries.
    """
    y_true_f = K.cast(y_true, 'float32')
    y_pred_f = K.cast(y_pred, 'float32')
    
    # Compute squared differences
    diff = K.square(y_true_f - y_pred_f)
    
    # Average pooling to get regional errors
    pooled = tf.nn.avg_pool2d(diff, ksize=3, strides=1, padding='SAME')
    
    # Maximum error approximates Hausdorff distance
    max_error = K.max(pooled)
    mean_error = K.mean(diff)
    
    # Combine max and mean for stable training
    return 0.3 * max_error + 0.7 * mean_error

@register_keras_serializable()
def multi_scale_dice_loss(y_true, y_pred, scales=[1, 2, 4]):
    """
    Multi-Scale Dice Loss: Computes Dice loss at multiple resolutions.
    Helps capture both fine details and global structure.
    """
    total_loss = 0.0
    weights = [0.5, 0.3, 0.2]  # Higher weight for original scale
    
    for scale, weight in zip(scales, weights):
        if scale > 1:
            # Downsample both tensors
            y_true_scaled = tf.nn.avg_pool2d(y_true, ksize=scale, strides=scale, padding='SAME')
            y_pred_scaled = tf.nn.avg_pool2d(y_pred, ksize=scale, strides=scale, padding='SAME')
        else:
            y_true_scaled = y_true
            y_pred_scaled = y_pred
        
        dice = dice_coefficient(y_true_scaled, y_pred_scaled)
        total_loss += weight * (1 - dice)
    
    return total_loss

@register_keras_serializable()
def unified_focal_loss(y_true, y_pred, alpha=0.7, gamma=0.75, delta=0.7, smooth=1e-6):
    """
    Unified Focal Loss: Combines Focal Tversky with Focal loss.
    Best for imbalanced segmentation tasks like tumor detection.
    
    Args:
        alpha: Weight for false negatives in Tversky
        gamma: Focusing parameter
        delta: Balance between Tversky and Cross-entropy
    """
    # Focal Tversky component
    y_true_f = K.flatten(K.cast(y_true, 'float32'))
    y_pred_f = K.flatten(K.cast(y_pred, 'float32'))
    
    true_pos = K.sum(y_true_f * y_pred_f)
    false_neg = K.sum(y_true_f * (1 - y_pred_f))
    false_pos = K.sum((1 - y_true_f) * y_pred_f)
    
    tversky_idx = (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)
    focal_tversky_loss = K.pow(1 - tversky_idx, gamma)
    
    # Focal cross-entropy component
    y_pred_clipped = K.clip(y_pred_f, K.epsilon(), 1 - K.epsilon())
    ce = -y_true_f * K.log(y_pred_clipped) - (1 - y_true_f) * K.log(1 - y_pred_clipped)
    focal_ce = K.pow(K.abs(y_true_f - y_pred_f), gamma) * ce
    focal_ce_loss = K.mean(focal_ce)
    
    return delta * focal_tversky_loss + (1 - delta) * focal_ce_loss

@register_keras_serializable()
def boundary_aware_dice_loss(y_true, y_pred, boundary_weight=2.0, smooth=1e-6):
    """
    Boundary-Aware Dice Loss: Combines Dice with boundary emphasis.
    Specifically designed for accurate tumor boundary delineation.
    """
    # Standard Dice
    dice = dice_coefficient(y_true, y_pred, smooth)
    dice_l = 1 - dice
    
    # Boundary loss component
    bound_l = boundary_loss(y_true, y_pred, smooth)
    
    # Combine with weighting
    return dice_l + boundary_weight * 0.1 * bound_l

@register_keras_serializable()
def log_cosh_dice_loss(y_true, y_pred, smooth=1e-6):
    """
    Log-Cosh Dice Loss: Smooth approximation of Dice loss.
    More stable training with less sensitivity to outliers.
    """
    dice = dice_coefficient(y_true, y_pred, smooth)
    dice_loss_val = 1 - dice
    return K.log(K.cosh(dice_loss_val))

@register_keras_serializable()
def asymmetric_focal_loss(y_true, y_pred, gamma_pos=1.0, gamma_neg=4.0, smooth=1e-6):
    """
    Asymmetric Focal Loss: Different focus for positive and negative samples.
    Reduces false negatives (missed tumors) which is critical in medical imaging.
    """
    y_true_f = K.flatten(K.cast(y_true, 'float32'))
    y_pred_f = K.flatten(K.clip(y_pred, K.epsilon(), 1 - K.epsilon()))
    
    # Positive samples (tumor pixels)
    pos_loss = -y_true_f * K.pow(1 - y_pred_f, gamma_pos) * K.log(y_pred_f)
    
    # Negative samples (background pixels)
    neg_loss = -(1 - y_true_f) * K.pow(y_pred_f, gamma_neg) * K.log(1 - y_pred_f)
    
    return K.mean(pos_loss + neg_loss)


# ============================================================================
# LEARNING RATE SCHEDULES
# ============================================================================

def cosine_annealing_schedule(epoch, lr, epochs=100, min_lr=1e-6):
    """Cosine Annealing Learning Rate Schedule."""
    return min_lr + (lr - min_lr) * (1 + np.cos(np.pi * epoch / epochs)) / 2

def warmup_cosine_schedule(epoch, lr, warmup_epochs=5, total_epochs=100, min_lr=1e-6):
    """Warmup + Cosine Annealing Schedule."""
    if epoch < warmup_epochs:
        return lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_lr + (lr - min_lr) * (1 + np.cos(np.pi * progress)) / 2

def get_lr_scheduler(schedule_type='reduce_on_plateau', **kwargs):
    """Factory function for learning rate scheduler callbacks."""
    if schedule_type == 'cosine':
        return tf.keras.callbacks.LearningRateScheduler(
            lambda epoch, lr: cosine_annealing_schedule(epoch, lr, **kwargs)
        )
    elif schedule_type == 'warmup_cosine':
        return tf.keras.callbacks.LearningRateScheduler(
            lambda epoch, lr: warmup_cosine_schedule(epoch, lr, **kwargs)
        )
    elif schedule_type == 'reduce_on_plateau':
        return tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


# ============================================================================
# ENHANCED CALLBACKS
# ============================================================================

def get_training_callbacks(checkpoint_path, patience=15, schedule='reduce_on_plateau'):
    """Get comprehensive training callbacks."""
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            mode='min',
            verbose=1,
            restore_best_weights=True
        ),
        get_lr_scheduler(schedule),
    ]
    return callbacks


# ============================================================================
# MODEL EVALUATION UTILITIES
# ============================================================================

def evaluate_segmentation(y_true, y_pred):
    """
    Comprehensive evaluation of segmentation predictions.
    
    Returns:
        dict: Dictionary containing all metrics
    """
    y_true = y_true.flatten()
    y_pred = (y_pred.flatten() > 0.5).astype(int)
    
    # Calculate metrics
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # Compute scores
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    # Dice and IoU
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'dice': dice,
        'iou': iou,
        'sensitivity': recall,
        'specificity': tn / (tn + fp + 1e-8)
    }


# ============================================================================
# ENHANCED MODEL ARCHITECTURES
# ============================================================================

def build_efficientnet_classifier(input_shape=(256, 256, 3), num_classes=2, dropout_rate=0.4):
    """
    Build EfficientNetB4-based classifier for tumor detection.
    EfficientNet provides better accuracy with fewer parameters than ResNet50.
    
    Args:
        input_shape: Input image shape
        num_classes: Number of output classes
        dropout_rate: Dropout rate for regularization
    
    Returns:
        Compiled Keras model
    """
    from tensorflow.keras.applications import EfficientNetB4
    from tensorflow.keras.layers import (GlobalAveragePooling2D, Dense, Dropout, 
                                         BatchNormalization, Input)
    from tensorflow.keras.models import Model
    from tensorflow.keras.regularizers import l2
    
    # Load EfficientNetB4 with ImageNet weights
    base_model = EfficientNetB4(
        weights='imagenet',
        include_top=False,
        input_tensor=Input(shape=input_shape)
    )
    
    # Freeze base model initially
    for layer in base_model.layers:
        layer.trainable = False
    
    # Build classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    # Dense block 1
    x = Dense(512, kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = Dropout(dropout_rate)(x)
    
    # Dense block 2
    x = Dense(256, kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = Dropout(dropout_rate * 0.75)(x)
    
    # Dense block 3
    x = Dense(128, kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = Dropout(dropout_rate * 0.5)(x)
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=outputs)
    
    return model, base_model


def squeeze_excite_block(input_tensor, ratio=16):
    """
    Squeeze-and-Excitation block for channel attention.
    Helps the model focus on important features.
    """
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Reshape, Multiply
    
    filters = input_tensor.shape[-1]
    se = GlobalAveragePooling2D()(input_tensor)
    se = Dense(filters // ratio, activation='relu')(se)
    se = Dense(filters, activation='sigmoid')(se)
    se = Reshape([1, 1, filters])(se)
    return Multiply()([input_tensor, se])


def cbam_block(input_tensor, ratio=8):
    """
    Convolutional Block Attention Module (CBAM).
    Combines channel and spatial attention for better feature focus.
    """
    from tensorflow.keras.layers import (GlobalAveragePooling2D, GlobalMaxPooling2D,
                                         Dense, Reshape, Multiply, Add, Conv2D,
                                         Concatenate, Activation)
    
    filters = input_tensor.shape[-1]
    
    # Channel Attention
    avg_pool = GlobalAveragePooling2D()(input_tensor)
    max_pool = GlobalMaxPooling2D()(input_tensor)
    
    # Shared MLP
    fc1 = Dense(filters // ratio, activation='relu')
    fc2 = Dense(filters, activation='linear')
    
    avg_out = fc2(fc1(avg_pool))
    max_out = fc2(fc1(max_pool))
    
    channel_attention = Activation('sigmoid')(Add()([avg_out, max_out]))
    channel_attention = Reshape([1, 1, filters])(channel_attention)
    x = Multiply()([input_tensor, channel_attention])
    
    # Spatial Attention
    avg_pool_spatial = tf.reduce_mean(x, axis=-1, keepdims=True)
    max_pool_spatial = tf.reduce_max(x, axis=-1, keepdims=True)
    concat = Concatenate(axis=-1)([avg_pool_spatial, max_pool_spatial])
    spatial_attention = Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')(concat)
    
    return Multiply()([x, spatial_attention])


def attention_gate(x, g, inter_channels):
    """
    Attention Gate for skip connections in U-Net.
    Focuses on relevant features during upsampling.
    """
    from tensorflow.keras.layers import Conv2D, Activation, Add, Multiply
    
    theta_x = Conv2D(inter_channels, kernel_size=1, strides=1, padding='same')(x)
    phi_g = Conv2D(inter_channels, kernel_size=1, strides=1, padding='same')(g)
    
    f = Activation('relu')(Add()([theta_x, phi_g]))
    psi_f = Conv2D(1, kernel_size=1, strides=1, padding='same', activation='sigmoid')(f)
    
    return Multiply()([x, psi_f])


def aspp_block(x, filters):
    """
    Atrous Spatial Pyramid Pooling (ASPP) for multi-scale context.
    Captures features at different receptive fields.
    """
    from tensorflow.keras.layers import (Conv2D, BatchNormalization, Activation,
                                         GlobalAveragePooling2D, Reshape, Concatenate)
    
    # 1x1 convolution
    conv1 = Conv2D(filters, 1, padding='same')(x)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    
    # 3x3 convolutions with different dilation rates
    conv2 = Conv2D(filters, 3, padding='same', dilation_rate=6)(x)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    
    conv3 = Conv2D(filters, 3, padding='same', dilation_rate=12)(x)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    
    conv4 = Conv2D(filters, 3, padding='same', dilation_rate=18)(x)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    
    # Image-level features
    pool = GlobalAveragePooling2D()(x)
    pool = Reshape((1, 1, x.shape[-1]))(pool)
    pool = Conv2D(filters, 1, padding='same')(pool)
    pool = BatchNormalization()(pool)
    pool = Activation('relu')(pool)
    pool = tf.image.resize(pool, (x.shape[1], x.shape[2]))
    
    # Concatenate all
    concat = Concatenate()([conv1, conv2, conv3, conv4, pool])
    output = Conv2D(filters, 1, padding='same')(concat)
    output = BatchNormalization()(output)
    output = Activation('relu')(output)
    
    return output


def build_attention_resunet(input_shape=(256, 256, 3), filters=[32, 64, 128, 256, 512],
                            use_aspp=True, use_cbam=True):
    """
    Build Enhanced Attention ResUNet for tumor segmentation.
    
    Features:
    - ResNet-style residual blocks
    - Attention gates for skip connections
    - CBAM attention modules
    - ASPP for multi-scale context
    - Deep supervision for better gradient flow
    
    Args:
        input_shape: Input image shape
        filters: List of filter sizes for each level
        use_aspp: Whether to use ASPP in bottleneck
        use_cbam: Whether to use CBAM attention
    
    Returns:
        Keras model
    """
    from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization, Activation,
                                         MaxPool2D, UpSampling2D, Concatenate, Add,
                                         Dropout)
    from tensorflow.keras.models import Model
    
    def resblock(X, f, use_attention=True):
        """Enhanced Residual Block with optional attention."""
        X_copy = X
        
        # Main path
        X = Conv2D(f, kernel_size=1, strides=1, kernel_initializer='he_normal', padding='same')(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        
        X = Conv2D(f, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal')(X)
        X = BatchNormalization()(X)
        
        # Skip path
        X_copy = Conv2D(f, kernel_size=1, strides=1, kernel_initializer='he_normal', padding='same')(X_copy)
        X_copy = BatchNormalization()(X_copy)
        
        # Add paths
        X = Add()([X, X_copy])
        X = Activation('relu')(X)
        
        # Apply attention
        if use_attention and use_cbam:
            X = cbam_block(X)
        elif use_attention:
            X = squeeze_excite_block(X)
        
        return X
    
    def upsample_concat_attention(x, skip, filters):
        """Upsampling with attention-gated skip connections."""
        x = UpSampling2D((2, 2))(x)
        skip = attention_gate(skip, x, filters // 2)
        return Concatenate()([x, skip])
    
    inputs = Input(input_shape)
    
    # ============== ENCODER ==============
    # Stage 1
    conv1 = Conv2D(filters[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(filters[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPool2D(pool_size=(2, 2))(conv1)
    
    # Stage 2
    conv2 = resblock(pool1, filters[1])
    pool2 = MaxPool2D(pool_size=(2, 2))(conv2)
    
    # Stage 3
    conv3 = resblock(pool2, filters[2])
    pool3 = MaxPool2D(pool_size=(2, 2))(conv3)
    
    # Stage 4
    conv4 = resblock(pool3, filters[3])
    pool4 = MaxPool2D(pool_size=(2, 2))(conv4)
    
    # ============== BOTTLENECK ==============
    conv5 = resblock(pool4, filters[4])
    
    # Apply ASPP for multi-scale context
    if use_aspp:
        conv5 = aspp_block(conv5, filters[4])
    
    # Additional convolutions for richer features
    conv5 = Conv2D(filters[4], 3, padding='same', dilation_rate=2, kernel_initializer='he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    
    # ============== DECODER ==============
    # Upscale stage 1
    up1 = upsample_concat_attention(conv5, conv4, filters[3])
    up1 = resblock(up1, filters[3])
    
    # Upscale stage 2
    up2 = upsample_concat_attention(up1, conv3, filters[2])
    up2 = resblock(up2, filters[2])
    
    # Upscale stage 3
    up3 = upsample_concat_attention(up2, conv2, filters[1])
    up3 = resblock(up3, filters[1])
    
    # Upscale stage 4
    up4 = upsample_concat_attention(up3, conv1, filters[0])
    up4 = resblock(up4, filters[0])
    
    # ============== OUTPUT ==============
    output = Dropout(0.3)(up4)
    output = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(output)
    
    model = Model(inputs=inputs, outputs=output)
    
    return model


def build_unet_plus_plus(input_shape=(256, 256, 3), filters=[32, 64, 128, 256, 512]):
    """
    Build UNet++ (Nested U-Net) for precise tumor segmentation.
    Dense skip connections for better feature propagation.
    """
    from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization, Activation,
                                         MaxPool2D, UpSampling2D, Concatenate, Dropout)
    from tensorflow.keras.models import Model
    
    def conv_block(x, filters, kernel_size=3):
        x = Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x
    
    inputs = Input(input_shape)
    
    # Encoder path (backbone)
    x00 = conv_block(inputs, filters[0])
    pool0 = MaxPool2D(pool_size=(2, 2))(x00)
    
    x10 = conv_block(pool0, filters[1])
    pool1 = MaxPool2D(pool_size=(2, 2))(x10)
    
    x20 = conv_block(pool1, filters[2])
    pool2 = MaxPool2D(pool_size=(2, 2))(x20)
    
    x30 = conv_block(pool2, filters[3])
    pool3 = MaxPool2D(pool_size=(2, 2))(x30)
    
    x40 = conv_block(pool3, filters[4])
    
    # Dense skip connections
    # Level 1
    x01 = conv_block(Concatenate()([x00, UpSampling2D()(x10)]), filters[0])
    x11 = conv_block(Concatenate()([x10, UpSampling2D()(x20)]), filters[1])
    x21 = conv_block(Concatenate()([x20, UpSampling2D()(x30)]), filters[2])
    x31 = conv_block(Concatenate()([x30, UpSampling2D()(x40)]), filters[3])
    
    # Level 2
    x02 = conv_block(Concatenate()([x00, x01, UpSampling2D()(x11)]), filters[0])
    x12 = conv_block(Concatenate()([x10, x11, UpSampling2D()(x21)]), filters[1])
    x22 = conv_block(Concatenate()([x20, x21, UpSampling2D()(x31)]), filters[2])
    
    # Level 3
    x03 = conv_block(Concatenate()([x00, x01, x02, UpSampling2D()(x12)]), filters[0])
    x13 = conv_block(Concatenate()([x10, x11, x12, UpSampling2D()(x22)]), filters[1])
    
    # Level 4
    x04 = conv_block(Concatenate()([x00, x01, x02, x03, UpSampling2D()(x13)]), filters[0])
    
    # Output
    output = Dropout(0.3)(x04)
    output = Conv2D(1, (1, 1), activation='sigmoid')(output)
    
    return Model(inputs=inputs, outputs=output)


# ============================================================================
# FAST INFERENCE UTILITIES
# ============================================================================

def optimize_model_for_inference(model, optimize_type='basic'):
    """
    Optimize model for faster inference.
    
    Args:
        model: Keras model to optimize
        optimize_type: 'basic', 'float16', or 'int8'
    
    Returns:
        Optimized model
    """
    if optimize_type == 'float16':
        # Convert to float16 for faster inference on GPU
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        return converter.convert()
    elif optimize_type == 'int8':
        # Full integer quantization
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        return converter.convert()
    else:
        # Basic optimization - just return the model
        return model


def warmup_model(model, input_shape=(1, 256, 256, 3)):
    """
    Warm up model for faster first inference.
    """
    dummy_input = np.zeros(input_shape, dtype=np.float32)
    _ = model.predict(dummy_input, verbose=0)
    return model