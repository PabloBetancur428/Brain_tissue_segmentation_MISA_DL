import torch
import random
import numpy as np
from torchvision.transforms import Compose
import torchvision.transforms.functional as TF
import cv2
from scipy.ndimage import gaussian_filter, map_coordinates

    
#Best normalization for consistent visual scaling and values between 0 and 1
class MinMaxNormalize:
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        min_val, max_val = image.min(), image.max()
        image = (image - min_val) / (max_val - min_val + 1e-8)  # Scale to [0, 1]
        return {'image': image, 'label': label}

# Example: Random Flip Transformation
# Random Flip Transformation
class RandomFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if np.random.random() < self.p:
            axis = np.random.choice([0, 1])
            image = np.flip(image, axis=axis).copy()
            label = np.flip(label, axis=axis).copy()
        return {'image': image, 'label': label}

# Random Rotation
class RandomRotate:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if np.random.random() < self.p:
            k = np.random.choice([0, 2])  # Rotate 0, 90, 180, or 270 degrees
            image = np.rot90(image, k, axes=(0, 1)).copy()
            label = np.rot90(label, k, axes=(0, 1)).copy()
        return {'image': image, 'label': label}
    
class GaussianBlur:
    def __init__(self, kernel_size=5, sigma=(0.1, 2.0), p=0.5):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.p = p

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if np.random.random() < self.p:
            sigma = np.random.uniform(self.sigma[0], self.sigma[1])
            image = cv2.GaussianBlur(image, (self.kernel_size, self.kernel_size), sigma)
        return {'image': image, 'label': label}

class Cutout:
    def __init__(self, size=16, p=0.5):
        self.size = size
        self.p = p

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if np.random.random() < self.p:
            h, w = image.shape[:2]
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            x1 = np.clip(x - self.size // 2, 0, w)
            x2 = np.clip(x + self.size // 2, 0, w)
            y1 = np.clip(y - self.size // 2, 0, h)
            y2 = np.clip(y + self.size // 2, 0, h)
            image[y1:y2, x1:x2] = 0
        return {'image': image, 'label': label}
    
class ElasticTransform:
    def __init__(self, alpha=1000, sigma=30, p=0.5):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if np.random.random() < self.p:
            shape = image.shape
            dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma) * self.alpha
            dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma) * self.alpha
            x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
            indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
            image = map_coordinates(image, indices, order=1).reshape(shape)
            label = map_coordinates(label, indices, order=0).reshape(shape)
        return {'image': image, 'label': label}
    
class RandomIntensityScale:
    def __init__(self, scale_range=(0.95, 1.05), shift_range=(-0.02, 0.02), p=0.5):
        self.scale_range = scale_range
        self.shift_range = shift_range
        self.p = p

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if np.random.random() < self.p:
            scale = np.random.uniform(*self.scale_range)
            shift = np.random.uniform(*self.shift_range)
            image = (image * scale) + shift
        return {'image': image, 'label': label}
    
# Example: Compose Training Transformations
def get_training_transformations():
    return Compose([
        RandomRotate(p = 0.3),
        RandomFlip(p = 0.3),
        RandomIntensityScale(p = 0.5),
        Cutout(p = 0.5),
        MinMaxNormalize(),  # Normalize
    ])

# Example: Validation Transformations
def get_validation_transformations():
    return Compose([
        MinMaxNormalize(),  # Only normalize
    ])

def get_test_transformations(): #Only normalization to ensure the data matches the same distribution as the training and validation inputs.
    return Compose([
        MinMaxNormalize(),  # Only normalize
    ])

