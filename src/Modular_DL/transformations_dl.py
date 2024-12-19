import torch
import random
import numpy as np
from torchvision.transforms import Compose
import torchvision.transforms.functional as TF
import cv2

    
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
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if np.random.rand() > 0.5:  # Random horizontal flip
            image = np.fliplr(image).copy()
            label = np.fliplr(label).copy()
        if np.random.rand() > 0.5:  # Random vertical flip
            image = np.flipud(image).copy()
            label = np.flipud(label).copy()
        return {'image': image, 'label': label}

# Random Rotation
class RandomRotate:
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        k = np.random.choice([0, 1, 2, 3])  # Rotate 0, 90, 180, or 270 degrees
        image = np.rot90(image, k, axes=(0, 1)).copy()
        label = np.rot90(label, k, axes=(0, 1)).copy()
        return {'image': image, 'label': label}
    
class RandomIntensityScale:
    def __init__(self, scale_range=(0.9, 1.1), shift_range=(-0.1, 0.1)):
        """
        Parameters:
        - scale_range (tuple): A tuple (min_scale, max_scale) specifying the range from which 
          a random scale factor is chosen. The image intensities are multiplied by this factor.
        - shift_range (tuple): A tuple (min_shift, max_shift) specifying the range for a random 
          intensity shift. After scaling, a shift is added to the intensities.

        For example, if scale_range=(0.9, 1.1), then at runtime the method may pick a scale of 1.05, 
        which means image = image * 1.05.
        
        If shift_range=(-0.1, 0.1), then it might pick shift=0.05, which means image = image + 0.05 
        after scaling.
        """
        self.scale_range = scale_range
        self.shift_range = shift_range

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        scale = np.random.uniform(*self.scale_range)   # Random scale factor
        shift = np.random.uniform(*self.shift_range)   # Random intensity offset
        image = (image * scale) + shift
        return {'image': image, 'label': label}

class RandomCrop:
    def __init__(self, crop_size=(128,128)):
        """
        Parameters:
        - crop_size (tuple): The desired output (height, width). A random region of the original 
          image/label is extracted to this size.

        This helps the model focus on smaller patches, potentially increasing data variability 
        and reducing overfitting.
        """
        self.crop_size = crop_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        h, w = image.shape
        ch, cw = self.crop_size
        
        if ch > h or cw > w:
            # If desired crop is larger than image, just return original or consider padding
            return {'image': image, 'label': label}

        # Random top-left corner for cropping
        top = np.random.randint(0, h - ch + 1)
        left = np.random.randint(0, w - cw + 1)

        image_cropped = image[top:top+ch, left:left+cw]
        label_cropped = label[top:top+ch, left:left+cw]

        return {'image': image_cropped, 'label': label_cropped}
    
class GaussianBlur:
    def __init__(self, kernel_size=3, sigma=1.0):
        """
        Parameters:
        - kernel_size (int): Size of the Gaussian kernel. Must be odd. Larger kernels blur more.
        - sigma (float): Standard deviation of the Gaussian. Higher sigma = more blurring.

        Gaussian blur smooths out noise and small details, making the model robust to 
        differences in image sharpness.
        """
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # Gaussian blur only on image, not label (labels should remain crisp)
        blurred_image = cv2.GaussianBlur(image, (self.kernel_size, self.kernel_size), self.sigma)
        return {'image': blurred_image, 'label': label}
    
# Example: Compose Training Transformations
def get_training_transformations():
    return Compose([
        RandomFlip(),
        RandomRotate(),
        RandomIntensityScale(scale_range=(0.9,1.1), shift_range=(-0.05,0.05)),
        GaussianBlur(kernel_size = 3, sigma = 1.0),
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

