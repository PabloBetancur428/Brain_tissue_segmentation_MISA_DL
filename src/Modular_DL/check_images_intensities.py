import matplotlib.pyplot as plt
import numpy as np
import torch

def check_dataloader_images(dataloader, num_samples=1):
    """
    Check the range of intensities, histogram of intensities, and visualize image slices and labels.
    
    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
        num_samples (int): Number of samples to check (default is 1).
    """
    for idx, (images, labels) in enumerate(dataloader):
        if idx >= num_samples:
            break
        
        # Extract single slice
        image = images[0, 0].numpy()  # Shape: [H, W]
        label = labels[0].numpy()     # Shape: [H, W]
        
        # Intensity Range
        print(f"Image Intensity Range: Min = {image.min():.4f}, Max = {image.max():.4f}")
        
        # Histogram of intensities
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.hist(image.flatten(), bins=50, color='blue', alpha=0.7)
        plt.title("Image Intensity Histogram")
        plt.xlabel("Intensity")
        plt.ylabel("Frequency")
        
        # Visualize image slice
        plt.subplot(1, 3, 2)
        plt.imshow(image, cmap='gray')
        plt.title("Image Slice")
        plt.axis("off")
        
        # Visualize label
        plt.subplot(1, 3, 3)
        plt.imshow(label, cmap='jet')
        plt.title("Label Slice")
        plt.axis("off")
        
        plt.tight_layout()
        plt.show()

        # Print a summary for sanity check
        print(f"Label Unique Values: {np.unique(label)}")

