import matplotlib.pyplot as plt

# Function to plot sagittal view slices
def plot_dataloader_sagittal(dataloader):
    """
    Plot sagittal slices and corresponding labels for a single batch.
    Args:
        dataloader: PyTorch DataLoader object.
    """
    # Fetch a single batch
    for images, labels in dataloader:
        # Convert to NumPy arrays
        images = images.numpy()  # Shape: [Batch, 1, H, W]
        labels = labels.numpy()  # Shape: [Batch, H, W]

        # Plot each slice in the batch
        batch_size = images.shape[0]
        plt.figure(figsize=(8, 4 * batch_size))
        
        for i in range(batch_size):  # Iterate through batch
            plt.subplot(batch_size, 2, 2 * i + 1)
            plt.title(f"Image Slice {i+1} (Sagittal View)")
            plt.imshow(images[i, 0, :, :], cmap="gray")  # Single slice
            plt.axis("off")

            plt.subplot(batch_size, 2, 2 * i + 2)
            plt.title(f"Label Slice {i+1}")
            plt.imshow(labels[i, :, :], cmap="jet")  # Corresponding label
            plt.axis("off")

        plt.tight_layout()
        plt.show()
        break  # Stop after the first batch
