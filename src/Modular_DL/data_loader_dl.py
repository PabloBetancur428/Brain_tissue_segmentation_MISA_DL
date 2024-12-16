import torch
from torch.utils.data import Dataset
import numpy as np
import SimpleITK as sitk

class Dataloader_2D(Dataset):
    def __init__(self, image_files, label_files = None, transform=None, test_mode = False):
        """
        Initialize the Dataloader to load 2D slices from 3D medical images.
        Args:
            image_files (list): List of paths to image files.
            label_files (list): List of paths to corresponding label files.
            transform (callable): Optional transformations applied to slices.
        """
        self.transform = transform
        self.test_mode = test_mode
        #self.image_label_pairs = list(zip(image_files, label_files))
        self.image_files = image_files
        self.label_files = label_files if not test_mode else None

        self.slice_indices = []  # To index slices across volumes
        self.volumes_cache = []  # Cache to hold image and label data

        # Load all volumes and build an index for slices
        for i, img_file in enumerate(image_files):
            # Load images and labels using SimpleITK
            img_nii = sitk.ReadImage(img_file)
            img_data = sitk.GetArrayFromImage(img_nii)  # Shape: [Depth, H, W]
            # Correct flipping
            img_data = np.flipud(img_data)  # Vertical flip
            img_data = np.fliplr(img_data)  # Horizontal flip

            if not test_mode:
                lbl_nii = sitk.ReadImage(label_files[i])
                # Convert to numpy arrays
                lbl_data = sitk.GetArrayFromImage(lbl_nii)  # Shape: [Depth, H, W]
                # Correct flipping
                lbl_data = np.flipud(lbl_data)  # Apply same flipping to labels
                lbl_data = np.fliplr(lbl_data)
            else:
                lbl_data = None
           

            # Cache the volumes
            self.volumes_cache.append((img_data, lbl_data))

            # Index slices for iteration
            depth = img_data.shape[0]  # Number of slices along Depth
            for d in range(depth):
                self.slice_indices.append((i, d))

    def __len__(self):
        return len(self.slice_indices)

    def __getitem__(self, idx):
        """
        Extract a specific slice from the cached volumes.
        Args:
            idx (int): Index of the slice.
        Returns:
            tuple: (image_tensor, label_tensor) - PyTorch tensors.
        """
        vol_i, slice_i = self.slice_indices[idx]
        img_data, lbl_data = self.volumes_cache[vol_i]

        # Extract the corresponding slice
        img_slice = img_data[slice_i, :, :].copy()  # Shape: [H, W]

        if self.test_mode:
            lbl_slice = np.zeros_like(img_slice, dtype=np.int64) #Dummy label
        else:
            lbl_slice = lbl_data[slice_i, :, :].copy()


        # Prepare the sample
        sample = {'image': img_slice, 'label': lbl_slice}

        # Apply transformations if specified
        if self.transform:
            sample = self.transform(sample)

        # Convert to PyTorch tensors
        image_tensor = torch.tensor(sample['image'], dtype=torch.float32).unsqueeze(0)  # Add channel dim
        label_tensor = torch.tensor(sample['label'], dtype=torch.long)

        return image_tensor, label_tensor
