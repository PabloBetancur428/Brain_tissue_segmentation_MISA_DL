import torch
from torch.utils.data import Dataset
import numpy as np
import SimpleITK as sitk
import random

class PatchDataloader_2D(Dataset):
    def __init__(
        self,
        image_files, 
        label_files=None,
        transform=None,
        test_mode=False,
        patch_size=(128, 128),
        num_patches_per_slice=5
    ):
        """
        A patch-based dataset loader. For each 3D volume, we:
         1) Load it & cache it (like in Dataloader_2D).
         2) Index each valid slice (skip empty ones).
         3) For each slice, replicate 'num_patches_per_slice' times in the index.
            On __getitem__, we load the slice from cache, then randomly sample
            a sub-region (patch) of size `patch_size`.

        Args:
            image_files (list[str]): Paths to 3D image files (NIfTI, etc.).
            label_files (list[str]): Corresponding label/seg files.
            transform (callable): Optional, transforms a dict {image, label}.
            test_mode (bool): If True, no labels are used (dummy labels).
            patch_size (tuple): (height, width) of the random patch to extract.
            num_patches_per_slice (int): # of random patches per slice, per epoch.
        """
        self.transform = transform
        self.test_mode = test_mode
        self.patch_size = patch_size
        self.num_patches_per_slice = num_patches_per_slice

        self.image_files = image_files
        self.label_files = label_files if not test_mode else None

        # Cache volumes: list of (img_data, lbl_data)
        self.volumes_cache = []

        # Indices for sampling: list of (volume_index, slice_index, patch_index)
        self.slice_patch_indices = []

        # ---------------------------
        # 1) LOAD & CACHE VOLUMES
        # ---------------------------
        for i, img_file in enumerate(image_files):
            # Load volume
            img_nii = sitk.ReadImage(img_file)
            img_data = sitk.GetArrayFromImage(img_nii)  # shape [Depth, H, W]

            # "Correct flipping" (from your original code)
            img_data = np.flipud(img_data)
            img_data = np.fliplr(img_data)

            if not test_mode:
                lbl_nii = sitk.ReadImage(label_files[i])
                lbl_data = sitk.GetArrayFromImage(lbl_nii)
                lbl_data = np.flipud(lbl_data)
                lbl_data = np.fliplr(lbl_data)
            else:
                lbl_data = None

            # Cache them
            self.volumes_cache.append((img_data, lbl_data))

            # Build a slice index just like in Dataloader_2D
            depth = img_data.shape[0]
            for d in range(depth):
                img_slice = img_data[d, :, :]
                lbl_slice = lbl_data[d, :, :] if lbl_data is not None else None

                # Skip empty slices
                if not test_mode:
                    if img_slice.sum() == 0 and lbl_slice.sum() == 0:
                        continue
                else:
                    if img_slice.sum() == 0:
                        continue

                # For each valid slice, replicate it for multiple patches
                for patch_i in range(num_patches_per_slice):
                    self.slice_patch_indices.append((i, d, patch_i))

    def __len__(self):
        # total = (# valid slices) * num_patches_per_slice
        return len(self.slice_patch_indices)

    def __getitem__(self, idx):
        # Which volume, which slice, which patch
        vol_i, slice_i, patch_i = self.slice_patch_indices[idx]
        img_data, lbl_data = self.volumes_cache[vol_i]

        # Extract the slice (2D). shape [H, W]
        img_slice = img_data[slice_i, :, :].copy()

        if self.test_mode:
            lbl_slice = np.zeros_like(img_slice, dtype=np.int64)
        else:
            lbl_slice = lbl_data[slice_i, :, :].copy()

        # ---------------------------
        # 2) RANDOM PATCH SAMPLING
        # ---------------------------

        for _ in range(10): #try 10 times until finding a patch that is not empty
            patch_img, patch_lbl = self._random_patch(img_slice, lbl_slice)

            if patch_img.sum() > 500 and (patch_lbl is not None and patch_lbl.sum() > 100):
                break

        # Make a sample dict for transforms
        sample = {
            'image': patch_img,
            'label': patch_lbl
        }

        # Apply transformations if provided
        if self.transform:
            sample = self.transform(sample)

        # Convert to PyTorch tensors
        image_tensor = torch.tensor(sample['image'], dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(sample['label'], dtype=torch.long)

        return image_tensor, label_tensor

    def _random_patch(self, img_slice, lbl_slice):
        """
        Extracts a random patch from the 2D slice.
        If patch is bigger than the slice, returns the slice as-is
        (or consider zero-padding).
        """
        h, w = img_slice.shape
        ph, pw = self.patch_size

        if ph > h or pw > w:
            # If slice is smaller than patch, just return entire slice.
            return img_slice, lbl_slice

        # random top-left coordinate
        top = random.randint(0, h - ph)
        left = random.randint(0, w - pw)

        patch_img = img_slice[top:top+ph, left:left+pw]
        patch_lbl = lbl_slice[top:top+ph, left:left+pw]

        return patch_img, patch_lbl
