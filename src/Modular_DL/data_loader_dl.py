import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import numpy as np
import nibabel as nib
from nibabel.orientations import aff2axcodes, axcodes2ornt, ornt_transform
import os

class ImageFolderWIthPaths(datasets.ImageFolder):

    def __getitem__(self, index):

        image, label = super(ImageFolderWIthPaths, self).__getitem__(index)

        path = self.samples[index][0]

        return image, label, path

class Dataloader_2D(Dataset):

    def __init__(self, data_folder, image_files, label_files, transform = None):
        self.data_folder = data_folder
        self.transform = transform

        #image_files[i] correspondons to label_files[i]

        self.image_label_pairs = list(zip(image_files, label_files))

        #compute indexing for slices

        self.slice_indices = []
        self.volumes_cache = []

        #Load all volumes and build an index of slices

        for vol_i, (img_f, lbl_f) in enumerate(self.image_label_pairs):
            img_path = os.path.join(data_folder, img_f)
            lbl_path = os.path.join(data_folder, lbl_f)

            img_nii = nib.load(img_path)
            lbl_nii = nib.load(lbl_path)

            img_data = img_nii.get_fdata(dtype = np.float32)
            lbl_data = lbl_nii.get_fdata(dtype = np.float32).astype(np.int64)

            self.volumes_cache.append((img_data, lbl_data))

            depth = img_data.shape[0]

            for d in range(depth):
                self.slice_indices.append((vol_i, d))

    def __len__(self):
        return len(self.slice_indices)
    
    def __getitem__(self, idx):

        vol_i, slice_i = self.slice_indices[idx]
        img_data, lbl_data = self.volumes_cache[vol_i]

        #Extract slice

        img_slice = img_data[slice_i, :, :] #H,W
        lbl_slice = lbl_data[slice_i, :, :]
                             

        #sample dir
        sample = {'image':img_slice, 'label':lbl_slice}

        if self.transform:
            sample = self.transform(sample)

        
        image_tensor = torch.tensor(sample['image'], dtype = torch.float32).unsqueeze(0) # to add a channel dimension since image is gray
        label_tensor = torch.tensor(sample['label'], dtype = torch.long)

        print(f"img tensor {image_tensor.shape}", flush=True)
        print(f"label tensor {label_tensor.shape}", flush=True)

        return image_tensor, label_tensor