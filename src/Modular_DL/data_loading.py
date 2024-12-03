import os
import nibabel as nib
import numpy as np

def load_nifti(file_path):

    nii = nib.load(file_path)
    data = nii.get_fdata()
    affine = nii.affine
    header = nii.header
    voxel_spacing = header.get_zooms()

    return data, affine, voxel_spacing