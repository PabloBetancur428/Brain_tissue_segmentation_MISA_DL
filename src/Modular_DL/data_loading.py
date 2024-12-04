import os
import nibabel as nib
import numpy as np
import SimpleITK as sitk

def load_nifti(file_path):

    nii = nib.load(file_path)
    data = nii.get_fdata()
    affine = nii.affine
    header = nii.header
    voxel_spacing = tuple(float(x) for x in header.get_zooms())

    return data, affine, voxel_spacing

def resample_image(image, target_spacing = (1.0,1.0,1.0) , is_label = False):

    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_size = [
        int(np.round(original_size[i] * (original_spacing[i] / target_spacing[i])))
        for i in range(3)
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(target_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)

    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(image)
