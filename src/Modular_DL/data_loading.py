import os
import nibabel as nib
import numpy as np
import SimpleITK as sitk

# def load_nifti(file_path):

#     nii = nib.load(file_path)
#     data = nii.get_fdata()
#     affine = nii.affine
#     header = nii.header
#     voxel_spacing = tuple(float(x) for x in header.get_zooms())

#     return data, affine, voxel_spacing


def load_nifti(file_path):
    """Load a NIFTI file and return the image data, affine, and voxel spacing."""
    image_sitk = sitk.ReadImage(file_path)
    data = sitk.GetArrayFromImage(image_sitk)
    affine = np.array(image_sitk.GetDirection())  # Orientation matrix
    voxel_spacing = image_sitk.GetSpacing()  # Spacing in mm
    return data, image_sitk, voxel_spacing

def resample_image(image, target_spacing, is_label=False):
    """Resample the image to the target spacing."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(
        [
            int(np.round(image.GetSize()[i] * (image.GetSpacing()[i] / target_spacing[i])))
            for i in range(3)
        ]
    )
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)
    return resampler.Execute(image)

def convert_to_ras(image_sitk):
    """Convert the image's orientation from LPS to RAS."""
    direction = np.array(image_sitk.GetDirection()).reshape(3, 3)
    lps_to_ras = np.diag([-1, -1, 1])  # Convert from LPS to RAS
    new_direction = np.dot(lps_to_ras, direction)
    image_sitk.SetDirection(new_direction.flatten())
    return image_sitk