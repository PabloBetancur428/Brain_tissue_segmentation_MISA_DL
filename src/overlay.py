#Code to plot a mask overlayed over the intensity image

import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib


def display_overlay(image, mask, slice_index):

    plt.figure(figsize = (8, 8))
    plt.imshow(image[:,:,slice_index], cmap = 'gray')
    plt.imshow(mask[:,:,slice_index], alpha = 0.5, cmap = 'jet')
    plt.title(f"Overlay of Image and Mask slice {slice_index}")
    plt.axis('off')
    plt.show()

def load_nifti_file(file_path): 

    nii_img = nib.load(file_path)
    return nii_img.get_fdata(dtype = np.float32)


if __name__ == '__main__':

    image_path = r"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MISSSSSA\Final_project\TrainingValidationTestSets\Training_Set\IBSR_01\IBSR_01.nii.gz".replace("\\", "/")
    mask_path = r"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MISSSSSA\Final_project\TrainingValidationTestSets\Training_Set\IBSR_01\IBSR_01_seg.nii.gz".replace("\\", "/")

    sample_image = load_nifti_file(image_path)
    mask_image = load_nifti_file(mask_path)

    middle_slice = sample_image.shape[2] // 2

    display_overlay(sample_image, mask_image, middle_slice)