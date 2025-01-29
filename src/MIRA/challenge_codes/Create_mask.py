import SimpleITK as sitk
import numpy as np
import os
from scipy.ndimage import label, binary_closing

# Function to create a refined lung mask from a CT image
def create_refined_lung_mask(ct_image, lower_threshold=100, upper_threshold=500, closing_size=5):
    """
    Generate a refined lung mask from a CT image using intensity thresholds,
    selecting the two largest connected components and performing morphological closing.

    Parameters:
    - ct_image: Input CT image (SimpleITK.Image).
    - lower_threshold: Minimum intensity value for the lung region.
    - upper_threshold: Maximum intensity value for the lung region.
    - closing_size: Size of the structuring element for morphological closing.

    Returns:
    - Refined lung mask as SimpleITK.Image.
    """

    # Convert the CT image to a numpy array
    ct_array = sitk.GetArrayFromImage(ct_image)

    # Threshold the image to create a binary mask
    mask_array = np.logical_and(ct_array >= lower_threshold, ct_array <= upper_threshold).astype(np.uint8)

    # Label connected components in the binary mask
    labeled_mask, num_features = label(mask_array)

    # Measure sizes of connected components
    component_sizes = np.bincount(labeled_mask.ravel())
    component_sizes[0] = 0  # Exclude background

    # Select the two largest connected components by size
    largest_labels = component_sizes.argsort()[-2:]  # Get indices of the two largest components, CHANGE THIS VALUE ACCORDING TO THE NUMBER OF CONNECTED COMPONENTS THAT YOU NEED
    two_largest_components_mask = np.isin(labeled_mask, largest_labels).astype(np.uint8)

    # Apply morphological closing to smooth the mask and fill small holes
    structuring_element = np.ones((closing_size, closing_size, closing_size), dtype=np.uint8)
    refined_mask_array = binary_closing(two_largest_components_mask, structure=structuring_element).astype(np.uint8)

    # Convert the refined mask back to a SimpleITK image
    refined_mask_image = sitk.GetImageFromArray(refined_mask_array)
    refined_mask_image.CopyInformation(ct_image)  # Retain spatial information of the original CT image

    return refined_mask_image

# Main function to generate masks for inhale and exhale moments
def main():
    moments = ["inhale", "exhale"]  # Define moments for processing
    val = "1"  # Set the patient identifier

    # Define file paths for input images and output folder
    inhale_path = fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MIRRRRA\Final_project\Training data-20241123\copd{val}\copd{val}\copd{val}_iBHCT.nii.gz".replace("\\", "/")
    exhale_path = fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MIRRRRA\Final_project\Training data-20241123\copd{val}\copd{val}\copd{val}_eBHCT.nii.gz".replace("\\", "/")
    output_folder = fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MIRRRRA\Final_project\Training data-20241123\copd{val}\copd{val}".replace("\\", "/")

    # Process each moment (inhale and exhale)
    for moment in moments:
        if moment == 'inhale':
            # Define output path for inhale mask
            output_path = os.path.join(output_folder, f"copd{val}_mask_iBHCT_jj.nii")

            # Read the inhale CT image
            ct_image = sitk.ReadImage(inhale_path)

            # Create a refined lung mask for the inhale image
            lung_mask = create_refined_lung_mask(ct_image, lower_threshold=100, upper_threshold=500, closing_size=7)

            # Save the generated mask to the specified output path
            sitk.WriteImage(lung_mask, output_path)
            print(f"Inhale mask saved at: {output_path}")

        elif moment == 'exhale':
            # Define output path for exhale mask
            output_path = os.path.join(output_folder, f"copd{val}_mask_eBHCT_jj.nii")

            # Read the exhale CT image
            ct_image = sitk.ReadImage(exhale_path)

            # Create a refined lung mask for the exhale image
            lung_mask = create_refined_lung_mask(ct_image, lower_threshold=100, upper_threshold=500, closing_size=7)

            # Save the generated mask to the specified output path
            sitk.WriteImage(lung_mask, output_path)
            print(f"Exhale mask saved at: {output_path}")

# Entry point of the script
if __name__ == '__main__':
    main()
