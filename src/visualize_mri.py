import numpy as np
import nibabel as nib
import pyvista as pv

# Load the MRI file and extract data
def load_mri(file_path):
    """
    Load the MRI file and reduce it to 3D by selecting the first slice along the last dimension.
    
    Parameters:
    - file_path: Path to the NIfTI (.nii or .nii.gz) file.
    
    Returns:
    - A 3D NumPy array of shape (x, y, z).
    """
    sample_image = nib.load(file_path).get_fdata(dtype=np.float32)
    reduced_array = sample_image[..., 0]  # Take the first volume if 4D
    return reduced_array

# Visualize the 3D data
def visualize_3d(image_data, voxel_spacing=(1.0, 1.0, 1.0)):
    """
    Visualizes 3D volumetric data using PyVista.
    
    Parameters:
    - image_data: 3D NumPy array containing the volumetric data.
    - voxel_spacing: Tuple of (dx, dy, dz) voxel spacing.
    """
    grid = pv.ImageData()
    grid.dimensions = np.array(image_data.shape)
    grid.spacing = voxel_spacing  # Set voxel spacing
    grid.point_data["values"] = image_data.flatten(order="F")  # Assign flattened data

    # Define opacity transfer function
    opacity = [0, 0.1, 0.3, 0.6, 0.9, 1.0]

    # Plot the 3D volume
    grid.plot(volume=True, opacity=opacity, cmap="viridis")

if __name__ == "__main__":
    # File path to the NIfTI file
    sample_folder = r"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MISSSSSA\Final_project\TrainingValidationTestSets\Training_Set\IBSR_01\IBSR_01.nii.gz".replace("\\", "/")

    # Load and visualize
    voxel_spacing = (0.9375, 1.5, 0.9375)  # Example voxel spacing
    reduced_image = load_mri(sample_folder)
    visualize_3d(reduced_image, voxel_spacing)