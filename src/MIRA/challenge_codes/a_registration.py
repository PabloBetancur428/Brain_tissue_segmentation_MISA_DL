import os
import subprocess
import numpy as np
import nibabel as nib

# Function to load landmarks from a text file
def load_landmarks(landmark_path, origin_one=True):
    """
    Load landmarks from a txt file. Each line: x y z
    If origin_one is True, convert to zero-based indexing.
    Returns a Nx3 numpy array.
    """
    landmarks = []
    with open(landmark_path, "r") as f:
        for line in f:
            if line.strip():  # Ensure the line is not empty
                x_str, y_str, z_str = line.strip().split()  # Split the line into x, y, z components
                x, y, z = float(x_str), float(y_str), float(z_str)  # Convert strings to floats
                if origin_one:
                    # Convert from 1-based to 0-based by subtracting 1
                    x -= 1
                    y -= 1
                    z -= 1
                landmarks.append([x, y, z])  # Append the processed landmark to the list
    return np.array(landmarks)  # Return landmarks as a numpy array

# Function to save points in a format compatible with transformix
def save_points_for_transformix(landmarks, output_path):
    num_points = landmarks.shape[0]  # Get the number of points
    with open(output_path, 'w') as f:
        f.write("index\n")  # Write header
        f.write(f"{num_points}\n")  # Write number of points
        for i in range(num_points):
            x, y, z = landmarks[i]
            f.write(f"{x} {y} {z}\n")  # Write each landmark in the required format

# Function to extract non-zero voxel coordinates from a NIfTI mask
def extract_landmarks_from_nifti(landmark_data):
    """
    Extract non-zero voxel coordinates from a NIfTI binary mask.
    """
    # Get indices of non-zero voxels
    coordinates = np.argwhere(landmark_data > 0)
    return coordinates

# Function to run the elastix registration
def run_elastix(fixed_image, moving_image, mask_fixed, param_file, output_dir, mask_moving=None):
    """
    Run elastix registration using subprocess.
    fixed_image: path to fixed image (.nii or .mhd, etc.)
    moving_image: path to moving image
    param_file: path to elastix parameter file (.txt)
    output_dir: directory for elastix output
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create the output directory if it doesn't exist
    
    # Define the elastix command with or without the moving mask
    if mask_moving:
        cmd = [
            fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MIRRRRA\Labs\ATLAS\elastix-5.0.0-win64\elastix.exe".replace("\\","/"),
            "-f", fixed_image,
            "-m", moving_image,
            "-fMask", mask_fixed,
            "-mMask", mask_moving,
            "-out", output_dir,
            "-p", param_file
        ]
    else:
        cmd = [
            fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MIRRRRA\Labs\ATLAS\elastix-5.0.0-win64\elastix.exe".replace("\\","/"),
            "-f", fixed_image,
            "-m", moving_image,
            "-fMask", mask_fixed,
            "-out", output_dir,
            "-p", param_file
        ]

    print("Running elastix:", " ".join(cmd))  # Print the command for debugging
    subprocess.run(cmd, check=True)  # Execute the command
    print("Elastix registration completed.")  # Confirmation message

# Function to apply transformation using transformix
def run_transformix(moving_points_path, transform_parameter_file, output_dir):
    """
    Apply the computed transformation to a set of points using transformix.
    moving_points_path: txt file with points in elastix format
    transform_parameter_file: the transform parameters from elastix output
    output_dir: directory to store the transformed points
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create the output directory if it doesn't exist

    # Define the transformix command
    cmd = [
        fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MIRRRRA\Labs\ATLAS\elastix-5.0.0-win64\transformix.exe".replace("\\","/"),
        "-def", moving_points_path,
        "-tp", transform_parameter_file,
        "-out", output_dir
    ]

    print("Running transformix:", " ".join(cmd))  # Print the command for debugging
    subprocess.run(cmd, check=True)  # Execute the command
    print("Transformix point transformation completed.")  # Confirmation message

# Function to save transformed landmarks as a NIfTI image
def save_transformed_landmarks_as_nifti(reference_image, transformed_points_path, output_path):
    """
    Save the transformed landmarks back as a NIfTI image.
    """
    # Load transformed points from the output file
    transformed_points = np.loadtxt(transformed_points_path, skiprows=2)[:, :3]
    transformed_image_data = np.zeros(reference_image.shape, dtype=np.uint8)  # Initialize an empty image

    for x, y, z in transformed_points:
        # Assign transformed points to the appropriate voxel
        transformed_image_data[int(round(x)), int(round(y)), int(round(z))] = 1

    # Create and save the NIfTI image
    transformed_img = nib.Nifti1Image(transformed_image_data, reference_image.affine)
    nib.save(transformed_img, output_path)
    print(f"Transformed landmarks saved to {output_path}")

# Main function for the registration pipeline
def main(val=1, mask_moving=True):
    # Paths to input images and masks
    fixed_image_path = fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MIRRRRA\Final_project\Training data-20241123\copd{val}\copd{val}\copd{val}_iBHCT.nii.gz".replace("\\", "/")
    moving_image_path = fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MIRRRRA\Final_project\Training data-20241123\copd{val}\copd{val}\copd{val}_eBHCT.nii.gz".replace("\\", "/")
    fixed_landmarks_path = fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MIRRRRA\Final_project\Training data-20241123\copd{val}\copd{val}\copd{val}_300_iBH_xyz_r1.txt".replace("\\", "/")
    mask_fixed = fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MIRRRRA\Final_project\Training data-20241123\copd{val}\copd{val}\copd{val}_mask_iBHCT_jj.nii".replace("\\", "/")
    if mask_moving:
        mask_moving = fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MIRRRRA\Final_project\Training data-20241123\copd{val}\copd{val}\copd{val}_mask_eBHCT_jj.nii".replace("\\", "/")
    else:
        mask_moving = None

    # Paths for output
    output_path_landmarks = fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MIRRRRA\Final_project\training_pruebas_trans".replace("\\", "/")
    
    param_file = fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MIRRRRA\Final_project\Transform_params\Parameters_BSpline_final.txt".replace("\\", "/")
    print(param_file)
    
    output_dir = fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MIRRRRA\Final_project\training_pruebas_trans".replace("\\", "/")

    # Load fixed landmarks
    fixed_landmarks = load_landmarks(fixed_landmarks_path)

    # Save the fixed landmarks for transformix
    save_points_for_transformix(fixed_landmarks, output_path_landmarks + f"/fixed_landmarks_{val}.txt")

    # Load the saved landmarks for transformix
    fixed_landmarks_1 = fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MIRRRRA\Final_project\training_pruebas_trans\fixed_landmarks_{val}.txt".replace("\\","/")

    # Run elastix for intensity-based registration
    run_elastix(fixed_image_path, moving_image_path, mask_fixed, param_file, output_dir, mask_moving=mask_moving)

    # Define transform parameter file path and output directory for transformix
    transform_parameter_file = os.path.join(output_dir, "TransformParameters.0.txt")
    transformed_points_dir = os.path.join(output_dir, "transformed_points")

    # Run transformix to transform the fixed landmarks
    run_transformix(fixed_landmarks_1, transform_parameter_file, transformed_points_dir)

    print(f"Transform done for patient {val}")

# Entry point of the script
if __name__ == "__main__":
    main()
