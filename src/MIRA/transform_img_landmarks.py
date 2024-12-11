import os
import subprocess
import numpy as np
import nibabel as nib


def load_nifti_image(image_path):
    """
    Load a NIfTI image and return the image object, its numpy array, and affine matrix.
    """
    img = nib.load(image_path)
    data = img.get_fdata()
    affine = img.affine
    return img, data, affine


def extract_landmarks_from_nifti(landmark_data):
    """
    Extract non-zero voxel coordinates from a NIfTI binary mask.
    """
    # Get indices of non-zero voxels
    coordinates = np.argwhere(landmark_data > 0)
    return coordinates



def run_elastix(fixed_image, moving_image, param_file, output_dir):
    """
    Run elastix registration using subprocess.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cmd = [
        fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MIRRRRA\Labs\ATLAS\elastix-5.0.0-win64\elastix.exe".replace("\\","/"),
        "-f", fixed_image,
        "-m", moving_image,
        "-out", output_dir,
        "-p", param_file
    ]

    print("Running elastix:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("Elastix registration completed.")


def run_transformix(moving_points_path, transform_parameter_file, output_dir):
    """
    Apply the computed transformation to a set of points using transformix.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cmd = [
        fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MIRRRRA\Labs\ATLAS\elastix-5.0.0-win64\transformix.exe".replace("\\","/"),
        "-in", moving_points_path,
        "-tp", transform_parameter_file,
        "-out", output_dir
    ]

    print("Running transformix:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("Transformix point transformation completed.")





def main():



    fixed_image_path = fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MIRRRRA\Final_project\Training data-20241123\copd1\copd1\copd1_eBHCT.nii\copd1_eBHCT.nii".replace("\\", "/")
    moving_image_path = fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MIRRRRA\Final_project\Training data-20241123\copd1\copd1\copd1_iBHCT.nii\copd1_iBHCT.nii".replace("\\", "/")
    fixed_landmarks_path = fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MIRRRRA\Final_project\Training data-20241123\copd1\copd1\copd1_300_eBH_xyz_r1.nii".replace("\\", "/")
    moving_landmarks_path = fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MIRRRRA\Final_project\Training data-20241123\copd1\copd1\copd1_300_iBH_xyz_r1.nii".replace("\\", "/")
    param_file = fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MIRRRRA\Final_project\Transform_params\Par0054_sstvd.txt".replace("\\", "/")
    output_dir = fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MIRRRRA\Final_project\training_pruebas_trans".replace("\\", "/")

    # Step 1: Load Images
    fixed_img, fixed_data, fixed_affine = load_nifti_image(fixed_image_path)
    moving_img, moving_data, moving_affine = load_nifti_image(moving_image_path)

    # Step 2: Load Landmark NIfTI Images
    _, fixed_landmark_data, _ = load_nifti_image(fixed_landmarks_path)
    _, moving_landmark_data, _ = load_nifti_image(moving_landmarks_path)

    # Step 5: Run Elastix Registration
    run_elastix(fixed_image_path, moving_image_path, param_file, output_dir)

    # Step 6: Transform landmarks using Transformix
    transform_parameter_file = os.path.join(output_dir, "TransformParameters.0.txt")
    transformed_points_dir = os.path.join(output_dir, "transformed_points.nii")
    run_transformix(moving_landmarks_path, transform_parameter_file, transformed_points_dir)



if __name__ == "__main__":
    main()
