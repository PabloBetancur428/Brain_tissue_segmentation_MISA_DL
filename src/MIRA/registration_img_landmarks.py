import os
import subprocess
import numpy as np
import nibabel as nib


def load_nifti_image(image_path):
    """
    Load a NIfTI image and return the image object and its numpy array.
    """
    img = nib.load(image_path)
    data = img.get_fdata()
    affine = img.affine
    return img, data, affine


def load_landmarks(landmark_path, origin_one=True):
    """
    Load landmarks from a txt file. Each line: x y z
    If origin_one is True, convert to zero-based indexing.
    Returns a Nx3 numpy array.
    """
    landmarks = []
    with open(landmark_path, "r") as f:
        for line in f:
        
            if line.strip():
                x_str, y_str, z_str = line.strip().split()
                x, y, z = float(x_str), float(y_str), float(z_str)
                if origin_one:
                    # Convert from 1-based to 0-based by subtracting 1
                    x -= 1
                    y -= 1
                    z -= 1
                landmarks.append([x, y, z])
    return np.array(landmarks)


def save_points_for_transformix(landmarks, output_path):
    num_points = landmarks.shape[0]
    with open(output_path, 'w') as f:
        f.write("index\n")
        f.write(f"{num_points}\n")
        for i in range(num_points):
            x, y, z = landmarks[i]
            f.write(f"{x} {y} {z}\n")

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
    fixed_image: path to fixed image (.nii or .mhd, etc.)
    moving_image: path to moving image
    param_file: path to elastix parameter file (.txt)
    output_dir: directory for elastix output
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
    moving_points_path: txt file with points in elastix format
    transform_parameter_file: the transform parameters from elastix output
    output_dir: directory to store the transformed points
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cmd = [
        fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MIRRRRA\Labs\ATLAS\elastix-5.0.0-win64\transformix.exe".replace("\\","/"),
        "-def", moving_points_path,
        "-tp", transform_parameter_file,
        "-out", output_dir
    ]

    print("Running transformix:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("Transformix point transformation completed.")

def save_transformed_landmarks_as_nifti(reference_image, transformed_points_path, output_path):
    """
    Save the transformed landmarks back as a NIfTI image.
    """
    # Load transformed points
    transformed_points = np.loadtxt(transformed_points_path, skiprows=2)[:, :3]
    transformed_image_data = np.zeros(reference_image.shape, dtype=np.uint8)

    for x, y, z in transformed_points:
        transformed_image_data[int(round(x)), int(round(y)), int(round(z))] = 1

    transformed_img = nib.Nifti1Image(transformed_image_data, reference_image.affine)
    nib.save(transformed_img, output_path)
    print(f"Transformed landmarks saved to {output_path}")


def main():
    # Paths to your data
    #landmark_txt_file = fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MIRRRRA\Final_project\Training data-20241123\copd1\copd1\copd1_300_eBH_xyz_r1.txt".replace("\\", "/")
    #reference_image_path = fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MIRRRRA\Final_project\Training data-20241123\copd1\copd1\copd1_iBHCT.nii.gz".replace("\\", "/")
    #output_nifti_path = fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MIRRRRA\Final_project\Training data-20241123\copd1\copd1\copd1_300_eBH_xyz_r1.nii".replace("\\", "/")
    fixed_image_path = fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MIRRRRA\Final_project\Training data-20241123\copd1\copd1\copd1_eBHCT.nii\copd1_eBHCT.nii".replace("\\", "/")
    moving_image_path = fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MIRRRRA\Final_project\Training data-20241123\copd1\copd1\copd1_iBHCT.nii\copd1_iBHCT.nii".replace("\\", "/")
    fixed_landmarks_path = fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MIRRRRA\Final_project\Training data-20241123\copd1\copd1\copd1_300_eBH_xyz_r1.txt".replace("\\", "/")
    moving_landmarks_path = fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MIRRRRA\Final_project\Training data-20241123\copd1\copd1\copd1_300_iBH_xyz_r1.txt".replace("\\", "/")
    output_path_landmarks = fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MIRRRRA\Final_project\training_pruebas_trans".replace("\\", "/")
    param_file = fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MIRRRRA\Final_project\Transform_params\par_007\Parameters.MI.RP.Bspline_tuned.txt".replace("\\", "/")
    output_dir = fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MIRRRRA\Final_project\training_pruebas_trans".replace("\\", "/")

    # Step 1: Load Images (just to ensure they exist and shape)
    fixed_img, fixed_data, fixed_affine = load_nifti_image(fixed_image_path)
    moving_img, moving_data, moving_affine = load_nifti_image(moving_image_path)

    # Step 2: Load Landmarks
    # Assume they are provided with origin 1, so we convert to 0-based
    #fixed_landmarks = load_nifti_image(fixed_landmarks_path)
    #moving_landmarks, moving_landmarks_data, moving_landmarks_affine = load_nifti_image(moving_landmarks_path)

    fixed_landmarks = load_landmarks(fixed_landmarks_path)
    moving_landmarks = load_landmarks(moving_landmarks_path)

    save_points_for_transformix(fixed_landmarks, output_path_landmarks + "/fixed_landmarks_1.txt")
    save_points_for_transformix(moving_landmarks, output_path_landmarks + "/moving_landmarks_1.txt")

    #Moving

    moving_landmarks_1 = fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MIRRRRA\Final_project\training_pruebas_trans\moving_landmarks_1.txt".replace("\\","/")
    fixed_landmarks_1 = fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MIRRRRA\Final_project\training_pruebas_trans\fixed_landmarks_1.txt".replace("\\","/")
    #print(train_landmarks)

    # Step 3: (Optional) Save moving landmarks in elastix format for transformix later
    # If you want to see how they transform after registration
    #moving_points_elastix = os.path.join(output_dir, "moving_points.txt")
    #save_landmarks_to_elastix_format(moving_landmarks, moving_points_elastix)

    # Step 4: Run Elastix Registration
    # This will produce TransformParameters.0.txt in output_dir
    run_elastix(fixed_image_path, moving_image_path, param_file, output_dir)

    # Step 5: After registration, we have a transform that maps from moving to fixed space.
    # Let's assume we want to transform the moving landmarks to fixed space:
    transform_parameter_file = os.path.join(output_dir, "TransformParameters.0.txt")
    transformed_points_dir = os.path.join(output_dir, "transformed_points")

    #I should set fixedlandmarks
    #dia de challenge 
    run_transformix(fixed_landmarks_1, transform_parameter_file, transformed_points_dir)

    # The result will be a file named outputpoints.txt in transformed_points_dir with the moved landmark locations.


if __name__ == "__main__":
    main()
