import os
import subprocess
import numpy as np
import nibabel as nib
import re


def update_transform_file(transform_file_path):
    """Modify the interpolation method in the TransformParameters file."""
    updated_lines = []
    with open(transform_file_path, 'r') as f:
        for line in f:
            # Replace ResampleInterpolator
            if line.startswith("(ResampleInterpolator"):
                updated_lines.append("(ResampleInterpolator \"FinalNearestNeighborInterpolator\")\n")
            # Replace FinalBSplineInterpolationOrder
            elif line.startswith("(FinalBSplineInterpolationOrder"):
                updated_lines.append("(FinalBSplineInterpolationOrder 0)\n")
            else:
                updated_lines.append(line)
    
    # Overwrite the file with updated lines
    with open(transform_file_path, 'w') as f:
        f.writelines(updated_lines)
    print(f"Updated interpolation method in: {transform_file_path}")



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
        "-in", moving_points_path,
        "-tp", transform_parameter_file,
        "-out", output_dir
    ]

    print("Running transformix:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("Transformix point transformation completed.")



def main():
    #Fixed = i, Moving = e
    #For training
    #folder_path = fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MISSSSSA\Final_project\TrainingValidationTestSets\Training_resampled\TrainingImg".replace("\\", "/")
    #For validation
    #folder_path = fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MISSSSSA\Final_project\TrainingValidationTestSets\Validation_resampled\ValidationImg".replace("\\","/")
    #For test
    folder_path = fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MISSSSSA\Final_project\TrainingValidationTestSets\Test_resampled\TestImg".replace("\\", "/")
    for file_name in os.listdir(folder_path):
        if file_name.startswith("resampled_IBSR") and file_name.endswith(".gz"):
            match = re.search(r"resampled_IBSR_(\d+)", file_name)
            if match:
                val = match.group(1)
            if val != "07":
                #Intensity images
                fixed_image_path = fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MISSSSSA\Final_project\TrainingValidationTestSets\Training_resampled\TrainingImg\resampled_IBSR_07.nii.gz".replace("\\", "/")
                #Always change the moving according to your dataset: train, val, or test
                moving_image_path = fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MISSSSSA\Final_project\TrainingValidationTestSets\Test_resampled\TestImg\resampled_IBSR_{val}.nii.gz".replace("\\", "/")
                    
                #There are no masks for test
                #moving_seg_path = fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MISSSSSA\Final_project\TrainingValidationTestSets\Validation_resampled\ValidationLabel\resampled_IBSR_{val}_seg.nii.gz".replace("\\", "/")
                #Mask I might use this later
                #mask_fixed = fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MIRRRRA\Final_project\Training data-20241123\copd{val}\copd{val}\copd{val}_mask_iBHCT.nii".replace("\\", "/")

                param_file = fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MIRRRRA\Final_project\Transform_params\Parameters_Rigid.txt".replace("\\", "/")
                
                output_dir = fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MISSSSSA\Final_project\TrainingValidationTestSets\Test_resampled\TestImg\Registered\IBSR_{val}".replace("\\", "/")

                os.makedirs(output_dir, exist_ok= True)

                #Intensity based registration
                run_elastix(fixed_image_path, moving_image_path,  param_file, output_dir)
                
                #Transform masks
                
                transform_param_file = os.path.join(output_dir, "TransformParameters.0.txt")
                if os.path.exists(transform_param_file):
                        update_transform_file(transform_param_file)
                        
                else:
                        print(f"Error: {transform_param_file} not found!")
                        continue
                seg_output_dir = os.path.join(output_dir, f"IBSR_{val}_seg")
                #run_transformix(moving_seg_path, transform_param_file, seg_output_dir)


                print(f"Transform done for patient {val}")
                output_check = output_dir.replace("/","\\")
                print(f"Saved in {output_check}")


if __name__ == "__main__":
    main()

    
