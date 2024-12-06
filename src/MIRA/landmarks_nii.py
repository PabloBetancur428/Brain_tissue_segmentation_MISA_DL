import numpy as np
import SimpleITK as sitk


def load_landmarks_with_tabs(file_path):
    landmarks = []
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line by tab
            parts = line.strip().split("\t")
            # Filter out empty strings and ensure we only keep the first three columns
            numeric_parts = [float(part) for part in parts[:3] if part.strip()]
            # Append to landmarks if we have all three values (x, y, z)
            if len(numeric_parts) == 3:
                landmarks.append(numeric_parts)
            else:
                print(f"Skipping invalid or incomplete line: {line.strip()}")
    return np.array(landmarks)


def create_nifti(landmarks, reference_image_path, output_nifti_path):
    #Load Reference
    reference_image = sitk.ReadImage(reference_image_path)
    image_size = reference_image.GetSize()
    image_spacing = reference_image.GetSpacing()
    image_origin = reference_image.GetOrigin()

    #Create an empty image
    landmark_image = sitk.Image(image_size, sitk.sitkUInt8)
    landmark_image.SetSpacing(image_spacing)
    landmark_image.SetOrigin(image_origin)

    #Draw points from the coordinates of the real image
    for coord in landmarks:
        index = reference_image.TransformPhysicalPointToIndex(coord.tolist())
        if all(0 <= idx < sz for idx, sz in zip(index, image_size)):
            landmark_image[index] = 1  # Mark the voxel

    #Save
    sitk.WriteImage(landmark_image, output_nifti_path)
    print(f"Landmark Nifti saved to {output_nifti_path}")


if __name__ == "__main__":

    landmark_txt_file = fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MIRRRRA\Final_project\Training data-20241123\copd3\copd3\copd3_300_eBH_xyz_r1.txt".replace("\\", "/")
    reference_image_path = fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MIRRRRA\Final_project\Training data-20241123\copd3\copd3\copd3_eBHCT.nii.gz".replace("\\", "/")
    output_nifti_path = fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MIRRRRA\Final_project\Training data-20241123\copd3\copd3\copd3_300_eBH_xyz_r1.nii".replace("\\", "/")

    landmarks = load_landmarks_with_tabs(landmark_txt_file)
    create_landmark_nifti = create_nifti(landmarks, reference_image_path, output_nifti_path)