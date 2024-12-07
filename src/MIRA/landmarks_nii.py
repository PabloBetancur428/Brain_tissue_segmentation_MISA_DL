import numpy as np
import SimpleITK as sitk
import nibabel as nib


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
                print(numeric_parts)
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
        #print(f"Coordinate: {coord}, Index: {index}, Image Size: {image_size}")
        #print("a")
        if all(0 <= idx < sz for idx, sz in zip(index, image_size)):
            print("f")
            landmark_image[index] = 1  # Mark the voxel

    #Save
    sitk.WriteImage(landmark_image, output_nifti_path)
    print(f"Landmark Nifti saved to {output_nifti_path}")


if __name__ == "__main__":

    landmark_txt_file = fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MIRRRRA\Final_project\Training data-20241123\copd3\copd3\copd3_300_eBH_xyz_r1.txt".replace("\\", "/")
    reference_image_path = fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MIRRRRA\Final_project\Training data-20241123\copd3\copd3\copd3_eBHCT.nii.gz".replace("\\", "/")
    output_nifti_path = fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MIRRRRA\Final_project\Training data-20241123\copd3\copd3\3333copd3_300_eBH_xyz_r1.nii".replace("\\", "/")

    ref_img = nib.load(reference_image_path)
    ref_data = ref_img.get_fdata()
    ref_affine = ref_img.affine
    ref_shape = ref_data.shape

    landmark_volume = np.zeros(ref_shape, dtype = np.int16)
    with open(landmark_txt_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            x, y, z = [int(part.split(".")[0]) - 1 for part in parts[:3] if part.strip()] #♠ we have to substract one bc of different origin between image and landmarks


            if (0<= x < ref_shape[0] and (0<= y < ref_shape[1]) and (0<= z < ref_shape[2])):
                landmark_volume[x, y, z] = 1
            
            else:
                print(f"Landmark {x}, {y}, {z} is out of image bounds")
    
    landmark_img = nib.Nifti1Image(landmark_volume, ref_affine)
    nib.save(landmark_img, output_nifti_path)
