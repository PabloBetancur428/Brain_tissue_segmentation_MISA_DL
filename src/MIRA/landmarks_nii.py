#Notebook to convert points to images
import numpy as np
import SimpleITK as sitk
import nibabel as nib


if __name__ == "__main__":

    val = "4"
    landmark_txt_file = fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MIRRRRA\Final_project\Training data-20241123\copd{val}\copd{val}\copd{val}_300_iBH_xyz_r1.txt".replace("\\", "/")
    reference_image_path = fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MIRRRRA\Final_project\Training data-20241123\copd{val}\copd{val}\copd{val}_iBHCT.nii.gz".replace("\\", "/")
    output_nifti_path = fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MIRRRRA\Final_project\Training data-20241123\copd{val}\copd{val}\copd{val}_300_iBH_xyz_r1.nii".replace("\\", "/")

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
    print("pathoutput:", output_nifti_path)
