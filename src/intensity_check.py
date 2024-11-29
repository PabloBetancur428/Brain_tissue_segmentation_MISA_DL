import numpy as np
from overlay import load_nifti_file

def intensity_statistics(image):
    mean_intensity = np.mean(image)
    median_intensity = np.median(image)
    std_intensity = np.std(image)

    print(f"Mean intensity: {mean_intensity}")
    print(f"median_intensity: {median_intensity}")
    print(f"std_intensity: {std_intensity}")


if __name__ == '__main__':

    image_path = r"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MISSSSSA\Final_project\TrainingValidationTestSets\Training_Set\IBSR_01\IBSR_01.nii.gz".replace("\\", "/")
    
    sample_image = load_nifti_file(image_path)
    

    intensity_statistics(sample_image)