from overlay import load_nifti_file
import matplotlib.pyplot as plt

def compary_intensity_histograms(image_paths):

    for path in image_paths:
        image = load_nifti_file(path)
        image = image[image > 0]

        plt.hist(image.flatten(), bins = 100, alpha = 0.5, label = path.split("/")[-1])
        
    plt.title("Intensity Histogram Across Subjects")
    plt.xlabel('Intensity')
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    image_paths = [r"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MISSSSSA\Final_project\TrainingValidationTestSets\Training_Set\IBSR_01\IBSR_01.nii.gz".replace("\\", "/"),
                   r"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MISSSSSA\Final_project\TrainingValidationTestSets\Training_Set\IBSR_03\IBSR_03.nii.gz".replace("\\", "/"),
                   r"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MISSSSSA\Final_project\TrainingValidationTestSets\Training_Set\IBSR_07\IBSR_07.nii.gz".replace("\\", "/"),
                   r"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MISSSSSA\Final_project\TrainingValidationTestSets\Training_Set\IBSR_16\IBSR_16.nii.gz".replace("\\", "/")
                   ]
    
    
    compary_intensity_histograms(image_paths)