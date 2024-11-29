import numpy as np
from overlay import load_nifti_file
import matplotlib.pyplot as plt

def class_distributions(mask):
    unique, counts = np.unique(mask, return_counts = True)
    distribution = dict(zip(unique, counts))
    print("Class distribution:", distribution)
    return distribution

def plot_class_distributions(distribution):
    labels = list(distribution.keys())
    counts = list(distribution.values())

    plt.bar(labels, counts)
    plt.title("Class distribution")
    plt.xlabel("Class labels")
    plt.ylabel("Voxel count")
    plt.show()

if __name__ == '__main__':

    mask_path = r"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MISSSSSA\Final_project\TrainingValidationTestSets\Training_Set\IBSR_01\IBSR_01_seg.nii.gz".replace("\\", "/")

    mask_image = load_nifti_file(mask_path)
    distributions = class_distributions(mask_image)

    plot_class_distributions(distributions)

    