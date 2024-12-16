import SimpleITK as sitk
import numpy as np
import os

def create_lung_mask(ct_image, kernel):
    #smooth_img = sitk.SmoothingRecursiveGaussian(ct_image) #reduces noise in the image
    #equalized_image = sitk.AdaptiveHistogramEqualization(smooth_img, alpha=0.3, beta=0.3) #enhances contrast
    
    threshold = sitk.BinaryThreshold(ct_image, lowerThreshold = 160, upperThreshold = 600, insideValue=1, outsideValue=0)
    #threshold = sitk.OtsuThreshold(ct_image, insideValue=1, outsideValue=0)

    #Morphological operations to clean up the holes of the thresholded image
    fill_holes = sitk.BinaryFillholeImageFilter()
    mask = fill_holes.Execute(threshold)

    kernel_radius = kernel
    foreground_value = 1

    mask_after_fill = sitk.BinaryMorphologicalClosing(mask, kernel_radius,foreground_value)
    #mask_cleaned = sitk.BinaryMorphologicalOpening(mask_after_fill, 5)

    #Only keep largest connected components (lungs)
    cc = sitk.ConnectedComponent(mask_after_fill)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc)
    #Does this returns the largest connected component in pos [0]

   
    labels = stats.GetLabels()
    

    if not labels:
        print("WARNING: NO CONNECTED COMPONENTS IN MASK")
        return threshold

    largest_label = max(labels, key = lambda x : stats.GetPhysicalSize(x))
    print(f"Largest connected component label: {largest_label}")
    lung_mask = sitk.Equal(cc, largest_label)

    return lung_mask


def main():

    moments = ["inhale", "exhale"]
    val = "4"

    inhale_path = fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MIRRRRA\Final_project\Training data-20241123\copd{val}\copd{val}\copd{val}_iBHCT.nii.gz".replace("\\", "/")
    exhale_path = fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MIRRRRA\Final_project\Training data-20241123\copd{val}\copd{val}\copd{val}_eBHCT.nii.gz".replace("\\", "/")
    output_folder = fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MIRRRRA\Final_project\Training data-20241123\copd{val}\copd{val}".replace("\\", "/")

    for moment in moments:
        if moment == 'inhale':
            output_path = os.path.join(output_folder, f"copd{val}_mask_iBHCT.nii")
            ct_image = sitk.ReadImage(inhale_path)
            lung_mask = create_lung_mask(ct_image, 7)
            sitk.WriteImage(lung_mask, output_path)
            print(f"Inhale mask saved at: {output_path}")

        else:
            output_path = os.path.join(output_folder, f"copd{val}_mask_eBHCT.nii")
            ct_image = sitk.ReadImage(exhale_path)
            lung_mask = create_lung_mask(ct_image, 3)
            sitk.WriteImage(lung_mask, output_path)
            print(f"Exhale mask saved at: {output_path}")
            break



if __name__ == '__main__':
    main()