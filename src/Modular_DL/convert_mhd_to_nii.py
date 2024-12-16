import SimpleITK as sitk
import os




folder = fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MISSSSSA\Final_project\TrainingValidationTestSets\Test_resampled\TestImg\Registered".replace("\\", "/")


for folder_name in os.listdir(folder):
    folder_img_path = os.path.join(folder, folder_name)
    for file_name in os.listdir(folder_img_path):
        
        if file_name == "result.0.mhd":

            img_path = os.path.join(folder_img_path, file_name)
            image = sitk.ReadImage(img_path)

            #Set image orientation
            lpi_direction = [-1, 0, 0, 0, -1, 0, 0, 0, 1]
            image.SetDirection(lpi_direction)
            new_name = f"Registered_{folder_name}_test.nii" #Set traing, val, or test accordingly
            
            output_img_path = os.path.join(folder_img_path, new_name)
            sitk.WriteImage(image, output_img_path)

        elif os.path.isdir(os.path.join(folder_img_path, file_name)):
            folder_seg = os.path.join(folder_img_path, file_name)
            for img in os.listdir(folder_seg):
                if img == "result.mhd":
                    seg_path = os.path.join(folder_seg, img)
                    seg = sitk.ReadImage(seg_path)
                    lpi_direction = [-1, 0, 0, 0, -1, 0, 0, 0, 1]
                    seg.SetDirection(lpi_direction)
                    new_seg = f"Registered_val_seg_{folder_name}.nii"
                    output_seg = os.path.join(folder_seg, new_seg)
                    sitk.WriteImage(seg, output_seg)




        
        
           
