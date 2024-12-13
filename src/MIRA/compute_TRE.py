#Target Registration Error
#Compute euclidean distance between both of them

import numpy as np

def load_landmarks(file_path, voxel_spacing):

    landmarks = []
    vx = voxel_spacing[0]
    vy = voxel_spacing[1]
    vz = voxel_spacing[2]
    with open(file_path, "r") as f:
        for line in f:
            
            part = line.strip().split("\t")

            if len(part) >= 3:
                x, y, z = [float(part) - 1 for part in part[:3] if part.strip()] #Removes one to account for origin in 1,1,1
                
                landmarks.append([x * vx, y * vy , z * vz])

    return np.array(landmarks)


def load_landmarks_afterreg(file_path, voxel_spacing):

    landmarks = []
    vx = voxel_spacing[0]
    vy = voxel_spacing[1]
    vz = voxel_spacing[2]
    with open(file_path, "r") as f:
        for line in f:
            
            part = line.strip().split(" ")

            if len(part) >= 3:
                x, y, z = [float(part) for part in part[:3] if part.strip()] #Removes one to account for origin in 1,1,1
                
                landmarks.append([x * vx, y * vy , z * vz])

    return np.array(landmarks)

def calculate_tre(landmarks_inhale, landmarks_exhale):
    """
    Calculate the Target Registration Error (TRE) between two sets of landmarks.

    Parameters:
    - landmarks_inhale: numpy array of shape (N, 3) containing inhale landmarks (x, y, z).
    - landmarks_exhale: numpy array of shape (N, 3) containing exhale landmarks (x, y, z).

    Returns:
    - tre: numpy array of shape (N,) containing TRE values for each landmark pair.
    - mean_tre: float, the mean TRE over all landmark pairs.
    """

    assert landmarks_inhale.shape == landmarks_exhale.shape

    diffs = landmarks_inhale - landmarks_exhale
    squared_diffs = diffs ** 2
    squared_distances = np.sum(squared_diffs, axis = 1)
    distances = np.sqrt(squared_distances)

    mean_tre = np.mean(distances)

    return distances, mean_tre


if __name__ == "__main__":
    print("\n")
    voxel_spacing_vals = [[0.625, 0.625, 2.5], [0.645, 0.645, 2.5], [0.652, 0.652, 2.5], [0.590, 0.590, 2.5]]
    #files_numer = 1 #This is for the for loop
    
    #for i in range(files_numer): #Use the for loop if you are going to run the four images
       
    #landmarks_ex_path = fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MIRRRRA\Final_project\Training data-20241123\copd{i + 1}\copd{i + 1}\copd{i + 1}_300_eBH_xyz_r1.txt".replace("\\", "/")
    #landmarks_in_path = fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MIRRRRA\Final_project\Training data-20241123\copd{i + 1}\copd{i + 1}\copd{i + 1}_300_iBH_xyz_r1.txt".replace("\\", "/")
    val = "4"
    landmarks_ex_path = fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MIRRRRA\Final_project\Training data-20241123\copd{val}\copd{val}\copd{val}_300_eBH_xyz_r1.txt".replace("\\","/")
    landmarks_in_path = fr"C:/Users/User/Desktop/UDG_old_pc/UDG/Subjects/MIRRRRA/Final_project/training_pruebas_trans/transformed_points/outputpoints_{val}_HPJFine.txt".replace("\\","/")

    #Load voxel spacing for getting TRE in mm
    voxel_spacing = voxel_spacing_vals[int(val)-1]
    print(voxel_spacing)
    landmarks_exhale = load_landmarks(landmarks_ex_path,voxel_spacing)
    #landmarks_inhale = load_landmarks(landmarks_in_path, voxel_spacing) #This is for the original values, diff between inhale and exhale
    landmarks_inhale = load_landmarks_afterreg(landmarks_in_path, voxel_spacing)
    print("EX: ", landmarks_exhale.shape)
    print("IN: ", landmarks_inhale)
    tre, mean_tre = calculate_tre(landmarks_exhale, landmarks_inhale)

    #print(f"TRE for each landmark: {tre}")
    print(f"Mean TRE: {mean_tre:.4f} mm for copd{val}")
    print("\n")