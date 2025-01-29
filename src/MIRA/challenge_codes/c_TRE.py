# Import necessary libraries
import numpy as np

def load_landmarks(file_path, voxel_spacing):
    """
    Load landmarks from a file, convert them to zero-based indexing, and apply voxel spacing.

    Parameters:
    - file_path: Path to the landmark file.
    - voxel_spacing: List of voxel spacing values [vx, vy, vz].

    Returns:
    - landmarks: numpy array of shape (N, 3) containing processed landmarks.
    """
    landmarks = []
    vx = voxel_spacing[0]
    vy = voxel_spacing[1]
    vz = voxel_spacing[2]
    with open(file_path, "r") as f:
        for line in f:
            part = line.strip().split("\t")  # Split line by tabs
            if len(part) >= 3:
                # Convert to zero-based indexing and apply voxel spacing
                x, y, z = [float(part) - 1 for part in part[:3] if part.strip()]
                landmarks.append([x * vx, y * vy, z * vz])
    return np.array(landmarks)

def load_landmarks_afterreg(file_path, voxel_spacing):
    """
    Load landmarks after registration from a file and apply voxel spacing.

    Parameters:
    - file_path: Path to the landmark file.
    - voxel_spacing: List of voxel spacing values [vx, vy, vz].

    Returns:
    - landmarks: numpy array of shape (N, 3) containing processed landmarks.
    """
    landmarks = []
    vx = voxel_spacing[0]
    vy = voxel_spacing[1]
    vz = voxel_spacing[2]
    with open(file_path, "r") as f:
        for line in f:
            part = line.strip().split(" ")  # Split line by spaces
            if len(part) >= 3:
                # Parse coordinates and apply voxel spacing
                x, y, z = [float(part) for part in part[:3] if part.strip()]
                landmarks.append([x * vx, y * vy, z * vz])
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
    assert landmarks_inhale.shape == landmarks_exhale.shape  # Ensure both sets have the same shape

    diffs = landmarks_inhale - landmarks_exhale  # Calculate differences
    squared_diffs = diffs ** 2  # Square the differences
    squared_distances = np.sum(squared_diffs, axis=1)  # Sum squared differences along the axis
    distances = np.sqrt(squared_distances)  # Take the square root to get Euclidean distances

    mean_tre = np.mean(distances)  # Calculate the mean TRE

    return distances, mean_tre

if __name__ == "__main__":
    print("\n")
    # Voxel spacing values for different datasets
    voxel_spacing_vals = [[0.625, 0.625, 2.5], [0.645, 0.645, 2.5], [0.652, 0.652, 2.5], [0.590, 0.590, 2.5]]
    
    val = "1"  # Specify the dataset index
    # Paths to fixed and transformed landmarks
    landmarks_fix_path = fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MIRRRRA\Final_project\Training data-20241123\copd{val}\copd{val}\copd{val}_300_eBH_xyz_r1.txt".replace("\\", "/")
    landmarks_moved_path = fr"C:/Users/User/Desktop/UDG_old_pc/UDG/Subjects/MIRRRRA/Final_project/training_pruebas_trans/transformed_points/outputpoints_{val}_prueba_challenge.txt".replace("\\", "/")

    # Load voxel spacing for the specified dataset
    voxel_spacing = voxel_spacing_vals[int(val) - 1]
    print(voxel_spacing)  # Print voxel spacing for verification

    # Load landmarks for exhale (fixed) and inhale (transformed)
    landmarks_exhale = load_landmarks(landmarks_fix_path, voxel_spacing)
    landmarks_inhale = load_landmarks_afterreg(landmarks_moved_path, voxel_spacing)
    print("EX: ", landmarks_exhale.shape)  # Print shape of exhale landmarks
    print("IN: ", landmarks_inhale)  # Print inhale landmarks

    # Calculate TRE and mean TRE
    tre, mean_tre = calculate_tre(landmarks_exhale, landmarks_inhale)

    # Print results
    print(f"Mean TRE: {mean_tre:.4f} mm for copd{val}")
    print("\n")
