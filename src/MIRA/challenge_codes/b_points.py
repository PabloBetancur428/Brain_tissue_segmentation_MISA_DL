# Function to convert transformix output to a txt file in x, y, z format
def extract_output_points(output_points_file, output_xyz_file, challenge=False):
    """
    Extract 'OutputPoint' from the transformix output file and save as x, y, z format.

    Parameters:
    - output_points_file: Path to the transformix outputpoints.txt file.
    - output_xyz_file: Path to save the extracted x, y, z points.
    - challenge: Boolean to determine if the landmarks need adjustment for zero-based indexing.
    """
    output_points = []

    # Read the transformix output file
    with open(output_points_file, "r") as file:
        if not challenge:
            for line in file:
                # Check for lines containing 'OutputIndexFixed'
                if "OutputIndexFixed" in line:
                    # Extract the coordinates inside [ ]
                    point_str = line.split("=")[-3][:15].strip()  # Get the part after '='
                    point_str = point_str.replace("[", "").replace("]", "")  # Remove brackets
                    coords = [int(coord) for coord in point_str.split()]  # Convert to integers
                    output_points.append(coords)  # Append the coordinates to the list
        else:
            for line in file:
                # Check for lines containing 'OutputIndexFixed'
                if "OutputIndexFixed" in line:
                    # Extract the coordinates inside [ ]
                    point_str = line.split("=")[-3][:15].strip()  # Get the part after '='
                    point_str = point_str.replace("[", "").replace("]", "")  # Remove brackets
                    coords = [int(coord) + 1 for coord in point_str.split()]  # Adjust for one-based indexing
                    output_points.append(coords)  # Append the coordinates to the list

    # Save the extracted points to a new file
    with open(output_xyz_file, "w") as file:
        for point in output_points:
            file.write(f"{point[0]} {point[1]} {point[2]}\n")  # Write each point in x, y, z format

    print(f"Extracted points saved to {output_xyz_file}")  # Confirmation message


if __name__ == "__main__":
    # Path to the transformix output file
    output_points_file = fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MIRRRRA\Final_project\training_pruebas_trans\transformed_points\outputpoints.txt".replace("\\", "/")

    # Value to differentiate outputs
    val = "1"
    
    # Path to save the processed output points file
    output_xyz_file = fr"C:/Users/User/Desktop/UDG_old_pc/UDG/Subjects/MIRRRRA/Final_project/training_pruebas_trans/transformed_points/outputpoints_{val}_prueba_challenge.txt".replace("\\", "/")

    # Call the function to process the transformix output
    extract_output_points(output_points_file, output_xyz_file)
