#Convert trasnformix output to txt file in form of x, y, z
def extract_output_points(output_points_file, output_xyz_file):
    """
    Extract 'OutputPoint' from the transformix output file and save as x, y, z format.

    Parameters:
    - outputpoints_file: Path to the transformix outputpoints.txt file.
    - output_xyz_file: Path to save the extracted x, y, z points.
    """
    output_points = []

    # Read the transformix output file
    with open(output_points_file, "r") as file:
        for line in file:
            #print(line)
            if "OutputIndexFixed" in line:
                # Extract the coordinates inside [ ]
                point_str = line.split("=")[-3][:15].strip()  # Get the part after '='
                print(point_str)
                point_str = point_str.replace("[", "").replace("]", "")  # Remove brackets
                coords = [int(coord) for coord in point_str.split()]  # Convert to floats
                output_points.append(coords)

    # Save the extracted points to a new file
    with open(output_xyz_file, "w") as file:
        for point in output_points:
            file.write(f"{point[0]} {point[1]} {point[2]}\n")

    print(f"Extracted points saved to {output_xyz_file}")


if __name__ == "__main__":

    output_points_file = fr"C:\Users\User\Desktop\UDG_old_pc\UDG\Subjects\MIRRRRA\Final_project\training_pruebas_trans\transformed_points\outputpoints.txt".replace("\\", "/")

    val = "4"
    output_xyz_file = fr"C:/Users/User/Desktop/UDG_old_pc/UDG/Subjects/MIRRRRA/Final_project/training_pruebas_trans/transformed_points/outputpoints_{val}_HPJFine.txt".replace("\\", "/")


    extract_output_points(output_points_file, output_xyz_file)
    
