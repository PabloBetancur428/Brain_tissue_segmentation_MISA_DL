import torch

if torch.cuda.is_available():
    print("CUDA is available!")
    device = torch.cuda.get_device_name(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory
    memory_in_gb = total_memory / 1e9  # Convert bytes to GB

    print(f"Device Name: {device}")
    print(f"Total Memory: {memory_in_gb:.2f} GB")
    print(f"Number of Devices: {torch.cuda.device_count()}")
else:
    print("CUDA is not available.")