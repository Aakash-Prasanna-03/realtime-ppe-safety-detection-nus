import torch
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("Total VRAM (MB):", torch.cuda.get_device_properties(0).total_memory // (1024 * 1024))
else:
    print("CUDA not available")