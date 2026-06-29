import torch

print(torch.cuda.is_available())   # True/False
print(torch.cuda.device_count())   # number of GPUs
print(torch.cuda.get_device_name(0))  # GPU name (if available)