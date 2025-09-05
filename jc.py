import torch

# 检查 PyTorch 是否检测到 CUDA
print(f"PyTorch 是否支持 CUDA: {torch.cuda.is_available()}")

# 如果支持，打印 CUDA 版本
if torch.cuda.is_available():
    print(f"PyTorch 检测到的 CUDA 版本: {torch.version.cuda}")
    print(f"当前 GPU 设备名称: {torch.cuda.get_device_name(0)}")

# 检查当前设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"当前使用的设备: {device}")