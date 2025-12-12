"""
 作者 lgf
 日期 2025/12/12
"""
import torch
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"CUDA 版本: {torch.version.cuda}")
print(f"GPU 数量: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

print("\n=== 环境检查 ===")
print(f"OpenCV: {cv2.__version__}")
print(f"NumPy: {np.__version__}")
print(f"ONNX Runtime Providers: {ort.get_available_providers()}")
print(f"Ultralytics 已安装")