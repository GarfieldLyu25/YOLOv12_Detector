"""
 作者 lgf
 日期 2025/12/12
"""
# export_model.py
from ultralytics import YOLO

# 加载已训练好的模型
model = YOLO('../model/yolo11n.pt')  # 或你的best.pt

# ========== 方式1: 基础导出 ==========
model.export(
    format='onnx',
    imgsz=640,           # 输入尺寸
    dynamic=False,       # False=固定尺寸, True=动态尺寸
    simplify=True,       # 简化ONNX图
    opset=12,           # ONNX opset版本

)


# ========== 方式2: 动态batch导出 ==========
# model.export(
#     format='onnx',
#     imgsz=640,
#     dynamic=True,        # 支持动态batch
#     simplify=True,
# )

# ========== 方式3: 多尺寸输入导出 ==========
# model.export(
#     format='onnx',
#     imgsz=[640, 480, 320],  # 支持多种输入尺寸
#     dynamic=True,
#     simplify=True,
# )