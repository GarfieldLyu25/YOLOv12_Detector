"""
作者 lgf
日期 2025/12/12
"""
import cv2
import numpy as np
from ultralytics import YOLO

# 加载模型
model = YOLO('../model/yolo11n.pt')

# COCO 类别名称
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# 打开摄像头
cap = cv2.VideoCapture(0)

print("="*60)
print("YOLOv11 实时检测")
print("按 'q' 退出")
print("="*60)

frame_count = 0
import time

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 推理
    start_time = time.time()
    results = model(frame, verbose=False, conf=0.25, iou=0.45)
    inference_time = (time.time() - start_time) * 1000

    # 获取检测结果
    result = results[0]
    boxes = result.boxes

    # 手动绘制
    for box in boxes:
        # 获取坐标
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        # 获取置信度和类别
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        class_name = COCO_CLASSES[cls] if cls < len(COCO_CLASSES) else f"Class {cls}"

        # 生成颜色
        np.random.seed(cls)
        color = tuple(np.random.randint(0, 255, 3).tolist())

        # 绘制边框
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # 绘制标签
        label = f"{class_name}: {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 显示 FPS
    fps = 1000 / inference_time if inference_time > 0 else 0
    cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 显示检测数
    cv2.putText(frame, f'Detections: {len(boxes)}', (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

    # 显示结果
    cv2.imshow('YOLOv11 Detection', frame)

    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n总帧数: {frame_count}")

# yolo touch原生 60-70 fps
