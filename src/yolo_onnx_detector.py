"""
 作者 lgf
 日期 2025/12/12
"""
import cv2
import numpy as np
import onnxruntime as ort
from typing import List,Tuple,Dict
import time


class YOLOv12ONNX:
    def __init__(
            self,
            model_path: str,
            conf_threshold: float = 0.25,
            iou_threshold: float = 0.45,
            class_names: List[str] = None,
    ):
        """
        初始化YOLOv12 ONNX检测器

        Args:
            model_path: ONNX模型路径
            conf_threshold: 置信度阈值
            iou_threshold: NMS的IOU阈值
            class_names: 类别名称列表
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = class_names or []

        # 初始化ONNX Runtime session
        providers = [
            ('CUDAExecutionProvider',{
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }),
            'CPUExecutionProvider',
        ]

        self.session = ort.InferenceSession(
            model_path,
            providers=providers
        )

        # 获取模型元数据
        self.get_model_info()

        print(f"✓ 模型加载成功: {model_path}")
        print(f"  输入: {self.input_name}, shape: {self.input_shape}")
        print(f"  输出: {self.output_names}")
        print(f"  运行设备: {self.session.get_providers()[0]}")

    def get_model_info(self):
        """获取模型输入输出信息"""
        # 输入信息
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        self.input_shape = model_input.shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

        # 输出信息
        self.output_names = [out.name for out in self.session.get_outputs()]

    def letterbox(
            self,
            image: np.ndarray,
            new_shape: Tuple[int,int] = (640,640),
            color: Tuple[int,int,int] = (114,114,114),
            auto: bool = False,
            scaleFill: bool = False,
            scaleup: bool = True,
            stride: int = 32,
    ) -> Tuple[np.ndarray,Tuple[float,float],Tuple[int,int]]:
        """
        Resize并padding图像（保持宽高比）

        Returns:
            resized_image: 处理后的图像
            ratio: (width_ratio, height_ratio)
            padding: (left_pad, top_pad)
        """
        shape = image.shape[:2]  # 当前shape [height, width]

        if isinstance(new_shape,int):
            new_shape = (new_shape,new_shape)

        # 计算缩放比例
        r = min(new_shape[0] / shape[0],new_shape[1] / shape[1])
        if not scaleup:  # 只缩小，不放大
            r = min(r,1.0)

        # 计算padding
        ratio = r,r  # width, height ratios
        new_unpad = int(round(shape[1] * r)),int(round(shape[0] * r))
        dw,dh = new_shape[1] - new_unpad[0],new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # 最小矩形padding
            dw,dh = np.mod(dw,stride),np.mod(dh,stride)
        elif scaleFill:  # 拉伸
            dw,dh = 0.0,0.0
            new_unpad = (new_shape[1],new_shape[0])
            ratio = new_shape[1] / shape[1],new_shape[0] / shape[0]

        dw /= 2  # 分配到两边
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            image = cv2.resize(image,new_unpad,interpolation=cv2.INTER_LINEAR)

        top,bottom = int(round(dh - 0.1)),int(round(dh + 0.1))
        left,right = int(round(dw - 0.1)),int(round(dw + 0.1))
        image = cv2.copyMakeBorder(
            image,top,bottom,left,right,
            cv2.BORDER_CONSTANT,value=color
        )

        return image,ratio,(dw,dh)

    def preprocess(self,image: np.ndarray) -> np.ndarray:
        """
        预处理图像

        Args:
            image: BGR格式的OpenCV图像

        Returns:
            处理后的tensor [1, 3, H, W]
        """
        # 保存原始尺寸
        self.img_height,self.img_width = image.shape[:2]

        # Letterbox resize
        img,self.ratio,self.padding = self.letterbox(
            image,
            new_shape=(self.input_height,self.input_width),
            auto=False
        )

        # 转换颜色空间 BGR -> RGB
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        # 归一化到 [0, 1]
        img = img.astype(np.float32) / 255.0

        # HWC -> CHW
        img = img.transpose(2,0,1)

        # 添加batch维度
        img = np.expand_dims(img,axis=0)

        # 确保连续内存
        img = np.ascontiguousarray(img)

        return img

    def postprocess(self,outputs: List[np.ndarray]) -> List[Dict]:
        """
        后处理检测结果

        Args:
            outputs: 模型输出

        Returns:
            检测结果列表，每个结果包含 {box, score, class_id, class_name}
        """
        # YOLOv8/v12输出格式: [1, 84, 8400] 或 [1, num_classes+4, num_boxes]
        # 其中 84 = 4 (bbox) + 80 (classes)

        predictions = outputs[0]  # [1, 84, 8400]
        predictions = predictions[0]  # [84, 8400]

        # 转置为 [8400, 84]
        predictions = predictions.T

        # 分离box和scores
        boxes = predictions[:,:4]  # [8400, 4] - x, y, w, h
        scores = predictions[:,4:]  # [8400, 80] - class scores

        # 获取每个框的最大类别分数和ID
        class_ids = np.argmax(scores,axis=1)  # [8400]
        confidences = scores[np.arange(len(scores)),class_ids]  # [8400]

        # 过滤低置信度
        mask = confidences > self.conf_threshold
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        # 转换box格式: (x_center, y_center, w, h) -> (x1, y1, x2, y2)
        boxes_xyxy = self.xywh2xyxy(boxes)

        # 缩放到原图坐标
        boxes_xyxy = self.scale_boxes(boxes_xyxy)

        # NMS
        indices = self.nms(boxes_xyxy,confidences,self.iou_threshold)

        # 组装结果
        results = []
        for i in indices:
            x1,y1,x2,y2 = boxes_xyxy[i].astype(int)
            results.append({
                'box': [x1,y1,x2,y2],
                'score': float(confidences[i]),
                'class_id': int(class_ids[i]),
                'class_name': self.class_names[class_ids[i]] if class_ids[i] < len(
                    self.class_names) else f'class_{class_ids[i]}'
            })

        return results

    def xywh2xyxy(self,boxes: np.ndarray) -> np.ndarray:
        """转换box格式 (x_center, y_center, w, h) -> (x1, y1, x2, y2)"""
        xyxy = np.copy(boxes)
        xyxy[:,0] = boxes[:,0] - boxes[:,2] / 2  # x1
        xyxy[:,1] = boxes[:,1] - boxes[:,3] / 2  # y1
        xyxy[:,2] = boxes[:,0] + boxes[:,2] / 2  # x2
        xyxy[:,3] = boxes[:,1] + boxes[:,3] / 2  # y2
        return xyxy

    def scale_boxes(self,boxes: np.ndarray) -> np.ndarray:
        """缩放boxes到原图尺寸"""
        # 去除padding
        boxes[:,[0,2]] -= self.padding[0]  # x padding
        boxes[:,[1,3]] -= self.padding[1]  # y padding

        # 缩放到原图
        boxes[:,[0,2]] /= self.ratio[0]
        boxes[:,[1,3]] /= self.ratio[1]

        # 裁剪到图像边界
        boxes[:,[0,2]] = boxes[:,[0,2]].clip(0,self.img_width)
        boxes[:,[1,3]] = boxes[:,[1,3]].clip(0,self.img_height)

        return boxes

    def nms(
            self,
            boxes: np.ndarray,
            scores: np.ndarray,
            iou_threshold: float
    ) -> List[int]:
        """Non-Maximum Suppression"""
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i],x1[order[1:]])
            yy1 = np.maximum(y1[i],y1[order[1:]])
            xx2 = np.minimum(x2[i],x2[order[1:]])
            yy2 = np.minimum(y2[i],y2[order[1:]])

            w = np.maximum(0.0,xx2 - xx1)
            h = np.maximum(0.0,yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return keep

    def detect(self,image: np.ndarray) -> Tuple[List[Dict],float]:
        """
        检测图像中的目标

        Args:
            image: BGR格式的OpenCV图像

        Returns:
            (检测结果列表, 推理时间ms)
        """
        # 预处理
        input_tensor = self.preprocess(image)

        # 推理
        start_time = time.time()
        outputs = self.session.run(
            self.output_names,
            {self.input_name: input_tensor}
        )
        inference_time = (time.time() - start_time) * 1000

        # 后处理
        results = self.postprocess(outputs)

        return results,inference_time

    def draw_detections(
            self,
            image: np.ndarray,
            detections: List[Dict],
            line_thickness: int = None
    ) -> np.ndarray:
        """
        在图像上绘制检测结果

        Args:
            image: 原始图像
            detections: 检测结果
            line_thickness: 线条粗细

        Returns:
            绘制后的图像
        """
        img = image.copy()

        # 自适应线条粗细
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1

        for det in detections:
            x1,y1,x2,y2 = det['box']
            score = det['score']
            class_name = det['class_name']

            # 生成颜色（基于class_id）
            color = self.get_color(det['class_id'])

            # 绘制边框
            cv2.rectangle(img,(x1,y1),(x2,y2),color,thickness=tl)

            # 绘制标签
            label = f"{class_name} {score:.2f}"
            tf = max(tl - 1,1)  # 字体粗细
            t_size = cv2.getTextSize(label,0,fontScale=tl / 3,thickness=tf)[0]
            c2 = x1 + t_size[0],y1 - t_size[1] - 3
            cv2.rectangle(img,(x1,y1),c2,color,-1,cv2.LINE_AA)
            cv2.putText(
                img,label,(x1,y1 - 2),0,tl / 3,
                [225,255,255],thickness=tf,lineType=cv2.LINE_AA
            )

        return img

    @staticmethod
    def get_color(class_id: int) -> Tuple[int,int,int]:
        """为每个类别生成固定颜色"""
        np.random.seed(class_id)
        return tuple(np.random.randint(0,255,3).tolist())


# ==================== 使用示例 ====================
if __name__ == '__main__':
    # COCO类别名称
    COCO_CLASSES = [
        'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
        'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat',
        'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack',
        'umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball',
        'kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket',
        'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
        'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair',
        'couch','potted plant','bed','dining table','toilet','tv','laptop','mouse',
        'remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator',
        'book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
    ]

    # 初始化检测器
    detector = YOLOv12ONNX(
        model_path='../model/yolo11n.onnx',
        conf_threshold=0.25,
        iou_threshold=0.45,
        class_names=COCO_CLASSES
    )

    # ========== 图像检测 ==========
    # image = cv2.imread('img/three-cat.jpg')
    # detections,inference_time = detector.detect(image)
    #
    # print(f"\n检测到 {len(detections)} 个目标，推理时间: {inference_time:.2f}ms")
    # for i,det in enumerate(detections):
    #     print(f"  [{i + 1}] {det['class_name']}: {det['score']:.3f}, box: {det['box']}")

    # 绘制结果
    # result_img = detector.draw_detections(image,detections)
    # cv2.imwrite('result/result.jpg',result_img)


    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_count = 0
    total_time = 0

    while cap.isOpened():
        ret,frame = cap.read()
        if not ret:
            break

        detections,inference_time = detector.detect(frame)
        result_frame = detector.draw_detections(frame,detections)

        # 显示FPS
        total_time += inference_time
        frame_count += 1
        avg_fps = 1000 / (total_time / frame_count)
        cv2.putText(
            result_frame,f'FPS: {avg_fps:.1f}',(10,30),
            cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2
        )

        cv2.imshow('Detection',result_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

    cv2.destroyAllWindows()

    print(f"\n平均FPS: {avg_fps:.2f}")



# python onnx 120fps