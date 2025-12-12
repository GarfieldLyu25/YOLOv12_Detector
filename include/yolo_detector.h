#ifndef YOLO_DETECTOR_H
#define YOLO_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <memory>

// 检测结果结构
struct Detection {
    cv::Rect box;           // 边界框
    float confidence;       // 置信度
    int class_id;          // 类别ID
    std::string class_name; // 类别名称
};

// YOLO 检测器类
class YOLODetector {
public:
    /**
     * 构造函数
     * @param model_path ONNX模型路径
     * @param conf_threshold 置信度阈值
     * @param iou_threshold NMS的IOU阈值
     * @param class_names 类别名称列表
     * @param use_cuda 是否使用CUDA加速
     */
    YOLODetector(
        const std::string& model_path,
        float conf_threshold = 0.25f,
        float iou_threshold = 0.45f,
        const std::vector<std::string>& class_names = {},
        bool use_cuda = true
    );

    /**
     * 析构函数
     */
    ~YOLODetector();

    /**
     * 检测图像中的目标
     * @param image 输入图像
     * @param inference_time 输出推理时间(ms)
     * @return 检测结果列表
     */
    std::vector<Detection> detect(const cv::Mat& image, float& inference_time);

    /**
     * 在图像上绘制检测结果
     * @param image 输入图像
     * @param detections 检测结果
     * @return 绘制后的图像
     */
    cv::Mat drawDetections(const cv::Mat& image, const std::vector<Detection>& detections);

    /**
     * 获取模型输入尺寸
     */
    cv::Size getInputSize() const { return cv::Size(input_width_, input_height_); }

private:
    // ONNX Runtime 相关
    Ort::Env env_;
    Ort::Session session_;
    Ort::AllocatorWithDefaultOptions allocator_;
    Ort::MemoryInfo memory_info_;

    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::vector<int64_t> input_shape_;

    // 模型参数
    int input_width_;
    int input_height_;
    float conf_threshold_;
    float iou_threshold_;
    std::vector<std::string> class_names_;

    // 图像信息
    int orig_width_;
    int orig_height_;
    float ratio_;
    float dw_;
    float dh_;

    // 私有方法
    void initializeModel(const std::string& model_path, bool use_cuda);
    cv::Mat letterbox(const cv::Mat& image);
    std::vector<float> preprocess(const cv::Mat& image);
    std::vector<Detection> postprocess(const std::vector<Ort::Value>& outputs);

    void xywh2xyxy(float* boxes, int num_boxes);
    void scaleBoxes(std::vector<cv::Rect>& boxes);
    std::vector<int> nms(const std::vector<cv::Rect>& boxes,
                         const std::vector<float>& scores,
                         float iou_threshold);

    static cv::Scalar getColor(int class_id);
};

#endif // YOLO_DETECTOR_H