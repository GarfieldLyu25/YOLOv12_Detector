#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

// 单个检测结果
struct Detection {
    cv::Rect2f box;   // xyxy
    int class_id;
    float score;
};

class YoloDetector {
public:
    struct Config {
        std::string model_path;
        int input_w = 640;
        int input_h = 640;
        int num_classes = 80;
        float conf_thresh = 0.25f;
        float iou_thresh = 0.45f;
        bool letterbox = true;   // 保持比例填充
        bool use_cpu = true;     // 仅 CPU 推理；如需 GPU 可后续扩展 EP
    };

    explicit YoloDetector(const Config& cfg);
    ~YoloDetector() = default;

    // 对单张图像进行检测
    std::vector<Detection> detect(const cv::Mat& bgr_image);

private:
    Config cfg_;

    // ORT
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_info_;
    std::vector<std::string> input_name_storage_;
    std::vector<std::string> output_name_storage_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;

    // 预处理：BGR -> RGB，resize/letterbox，归一化，NCHW
    struct PreprocResult {
        cv::Mat processed;   // 预处理后图像（RGB）
        float scale;         // 缩放比例
        int pad_w;           // 左右总填充
        int pad_h;           // 上下总填充
    };
    PreprocResult preprocess(const cv::Mat& bgr) const;

    // 后处理：从输出张量解析框，映射回原图坐标，做 NMS
    std::vector<Detection> postprocess(const cv::Size& orig_size,
                                       float scale, int pad_w, int pad_h,
                                       const float* output, const std::vector<int64_t>& shape) const;

    // NMS
    static std::vector<int> nms_indices(const std::vector<cv::Rect2f>& boxes,
                                        const std::vector<float>& scores,
                                        float iou_thresh);

    // 辅助：获取输出形状
    static std::vector<int64_t> get_tensor_shape(const Ort::Value& val);
}
;