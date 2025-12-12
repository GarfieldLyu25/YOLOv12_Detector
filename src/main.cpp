#include "D:/Garfield_Lyu/postgraduate/project/YOLOv12-ONNX-TensorRT/include/yolo_detector.h"
#include <iostream>

static void draw_detections(cv::Mat& img, const std::vector<Detection>& dets) {
    for (const auto& d : dets) {
        cv::rectangle(img, d.box, cv::Scalar(0, 255, 0), 2);
        char text[64];
        std::snprintf(text, sizeof(text), "id:%d score:%.2f", d.class_id, d.score);
        cv::putText(img, text, cv::Point((int)d.box.x, (int)d.box.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    }
}

int main(int argc, char** argv) {
    try {
        std::cout << "========================================\n";
        std::cout << "    YOLOv12 C++ 检测器\n";
        std::cout << "========================================\n\n";

        std::string model_path = "model/yolo11n.onnx"; // 按需修改
        std::string image_path = argc >= 2 ? argv[1] : "assets/test.jpg";

        std::cout << "========================================\n";
        std::cout << "初始化 YOLO 检测器\n";
        std::cout << "========================================\n";
        YoloDetector::Config cfg;
        cfg.model_path = model_path;
        cfg.use_cpu = true;
        cfg.input_w = 640;
        cfg.input_h = 640;
        cfg.num_classes = 80;
        cfg.conf_thresh = 0.25f;
        cfg.iou_thresh = 0.45f;
        cfg.letterbox = true;

        std::cout << u8"\u2139 使用 CPU 推理\n";
        YoloDetector detector(cfg);
        std::cout << u8"\u2714 模型加载成功: " << model_path << "\n";
        std::cout << "  输入尺寸: " << cfg.input_w << "x" << cfg.input_h << "\n";
        std::cout << "  类别数量: " << cfg.num_classes << "\n";
        std::cout << "========================================\n";

        cv::Mat img = cv::imread(image_path);
        if (img.empty()) {
            std::cerr << "无法读取图像: " << image_path << "\n";
            return 1;
        }

        auto dets = detector.detect(img);
        std::cout << "检测到目标数量: " << dets.size() << "\n";

        draw_detections(img, dets);
        cv::imwrite("output.jpg", img);
        std::cout << "结果已保存到 output.jpg\n";

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "异常: " << ex.what() << "\n";
        return 2;
    }
}