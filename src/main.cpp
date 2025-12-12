#include "yolo_detector.h"
#include "utils.h"
#include <iostream>
#include <chrono>
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <windows.h>
#include <algorithm>  // 添加这一行

void detectImage(const std::string& model_path, const std::string& image_path) {
    std::cout << "\n========== 图像检测模式 ==========\n" << std::endl;

    // 初始化检测器
    YOLODetector detector(model_path, 0.25f, 0.45f, COCO_CLASSES, true);

    // 读取图像
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "错误: 无法读取图像 " << image_path << std::endl;
        return;
    }

    // 检测
    float inference_time;
    auto detections = detector.detect(image, inference_time);

    std::cout << "Found " << detections.size() << " objects" << std::endl;
    std::cout << "Inference time: " << inference_time << " ms" << std::endl;
    std::cout << "FPS: " << 1000.0f / inference_time << std::endl;

    // 打印结果
    for (size_t i = 0; i < detections.size(); i++) {
        const auto& det = detections[i];
        std::cout << "  [" << i + 1 << "] " << det.class_name
                  << ": " << det.confidence * 100 << "%"
                  << " at [" << det.box.x << ", " << det.box.y << ", "
                  << det.box.width << ", " << det.box.height << "]" << std::endl;
    }

    // 绘制并保存
    cv::Mat result = detector.drawDetections(image, detections);
    cv::imwrite("result.jpg", result);
    std::cout << "\n结果已保存到 result.jpg" << std::endl;
}

void detectVideo(const std::string& model_path, const std::string& video_path) {
    std::cout << "\n========== 视频检测模式 ==========\n" << std::endl;

    // 初始化检测器
    YOLODetector detector(model_path, 0.25f, 0.45f, COCO_CLASSES, true);

    // 打开视频
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "错误: 无法打开视频 " << video_path << std::endl;
        return;
    }

    // 获取视频信息
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);

    std::cout << "视频分辨率: " << width << "x" << height << std::endl;
    std::cout << "视频FPS: " << fps << std::endl;
    std::cout << "\n按 'q' 退出\n" << std::endl;

    // 创建窗口
    cv::namedWindow("Detection", cv::WINDOW_NORMAL);

    // FPS 统计
    std::vector<float> fps_list;
    int frame_count = 0;

    while (true) {
        cv::Mat frame;
        if (!cap.read(frame)) {
            break;
        }

        // 检测
        float inference_time;
        auto detections = detector.detect(frame, inference_time);

        // 绘制
        cv::Mat result = detector.drawDetections(frame, detections);

        // 计算并显示 FPS
        float current_fps = 1000.0f / inference_time;
        fps_list.push_back(current_fps);
        frame_count++;

        // 计算平均 FPS（最近30帧）
        int window = (30 < static_cast<int>(fps_list.size())) ? 30 : static_cast<int>(fps_list.size());
        float avg_fps = 0.0f;
        for (int i = fps_list.size() - window; i < fps_list.size(); i++) {
            avg_fps += fps_list[i];
        }
        avg_fps /= window;

        // 显示信息
        std::string fps_text = "FPS: " + std::to_string(static_cast<int>(avg_fps));
        cv::putText(result, fps_text, cv::Point(10, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

        std::string det_text = "Detections: " + std::to_string(detections.size());
        cv::putText(result, det_text, cv::Point(10, 60),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);

        // 显示
        cv::imshow("Detection", result);

        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    // 统计
    if (!fps_list.empty()) {
        float sum = 0.0f;
        for (float f : fps_list) sum += f;
        std::cout << "\n平均 FPS: " << sum / fps_list.size() << std::endl;
    }
}

void detectCamera(const std::string& model_path, int camera_id) {
    std::cout << "\n========== Camera Detection Mode ==========\n" << std::endl;

    // Initialize detector
    YOLODetector detector(model_path, 0.25f, 0.45f, COCO_CLASSES, true);

    // Open camera
    cv::VideoCapture cap;

    // 尝试多种方式打开摄像头
    std::cout << "Trying to open camera " << camera_id << "..." << std::endl;

    // 方法1：直接打开
    cap.open(camera_id, cv::CAP_ANY);

    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open camera using CAP_ANY" << std::endl;

        // 方法2：尝试使用DirectShow
        cap.open(camera_id + cv::CAP_DSHOW);
        std::cout << "Trying DirectShow..." << std::endl;
    }

    if (!cap.isOpened()) {
        // 方法3：尝试MSMF
        cap.open(camera_id + cv::CAP_MSMF);
        std::cout << "Trying MSMF..." << std::endl;
    }

    if (!cap.isOpened()) {
        std::cerr << "Failed to open camera " << camera_id << std::endl;
        std::cout << "Available camera backends:" << std::endl;
        std::cout << "  0: CAP_ANY" << std::endl;
        std::cout << "  700: CAP_DSHOW" << std::endl;
        std::cout << "  1400: CAP_MSMF" << std::endl;
        return;
    }

    std::cout << "✓ Camera opened successfully" << std::endl;

    // Set camera parameters - 更简单的设置
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);  // 改为较低的分辨率
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_FPS, 30);

    // 禁用硬件加速（使用CPU处理）
    cap.set(cv::CAP_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_NONE);

    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);

    std::cout << "Camera resolution: " << width << "x" << height << std::endl;
    std::cout << "Camera FPS: " << fps << std::endl;
    std::cout << "\nPress 'q' to quit, 'p' to pause, 's' to save screenshot\n" << std::endl;

    // 测试读取一帧
    cv::Mat test_frame;
    if (!cap.read(test_frame)) {
        std::cerr << "Error: Cannot read frame from camera" << std::endl;
        cap.release();
        return;
    }
    std::cout << "✓ Frame read successfully: "
              << test_frame.cols << "x" << test_frame.rows << std::endl;

    // Create window
    cv::namedWindow("Camera Detection", cv::WINDOW_AUTOSIZE);

    // FPS statistics
    std::vector<float> fps_list;
    int frame_count = 0;
    bool paused = false;

    cv::Mat last_result;

    while (true) {
        cv::Mat frame;

        if (!paused) {
            if (!cap.read(frame)) {
                std::cerr << "Warning: Failed to read frame" << std::endl;
                break;
            }

            // 检查帧是否有效
            if (frame.empty()) {
                std::cerr << "Warning: Empty frame received" << std::endl;
                continue;
            }

            // Detect
            float inference_time;
            auto detections = detector.detect(frame, inference_time);

            // Draw
            last_result = detector.drawDetections(frame, detections);

            // Calculate FPS
            float current_fps = 1000.0f / inference_time;
            fps_list.push_back(current_fps);
            frame_count++;

            // Display info
            std::string fps_text = "FPS: " + std::to_string(static_cast<int>(current_fps));
            cv::putText(last_result, fps_text, cv::Point(10, 30),
                       cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

            std::string inf_text = "Inference: " +
                std::to_string(static_cast<int>(inference_time)) + "ms";
            cv::putText(last_result, inf_text, cv::Point(10, 60),
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);

            std::string det_text = "Detections: " + std::to_string(detections.size());
            cv::putText(last_result, det_text, cv::Point(10, 90),
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);

        } else {
            // Paused state
            if (!last_result.empty()) {
                cv::putText(last_result, "PAUSED",
                           cv::Point(width / 2 - 100, height / 2),
                           cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 255), 3);
            }
        }

        // Display
        if (!last_result.empty()) {
            cv::imshow("Camera Detection", last_result);
        }

        // Key handling
        char key = static_cast<char>(cv::waitKey(1));
        if (key == 'q' || key == 27) {  // 27 is ESC
            break;
        } else if (key == 'p') {
            paused = !paused;
            std::cout << (paused ? "Paused" : "Resumed") << std::endl;
        } else if (key == 's') {
            if (!last_result.empty()) {
                std::string filename = "screenshot_" +
                    std::to_string(std::time(nullptr)) + ".jpg";
                cv::imwrite(filename, last_result);
                std::cout << "Screenshot saved: " << filename << std::endl;
            }
        }
    }

    cap.release();
    cv::destroyAllWindows();

    // Statistics
    if (!fps_list.empty()) {
        float sum = 0.0f;
        for (float f : fps_list) sum += f;
        std::cout << "\nTotal frames: " << frame_count << std::endl;
        std::cout << "Average FPS: " << sum / fps_list.size() << std::endl;
    }
}

int main(int argc, char* argv[]) {
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
    std::cout << R"(
========================================
    YOLOv12 C++ 检测器
========================================
)" << std::endl;

    if (argc < 3) {
        std::cout << "用法:" << std::endl;
        std::cout << "  图像检测: " << argv[0] << " <model_path> <image_path>" << std::endl;
        std::cout << "  视频检测: " << argv[0] << " <model_path> <video_path>" << std::endl;
        std::cout << "  摄像头检测: " << argv[0] << " <model_path> camera [camera_id]" << std::endl;
        std::cout << "\n示例:" << std::endl;
        std::cout << "  " << argv[0] << " yolo11n.onnx test.jpg" << std::endl;
        std::cout << "  " << argv[0] << " yolo11n.onnx video.mp4" << std::endl;
        std::cout << "  " << argv[0] << " yolo11n.onnx camera 0" << std::endl;
        return -1;
    }

    std::string model_path = argv[1];
    std::string source = argv[2];

    try {
        if (source == "camera") {
            int camera_id = (argc > 3) ? std::stoi(argv[3]) : 0;
            detectCamera(model_path, camera_id);
        } else if (source.find(".mp4") != std::string::npos ||
                   source.find(".avi") != std::string::npos) {
            detectVideo(model_path, source);
        } else {
            detectImage(model_path, source);
        }
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}


// D:\Garfield_Lyu\postgraduate\project\YOLOv12-ONNX-TensorRT\cmake-build-debug\yolo_detector.exe model/yolo11n.onnx camera 0
// D:\Garfield_Lyu\postgraduate\project\YOLOv12-ONNX-TensorRT\cmake-build-debug\yolo_detector.exe model/yolo11n.onnx img/three-cat.jpg