#include "yolo_detector.h"
#include "utils.h"
#include <algorithm>
#include <numeric>
#include <iostream>

YOLODetector::YOLODetector(
    const std::string& model_path,
    float conf_threshold,
    float iou_threshold,
    const std::vector<std::string>& class_names,
    bool use_cuda
) : env_(ORT_LOGGING_LEVEL_WARNING, "YOLOv12"),
    memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
    conf_threshold_(conf_threshold),
    iou_threshold_(iou_threshold),
    class_names_(class_names)
{
    initializeModel(model_path, use_cuda);
}

YOLODetector::~YOLODetector() {
    // 清理资源
    for (auto name : input_names_) {
        delete[] name;
    }
    for (auto name : output_names_) {
        delete[] name;
    }
}

void YOLODetector::initializeModel(const std::string& model_path, bool use_cuda) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "初始化 YOLO 检测器" << std::endl;
    std::cout << "========================================" << std::endl;

    // 配置会话选项
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(4);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef USE_CUDA
    if (use_cuda) {
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = 0;
        cuda_options.arena_extend_strategy = 0;
        cuda_options.gpu_mem_limit = 2ULL * 1024 * 1024 * 1024; // 2GB
        cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
        cuda_options.do_copy_in_default_stream = 1;

        try {
            session_options.AppendExecutionProvider_CUDA(cuda_options);
            std::cout << "✓ CUDA 加速已启用" << std::endl;
        } catch (const Ort::Exception& e) {
            std::cerr << "⚠ CUDA 初始化失败，使用 CPU: " << e.what() << std::endl;
        }
    }
#else
    std::cout << "ℹ 使用 CPU 推理" << std::endl;
#endif

    // 创建会话
#ifdef _WIN32
    // Windows: 需要转换为宽字符
    std::wstring model_path_w(model_path.begin(), model_path.end());
    session_ = std::make_unique<Ort::Session>(env_, model_path_w.c_str(), session_options);
#else
    session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options);
#endif

    // 获取输入信息
    size_t num_input_nodes = session_->GetInputCount();
    for (size_t i = 0; i < num_input_nodes; i++) {
        auto input_name = session_->GetInputNameAllocated(i, allocator_);
        input_names_.push_back(input_name.get());

        auto type_info = session_->GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        input_shape_ = tensor_info.GetShape();
    }

    input_width_ = static_cast<int>(input_shape_[3]);
    input_height_ = static_cast<int>(input_shape_[2]);

    // 获取输出信息
    size_t num_output_nodes = session_->GetOutputCount();
    for (size_t i = 0; i < num_output_nodes; i++) {
        auto output_name = session_->GetOutputNameAllocated(i, allocator_);
        output_names_.push_back(output_name.get());
    }

    std::cout << "✓ 模型加载成功: " << model_path << std::endl;
    std::cout << "  输入尺寸: " << input_width_ << "x" << input_height_ << std::endl;
    std::cout << "  类别数量: " << class_names_.size() << std::endl;
    std::cout << "========================================\n" << std::endl;
}

cv::Mat YOLODetector::letterbox(const cv::Mat& image) {
    orig_width_ = image.cols;
    orig_height_ = image.rows;

    // 计算缩放比例
    ratio_ = std::min(
        static_cast<float>(input_width_) / orig_width_,
        static_cast<float>(input_height_) / orig_height_
    );

    int new_width = static_cast<int>(orig_width_ * ratio_);
    int new_height = static_cast<int>(orig_height_ * ratio_);

    // Resize
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_width, new_height), 0, 0, cv::INTER_LINEAR);

    // 计算 padding
    dw_ = (input_width_ - new_width) / 2.0f;
    dh_ = (input_height_ - new_height) / 2.0f;

    int top = static_cast<int>(std::round(dh_ - 0.1));
    int bottom = static_cast<int>(std::round(dh_ + 0.1));
    int left = static_cast<int>(std::round(dw_ - 0.1));
    int right = static_cast<int>(std::round(dw_ + 0.1));

    // Padding
    cv::Mat padded;
    cv::copyMakeBorder(
        resized, padded,
        top, bottom, left, right,
        cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114)
    );

    return padded;
}

std::vector<float> YOLODetector::preprocess(const cv::Mat& image) {
    // Letterbox
    cv::Mat processed = letterbox(image);

    // BGR -> RGB
    cv::cvtColor(processed, processed, cv::COLOR_BGR2RGB);

    // 归一化并转换为 CHW 格式
    std::vector<float> input_tensor;
    input_tensor.resize(3 * input_height_ * input_width_);

    for (int c = 0; c < 3; c++) {
        for (int h = 0; h < input_height_; h++) {
            for (int w = 0; w < input_width_; w++) {
                input_tensor[c * input_height_ * input_width_ + h * input_width_ + w] =
                    processed.at<cv::Vec3b>(h, w)[c] / 255.0f;
            }
        }
    }

    return input_tensor;
}

std::vector<Detection> YOLODetector::detect(const cv::Mat& image, float& inference_time) {
    Timer timer;

    // 预处理
    std::vector<float> input_tensor = preprocess(image);

    // 创建输入 tensor
    std::vector<int64_t> input_shape = {1, 3, input_height_, input_width_};

    Ort::Value input_tensor_ort = Ort::Value::CreateTensor<float>(
        memory_info_,
        input_tensor.data(),
        input_tensor.size(),
        input_shape.data(),
        input_shape.size()
    );

    // 推理
    Timer infer_timer;
    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        input_names_.data(),
        &input_tensor_ort,
        1,
        output_names_.data(),
        output_names_.size()
    );
    inference_time = static_cast<float>(infer_timer.elapsed());

    // 后处理
    auto detections = postprocess(output_tensors);

    return detections;
}

void YOLODetector::xywh2xyxy(float* boxes, int num_boxes) {
    for (int i = 0; i < num_boxes; i++) {
        float x_center = boxes[i * 4 + 0];
        float y_center = boxes[i * 4 + 1];
        float width = boxes[i * 4 + 2];
        float height = boxes[i * 4 + 3];

        boxes[i * 4 + 0] = x_center - width / 2;   // x1
        boxes[i * 4 + 1] = y_center - height / 2;  // y1
        boxes[i * 4 + 2] = x_center + width / 2;   // x2
        boxes[i * 4 + 3] = y_center + height / 2;  // y2
    }
}

void YOLODetector::scaleBoxes(std::vector<cv::Rect>& boxes) {
    for (auto& box : boxes) {
        // 去除 padding
        box.x = static_cast<int>((box.x - dw_) / ratio_);
        box.y = static_cast<int>((box.y - dh_) / ratio_);
        box.width = static_cast<int>(box.width / ratio_);
        box.height = static_cast<int>(box.height / ratio_);

        // 裁剪到图像边界
        box.x = std::max(0, std::min(box.x, orig_width_));
        box.y = std::max(0, std::min(box.y, orig_height_));
        box.width = std::min(box.width, orig_width_ - box.x);
        box.height = std::min(box.height, orig_height_ - box.y);
    }
}

std::vector<int> YOLODetector::nms(
    const std::vector<cv::Rect>& boxes,
    const std::vector<float>& scores,
    float iou_threshold
) {
    std::vector<int> indices(scores.size());
    std::iota(indices.begin(), indices.end(), 0);

    // 按分数排序
    std::sort(indices.begin(), indices.end(),
        [&scores](int i1, int i2) { return scores[i1] > scores[i2]; });

    std::vector<int> keep;
    while (!indices.empty()) {
        int current = indices[0];
        keep.push_back(current);

        std::vector<int> new_indices;
        for (size_t i = 1; i < indices.size(); i++) {
            int idx = indices[i];

            // 计算 IOU
            int x1 = std::max(boxes[current].x, boxes[idx].x);
            int y1 = std::max(boxes[current].y, boxes[idx].y);
            int x2 = std::min(boxes[current].x + boxes[current].width,
                            boxes[idx].x + boxes[idx].width);
            int y2 = std::min(boxes[current].y + boxes[current].height,
                            boxes[idx].y + boxes[idx].height);

            int w = std::max(0, x2 - x1);
            int h = std::max(0, y2 - y1);
            int inter = w * h;

            int area1 = boxes[current].width * boxes[current].height;
            int area2 = boxes[idx].width * boxes[idx].height;
            float iou = static_cast<float>(inter) / (area1 + area2 - inter);

            if (iou <= iou_threshold) {
                new_indices.push_back(idx);
            }
        }

        indices = new_indices;
    }

    return keep;
}

std::vector<Detection> YOLODetector::postprocess(const std::vector<Ort::Value>& outputs) {
    // 获取输出数据
    const float* output_data = outputs[0].GetTensorData<float>();
    auto output_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();

    // output shape: [1, 84, 8400] 或 [1, num_classes+4, num_boxes]
    int num_classes = static_cast<int>(output_shape[1]) - 4;
    int num_boxes = static_cast<int>(output_shape[2]);

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;

    // 遍历所有检测框
    for (int i = 0; i < num_boxes; i++) {
        // 解析 box 坐标
        float x_center = output_data[i];
        float y_center = output_data[num_boxes + i];
        float width = output_data[2 * num_boxes + i];
        float height = output_data[3 * num_boxes + i];

        // 找到最大类别分数
        float max_score = 0.0f;
        int max_class_id = 0;

        for (int c = 0; c < num_classes; c++) {
            float score = output_data[(4 + c) * num_boxes + i];
            if (score > max_score) {
                max_score = score;
                max_class_id = c;
            }
        }

        // 过滤低置信度
        if (max_score > conf_threshold_) {
            boxes.push_back(cv::Rect(
                static_cast<int>(x_center - width / 2),
                static_cast<int>(y_center - height / 2),
                static_cast<int>(width),
                static_cast<int>(height)
            ));
            confidences.push_back(max_score);
            class_ids.push_back(max_class_id);
        }
    }

    // 缩放到原图
    scaleBoxes(boxes);

    // NMS
    std::vector<int> keep_indices = nms(boxes, confidences, iou_threshold_);

    // 组装结果
    std::vector<Detection> detections;
    for (int idx : keep_indices) {
        Detection det;
        det.box = boxes[idx];
        det.confidence = confidences[idx];
        det.class_id = class_ids[idx];
        det.class_name = (class_ids[idx] < class_names_.size()) ?
            class_names_[class_ids[idx]] : "class_" + std::to_string(class_ids[idx]);
        detections.push_back(det);
    }

    return detections;
}

cv::Mat YOLODetector::drawDetections(
    const cv::Mat& image,
    const std::vector<Detection>& detections
) {
    cv::Mat result = image.clone();

    for (const auto& det : detections) {
        // 颜色
        cv::Scalar color = getColor(det.class_id);

        // 边框
        cv::rectangle(result, det.box, color, 2);

        // 标签
        std::string label = det.class_name + " " +
            std::to_string(static_cast<int>(det.confidence * 100)) + "%";

        int baseline;
        cv::Size label_size = cv::getTextSize(
            label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline
        );

        cv::rectangle(
            result,
            cv::Point(det.box.x, det.box.y - label_size.height - 5),
            cv::Point(det.box.x + label_size.width, det.box.y),
            color, -1
        );

        cv::putText(
            result, label,
            cv::Point(det.box.x, det.box.y - 3),
            cv::FONT_HERSHEY_SIMPLEX, 0.5,
            cv::Scalar(255, 255, 255), 1
        );
    }

    return result;
}

cv::Scalar YOLODetector::getColor(int class_id) {
    cv::RNG rng(class_id * 100);
    return cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
}