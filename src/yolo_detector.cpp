#include "yolo_detector.h"
#include <algorithm>
#include <numeric>
#include <stdexcept>

static inline float sigmoid(float x) {
    return 1.f / (1.f + std::exp(-x));
}

YoloDetector::YoloDetector(const Config& cfg)
    : cfg_(cfg),
      env_(ORT_LOGGING_LEVEL_WARNING, "yolo"),
      session_options_(),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {

    // CPU 推理（默认）
    session_options_.SetIntraOpNumThreads(1);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    // 如需 GPU 可在此追加 EP，这里保持 CPU 简洁

    // 创建会话
    session_ = std::make_unique<Ort::Session>(env_, cfg_.model_path.c_str(), session_options_);

    // 记录输入/输出名（复制到 std::string，避免原分配器管理的指针跨作用域释放）
    Ort::AllocatorWithDefaultOptions allocator;
    size_t n_input = session_->GetInputCount();
    size_t n_output = session_->GetOutputCount();

    input_name_storage_.reserve(n_input);
    input_names_.reserve(n_input);
    for (size_t i = 0; i < n_input; ++i) {
        auto name_alloc = session_->GetInputNameAllocated(i, allocator);
        input_name_storage_.emplace_back(name_alloc.get());
        input_names_.push_back(input_name_storage_.back().c_str());
    }
    output_name_storage_.reserve(n_output);
    output_names_.reserve(n_output);
    for (size_t i = 0; i < n_output; ++i) {
        auto name_alloc = session_->GetOutputNameAllocated(i, allocator);
        output_name_storage_.emplace_back(name_alloc.get());
        output_names_.push_back(output_name_storage_.back().c_str());
    }

    // 尝试从模型输入形状同步 input_w/h（若模型是动态形状则保持配置值）
    auto typeinfo = session_->GetInputTypeInfo(0);
    auto tensor_info = typeinfo.GetTensorTypeAndShapeInfo();
    auto shape = tensor_info.GetShape();
    // 预期 [N, C, H, W] 或 [1,3,640,640]
    if (shape.size() == 4) {
        if (shape[2] > 0 && shape[3] > 0) {
            cfg_.input_h = static_cast<int>(shape[2]);
            cfg_.input_w = static_cast<int>(shape[3]);
        }
    }
}

YoloDetector::PreprocResult YoloDetector::preprocess(const cv::Mat& bgr) const {
    if (bgr.empty()) {
        throw std::runtime_error("Input image is empty");
    }

    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    int in_w = cfg_.input_w;
    int in_h = cfg_.input_h;

    float scale = 1.0f;
    int pad_w = 0, pad_h = 0;
    cv::Mat resized;

    if (cfg_.letterbox) {
        float r = std::min(in_w / (float)rgb.cols, in_h / (float)rgb.rows);
        int new_w = static_cast<int>(std::round(rgb.cols * r));
        int new_h = static_cast<int>(std::round(rgb.rows * r));
        cv::resize(rgb, resized, cv::Size(new_w, new_h));

        int dw = in_w - new_w;
        int dh = in_h - new_h;
        pad_w = dw;
        pad_h = dh;
        int top = dh / 2;
        int bottom = dh - top;
        int left = dw / 2;
        int right = dw - left;

        cv::copyMakeBorder(resized, resized, top, bottom, left, right,
                           cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
        scale = r;
    } else {
        cv::resize(rgb, resized, cv::Size(in_w, in_h));
        scale = static_cast<float>(in_w) / rgb.cols; // 粗略比例，非 letterbox 时用于反映
        pad_w = 0;
        pad_h = 0;
    }

    return { resized, scale, pad_w, pad_h };
}

std::vector<int64_t> YoloDetector::get_tensor_shape(const Ort::Value& val) {
    auto info = val.GetTensorTypeAndShapeInfo();
    return info.GetShape();
}

std::vector<Detection> YoloDetector::postprocess(const cv::Size& orig_size,
                                                 float scale, int pad_w, int pad_h,
                                                 const float* out, const std::vector<int64_t>& shape) const {
    // 兼容常见 YOLO 输出：
    // 1) [1, num, 85] -> 4 box + 80 cls
    // 2) [num, 85]
    // 若模型是不同布局，可能需要适配
    int64_t num = 0;
    int64_t dim = 0;
    if (shape.size() == 3) {
        num = shape[1];
        dim = shape[2];
    } else if (shape.size() == 2) {
        num = shape[0];
        dim = shape[1];
    } else {
        throw std::runtime_error("Unexpected output shape, expect 2D or 3D tensor");
    }

    if (dim < 5) { // 至少 4 + 1
        throw std::runtime_error("Output dim too small for YOLO format");
    }

    std::vector<cv::Rect2f> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;

    boxes.reserve(static_cast<size_t>(num));
    scores.reserve(static_cast<size_t>(num));
    class_ids.reserve(static_cast<size_t>(num));

    const int img_w = orig_size.width;
    const int img_h = orig_size.height;
    const int in_w = cfg_.input_w;
    const int in_h = cfg_.input_h;

    // 反映射到原图坐标（针对 letterbox）
    const float pad_w_half = pad_w * 0.5f;
    const float pad_h_half = pad_h * 0.5f;

    for (int64_t i = 0; i < num; ++i) {
        const float* row = out + i * dim;

        float x = row[0];
        float y = row[1];
        float w = row[2];
        float h = row[3];

        // 如果输出是中心点格式(cx, cy, w, h)，转为 xyxy
        float x1 = x - w * 0.5f;
        float y1 = y - h * 0.5f;
        float x2 = x + w * 0.5f;
        float y2 = y + h * 0.5f;

        // 映射回 letterbox 前的坐标
        // 先去除 padding，再除以 scale
        x1 = (x1 - pad_w_half) / std::max(scale, 1e-6f);
        y1 = (y1 - pad_h_half) / std::max(scale, 1e-6f);
        x2 = (x2 - pad_w_half) / std::max(scale, 1e-6f);
        y2 = (y2 - pad_h_half) / std::max(scale, 1e-6f);

        // 裁剪到原图
        x1 = std::clamp(x1, 0.f, (float)img_w - 1.f);
        y1 = std::clamp(y1, 0.f, (float)img_h - 1.f);
        x2 = std::clamp(x2, 0.f, (float)img_w - 1.f);
        y2 = std::clamp(y2, 0.f, (float)img_h - 1.f);

        // 分类得分：取最大类别
        int best_id = -1;
        float best_score = -1.f;

        // 有的模型会对类别用 sigmoid
        for (int c = 0; c < cfg_.num_classes && (4 + 1 + c) < dim; ++c) {
            float s = row[4 + c];
            // 如模型已是后处理后的概率，可以跳过 sigmoid；保守地套一层 sigmoid
            s = sigmoid(s);
            if (s > best_score) {
                best_score = s;
                best_id = c;
            }
        }

        if (best_score >= cfg_.conf_thresh) {
            boxes.emplace_back(cv::Rect2f(x1, y1, x2 - x1, y2 - y1));
            scores.emplace_back(best_score);
            class_ids.emplace_back(best_id);
        }
    }

    // NMS
    auto keep = nms_indices(boxes, scores, cfg_.iou_thresh);

    std::vector<Detection> dets;
    dets.reserve(keep.size());
    for (int idx : keep) {
        dets.push_back({boxes[idx], class_ids[idx], scores[idx]});
    }
    return dets;
}

std::vector<int> YoloDetector::nms_indices(const std::vector<cv::Rect2f>& boxes,
                                           const std::vector<float>& scores,
                                           float iou_thresh) {
    std::vector<int> order(boxes.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return scores[a] > scores[b];
    });

    std::vector<int> keep;
    std::vector<char> removed(boxes.size(), 0);

    for (size_t i = 0; i < order.size(); ++i) {
        int idx = order[i];
        if (removed[idx]) continue;
        keep.push_back(idx);
        for (size_t j = i + 1; j < order.size(); ++j) {
            int idx2 = order[j];
            if (removed[idx2]) continue;

            float inter = (boxes[idx] & boxes[idx2]).area();
            float union_ = boxes[idx].area() + boxes[idx2].area() - inter;
            float iou = union_ > 0 ? inter / union_ : 0.f;

            if (iou >= iou_thresh) {
                removed[idx2] = 1;
            }
        }
    }
    return keep;
}

std::vector<Detection> YoloDetector::detect(const cv::Mat& bgr_image) {
    auto pre = preprocess(bgr_image);
    cv::Mat img = pre.processed; // RGB

    // NCHW float32
    std::vector<float> input;
    input.resize(static_cast<size_t>(cfg_.input_w) * cfg_.input_h * 3);

    // OpenCV 是 HWC；我们需要 NCHW 且归一化到 [0,1]
    int h = img.rows, w = img.cols;
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < h; ++y) {
            const uchar* row_ptr = img.ptr<uchar>(y);
            for (int x = 0; x < w; ++x) {
                float v = row_ptr[3 * x + c] / 255.0f;
                input[c * (h * w) + y * w + x] = v;
            }
        }
    }

    std::array<int64_t, 4> input_shape = {1, 3, cfg_.input_h, cfg_.input_w};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_, input.data(), input.size(), input_shape.data(), input_shape.size());

    auto outputs = session_->Run(Ort::RunOptions{nullptr},
                                 input_names_.data(), &input_tensor, input_names_.size(),
                                 output_names_.data(), output_names_.size());

    // 取第一个输出
    if (outputs.empty()) {
        throw std::runtime_error("No outputs from ONNX Runtime session");
    }
    auto& out0 = outputs[0];

    auto shape = get_tensor_shape(out0);
    const float* out_ptr = out0.GetTensorData<float>(); // 不要手动释放

    auto dets = postprocess(bgr_image.size(), pre.scale, pre.pad_w, pre.pad_h, out_ptr, shape);
    return dets;
}