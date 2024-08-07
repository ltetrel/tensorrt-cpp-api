#pragma once

#include <filesystem>
#include <vector>
#include <unordered_map>
#include <string>
#include <opencv2/opencv.hpp>

#include "transforms.h"


enum class ModelType{
    darknet,
    netharn,
};

struct Model{
    ModelType type = ModelType::darknet;
};

struct ImagePreTransforms{
    ConvertColor convertColor = {ColorModel::BGR};
    ResizeImg resize = {{640, 640}, ResizeMethod::maintain_ar};
    CastImg cast = {Precision::FP32, true};
    NormalizeImg normalize = {{0.f, 0.f, 0.f}, {1.f, 1.f, 1.f}};
};

struct TargetPostTransforms{
    FilterBoxes filterBoxes = {0.01};
    ConvertBox convert = {BoxFormat::cxcywh};
    RescaleBox rescale = {{0.f, 0.f}, {640.f, 640.f}};
    ResizeBox resize = {{}, ResizeMethod::maintain_ar};
    NMS nms = {0.5, 1.0, 1.0};
};

struct TransformValueMapper{
    std::unordered_map<std::string, ColorModel> const colorModel = {
        {"RGB", ColorModel::RGB},
        {"BGR", ColorModel::BGR},
        {"GRAY", ColorModel::GRAY}
    };
    std::unordered_map<std::string, ResizeMethod> const resizeMethod = {
        {"maintain_ar", ResizeMethod::maintain_ar},
        {"scale", ResizeMethod::scale}
    };
    std::unordered_map<std::string, Precision> const imageType = {
        {"int", Precision::INT8},
        {"float", Precision::FP32}
    };
    std::unordered_map<std::string, BoxFormat> const boxFormat = {
        {"xyxy", BoxFormat::xyxy},
        {"cxcywh", BoxFormat::cxcywh}
    };
    std::unordered_map<std::string, ModelType> const modelType = {
        {"darknet", ModelType::darknet},
        {"netharn", ModelType::netharn}
    };
};

namespace{
    std::vector<std::string> cocoLabels = {
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic-light",
        "fire-hydrant",
        "stop-sign",
        "parking-meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports-ball",
        "kite",
        "baseball-bat",
        "baseball-glove",
        "skateboard",
        "surfboard",
        "tennis-racket",
        "bottle",
        "wine-glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot-dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted-plant",
        "bed",
        "dining-table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell-phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy-bear",
        "hair-drier",
        "toothbrush"
    };
    std::vector<std::vector<float>> cocoColors = {
        {0.86f, 0.08f, 0.24f},
        {0.47f, 0.04f, 0.13f},
        {0.00f, 0.00f, 0.56f},
        {0.00f, 0.00f, 0.90f},
        {0.42f, 0.00f, 0.89f},
        {0.00f, 0.24f, 0.39f},
        {0.00f, 0.31f, 0.39f},
        {0.00f, 0.00f, 0.27f},
        {0.00f, 0.00f, 0.75f},
        {0.98f, 0.67f, 0.12f},
        {0.39f, 0.67f, 0.12f},
        {0.86f, 0.86f, 0.00f},
        {0.69f, 0.45f, 0.69f},
        {0.98f, 0.00f, 0.12f},
        {0.65f, 0.16f, 0.16f},
        {1.00f, 0.30f, 1.00f},
        {0.00f, 0.89f, 0.99f},
        {0.71f, 0.71f, 1.00f},
        {0.00f, 0.32f, 0.00f},
        {0.47f, 0.65f, 0.62f},
        {0.43f, 0.30f, 0.00f},
        {0.68f, 0.22f, 1.00f},
        {0.78f, 0.39f, 0.00f},
        {0.28f, 0.00f, 0.46f},
        {1.00f, 0.70f, 0.94f},
        {0.00f, 0.49f, 0.36f},
        {0.82f, 0.00f, 0.59f},
        {0.74f, 0.82f, 0.71f},
        {0.00f, 0.86f, 0.69f},
        {1.00f, 0.39f, 0.64f},
        {0.36f, 0.00f, 0.29f},
        {0.52f, 0.51f, 1.00f},
        {0.31f, 0.71f, 1.00f},
        {0.00f, 0.89f, 0.00f},
        {0.68f, 1.00f, 0.95f},
        {0.18f, 0.35f, 1.00f},
        {0.53f, 0.53f, 0.40f},
        {0.57f, 0.58f, 0.68f},
        {1.00f, 0.82f, 0.73f},
        {0.77f, 0.89f, 1.00f},
        {0.67f, 0.53f, 0.00f},
        {0.43f, 0.25f, 0.21f},
        {0.81f, 0.54f, 1.00f},
        {0.59f, 0.00f, 0.37f},
        {0.04f, 0.31f, 0.24f},
        {0.33f, 0.41f, 0.20f},
        {0.29f, 0.25f, 0.41f},
        {0.65f, 0.77f, 0.40f},
        {0.82f, 0.76f, 0.82f},
        {1.00f, 0.43f, 0.25f},
        {0.00f, 0.56f, 0.58f},
        {0.70f, 0.00f, 0.76f},
        {0.82f, 0.39f, 0.42f},
        {0.02f, 0.47f, 0.00f},
        {0.89f, 1.00f, 0.80f},
        {0.58f, 0.73f, 0.82f},
        {0.60f, 0.27f, 0.00f},
        {0.01f, 0.37f, 0.63f},
        {0.64f, 1.00f, 0.00f},
        {0.47f, 0.00f, 0.67f},
        {0.00f, 0.71f, 0.78f},
        {0.00f, 0.65f, 0.47f},
        {0.72f, 0.51f, 0.35f},
        {0.37f, 0.13f, 0.00f},
        {0.51f, 0.45f, 0.53f},
        {0.43f, 0.51f, 0.52f},
        {0.65f, 0.29f, 0.46f},
        {0.86f, 0.56f, 0.73f},
        {0.31f, 0.82f, 0.45f},
        {0.70f, 0.35f, 0.24f},
        {0.25f, 0.27f, 0.06f},
        {0.50f, 0.65f, 0.45f},
        {0.23f, 0.41f, 0.42f},
        {0.56f, 0.42f, 0.18f},
        {0.77f, 0.67f, 0.00f},
        {0.37f, 0.21f, 0.31f},
        {0.50f, 0.30f, 1.00f},
        {0.79f, 0.22f, 0.00f},
        {0.96f, 0.00f, 0.48f},
        {0.75f, 0.64f, 0.82f}
    };
}

class CfgParser{
public:
    CfgParser() = default;
    CfgParser(std::filesystem::path cfgPath);
    void mSetImgSize(const cv::Size& size);
    const cv::Size mGetImgSize();

    ImagePreTransforms aImagePreTransforms;
    TargetPostTransforms aTargetPostTransforms;
    Model aModel;
    std::vector<std::string> aLabels = cocoLabels;
    std::vector<std::vector<float>> aColors = cocoColors;
private:
    ImagePreTransforms mParsePreProcessing(cv::FileStorage inputFs);
    TargetPostTransforms mParsePostProcessing(cv::FileStorage inputFs);
    Model mParseModel(cv::FileStorage inputFs);
    std::vector<std::string> mParseLabels(cv::FileStorage inputFs);
    std::vector<std::vector<float>> mParseColors(cv::FileStorage inputFs);
};
