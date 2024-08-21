#pragma once

#include <filesystem>
#include <vector>
#include <unordered_map>
#include <string>
#include <opencv2/opencv.hpp>
#include <memory>

#include "boundingBox.h"
#include "transforms.h"


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

enum class ModelType{
    darknet,
    netharn,
};

struct Model{
    ModelType type = ModelType::darknet;
    //TODO: engineType
};

// enum class ModelType{
//     yolov4,
//     yolov7,
//     cascadeFRCNN
// };

// enum class ModelBackend{
//     darknet,
//     netharn,
    
// };

// struct Model{
//     ModelType type = ModelType::yolov4;
//     ModelBackend backend = ModelBackend::darknet;
//     //TODO: engineType
// };

struct TargetPostTransforms{
    Transforms::FilterBBoxes filter;
    Transforms::ConvertBBox convert;
    Transforms::RescaleBBox rescale;
    Transforms::ResizeBBox resize;
    Transforms::NMSBBoxes nms;
};

using ITImgTransformPtr = std::shared_ptr<Transforms::ITransform<cv::cuda::GpuMat>>;
using ITBBoxTransformPtr = std::shared_ptr<Transforms::ITransform<BoundingBox>>;

class CfgParser{
public:
    CfgParser() = default;
    CfgParser(std::filesystem::path cfgPath);

    std::vector<ITImgTransformPtr> aImagePreTransforms;
    TargetPostTransforms aTargetPostTransforms;
    Model aModel;
    BoxFormat aBBoxSrcFormat;
    std::vector<std::string> aLabels = cocoLabels;
    std::vector<std::vector<float>> aColors = cocoColors;
private:
    std::vector<ITImgTransformPtr> mParsePreProcessing(cv::FileStorage inputFs);
    TargetPostTransforms mParsePostProcessing(cv::FileStorage inputFs);
    BoxFormat mParseBBoxFormat(cv::FileStorage inputFs);
    Model mParseModel(cv::FileStorage inputFs);
    std::vector<std::string> mParseLabels(cv::FileStorage inputFs);
    std::vector<std::vector<float>> mParseColors(cv::FileStorage inputFs);
};
