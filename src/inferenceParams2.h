#pragma once

#include "transforms.h"

// yaml reader

// FileStorage fs("/home/ltetrel/Documents/developments/ifremer/marine-beacon/tensorrt-cpp-api/models/inference_params.yaml", FileStorage::READ);
//     Mat r;
//     fs["data"] >> r;
//     std::cout << r << std::endl;
//     fs.release();

namespace inferenceParams2{

struct ResizeImg{
    // original height and width of the network
    cv::Size tgtSize;
    ResizeMethod method = ResizeMethod::scale; // defines resizing method ["maintain_ar" or "scale"]
};

struct ConvertColor{
    ColorModel model = ColorModel::RGB;
};

struct ToDtypeImg{
    Precision dtype = Precision::FP32;
    bool scale = true;
};

struct NormalizeImg{
    // Carefull with normalization parameters:
    // https://discuss.pytorch.org/t/discussion-why-normalise-according-to-imagenet-mean-and-std-dev-for-transfer-learning/115670/7
    cv::Vec3f mean = {0.485, 0.456, 0.406};
    cv::Vec3f std = {0.229, 0.224, 0.225};
};

struct FilterBoxes{
    float thresh = 0.1;  // probability for an object to exists (yolo objectness)
};

struct ConvertBox{
    BoxFormat srcFmt = BoxFormat::xyxy;  // can be either "xyxy", "cxcywh" or "xywh"
};

struct RescaleBox{
    cv::Vec2f offset = {0.f, 0.f};
    cv::Vec2f scale = {1.f, 1.f};
};

struct ResizeBox{
    cv::Size inpSize;
    cv::Size tgtSize;
    ResizeMethod method = ResizeMethod::scale;
};

struct NMS{
    float maxOverlap = 0.50;
    float nmsScaleFactor = 1.0;
    float outputScaleFactor = 1.0;
};

struct ImagePreTransforms{
    const ResizeImg resize = {{640, 640}, ResizeMethod::maintain_ar};
    const ConvertColor convertColor = {ColorModel::BGR};
    const ToDtypeImg toDtype = {Precision::FP32, true};
    const NormalizeImg normalize = {{0.f, 0.f, 0.f}, {1.f, 1.f, 1.f}};
};

struct TargetPostTransforms{
    const FilterBoxes filterBoxes = {0.01};
    const ConvertBox convert = {BoxFormat::cxcywh};
    const RescaleBox rescale = {{0.f, 0.f}, {640.f, 640.f}};
    ResizeBox resize = {{640, 640}, {}, {}};
    const NMS nms = {0.5, 1.0, 1.0};
};

enum class ModelType{
    darknet,
    netharn,
};

struct Model{
    ModelType modelType = ModelType::darknet;
};

const std::vector<std::string> classLabels = {
    "Scallop",
    "cucumaria_frondosa",
    "cucumaria_frondosa_juv"
};

const std::vector<std::vector<float>> colors = {
    {1.f, 0.f, 0.f},
    {0.f, 1.f, 0.f},
    {0.f, 0.f, 1.f}
};

};
