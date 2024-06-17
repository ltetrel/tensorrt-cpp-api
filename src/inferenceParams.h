#pragma once

namespace inferenceParams{

struct Resize{
    // defines resizing method ["maintain_ar" or "scale"]
    std::string method = "scale";
    // original height and width of the network
    size_t height = 640;
    size_t width = 640;
};


struct Normalize{
    // Carefull with normalization parameters:
    // https://discuss.pytorch.org/t/discussion-why-normalise-according-to-imagenet-mean-and-std-dev-for-transfer-learning/115670/7
    std::vector<float> mean = {0.485, 0.456, 0.406};
    std::vector<float> std = {0.229, 0.224, 0.225};
};

struct ToDtype{
    std::string dtype = "float";
    bool scale = true;
};

struct BoxConvert{
    // can be either "xyxy", "cxcywh" or "xywh"
    std::string srcFmt = "xyxy";
    std::string tgtFmt = "xywh";
};

struct Threshold{
    float prob = 0.2; // class probability
    float conf = 0.1; // probability for an object to exists (yolo objectness)
};

struct NMS{
    float maxOverlap = 0.50;
    float nmsScaleFactor = 1.0;
    float outputScaleFactor = 1.0;
};

struct ImagePreTransforms{
    const Resize resize = {"maintain_ar", 640, 640};
    const ToDtype toDtype = {"float", true};
    const Normalize normalize = {{0.f, 0.f, 0.f}, {1.f, 1.f, 1.f}};
};

struct TargetPostTransforms{
    const BoxConvert boxConvert = {"cxcywh"};
    const Normalize invertNormalize = {{0.f, 0.f}, {640.f, 640.f}};
    Resize invertResize = {};
    const Threshold threshold = {0.3, 0.010};
    const NMS nms = {0.5, 1.0, 1.0};
};

struct Model{
    std::string type = "darknet";
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
