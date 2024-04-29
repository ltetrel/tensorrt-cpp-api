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
    std::array<float, 3> mean = {0.485, 0.456, 0.406};
    std::array<float, 3> std = {0.229, 0.224, 0.225};
};

struct ToDtype{
    std::string dtype = "float";
    bool scale = true;
};

struct NMS{
    float threshold = 0.010;
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
    Resize invertResize = {};
    const NMS nms = {0.010, 0.50, 1.0, 1.0};
};

const std::vector<std::string> classLabels = {
    "Scallop",
    "cucumaria_frondosa",
    "cucumaria_frondosa_juv"
};

};
